import os

import numpy as np

try:
    # mmdet2.x
    from mmdet.datasets.builder import PIPELINES
except Exception:
    # mmdet3d / 兼容
    from mmdet3d.datasets.builder import PIPELINES

try:
    import torch
except Exception:
    torch = None

try:
    from PIL import Image
except Exception as exc:
    raise ImportError("Pillow is required for RAWLikeToSRGBISP debug image saving.") from exc

try:
    from projects.mmdet3d_plugin.bevformer.isp.unprocess_np import apply_ccm
except Exception as exc:
    raise ImportError(
        "Failed to import apply_ccm from unprocess_np; ensure the file exists in "
        "projects/mmdet3d_plugin/bevformer/isp/"
    ) from exc


def linear_to_srgb(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    a = 0.055
    return np.where(x01 <= 0.0031308, 12.92 * x01, (1 + a) * (x01 ** (1 / 2.4)) - a)


def smoothstep(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return x01 * x01 * (3.0 - 2.0 * x01)


@PIPELINES.register_module()
class RAWLikeToSRGBISP:
    """
    Forward ISP for RAW-like (linear RGB) -> sRGB.

    Expected input:
      - results['img']: list of RAW-like images (H, W, 3), float in [0, 1]
      - results['rawlike_meta']: list of metadata from RGB2RAWLikeUnprocessWoMosaic
    """

    def __init__(
        self,
        input_is_bgr=False,
        output_is_bgr=False,
        backend="torch",
        device="cuda",
        allow_cuda_in_worker=False,
        output_backend="torch",
        output_range="01",
        debug_save_dir=None,
        debug_save_count=0,
        debug_prefix="rawlike_to_srgb",
        apply_tone_mapping=True,
        apply_brightness_comp=True,
        clip=True,
        eps=1e-6,
    ):
        self.input_is_bgr = input_is_bgr
        self.output_is_bgr = output_is_bgr
        self.backend = backend
        self.device = device
        self.allow_cuda_in_worker = allow_cuda_in_worker
        self.output_backend = output_backend
        self.output_range = output_range
        self.debug_save_dir = debug_save_dir
        self.debug_save_count = int(debug_save_count)
        self.debug_prefix = debug_prefix
        self.apply_tone_mapping = apply_tone_mapping
        self.apply_brightness_comp = apply_brightness_comp
        self.clip = clip
        self.eps = float(eps)
        if self.output_range not in {"01", "255"}:
            raise ValueError("output_range must be '01' or '255'.")
        if self.debug_save_count < 0:
            raise ValueError("debug_save_count must be >= 0.")
        self._debug_saved = 0

    def _get_device(self):
        if self.backend != "torch":
            return None
        if torch is None:
            raise RuntimeError("torch is required for backend='torch'.")
        if self.device.startswith("cuda") and not self.allow_cuda_in_worker:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                return torch.device("cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)

    def _to_float01(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        return x

    def _to_float01_torch(self, img) -> torch.Tensor:
        if not torch.is_tensor(img):
            x = torch.from_numpy(np.asarray(img))
        else:
            x = img
        x = x.to(dtype=torch.float32)
        if torch.max(x) > 1.5:
            x = x / 255.0
        return x

    def _apply_gains(self, image: np.ndarray, meta: dict) -> np.ndarray:
        rgb_gain = float(meta.get("rgb_gain", 1.0))
        red_gain = float(meta.get("red_gain", 1.0))
        blue_gain = float(meta.get("blue_gain", 1.0))
        gains = np.array([red_gain, 1.0, blue_gain], dtype=np.float32) * rgb_gain
        return image * gains[None, None, :]

    def _apply_gains_torch(self, image: torch.Tensor, meta: dict) -> torch.Tensor:
        rgb_gain = float(meta.get("rgb_gain", 1.0))
        red_gain = float(meta.get("red_gain", 1.0))
        blue_gain = float(meta.get("blue_gain", 1.0))
        gains = torch.tensor([red_gain, 1.0, blue_gain], device=image.device, dtype=image.dtype)
        gains = gains * rgb_gain
        return image * gains.view(1, 1, 3)

    def _apply_brightness(self, image: np.ndarray, meta: dict) -> np.ndarray:
        if not self.apply_brightness_comp:
            return image
        gain = float(meta.get("gain", 1.0))
        if gain <= 0:
            return image
        return image / gain

    def _apply_brightness_torch(self, image: torch.Tensor, meta: dict) -> torch.Tensor:
        if not self.apply_brightness_comp:
            return image
        gain = float(meta.get("gain", 1.0))
        if gain <= 0:
            return image
        return image / gain

    def _apply_ccm(self, image: np.ndarray, meta: dict) -> np.ndarray:
        cam2rgb = meta.get("cam2rgb", None)
        if cam2rgb is None:
            cam2rgb = np.eye(3, dtype=np.float32)
        cam2rgb = np.asarray(cam2rgb, dtype=np.float32)
        if cam2rgb.shape != (3, 3):
            raise ValueError("cam2rgb must be a 3x3 matrix.")
        return apply_ccm(image, cam2rgb)

    def _apply_ccm_torch(self, image: torch.Tensor, meta: dict) -> torch.Tensor:
        cam2rgb = meta.get("cam2rgb", None)
        if cam2rgb is None:
            cam2rgb = np.eye(3, dtype=np.float32)
        cam2rgb = np.asarray(cam2rgb, dtype=np.float32)
        if cam2rgb.shape != (3, 3):
            raise ValueError("cam2rgb must be a 3x3 matrix.")
        cam2rgb = torch.from_numpy(cam2rgb).to(device=image.device, dtype=image.dtype)
        shape = image.shape
        image = image.reshape(-1, 3)
        image = torch.matmul(image, cam2rgb.T)
        return image.reshape(shape)

    def _linear_to_srgb_torch(self, x01: torch.Tensor) -> torch.Tensor:
        x01 = torch.clamp(x01, 0.0, 1.0)
        a = 0.055
        return torch.where(
            x01 <= 0.0031308,
            12.92 * x01,
            (1 + a) * torch.pow(x01, 1 / 2.4) - a,
        )

    def _smoothstep_torch(self, x01: torch.Tensor) -> torch.Tensor:
        x01 = torch.clamp(x01, 0.0, 1.0)
        return x01 * x01 * (3.0 - 2.0 * x01)

    def _maybe_save_debug(self, image, index, stage):
        if self.debug_save_count <= 0 or self.debug_save_dir is None:
            return
        if self._debug_saved >= self.debug_save_count:
            return
        os.makedirs(self.debug_save_dir, exist_ok=True)
        if torch is not None and torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        img = np.asarray(image, dtype=np.float32)
        if self.output_range == "01":
            img = img * 255.0
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        if self.output_is_bgr:
            img = img[..., ::-1]
        filename = f"{self.debug_prefix}_{self._debug_saved:03d}_{stage}_{index}.png"
        Image.fromarray(img).save(os.path.join(self.debug_save_dir, filename))
        self._debug_saved += 1

    def __call__(self, results):
        imgs = results["img"]
        assert isinstance(imgs, list), "BEVFormer multi-view expects results['img'] to be a list."

        metas = results.get("rawlike_meta", None)
        if metas is None:
            raise KeyError("rawlike_meta is required for RAWLikeToSRGBISP. Set keep_meta=True.")
        if isinstance(metas, dict):
            metas = [metas] * len(imgs)
        if isinstance(metas, list) and len(metas) == 0:
            metas = [{} for _ in range(len(imgs))]
        if len(metas) != len(imgs):
            if len(metas) == 1:
                metas = list(metas) * len(imgs)
            else:
                raise ValueError(
                    f"rawlike_meta length ({len(metas)}) must match img length ({len(imgs)})."
                )
        metas = [meta if meta is not None else {} for meta in metas]

        if self.backend == "torch":
            device = self._get_device()
            if device is None:
                raise RuntimeError("backend is not torch")
            out_imgs = []
            for img, meta in zip(imgs, metas):
                x01 = self._to_float01_torch(img).to(device=device)
                if self.input_is_bgr:
                    x01 = x01[..., [2, 1, 0]]

                x01 = self._apply_brightness_torch(x01, meta)
                x01 = self._apply_gains_torch(x01, meta)
                x01 = self._apply_ccm_torch(x01, meta)

                if self.apply_tone_mapping:
                    x01 = self._smoothstep_torch(x01)

                x01 = self._linear_to_srgb_torch(x01)

                if self.clip:
                    x01 = torch.clamp(x01, 0.0, 1.0)
                x01 = torch.maximum(x01, torch.tensor(self.eps, device=x01.device, dtype=x01.dtype))
                if self.output_range == "255":
                    x01 = x01 * 255.0
                    if self.clip:
                        x01 = torch.clamp(x01, 0.0, 255.0)

                if self.output_is_bgr:
                    x01 = x01[..., [2, 1, 0]]

                self._maybe_save_debug(x01, index=len(out_imgs), stage="torch")

                if self.output_backend == "numpy":
                    out_imgs.append(x01.detach().cpu().numpy().astype(np.float32))
                else:
                    out_imgs.append(x01)
        else:
            out_imgs = []
            for img, meta in zip(imgs, metas):
                x01 = self._to_float01(img)
                if self.input_is_bgr:
                    x01 = x01[..., ::-1]

                x01 = self._apply_brightness(x01, meta)
                x01 = self._apply_gains(x01, meta)
                x01 = self._apply_ccm(x01, meta)

                if self.apply_tone_mapping:
                    x01 = smoothstep(x01)

                x01 = linear_to_srgb(x01)

                if self.clip:
                    x01 = np.clip(x01, 0.0, 1.0)
                x01 = np.maximum(x01, self.eps)
                if self.output_range == "255":
                    x01 = x01 * 255.0
                    if self.clip:
                        x01 = np.clip(x01, 0.0, 255.0)

                if self.output_is_bgr:
                    x01 = x01[..., ::-1]
                self._maybe_save_debug(x01, index=len(out_imgs), stage="numpy")
                out_imgs.append(x01.astype(np.float32))

        results["img"] = out_imgs
        return results
