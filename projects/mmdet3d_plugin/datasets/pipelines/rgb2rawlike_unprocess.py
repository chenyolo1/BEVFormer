import hashlib
import numpy as np
import os
import cv2

try:
    import torch
except Exception:
    torch = None

try:
    # mmdet2.x
    from mmdet.datasets.builder import PIPELINES
except Exception:
    # mmdet3d / 兼容
    from mmdet3d.datasets.builder import PIPELINES

# def linear_to_srgb(x01: np.ndarray) -> np.ndarray:
#     a = 0.055
#     x01 = np.clip(x01, 0.0, 1.0)
#     return np.where(x01 <= 0.0031308, 12.92 * x01, (1 + a) * (x01 ** (1/2.4)) - a)



@PIPELINES.register_module()
class RGB2RAWLikeUnprocessWoMosaic:
    """
    Use AdaptiveISP's unprocess_wo_mosaic to convert sRGB (JPEG) to RAW-like (no mosaic)
    online inside OpenMMLab pipeline.

    Key points:
      - BEVFormer multi-view: results['img'] is a list of np.ndarray
      - mmcv通常读入为BGR；unprocess一般按RGB通道语义写，因此默认先BGR->RGB
      - unprocess函数可能带随机性；提供 deterministic=True 让每张图固定种子（利于val/test一致）
    """

    def __init__(
        self,
        input_is_bgr=True,
        clip=True,
        deterministic=False,
        base_seed=0,
        keep_meta=False,
        eps=1e-6,
        import_mode="third_party",  # "third_party" or "vendored"
        backend="numpy", # "numpy" or "torch"
        device="cuda",
        allow_cuda_in_worker=False,
    ):
        self.input_is_bgr = input_is_bgr
        self.clip = clip
        self.deterministic = deterministic
        self.base_seed = int(base_seed)
        self.keep_meta = keep_meta
        self.eps = float(eps)
        self.import_mode = import_mode
        self.backend = backend
        self.device = device
        self.allow_cuda_in_worker = allow_cuda_in_worker

        # Import unprocess_wo_mosaic
        if import_mode == "third_party":
            # 依赖：export PYTHONPATH=.../third_party/AdaptiveISP:$PYTHONPATH
            from isp.unprocess_np import unprocess_wo_mosaic  # noqa: F401
        elif import_mode == "vendored":
            # 你把 unprocess_np.py 放到 projects/mmdet3d_plugin/bevformer/isp/ 下的写法
            from projects.mmdet3d_plugin.bevformer.isp.unprocess_np import unprocess_wo_mosaic  # noqa: E501,F401
        else:
            raise ValueError(f"Unknown import_mode={import_mode}")

        self.unprocess_wo_mosaic = unprocess_wo_mosaic

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
        # mmcv to_float32=True 通常仍是 0~255
        if x.max() > 1.5:
            x = x / 255.0
        return x

    def _stable_seed_from_filename(self, filename: str) -> int:
        # 使用 md5 做稳定 hash（避免 python hash 不稳定）
        h = hashlib.md5(filename.encode("utf-8")).hexdigest()
        return (int(h[:8], 16) ^ self.base_seed) & 0x7fffffff

    def _call_unprocess(self, rgb01: np.ndarray, seed: int):
        if not self.deterministic:
            return self.unprocess_wo_mosaic(rgb01)

        # 若 unprocess 内部使用 np.random，全局设种子可实现“逐图确定性”
        st = np.random.get_state()
        np.random.seed(seed)
        try:
            out = self.unprocess_wo_mosaic(rgb01)
        finally:
            np.random.set_state(st)
        return out

    def _random_ccm_np(self, rng: np.random.RandomState) -> np.ndarray:
        xyz2cams = np.array([
            [[1.0234, -0.2969, -0.2266],
             [-0.5625, 1.6328, -0.0469],
             [-0.0703, 0.2188, 0.6406]],
            [[0.4913, -0.0541, -0.0202],
             [-0.613, 1.3513, 0.2906],
             [-0.1564, 0.2151, 0.7183]],
            [[0.838, -0.263, -0.0639],
             [-0.2887, 1.0725, 0.2496],
             [-0.0627, 0.1427, 0.5438]],
            [[0.6596, -0.2079, -0.0562],
             [-0.4782, 1.3016, 0.1933],
             [-0.097, 0.1581, 0.5181]],
        ], dtype=np.float32)
        weights = rng.uniform(1e-8, 1e8, size=(len(xyz2cams), 1, 1)).astype(np.float32)
        xyz2cam = np.sum(xyz2cams * weights, axis=0) / np.sum(weights, axis=0)
        rgb2xyz = np.array(
            [[0.4124564, 0.3575761, 0.1804375],
             [0.2126729, 0.7151522, 0.0721750],
             [0.0193339, 0.1191920, 0.9503041]],
            dtype=np.float32,
        )
        rgb2cam = np.matmul(xyz2cam, rgb2xyz)
        rgb2cam = rgb2cam / np.sum(rgb2cam, axis=-1, keepdims=True)
        return rgb2cam.astype(np.float32)

    def _random_gains_np(self, rng: np.random.RandomState):
        rgb_gain = 1.0 / rng.normal(0.8, 0.1)
        red_gain = rng.uniform(1.9, 2.4)
        blue_gain = rng.uniform(1.5, 1.9)
        return float(rgb_gain), float(red_gain), float(blue_gain)

    def _unprocess_batch_torch(self, rgb01: np.ndarray, seeds, add_noise=False):
        device = self._get_device()
        if device is None:
            raise RuntimeError("backend is not torch")

        rgb2cam_list = []
        rgb_gain_list = []
        red_gain_list = []
        blue_gain_list = []
        noise_shot_list = []
        noise_read_list = []

        for seed in seeds:
            rng = np.random.RandomState(seed) if self.deterministic else np.random
            rgb2cam_list.append(self._random_ccm_np(rng))
            rgb_gain, red_gain, blue_gain = self._random_gains_np(rng)
            rgb_gain_list.append(rgb_gain)
            red_gain_list.append(red_gain)
            blue_gain_list.append(blue_gain)
            if add_noise:
                shot = rng.uniform(0.0001, 0.012)
                log_shot = np.log(shot)
                log_read = 2.18 * log_shot + 1.20 + rng.normal(0, 0.26)
                noise_shot_list.append(shot)
                noise_read_list.append(np.exp(log_read))

        rgb2cam = torch.from_numpy(np.stack(rgb2cam_list)).to(device=device)
        rgb_gain = torch.tensor(rgb_gain_list, device=device, dtype=torch.float32)
        red_gain = torch.tensor(red_gain_list, device=device, dtype=torch.float32)
        blue_gain = torch.tensor(blue_gain_list, device=device, dtype=torch.float32)

        image = torch.from_numpy(rgb01).to(device=device, dtype=torch.float32)
        image = image * 0.9
        image = torch.clamp(image, 0.0, 1.0)
        image = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)
        image = torch.clamp(image, min=1e-8) ** 2.2

        image = image.reshape(image.shape[0], -1, 3)
        image = torch.einsum("nvc,nkc->nvk", image, rgb2cam)
        image = image.reshape(rgb01.shape)

        gains = torch.stack(
            (1.0 / red_gain, torch.ones_like(red_gain), 1.0 / blue_gain),
            dim=-1,
        ) / rgb_gain[:, None]
        gains = gains[:, None, None, :]
        gray = torch.mean(image, dim=-1, keepdim=True)
        inflection = 0.9
        mask = torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)
        mask = mask ** 2.0
        safe_gains = torch.maximum(mask + (1.0 - mask) * gains, gains)
        image = image * safe_gains

        image = torch.clamp(image, 0.0, 1.0)

        if add_noise:
            shot = torch.tensor(noise_shot_list, device=device, dtype=torch.float32)
            read = torch.tensor(noise_read_list, device=device, dtype=torch.float32)
            variance = image * shot[:, None, None, None] + read[:, None, None, None]
            noise = torch.randn_like(image) * torch.sqrt(variance)
            image = torch.clamp(image + noise, 0.0, 1.0)

        return image

    def __call__(self, results):
        imgs = results["img"]
        assert isinstance(imgs, list), "BEVFormer multi-view expects results['img'] to be a list."

        # 常见字段：results['img_filename'] 是 list[str]
        filenames = results.get("img_filename", None)
        if filenames is None:
            # 兼容一些数据集字段命名
            filenames = results.get("filename", None)

        seeds = []
        x01_list = []

        for i, im in enumerate(imgs):
            x01 = self._to_float01(im)
            if self.input_is_bgr:
                x01 = x01[..., ::-1]
            x01_list.append(x01)

            # seed（若拿不到文件名，就退化为 i）
            if isinstance(filenames, list) and i < len(filenames):
                seed = self._stable_seed_from_filename(str(filenames[i]))
            elif isinstance(filenames, str):
                seed = self._stable_seed_from_filename(filenames + f"#{i}")
            else:
                seed = (self.base_seed + i) & 0x7fffffff
            seeds.append(seed)

        out_imgs = []
        metas = [] if self.keep_meta else None

        if self.backend == "torch":
            batch = np.stack(x01_list, axis=0)
            rawlike = self._unprocess_batch_torch(batch, seeds)
            if self.clip:
                rawlike = torch.clamp(rawlike, 0.0, 1.0)
            rawlike = torch.maximum(rawlike, torch.tensor(self.eps, device=rawlike.device))
            rawlike = rawlike.detach().cpu().numpy()
            out_imgs.extend(list(rawlike))
        else:
            for x01, seed in zip(x01_list, seeds):
                ret = self._call_unprocess(x01, seed)
                if isinstance(ret, (tuple, list)):
                    rawlike = ret[0]
                    meta = ret[1:] if len(ret) > 1 else None
                else:
                    rawlike = ret
                    meta = None

                rawlike = np.asarray(rawlike, dtype=np.float32)
                if self.clip:
                    rawlike = np.clip(rawlike, 0.0, 1.0)
                rawlike = np.maximum(rawlike, self.eps)
                out_imgs.append(rawlike)
                if self.keep_meta:
                    metas.append(meta)

        results["img"] = out_imgs
        if self.keep_meta:
            results["rawlike_meta"] = metas

        # 注意：现在 results['img'] 已经是 RGB（如果 input_is_bgr=True）
        return results
