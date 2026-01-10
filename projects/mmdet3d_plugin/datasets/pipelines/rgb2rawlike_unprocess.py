import hashlib
import numpy as np
import os
import cv2

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
    ):
        self.input_is_bgr = input_is_bgr
        self.clip = clip
        self.deterministic = deterministic
        self.base_seed = int(base_seed)
        self.keep_meta = keep_meta
        self.eps = float(eps)
        self.import_mode = import_mode

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

    def __call__(self, results):
        imgs = results["img"]
        assert isinstance(imgs, list), "BEVFormer multi-view expects results['img'] to be a list."

        # 常见字段：results['img_filename'] 是 list[str]
        filenames = results.get("img_filename", None)
        if filenames is None:
            # 兼容一些数据集字段命名
            filenames = results.get("filename", None)

        out_imgs = []
        metas = [] if self.keep_meta else None

        for i, im in enumerate(imgs):
            x01 = self._to_float01(im)

            # if i == 0:  # 只看第一个view
            #     print(f"[DBG] before BGR2RGB: dtype={x01.dtype}, min={x01.min():.4f}, max={x01.max():.4f}, mean={x01.mean():.4f}")

            # BGR -> RGB（强烈建议，避免通道语义错位）
            if self.input_is_bgr:
                x01 = x01[..., ::-1]

            # seed（若拿不到文件名，就退化为 i）
            if isinstance(filenames, list) and i < len(filenames):
                seed = self._stable_seed_from_filename(str(filenames[i]))
            elif isinstance(filenames, str):
                seed = self._stable_seed_from_filename(filenames + f"#{i}")
            else:
                seed = (self.base_seed + i) & 0x7fffffff

            ret = self._call_unprocess(x01, seed)

            # 兼容两类返回：1) 直接返回img；2) 返回(img, meta)或(img, meta, ...)
            if isinstance(ret, (tuple, list)):
                rawlike = ret[0]
                meta = ret[1:] if len(ret) > 1 else None
            else:
                rawlike = ret
                meta = None

            rawlike = np.asarray(rawlike, dtype=np.float32)

            # if i == 0:
            #     print(f"[DBG] after unprocess: dtype={rawlike.dtype}, min={rawlike.min():.4f}, max={rawlike.max():.4f}, mean={rawlike.mean():.4f}")
            #     print(f"[DBG] after unprocess per-channel mean: {rawlike.reshape(-1,3).mean(axis=0)}")

            if self.clip:
                rawlike = np.clip(rawlike, 0.0, 1.0)

            # # ---------------- DEBUG SAVE (only once) ----------------
            # # 只在第一个样本的第一个 view 保存三张对比图，避免刷屏/频繁写盘
            # if i == 0 and (not hasattr(self, "_saved_dbg")):
            #     self._saved_dbg = True
            #     os.makedirs("vis_dbg", exist_ok=True)

            #     # 0) 原始输入图（im 通常是BGR, 0~255）
            #     im_bgr = im.copy()
            #     if im_bgr.dtype != np.uint8:
            #         im_bgr = np.clip(im_bgr, 0, 255).astype(np.uint8)
            #     cv2.imwrite("vis_dbg/0_input_bgr.jpg", im_bgr)

            #     # 1) RAW-like 线性域直接可视化（rawlike 是 RGB, [0,1]）
            #     raw01 = np.clip(rawlike, 0.0, 1.0).astype(np.float32)
            #     raw_linear_bgr = (raw01[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)  # RGB->BGR for cv2
            #     cv2.imwrite("vis_dbg/1_rawlike_linear_bgr.jpg", raw_linear_bgr)

            #     # 2) RAW-like 转回 sRGB 再可视化（更接近“正常观感”）
            #     srgb01 = linear_to_srgb(raw01)
            #     srgb_bgr = (srgb01[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
            #     cv2.imwrite("vis_dbg/2_rawlike_to_srgb_bgr.jpg", srgb_bgr)

            #     print("[DBG] Saved visualization images to ./vis_dbg/")
            # # --------------------------------------------------------

            rawlike = np.maximum(rawlike, self.eps)

            # if i == 0:
            #     print(f"[DBG] after clip/eps: min={rawlike.min():.6f}, max={rawlike.max():.6f}, mean={rawlike.mean():.6f}")
            #     print(f"[DBG] has_nan={np.isnan(rawlike).any()}, has_inf={np.isinf(rawlike).any()}")

            out_imgs.append(rawlike)
            if self.keep_meta:
                metas.append(meta)

        results["img"] = out_imgs
        if self.keep_meta:
            results["rawlike_meta"] = metas

        # 注意：现在 results['img'] 已经是 RGB（如果 input_is_bgr=True）
        return results
