"""SAM2 per-pixel segmentation -> seg-label map for the PointNet cloud (POINTCLOUD_EVAL.md).

Mirrors the SAM2 usage in ``orbbec/orbbec_segment_pointclouds.py`` (image predictor,
point-click prompts), generalized to the 3 deployment classes. Produces a (H, W) label
map with values in SEG_LABELS = {robot:0.0, peg:-1.0, hole:+1.0} and NaN for background,
aligned with the image you pass in. The label map must ultimately align with the DEPTH frame:
either segment the frame depth is already in (color image when depth is RealSense-aligned), or
segment the color image and reverse-warp the masks with ``warp_masks_to_depth_frame`` (the FFS
path -- SAM2 is far better on RGB than on raw IR, so we segment color, not the left-IR frame).

The SAM2 dependency is isolated in ``PointCloudSegmenter``; the mask->label composition
(``compose_label_map``) is pure-numpy and unit-tested without SAM2.
"""

from __future__ import annotations

import numpy as np

from diffusion_policy.real_world.pointcloud_builder import SEG_LABELS

# Compose order: later classes overwrite earlier ones where masks overlap.
# robot first so the (usually smaller) peg/hole win on the gripper-object boundary.
_COMPOSE_ORDER = ["robot", "peg", "hole"]


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    """Erode a bool mask by ``px`` (drops the mixed-pixel silhouette ring). cv2 if present."""
    if px <= 0:
        return mask
    try:
        import cv2
        k = np.ones((2 * px + 1,) * 2, np.uint8)
        return cv2.erode(mask.astype(np.uint8), k).astype(bool)
    except Exception:
        # cheap binary erosion fallback (separable min over a (2px+1) window)
        from scipy.ndimage import binary_erosion
        return binary_erosion(mask, iterations=px)


def compose_label_map(masks: dict, shape, erode: int = 3) -> np.ndarray:
    """Combine per-class bool masks into a (H, W) float label map (NaN = background).

    Args:
        masks: {class_name -> (H, W) bool}. Names must be keys of SEG_LABELS.
        shape: (H, W) of the output map.
        erode: erode each mask by N px before stamping (reduces flying-pixel rings).
    """
    label_map = np.full(shape, np.nan, np.float32)
    for name in _COMPOSE_ORDER:
        m = masks.get(name)
        if m is None:
            continue
        m = _erode(np.asarray(m, bool), erode)
        label_map[m] = SEG_LABELS[name]
    return label_map


class PointCloudSegmenter:
    """SAM2 image predictor -> per-class masks -> seg-label map."""

    def __init__(self, checkpoint: str, model_cfg: str, device: str = None):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=self.device))

    def mask(self, rgb: np.ndarray, pos, neg) -> np.ndarray:
        """Best-scoring SAM2 mask for one prompt set (mirrors orbbec segment())."""
        torch = self._torch
        points = np.array(pos + neg, np.float32)
        labels = np.array([1] * len(pos) + [0] * len(neg), np.int32)
        autocast = (torch.autocast(self.device, dtype=torch.bfloat16)
                    if self.device == "cuda" else torch.no_grad())
        with torch.inference_mode(), autocast:
            self.predictor.set_image(rgb)  # SAM2 expects RGB
            masks, scores, _ = self.predictor.predict(
                point_coords=points, point_labels=labels, multimask_output=True)
        return masks[int(np.argmax(scores))].astype(bool)

    def masks(self, rgb: np.ndarray, prompts: dict) -> dict:
        """Per-class bool masks for ``prompts`` (no compose/erode). See ``label_map``."""
        out = {}
        for name, (pos, neg) in prompts.items():
            if not pos:
                continue
            out[name] = self.mask(rgb, pos, list(neg))
        return out

    def label_map(self, rgb: np.ndarray, prompts: dict, erode: int = 3) -> np.ndarray:
        """Segment each class and compose the (H, W) seg-label map.

        Args:
            prompts: {class_name -> (pos_clicks, neg_clicks)} for classes present in the
                scene. Each *_clicks is a list of [x, y]. Omit a class to leave it absent.
        """
        h, w = rgb.shape[:2]
        return compose_label_map(self.masks(rgb, prompts), (h, w), erode=erode)


class StreamingSegmenter:
    """SAM2 streaming camera predictor: prompt the first frame once, then track each new frame.

    This is the real-time sibling of ``PointCloudSegmenter`` (image predictor, re-prompted per
    frame) for the moving-arm case: clicking the same pixel every frame drifts off a moving
    object, so instead we seed SAM2's memory bank once and let it propagate causally. Used by
    ``debug_pointcloud.py --video`` to de-risk the online seg + point-masking loop.

    Requires the streaming fork (``Gy920/segment-anything-2-real-time``), which adds
    ``build_sam2_camera_predictor`` to ``sam2.build_sam``. That package is a *superset* of the
    official one -- the image predictor (``PointCloudSegmenter``) and offline video predictor
    (``segment_pointclouds.py``) still import and run -- so it may replace the official ``sam2``
    in the perception env. See ``online_segmentation.md``.

    Multi-class: each prompted class gets its own SAM2 ``obj_id``; ``track`` returns one mask
    per class, keyed by name (composable with ``compose_label_map`` / ``warp_masks_to_depth_frame``
    exactly like the image-predictor masks).
    """

    def __init__(self, checkpoint: str, model_cfg: str, device: str = None):
        import torch
        from sam2.build_sam import build_sam2_camera_predictor
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = build_sam2_camera_predictor(model_cfg, checkpoint, device=self.device)
        self._obj_to_name: dict = {}  # SAM2 obj_id -> class name

    def _autocast(self):
        torch = self._torch
        return (torch.autocast(self.device, dtype=torch.bfloat16)
                if self.device == "cuda" else torch.no_grad())

    def start(self, rgb: np.ndarray, prompts: dict) -> dict:
        """Seed the first frame. ``prompts``: {class_name -> (pos, neg)} (clicks as [x, y]).

        Classes with no positive clicks are skipped. Returns the frame-0 masks dict
        {class_name -> (H, W) bool}.
        """
        torch = self._torch
        self._obj_to_name = {}
        last = None
        with torch.inference_mode(), self._autocast():
            self.predictor.load_first_frame(rgb)  # SAM2 expects RGB
            obj_id = 0
            for name, (pos, neg) in prompts.items():
                if not pos:
                    continue
                obj_id += 1
                self._obj_to_name[obj_id] = name
                points = np.array(list(pos) + list(neg), np.float32)
                labels = np.array([1] * len(pos) + [0] * len(neg), np.int32)
                last = self.predictor.add_new_prompt(
                    frame_idx=0, obj_id=obj_id, points=points, labels=labels)
        # add_new_prompt -> (frame_idx, obj_ids, video_res_masks); last call has every object.
        return self._masks_from(last[1], last[2]) if last is not None else {}

    def track(self, rgb: np.ndarray) -> dict:
        """Track the next frame using accumulated memory. Returns {class_name -> (H, W) bool}."""
        torch = self._torch
        with torch.inference_mode(), self._autocast():
            obj_ids, mask_logits = self.predictor.track(rgb)  # track -> (obj_ids, masks)
        return self._masks_from(obj_ids, mask_logits)

    def _masks_from(self, obj_ids, mask_logits) -> dict:
        """(obj_ids, (N,1,H,W) logits) -> {class_name -> (H, W) bool tensor}, dropping unknown ids.

        Masks stay on the SAM2 device (no host round-trip): the GPU pipeline
        (``warp_masks_to_depth_frame_torch`` / ``compose_label_map_torch``) consumes them directly.
        Call ``.cpu().numpy()`` yourself if you need host bools (e.g. for drawing).
        """
        out = {}
        for i, oid in enumerate(obj_ids):
            name = self._obj_to_name.get(int(oid))
            if name is not None:
                out[name] = mask_logits[i, 0] > 0.0  # (H, W) bool tensor on device
        return out


def warp_masks_to_depth_frame(masks: dict, depth: np.ndarray, K_depth: np.ndarray,
                              K_color: np.ndarray, T_depth_color: np.ndarray) -> dict:
    """Reverse-warp color-frame bool masks into the depth camera's pixel grid.

    When depth comes from FFS (left-IR frame) but SAM2 was run on the COLOR image, the masks
    live in the color frame and must be carried into the IR/depth frame before composing the
    label map (which must align pixel-for-pixel with ``depth``). For every depth pixel with
    valid depth we backproject to 3D in the depth frame, transform into the color frame, project
    into the color image, and sample the color mask there (nearest-neighbour).

    Args:
        masks: {class_name -> (Hc, Wc) bool} in the COLOR frame.
        depth: (H, W) metric depth in the depth/IR frame (0 or non-finite = invalid).
        K_depth, K_color: (3, 3) intrinsics of the depth and color cameras.
        T_depth_color: (4, 4) depth-frame -> color-frame rigid transform.

    Returns:
        {class_name -> (H, W) bool} on the depth grid (False wherever depth is invalid or the
        reprojection lands outside the color image / behind the color camera).
    """
    h, w = depth.shape
    vv, uu = np.indices((h, w))
    z = np.asarray(depth, np.float64)
    valid = np.isfinite(z) & (z > 0)

    # backproject depth pixels -> 3D points in the depth frame
    x = (uu - K_depth[0, 2]) / K_depth[0, 0] * z
    y = (vv - K_depth[1, 2]) / K_depth[1, 1] * z
    pts = np.stack([x, y, z], axis=-1)  # (H, W, 3)

    # depth frame -> color frame, then project into the color image
    R, t = T_depth_color[:3, :3], T_depth_color[:3, 3]
    pc = pts @ R.T + t
    zc = pc[..., 2]
    safe = valid & (zc > 0)
    zc_safe = np.where(safe, zc, 1.0)
    uc = np.round(pc[..., 0] / zc_safe * K_color[0, 0] + K_color[0, 2]).astype(np.int64)
    vc = np.round(pc[..., 1] / zc_safe * K_color[1, 1] + K_color[1, 2]).astype(np.int64)

    out = {}
    for name, m in masks.items():
        m = np.asarray(m, bool)
        hc, wc = m.shape
        in_b = safe & (uc >= 0) & (uc < wc) & (vc >= 0) & (vc < hc)
        warped = np.zeros((h, w), bool)
        warped[in_b] = m[vc[in_b], uc[in_b]]
        out[name] = warped
    return out


def _erode_torch(mask, px: int):
    """Erode a bool mask by ``px`` with a square SE (twin of ``_erode``): erosion = NOT dilate(NOT)."""
    import torch
    import torch.nn.functional as F
    if px <= 0:
        return mask
    inv = (~mask).float()[None, None]
    dil = F.max_pool2d(inv, kernel_size=2 * px + 1, stride=1, padding=px) > 0
    return ~dil[0, 0]


def compose_label_map_torch(masks: dict, shape, erode: int = 3, device=None):
    """torch twin of ``compose_label_map``: {name -> (H,W) bool tensor} -> (H,W) float32 label map
    (NaN = background) on ``device`` (defaults to the masks' device)."""
    import torch
    first = next(iter(masks.values()), None)
    dev = device or (first.device if first is not None else "cpu")
    label_map = torch.full(tuple(shape), float("nan"), dtype=torch.float32, device=dev)
    for name in _COMPOSE_ORDER:
        m = masks.get(name)
        if m is None:
            continue
        m = _erode_torch(m.bool(), erode)
        label_map[m] = SEG_LABELS[name]
    return label_map


def warp_masks_to_depth_frame_torch(masks: dict, depth, K_depth, K_color, T_depth_color) -> dict:
    """On-device twin of ``warp_masks_to_depth_frame``. ``depth`` is a (H,W) tensor on the device;
    ``masks`` are color-frame (Hc,Wc) bool tensors. Returns {name -> (H,W) bool tensor} on the
    depth grid. K_* / T_* are read as scalars, so plain numpy arrays are fine."""
    import torch
    h, w = depth.shape
    dev = depth.device
    vv, uu = torch.meshgrid(
        torch.arange(h, device=dev, dtype=torch.float32),
        torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    z = depth.to(torch.float32)
    valid = torch.isfinite(z) & (z > 0)

    x = (uu - float(K_depth[0, 2])) / float(K_depth[0, 0]) * z
    y = (vv - float(K_depth[1, 2])) / float(K_depth[1, 1]) * z
    pts = torch.stack([x, y, z], dim=-1)  # (H, W, 3) in the depth frame

    T = torch.as_tensor(T_depth_color, dtype=torch.float32, device=dev)
    pc = pts @ T[:3, :3].T + T[:3, 3]
    zc = pc[..., 2]
    safe = valid & (zc > 0)
    zc_safe = torch.where(safe, zc, torch.ones_like(zc))
    uc = torch.round(pc[..., 0] / zc_safe * float(K_color[0, 0]) + float(K_color[0, 2])).long()
    vc = torch.round(pc[..., 1] / zc_safe * float(K_color[1, 1]) + float(K_color[1, 2])).long()

    out = {}
    for name, m in masks.items():
        m = m.bool()
        hc, wc = m.shape
        in_b = safe & (uc >= 0) & (uc < wc) & (vc >= 0) & (vc < hc)
        warped = torch.zeros((h, w), dtype=torch.bool, device=dev)
        warped[in_b] = m[vc[in_b], uc[in_b]]
        out[name] = warped
    return out


def pick_prompts_interactive(rgb: np.ndarray, class_names=("robot", "peg", "hole")) -> dict:
    """One matplotlib window per class: LEFT-click positive, RIGHT-click negative.

    Returns {class_name -> (pos, neg)} for use with ``label_map``. Close each window to
    advance to the next class; a class with no positive clicks is treated as absent.
    """
    import matplotlib.pyplot as plt
    prompts = {}
    for name in class_names:
        pos, neg = [], []
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.imshow(rgb)
        ax.set_title(f"[{name}] LEFT-click = {name} (green) | RIGHT-click = background (red) "
                     f"| close window when done (no clicks = {name} absent)")

        def onclick(event, pos=pos, neg=neg, ax=ax):
            if event.inaxes != ax or event.xdata is None:
                return
            if event.button == 1:
                pos.append([float(event.xdata), float(event.ydata)])
                ax.plot(event.xdata, event.ydata, "o", color="lime", ms=8, mec="black")
            elif event.button == 3:
                neg.append([float(event.xdata), float(event.ydata)])
                ax.plot(event.xdata, event.ydata, "o", color="red", ms=8, mec="black")
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()
        if pos:
            prompts[name] = (pos, neg)
    return prompts
