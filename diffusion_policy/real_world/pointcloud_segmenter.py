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
