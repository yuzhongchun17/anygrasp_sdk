"""
text_tcp.py
-----------
Subscribes to the segmented RGBD stream published by SAM3 on tcp://localhost:5560,
builds a point cloud from each frame, and runs AnyGrasp to detect grasps.

Usage:
    python3 text_tcp.py --checkpoint_path /path/to/checkpoint.tar [options]
"""

import argparse
import sys
import os

import zmq
import msgpack
import numpy as np
import cv2

# AnyGrasp lives in the grasp_detection sub-directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "grasp_detection"))
from gsnet import AnyGrasp

# ── Camera intrinsics table (from cam_intrinsics.json) ───────────────────────
CAM_INTRINSICS = {
    "left":         dict(fx=690.12060546875,    fy=690.3575439453125,  cx=640.8505859375,   cy=361.0849304199219),
    "right":        dict(fx=691.2072143554688,  fy=691.3511962890625,  cx=639.8049926757812, cy=362.0535888671875),
    "right_camera": dict(fx=1034.4736328125,    fy=1034.5303955078125, cx=963.7049560546875, cy=544.0369873046875),
    "eye":          dict(fx=610.3641357421875,  fy=610.4464721679688,  cx=634.84619140625,   cy=362.1564025878906),
}

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='AnyGrasp checkpoint (.tar)')
parser.add_argument('--max_gripper_width', type=float, default=0.08,
                    help='UFACTORY xArm Gripper: position range 0-800 × 0.1 mm/unit = 80 mm max')
parser.add_argument('--gripper_height',    type=float, default=0.06,
                    help='UFACTORY xArm Gripper finger height ≈ 60 mm')
parser.add_argument('--top_down_grasp',    action='store_true')
parser.add_argument('--debug',             action='store_true', help='Open3D visualisation')
parser.add_argument('--cam', choices=list(CAM_INTRINSICS.keys()), default='left',
                    help='Camera name — sets fx/fy/cx/cy from built-in table')
# Per-axis overrides (only needed to deviate from the --cam defaults)
parser.add_argument('--fx', type=float, default=None)
parser.add_argument('--fy', type=float, default=None)
parser.add_argument('--cx', type=float, default=None)
parser.add_argument('--cy', type=float, default=None)
parser.add_argument('--depth_scale', type=float, default=1000.0,
                    help='Depth units → metres (1000 for mm, 1 for m)')
parser.add_argument('--zmq_addr', default='tcp://localhost:5560')
parser.add_argument('--top_k', type=int, default=20, help='Number of top grasps to keep')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

# Apply camera intrinsics from table, then allow per-axis CLI overrides
_intr = CAM_INTRINSICS[cfgs.cam]
if cfgs.fx is None: cfgs.fx = _intr['fx']
if cfgs.fy is None: cfgs.fy = _intr['fy']
if cfgs.cx is None: cfgs.cx = _intr['cx']
if cfgs.cy is None: cfgs.cy = _intr['cy']
print(f"[cam] {cfgs.cam}  fx={cfgs.fx}  fy={cfgs.fy}  cx={cfgs.cx}  cy={cfgs.cy}")

# ── AnyGrasp ─────────────────────────────────────────────────────────────────
print(f"[anygrasp] Loading checkpoint: {cfgs.checkpoint_path}")
anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()
print("[anygrasp] Network ready.")

# ── ZMQ subscriber ───────────────────────────────────────────────────────────
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.CONFLATE, 1)          # keep only the latest frame
sub.connect(cfgs.zmq_addr)
sub.setsockopt_string(zmq.SUBSCRIBE, "")
print(f"[zmq] Subscribed to {cfgs.zmq_addr}. Waiting for SAM3 frames… (Ctrl-C to quit)")


def depth_to_pointcloud(depth, color_rgb, fx, fy, cx, cy, scale):
    """Return (points, colors) arrays with background (depth==0) removed."""
    h, w = depth.shape
    xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))

    z = depth / scale                       # metres
    x = (xmap - cx) / fx * z
    y = (ymap - cy) / fy * z

    points = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    colors = color_rgb.astype(np.float32) / 255.0

    # Mask: keep only foreground pixels (depth > 0) within a sane range
    mask = (z > 0) & (z < 2.0)
    return points[mask].astype(np.float32), colors[mask].astype(np.float32)


# ── Main loop ─────────────────────────────────────────────────────────────────
frame_idx = 0
try:
    while True:
        # ---- receive --------------------------------------------------------
        try:
            raw = sub.recv(zmq.NOBLOCK)
        except zmq.error.Again:
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue

        data = msgpack.unpackb(raw)

        # ---- decode colour --------------------------------------------------
        print("[debug] keys:", list(data.keys()))
        color_buf = data['color_img']
        color_bgr = cv2.imdecode(np.frombuffer(color_buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        if color_bgr is None:
            print("[warn] Failed to decode colour frame — skipping.")
            continue
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # ---- decode depth ---------------------------------------------------
        depth_shape = data['depth_shape']   # [H, W]
        depth_raw   = data['depth_raw']
        prompt      = data['prompt']
        if isinstance(prompt, bytes):
            prompt = prompt.decode()

        if depth_shape[0] == 0 or len(depth_raw) == 0:
            print("[warn] Empty depth — skipping.")
            continue

        depth = np.frombuffer(depth_raw, dtype=np.uint16).reshape(depth_shape[0], depth_shape[1])

        frame_idx += 1
        print(f"\n[frame {frame_idx}] prompt='{prompt}'  depth shape={depth.shape}")

        # ---- build point cloud ----------------------------------------------
        points, colors = depth_to_pointcloud(
            depth, color_rgb,
            cfgs.fx, cfgs.fy, cfgs.cx, cfgs.cy, cfgs.depth_scale
        )

        if len(points) < 10:
            print("[warn] Too few foreground points — skipping.")
            continue

        print(f"[pcd] {len(points)} foreground points  "
              f"XYZ min={points.min(axis=0)}  max={points.max(axis=0)}")

        # ---- workspace limits from point cloud extents ----------------------
        lims = [
            float(points[:, 0].min()), float(points[:, 0].max()),
            float(points[:, 1].min()), float(points[:, 1].max()),
            float(points[:, 2].min()), float(points[:, 2].max()),
        ]

        # ---- AnyGrasp inference ---------------------------------------------
        gg, cloud = anygrasp.get_grasp(
            points, colors,
            lims=lims,
            apply_object_mask=True,
            dense_grasp=False,
            collision_detection=True,
        )

        if gg is None or len(gg) == 0:
            print("[anygrasp] No grasps detected after collision filtering.")
            continue

        gg = gg.nms().sort_by_score()
        gg_pick = gg[: cfgs.top_k]

        print(f"[anygrasp] {len(gg)} grasps found. Top-{len(gg_pick)} scores: {gg_pick.scores}")
        best = gg_pick[0]
        print(f"[anygrasp] Best grasp — score={best.score:.4f}  "
              f"width={best.width:.4f}m  translation={best.translation}")

        # ---- optional Open3D visualisation ----------------------------------
        if cfgs.debug:
            import open3d as o3d
            trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            cloud.transform(trans_mat)
            grippers = gg_pick.to_open3d_geometry_list()
            for g in grippers:
                g.transform(trans_mat)
            o3d.visualization.draw_geometries(
                [*grippers, cloud],
                window_name=f"AnyGrasp | frame {frame_idx} | prompt: {prompt}"
            )

        # ---- show colour frame (non-blocking) --------------------------------
        cv2.putText(color_bgr, f"prompt: {prompt}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(color_bgr, f"grasps: {len(gg)}  best={best.score:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("AnyGrasp — segmented RGB", color_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[info] Interrupted — exiting.")
finally:
    cv2.destroyAllWindows()
    sub.close()
    ctx.term()
    print("[info] Subscriber closed.")
