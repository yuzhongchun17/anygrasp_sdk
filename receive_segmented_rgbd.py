"""
receive_segmented_rgbd.py
-------------------------
Subscribes to the segmented RGBD stream published by stream_segment.py
(sam3 env) on tcp://127.0.0.1:5560 and visualises both colour and depth.

Controls:
  q  — quit
"""

import cv2
import zmq
import msgpack
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEG_SUB_ADDR = "tcp://127.0.0.1:5560"


class SimpleFPS:
    def __init__(self):
        self.prev_time = time.perf_counter()
        self.fps = 0.0

    def update(self):
        now = time.perf_counter()
        diff = now - self.prev_time
        if diff > 0:
            self.fps = self.fps * 0.9 + (1.0 / diff) * 0.1
        self.prev_time = now
        return self.fps


def main():
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)   # keep only the latest frame
    sub.connect(SEG_SUB_ADDR)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    logger.info(f"Subscribed to {SEG_SUB_ADDR}")
    logger.info("Waiting for segmented RGBD frames... (press 'q' to quit)")

    fps = SimpleFPS()

    while True:
        # ---- receive ----
        try:
            raw = sub.recv(zmq.NOBLOCK)
        except zmq.error.Again:
            # No message yet — still poll keyboard so the window stays responsive
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue

        data = msgpack.unpackb(raw)

        # ---- decode colour ----
        color_buf = data[b'color_img']
        color = cv2.imdecode(np.frombuffer(color_buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        if color is None:
            logger.warning("Failed to decode colour frame — skipping.")
            continue

        # ---- decode depth ----
        depth_shape = data[b'depth_shape']   # [H, W]
        depth_raw   = data[b'depth_raw']
        prompt      = data[b'prompt'].decode() if isinstance(data[b'prompt'], bytes) else data[b'prompt']
        timestamp   = data[b'timestamp']

        has_depth = (depth_shape[0] > 0 and depth_shape[1] > 0 and len(depth_raw) > 0)
        if has_depth:
            depth = np.frombuffer(depth_raw, dtype=np.uint16).reshape(depth_shape[0], depth_shape[1])

        current_fps = fps.update()
        latency_ms  = (time.time() - timestamp) * 1000

        # ---- visualise colour ----
        vis_color = color.copy()
        cv2.putText(vis_color, f"Prompt: {prompt}",       (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis_color, f"FPS: {current_fps:.1f}", (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_color, f"Lat: {latency_ms:.0f}ms",(10, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Segmented RGB", vis_color)

        # ---- visualise depth ----
        if has_depth:
            norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            vis_depth = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            # Black out background pixels (depth == 0) so the mask boundary is clear
            vis_depth[depth == 0] = 0
            cv2.putText(vis_depth, f"Depth ({depth.dtype})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Segmented Depth", vis_depth)

        logger.info(f"Frame received — prompt='{prompt}', latency={latency_ms:.0f}ms, fps={current_fps:.1f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    sub.close()
    ctx.term()
    logger.info("Subscriber exited cleanly.")


if __name__ == "__main__":
    main()
