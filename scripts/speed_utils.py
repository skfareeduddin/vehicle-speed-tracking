import numpy as np
import pandas as pd
import math
import os


class SpeedEstimator:

    def __init__(self, pixels_per_meter=8.0, fps=30.0):
        # You should tune pixels_per_meter for your specific video.
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.history = {}  # id -> list of (frame_idx, cx, cy)

    def update(self, frame_idx, tracks):
        speeds = {}
        for tid, box in tracks:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if tid not in self.history:
                self.history[tid] = []
            self.history[tid].append((frame_idx, cx, cy))

            if len(self.history[tid]) >= 2:
                (f0, x0, y0), (f1, x1c, y1c) = self.history[tid][-2:]
                dt_frames = f1 - f0
                if dt_frames > 0:
                    dx = x1c - x0
                    dy = y1c - y0
                    pixel_dist = math.sqrt(dx * dx + dy * dy)
                    meter_dist = pixel_dist / self.pixels_per_meter
                    dt = dt_frames / self.fps  # seconds
                    if dt > 0:
                        speed_mps = meter_dist / dt
                        speed_kmh = speed_mps * 3.6
                        speeds[tid] = speed_kmh
        return speeds

    def export_csv(self, csv_path):
        rows = []
        for tid, pts in self.history.items():
            if len(pts) < 2:
                continue
            total_pixel = 0.0
            total_dt_frames = 0
            for (f0, x0, y0), (f1, x1c, y1c) in zip(pts[:-1], pts[1:]):
                dx = x1c - x0
                dy = y1c - y0
                pixel_dist = math.sqrt(dx * dx + dy * dy)
                total_pixel += pixel_dist
                total_dt_frames += (f1 - f0)
            if total_dt_frames == 0:
                continue
            avg_pixel_per_frame = total_pixel / total_dt_frames
            meter_per_frame = avg_pixel_per_frame / self.pixels_per_meter
            speed_mps = meter_per_frame * self.fps
            speed_kmh = speed_mps * 3.6
            rows.append({"id": tid, "avg_speed_kmh": speed_kmh})

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        return df
