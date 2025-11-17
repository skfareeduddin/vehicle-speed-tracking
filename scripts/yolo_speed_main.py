import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import torch

from tracker import SimpleTracker
from speed_utils import SpeedEstimator


def run_speed_estimation(
    video_path,
    output_video_path,
    output_csv_path,
    yolo_model="yolov8n.pt",
    conf_thres=0.4,
    pixels_per_meter=8.0,
):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(yolo_model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: cannot open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    print("Video FPS:", fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Video size:", width, "x", height)

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    tracker = SimpleTracker(iou_threshold=0.3, max_lost=10)
    speed_estimator = SpeedEstimator(pixels_per_meter=pixels_per_meter, fps=fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    # pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing video")
    pbar = None

    classes_to_keep = [2, 3, 5, 7]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, verbose=False)[0]

        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box, cls, conf in zip(
                results.boxes.xyxy, results.boxes.cls, results.boxes.conf
            ):
                if conf < conf_thres:
                    continue
                cls_id = int(cls.item())
                if cls_id not in classes_to_keep:
                    continue
                x1, y1, x2, y2 = box.tolist()
                detections.append([x1, y1, x2, y2])

        tracks = tracker.update(detections)

        speeds = speed_estimator.update(frame_idx, tracks)

        for tid, box in tracks:
            x1, y1, x2, y2 = map(int, box)
            speed = speeds.get(tid, None)
            label = f"ID {tid}"
            if speed is not None:
                label += f" {speed:.1f} km/h"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        out.write(frame)

        # if total_frames > 0:
        #     pbar.update(1)

        pass

    cap.release()
    out.release()
    # pbar.close()

    print("Video processing done. Output saved at:", output_video_path)
    df = speed_estimator.export_csv(output_csv_path)
    print("Speed CSV saved at:", output_csv_path)
    print(df.head())


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(project_root, "data", "input_videos")
    output_dir = os.path.join(project_root, "data", "output_videos")

    videos = [f for f in os.listdir(input_dir) if f.lower().endswith(".avi")]

    print("Videos found:", videos)
    for VIDEO_FILENAME in videos:
        print("\nProcessing:", VIDEO_FILENAME)

        video_path = os.path.join(input_dir, VIDEO_FILENAME)
        output_video_path = os.path.join(
            output_dir, VIDEO_FILENAME.replace(".avi", "_out.mp4")
        )
        output_csv_path = os.path.join(
            output_dir, VIDEO_FILENAME.replace(".avi", "_speeds.csv")
        )

        run_speed_estimation(
            video_path=video_path,
            output_video_path=output_video_path,
            output_csv_path=output_csv_path,
            yolo_model="yolov8n.pt",
            conf_thres=0.4,
            pixels_per_meter=1.5,
        )

    print("\nAll videos processed successfully!")

