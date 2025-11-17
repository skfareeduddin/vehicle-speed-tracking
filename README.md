# Vehicle Tracking & Speed Estimation

This project performs automatic vehicle speed estimation from highway CCTV surveillance videos using YOLOv8 object detection, object tracking, and pixel displacement speed calculation. It supports batch processing of multiple video files and provides annotated output videos and per-vehicle speed CSV files.

---

## Features

- Vehicle detection using YOLOv8
- Object tracking with persistent IDs
- Speed estimation using pixel displacement → meters → km/h
- Batch processing of all input videos
- Exports annotated output videos and speed CSV files
- Jupyter notebook analytics for speed visualization

---

## Project Structure

```
vehicle-speed-tracking
├── data/
│ ├── input_videos/
│ └── output_videos/
├── scripts/
│ ├── yolo_speed_main.py
│ ├── tracker.py
│ └── speed_utils.py
├── notebook/
│ └── Vehicle_Speed_Tracking.ipynb
├── results/
│ ├── plots/ # Saved graphs
│ └── all_vehicle_speeds.csv
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/vehicle-speed-estimation-yolo.git
cd vehicle-speed-estimation-yolo
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Input Videos
Place video files in: `data/input_videos/`

### 4. Run Batch Speed Estimation
```bash
python scripts/yolo_speed_main.py
```

Output will be generated in: `data/output_videos/`

---

## Output Example

### Sample CSV Output
| id | avg_speed_kmh |
|----|---------------|
| 0  | 52.4          |
| 1  | 34.9          |
| 2  | 61.2          |

### Sample Video Overlay
```
ID 3 — 64.2 km/h
```

---

## Analysis Notebook

Run: `notebooks/speed_analysis.ipynb`


Includes:

- Speed histograms
- Fastest vehicle summary
- Optional speed heatmaps

---

## YOLO Model and Classes

Model used: **YOLOv8n** (auto-downloads via Ultralytics)

### Tracked Classes

| Class | Label       |
|-------|-------------|
| 2     | Car         |
| 3     | Motorcycle  |
| 5     | Bus         |
| 7     | Truck       |

---

## Configuration Parameters

| Parameter         | Description |
|------------------|-------------|
| conf_thres       | Confidence threshold for YOLO detections |
| pixels_per_meter | Calibration constant based on camera distance |
| iou_threshold    | IOU threshold for tracker ID assignment |
| max_lost         | Max frames to keep lost track alive |

---

## Requirements

```
ultralytics
opencv-python
numpy
pandas
matplotlib
seaborn
torch
tqdm
```

---

## Limitations

- Speed estimation is approximate and not legally verified
- Accuracy depends on camera position, angle, and lens distortion
- No homography-based calibration in this version
- Stationary vehicles may show small noise values
