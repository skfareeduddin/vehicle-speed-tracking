import numpy as np


def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou_val = interArea / float(boxAArea + boxBArea - interArea)
    return iou_val


class SimpleTracker:

    def __init__(self, iou_threshold=0.3, max_lost=10):
        self.next_id = 0
        self.tracks = {}
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def update(self, detections):
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks[self.next_id] = {
                    "box": det,
                    "lost": 0,
                }
                self.next_id += 1
        else:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["lost"] += 1

            for det in detections:
                best_iou = 0.0
                best_id = None
                for tid, t in self.tracks.items():
                    i = iou(det, t["box"])
                    if i > best_iou:
                        best_iou = i
                        best_id = tid

                if best_iou >= self.iou_threshold:
                    self.tracks[best_id]["box"] = det
                    self.tracks[best_id]["lost"] = 0
                else:
                    self.tracks[self.next_id] = {
                        "box": det,
                        "lost": 0,
                    }
                    self.next_id += 1

        delete_ids = [tid for tid, t in self.tracks.items() if t["lost"] > self.max_lost]
        for tid in delete_ids:
            del self.tracks[tid]

        outputs = []
        for tid, t in self.tracks.items():
            outputs.append((tid, t["box"]))
        return outputs