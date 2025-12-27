# YOLOv8 Detection Evaluation Script

# === INSTALL ULTRALYTICS ===
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

# === BASIC IMPORT CHECK ===
import torch
print("Torch version:", torch.__version__)

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "/content/drive/MyDrive/Number_Plate_Detection/best.pt"   # Add your trained model path over here

TEST_IMAGES_DIR = "/content/drive/MyDrive/Number_Plate_Detection/dataset/images/test"   # Add your dataset path over here
TEST_LABELS_DIR = "/content/drive/MyDrive/Number_Plate_Detection/dataset/labels/test"   # Add your dataset path over here

OUTPUT_CSV = "/content/drive/MyDrive/Number_Plate_Detection/recognition_output/yolov8_eval.csv"   # Add your output file path over here 

IOU_THRESHOLD = 0.5

# === HELPER FUNCTIONS ===

def xywhn_to_xyxy(xc, yc, w, h, img_w, img_h):
    """Convert YOLO normalized format to pixel box."""
    xc *= img_w
    yc *= img_h
    w  *= img_w
    h  *= img_h
    x1 = int(xc - w / 2)
    y1 = int(yc - h / 2)
    x2 = int(xc + w / 2)
    y2 = int(yc + h / 2)
    return x1, y1, x2, y2


def iou(boxA, boxB):
    """Compute Intersection over Union."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / (areaA + areaB - interArea)


def load_gt_box(label_path, img_w, img_h):
    """Load first ground-truth box from YOLO label file."""
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r") as f:
        line = f.readline().strip()
        if not line:
            return None

    _, xc, yc, w, h = map(float, line.split())
    return xywhn_to_xyxy(xc, yc, w, h, img_w, img_h)

# === EVALUATION ===

def evaluate_yolov8():
    model = YOLO(MODEL_PATH)

    image_files = [
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    results = []
    correct = 0
    total = 0
    iou_scores = []

    print(f"Found {len(image_files)} test images.")

    for img_name in image_files:
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        label_path = os.path.join(
            TEST_LABELS_DIR,
            os.path.splitext(img_name)[0] + ".txt"
        )

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        gt_box = load_gt_box(label_path, w, h)

        preds = model.predict(
            source=img_path,
            save=False,
            verbose=False
        )

        if not preds or preds[0].boxes is None:
            det_iou = 0.0
            det_correct = 0
        else:
            boxes = preds[0].boxes.xyxy.cpu().numpy()
            confs = preds[0].boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
            pred_box = boxes[best_idx].astype(int)

            if gt_box is not None:
                det_iou = iou(pred_box, gt_box)
                det_correct = int(det_iou >= IOU_THRESHOLD)
            else:
                det_iou = 0.0
                det_correct = 0

        correct += det_correct
        total += 1
        iou_scores.append(det_iou)

        results.append({
            "image": img_name,
            "iou": det_iou,
            "detected_correctly": det_correct
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n================ DETECTION SUMMARY ================")
    print(f"Total test images:     {total}")
    print(f"Detection accuracy:    {correct / total:.4f}")
    print(f"Mean IoU:              {np.mean(iou_scores):.4f}")
    print(f"Results saved to:      {OUTPUT_CSV}")
    print("==================================================")

# === RUN ===
if __name__ == "__main__":
    evaluate_yolov8()
