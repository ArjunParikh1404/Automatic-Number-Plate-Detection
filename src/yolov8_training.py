import subprocess
import sys

# === Install YOLOv8 (ultralytics) ===
def install_yolov8():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--upgrade"])

# === Check YOLO version ===
def check_yolo_version():
    subprocess.check_call(["yolo", "version"])

# === Train YOLOv8 model ===
def train_yolov8():
    subprocess.check_call([
        "yolo", "detect", "train",
        "data=/content/drive/MyDrive/Number_Plate_Detection/data.yaml",  # Add your data.yaml file's path here.
        "model=yolov8n.pt",
        "epochs=50",
        "imgsz=640",
        "batch=16",
        "name=number_plate_yolov8"
    ])

if __name__ == "__main__":
    # === Run all steps === 
    install_yolov8()
    check_yolo_version()
    train_yolov8()
