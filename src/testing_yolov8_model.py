import subprocess
import sys
import os
import shutil
from ultralytics import YOLO
from PIL import Image

# === Install ultralytics ===
subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

# === CONFIGURATION ===
MODEL_PATH = "/content/drive/MyDrive/Number_Plate_Detection/best.pt"   # Add your trained model path over here
TEST_IMAGE_PATH = "/content/drive/MyDrive/Number_Plate_Detection/dataset/images/val/Cars_524.jpg"  # Add your image path over here 

PREDICT_FOLDER = "runs/detect/predict"

# === CLEAN OLD PREDICTIONS ===
if os.path.exists(PREDICT_FOLDER):
    shutil.rmtree(PREDICT_FOLDER)

# === LOAD MODEL === 
model = YOLO(MODEL_PATH)

# === RUN PREDICTION === 
results = model.predict(
    source=TEST_IMAGE_PATH,
    save=True
)

# === DISPLAY / SHOW RESULT ===
if os.path.exists(PREDICT_FOLDER):
    pred_images = [
        f for f in os.listdir(PREDICT_FOLDER)
        if f.lower().endswith((".jpg", ".png"))
    ]

    if pred_images:
        output_image_path = os.path.join(PREDICT_FOLDER, pred_images[0])
        print(f"Prediction completed! Output saved at: {output_image_path}")

        # Show image (works on local machine)
        img = Image.open(output_image_path)
        img.show()
    else:
        print("No output image found in predict folder.")
else:
    print("Prediction folder not found.")
