# Automatic Number Plate Detection
Fine-tuned a YOLOv8 object detection model to accurately detect vehicle number plates from images, enabling automated traffic monitoring, surveillance, and intelligent transportation applications.

# Problem Statement
Manual monitoring of vehicle number plates is inefficient, error-prone, and not scalable for real-time traffic surveillance systems.
This project addresses the problem by building an automatic number plate detection system using a deep learning–based object detection model, enabling fast and accurate localization of license plates from vehicle images.

# Dataset
Name: Car Number Plate Dataset (YOLO Format)

Source: Kaggle

Link: https://www.kaggle.com/datasets/sujaymann/car-number-plate-dataset-yolo-format

Number of Samples: 433 annotated images

Data Type: Vehicle images with bounding box annotations

- Annotation Details
  Format: YOLO
  Annotation Structure: class label + normalized x-center, y-center, width, height

- Classes
  number_plate

**Note :** The dataset is not included in this repository.
Please refer to data/README.md for dataset download instructions and expected directory structure.

# Tech Stack

Programming Language: Python

Framework: Ultralytics YOLOv8

Libraries: PyTorch, OpenCV, NumPy, Ultralytics

Tools: Google Colab, Git

# Approach
1. Downloaded and analyzed the number plate detection dataset in YOLO format.
2. Organized the dataset into train, validation, and test splits.
3. Initialized a pre-trained YOLOv8 model.
4. Fine-tuned the model specifically for number plate detection.
5. Evaluated the trained model on a held-out test set using a custom evaluation script that computes Intersection over Union (IoU) and detection accuracy at an IoU threshold of 0.5.
6. Performed inference on test images to validate real-world performance.

# Model Used

YOLOv8 (You Only Look Once – Version 8)

The model was fine-tuned to detect only vehicle number plates, making it optimized for the ANPR (Automatic Number Plate Recognition) task.

# Results



# How to Run the Project

 **Note:** This project is implemented using Google Colab.  
The dataset is not included in this repository. Please refer to `data/README.md` for dataset details and expected directory structure.

# Option 1: Run Locally 

1. Clone the repository
git clone https://github.com/ArjunParikh1404/Automatic-Number-Plate-Detection.git
cd Automatic-Number-Plate-Detection

2. Install dependencies
pip install -r requirements.txt

3. Launch Jupyter Notebook
jupyter notebook

- Open and run the notebooks in the following order:
  1. notebook/01_train.ipynb
  2. notebook/02_test.ipynb
  3. notebook/03_evaluation.ipynb
  
# Option 2: Run on Google Colab

1. Open Google Colab
   
2. Upload or clone this repository
   
3. Install dependencies inside Colab: pip install -r requirements.txt

4. Upload the dataset to Colab or mount Google Drive

5. Update dataset paths inside the notebook if required

6. Run the notebooks in the same order as above

# Future Improvements
Integrate OCR (Optical Character Recognition) to extract text from detected number plates.

Enable real-time video inference for traffic surveillance cameras.

Deploy the model using FastAPI or Flask for real-world usage.

Optimize inference speed for edge devices.
