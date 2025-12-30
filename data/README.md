# Dataset Information

**Dataset used:** Car Number Plate Dataset (YOLO Format)

**Link:** https://www.kaggle.com/datasets/sujaymann/car-number-plate-dataset-yolo-format

**Number of samples:** 433 files 

**Data type:** Images of vehicles with license plates and corresponding YOLO annotations

**Features:**
- Images of vehicles with visible license plates
- Bounding box annotations of number plates (in YOLO format: class label + normalized x-center, y-center, width, height) 

**Annotation format:** YOLO format (Label, X_center, Y_center, Width, Height) 

**Target/Task:** Object Detection (Detect number plates in images)

# Dataset Usage Instructions

The dataset files (images and labels) are **not included** in this repository.

After downloading the dataset from Kaggle, organize it in the following structure to train the YOLOv8 model:

dataset/images/train  

dataset/images/val  

dataset/images/test  

dataset/labels/train  

dataset/labels/val  

dataset/labels/test  

Each image file must have a corresponding `.txt` annotation file with the same filename.

This structure is required by the `data.yaml` file used during training.

**Google Colab Users**

If you are running this project on Google Colab, you may:

Upload the dataset directly to the Colab environment, or

Store the dataset in Google Drive and mount it inside Colab

Ensure that the dataset directory structure matches the format shown above, and update dataset paths in the notebooks if necessary.
