# Models

This directory is intended to store trained model weights generated during experimentation and training.

**Model Weights Availability :**
The trained YOLOv8 model weights are not included in this repository.

**Reason :**
  1. The model is fine-tuned from YOLOv8, which is licensed under AGPL-3.0
  2. To avoid licensing and redistribution issues, trained weights are excluded

## How to Generate the Model Weights

You can train the model locally or on Google Colab by running the training notebook **notebook/01_train.ipynb**.

The trained weights (e.g., best.pt) will be automatically generated and saved during training.

### Notes

Generated model files are typically large and should not be committed to version control

If required, trained weights can be shared externally for research or evaluation purposes
