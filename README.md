Brain Tumor MRI Classification using Custom CNN & Transfer Learning
====================================================================

Project Overview
----------------
This project focuses on classifying brain MRI images into four categories:
1. Glioma
2. Meningioma
3. Pituitary
4. No Tumor

We use two approaches:
- A Custom Convolutional Neural Network (CNN)
- Transfer Learning with MobileNetV2

Dataset
-------
The dataset consists of MRI images organized into training, validation, and test folders, each containing:
- glioma/
- meningioma/
- pituitary/
- no_tumor/

Image size: resized to 224x224  
Color mode: grayscale for CNN, RGB for Transfer Learning

Project Structure
-----------------
- Tumour/                 # Dataset folder (train/valid/test)
- app.py                  # Streamlit application
- brain_tumor_model.h5    # Trained model (MobileNetV2)
- best_custom_cnn.h5      # Trained model (Custom CNN)
- README.txt              # This file

Model Evaluation
----------------
Custom CNN:
- Accuracy: 33%
- F1-score: 0.16
- Only glioma detected reasonably; other classes failed

Transfer Learning (MobileNetV2):
- Accuracy: 74%
- F1-score: 0.71
- Reliable across all tumor types with strong generalization

Conclusion: MobileNetV2 outperformed the custom CNN and is recommended for deployment.

Streamlit App Features
----------------------
- Upload an MRI image
- Get predicted tumor type
- See confidence scores
- Clean UI with sidebar info

Run the App
-----------
1. Install dependencies:
   pip install streamlit tensorflow keras pillow numpy h5py

2. Run the app:
   streamlit run app.py

Acknowledgments
---------------
This project was built using:
- TensorFlow & Keras
- Streamlit
- Pretrained MobileNetV2 from ImageNet
- Publicly available Brain MRI dataset

Author
------
Obuli Pranav
obulipranav@gmail.com
