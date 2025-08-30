Mahogany Plant Disease Prediction System using Machine Learning
📌 Project Overview

The Mahogany Plant Disease Prediction System is an AI-driven project aimed at detecting leaf spot disease and major pest infestations in mahogany plants. The system leverages machine learning models to classify plant images and provide disease/fertilizer recommendations for farmers.

Since no existing system is dedicated to mahogany plants, this project fills a crucial gap by helping improve yield and promoting sustainable farming practices.

🎯 Objectives

Detect and classify common mahogany plant diseases and pests.

Provide recommendations for crop management and fertilizer usage.

Develop a system that works offline for rural farmers with limited connectivity.

Contribute to agriculture technology solutions that support farmer livelihoods.

🛠️ Tech Stack

Programming Language: Python

Libraries: TensorFlow / Keras, OpenCV, NumPy, Pandas, Scikit-learn, Matplotlib

Database: (Optional) SQLite / Local Storage

Deployment: Streamlit / Tkinter (for offline desktop app)

📂 Dataset

The dataset is built from collected images of mahogany trees, including:

Leaf Spot Disease

Shoot Borer Pest

Other major pests/diseases

Due to dataset scarcity, data augmentation techniques (rotation, flipping, scaling, brightness adjustments) are used.

⚙️ Methodology

Data Collection & Preprocessing

Image acquisition (collected ~30+ shoot borer disease images, more to be added).

Data augmentation to increase dataset size.

Image resizing and normalization.

Model Development

Built ML models (CNN-based) for disease/pest classification.

Compared models for accuracy and efficiency.

Evaluation

Accuracy, Precision, Recall, F1-score metrics.

Deployment

Offline application for farmers (Streamlit/Tkinter).

Simple interface for uploading images and viewing disease predictions.

📊 Results (Sample)

Initial model trained but faced underfitting due to limited dataset.

Future improvements planned with transfer learning (ResNet, MobileNet, VGG16).

🚀 Future Scope

Build a larger mahogany-specific dataset.

Improve accuracy using transfer learning & ensemble methods.

Extend recommendations to include weather-based disease risk prediction.

Deploy on mobile app for farmers.
