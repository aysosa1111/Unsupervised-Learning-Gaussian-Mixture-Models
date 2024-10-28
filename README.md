# Gaussian Mixture Model on Olivetti Faces Dataset

This project applies a Gaussian Mixture Model (GMM) to perform clustering and anomaly detection on the Olivetti faces dataset. The code performs dimensionality reduction using PCA, optimizes GMM parameters, demonstrates clustering, generates new face images, and detects anomalies.

## Project Structure

-Data Loading: Utilizes the Olivetti faces dataset.

-Data Preprocessing: Splits the data into training, validation, and test sets using stratified sampling.

-PCA: Reduces dimensionality while retaining 99% of the variance.

-GMM Optimization: Determines the optimal covariance type and number of clusters using GridSearchCV and BIC.

-Clustering: Outputs both hard and soft clustering results.

-Face Generation: Generates new faces from the model.

-Anomaly Detection: Detects anomalies by evaluating modified images.

## Installation

-To run this project, install the required libraries:


pip install numpy matplotlib scikit-learn Pillow

## Usage

-Execute the script to perform all operations:



python path_to_script.py

Ensure you have the Olivetti dataset accessible or modify the script to download it directly using sklearn.datasets.fetch_olivetti_faces.

## Results

-The script will output clustering assignments and probabilities.

-It will generate images based on the learned model and display them.

-Anomaly detection results will be printed showing the effectiveness of the model in identifying altered images.
