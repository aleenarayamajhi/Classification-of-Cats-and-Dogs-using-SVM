# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 16:24:55 2023

@author: ar52624
"""


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from PIL import Image
import skimage.feature as skif
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import fetch_openml
import seaborn as sns
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Specify the directory path for both "dogs" and "cats" folders
folders = ["dogs", "cats"]

feat = []
label = []

# Iterate over the folders
for label_value, folder_name in enumerate(folders):
    # Specify the directory path for the current folder
    images = os.path.join(r"C:\\Users\ar52624\Desktop\Fall 2023\Pattern Recognition\CatsAndDogs\Project\dataset\processedimages", folder_name)
    
    # List all the files in the directory
    imagecount = os.listdir(images)
    
    # Iterate over the files and read each one
    for value in imagecount:
        # Construct the full file path
        image_path = os.path.join(images, value)
        
        # Check if the file is a regular file (not a directory)
        if os.path.isfile(image_path):
            # Open and read the file
            dataset = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Define HOG parameters
            orientations = 9  # Number of gradient orientations
            pixels_per_cell = (8, 8)  # Size of a cell in pixels
            cells_per_block = (2, 2)  # Number of cells in each block
            
            # Compute HOG features
            hog_features = hog(
                dataset,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm='L2-Hys'
            )
            
            # Compute LBP features
            radius = 4
            n_points = 6 * radius
            method = 'uniform'
            lbp_image = local_binary_pattern(dataset, n_points, radius, method=method)
            lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

            # Normalize the LBP histogram
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= (lbp_hist.sum() + 1e-8)
            
            # Combine HOG and LBP features
            features = np.concatenate((hog_features, lbp_hist))
            
            feat.append(features)
            label.append(label_value)

X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=42)


# Initialize the SVM classifier (you can choose the kernel type, e.g., linear, RBF, etc.)
clf = svm.SVC()
clf1 = svm.SVC(C=0.1, kernel='poly',degree=3,coef0=2)
clf1.fit(X_train, y_train)

# Calculate the training accuracy
y_pred = clf1.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred)
print(f"Training Accuracy: {training_accuracy * 100:.2f}%")
cv_scores = cross_val_score(clf1, feat, label, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)

# Calculate and print the mean and standard deviation of the scores
print("Mean Accuracy:", cv_scores.mean())

# Calculate accuracy

###################################################################################
###################################################################################
# Make predictions on the test data
y_pred = clf1.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'test Accuracy: {accuracy * 100:.2f}%')

# Generate the confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(confusion)

###############################################################################
class_names=[0 , 1]
fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(confusion, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')    
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.show()
###############################################################################
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
###############################################################################
# Initialize the KNN classifier with the desired number of neighbors (e.g., n_neighbors=5)
clf = KNeighborsClassifier(n_neighbors=5)

# Train the KNN classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'test Accuracy with KNN: {accuracy * 100:.2f}%')

# Generate the confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(confusion)

# Print the classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

