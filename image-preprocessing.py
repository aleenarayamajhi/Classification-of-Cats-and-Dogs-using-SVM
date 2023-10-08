# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:18:18 2023

@author: ar52624
"""

import os
import cv2

# Specify the directory paths and folder names for dogs and cats
categories = ["dogs", "cats"]

# Iterate over categories (dogs and cats)
for category in categories:
    # Specify the directory path for the current category
    images_dir = f'C:/Users/ar52624/Desktop/Fall 2023/Pattern Recognition/CatsAndDogs/Project/dataset/{category}/'
    
    # List all the files in the directory
    image_files = os.listdir(images_dir)
    
    # Define the folder where you are going to save your processed images
    processed_images_dir = f'C:/Users/ar52624/Desktop/Fall 2023/Pattern Recognition/CatsAndDogs/Project/dataset/processedimages/{category}/'
    
    # Iterate over the image files in the current category
    for image_file in image_files:
        # Construct the full file path
        image_path = os.path.join(images_dir, image_file)
        
        # Check if the file is a regular file (not a directory)
        if os.path.isfile(image_path):
            # Open and read the file
            dataset = cv2.imread(image_path)
            
            # Convert to grayscale
            convert_gray = cv2.cvtColor(dataset, cv2.COLOR_BGR2GRAY)
            
            # Resize the image
            convert_size = cv2.resize(convert_gray, (100, 100))
            
            # Save the processed image
            processed_image_path = os.path.join(processed_images_dir, image_file)
            cv2.imwrite(processed_image_path, convert_size)
