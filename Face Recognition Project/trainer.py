import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"  # Folder containing the images
training_data_path = "recognizer"

# Create the dataset folder if it doesn't exist
if not os.path.exists(path):
    print(f"The '{path}' folder does not exist. Creating the folder.")
    os.makedirs(path)  # Create the folder automatically

# Create the directory for saving the trained data if it doesn't exist
if not os.path.exists(training_data_path):
    os.makedirs(training_data_path)

def get_images_with_id(path):
    images_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]  # Ensure it's a file
    faces = []
    ids = []
    
    for single_image_path in images_paths:
        if single_image_path.endswith(".jpg") or single_image_path.endswith(".png"):  # Check if it's an image
            try:
                faceImg = Image.open(single_image_path).convert('L')  # Convert to grayscale
                faceNp = np.array(faceImg, np.uint8)
                # Correctly extract the ID (this assumes filenames are in the format user.{id}.{sample}.jpg)
                filename_parts = os.path.split(single_image_path)[-1].split(".")
                id = int(filename_parts[1])  # ID is in the second part of the filename
                print(f"Processing image with ID: {id}")
                faces.append(faceNp)
                ids.append(id)
                cv2.imshow("Training", faceNp)  # Display image for training (optional)
                cv2.waitKey(10)
            except Exception as e:
                print(f"Error loading image {single_image_path}: {e}")
    
    return np.array(ids), faces

# Start processing images and training the model
ids, faces = get_images_with_id(path)

# Train the recognizer
recognizer.train(faces, ids)

# Save the trained data
recognizer.save(os.path.join(training_data_path, "trainingdata.yml"))

cv2.destroyAllWindows()
