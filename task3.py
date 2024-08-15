import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Directory paths
cat_folder = 'path_to_cat_images'
dog_folder = 'path_to_dog_images'

# Initialize lists for images and labels
data = []
labels = []

# Image size for resizing
img_size = 64

# Load cat images
for img_name in os.listdir(cat_folder):
    img_path = os.path.join(cat_folder, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (img_size, img_size))
        data.append(img)
        labels.append(0)  # Label 0 for cats

# Load dog images
for img_name in os.listdir(dog_folder):
    img_path = os.path.join(dog_folder, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (img_size, img_size))
        data.append(img)
        labels.append(1)  # Label 1 for dogs

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Flatten the images
data = data.reshape(len(data), -1)

# Standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Initialize the SVM model
svm = SVC(kernel='linear', random_state=42)

# Train the model
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
