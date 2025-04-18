import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
# Set image dimensions
IMG_SIZE = 224
BATCH_SIZE = 32

# Paths to image datasets
ANEMIA_DIR = "Anemic"         # Path to anemia images
NON_ANEMIA_DIR = "Non_Anemic" # Path to non-anemia images

# Function to load images and labels
def load_images_from_directory(directory, label):
    images, labels = [], []
    for file in os.listdir(directory):
        img_path = os.path.join(directory, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to uniform size
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load anemia images
X_anemia, y_anemia = load_images_from_directory(ANEMIA_DIR, label=1)

# Load non-anemia images
X_non_anemia, y_non_anemia = load_images_from_directory(NON_ANEMIA_DIR, label=0)

# Data Augmentation for Non-Anemia (since it has fewer images)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

augmented_images = []
augmented_labels = []

for img in X_non_anemia:
    img = img.reshape((1,) + img.shape)  # Reshape for augmentation
    for _ in range(20):  # Create 20 augmented copies per image
        aug_img = next(datagen.flow(img, batch_size=1))[0]
        augmented_images.append(aug_img)
        augmented_labels.append(0)  # Non-Anemia label

# Convert lists to NumPy arrays
X_augmented = np.array(augmented_images)
y_augmented = np.array(augmented_labels)

# Merge all data
X = np.concatenate((X_anemia, X_non_anemia, X_augmented), axis=0)
y = np.concatenate((y_anemia, y_non_anemia, y_augmented), axis=0)

# Normalize pixel values
X = X / 255.0  # Scale pixel values to [0,1]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced',  classes=np.array([0, 1]), y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class Weights:", class_weights_dict)

# Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=BATCH_SIZE, validation_split=0.2, class_weight=class_weights_dict)

# Evaluate model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# Save model
model.save("anemia_classification_model.h5")
