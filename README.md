# MRI-BASED-ALZHEIMER-DETECTION-USING-PYTHON
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import os
# Load the saved model
model = tf.keras.models.load_model('Trained_models/LastModel.h5')

# Get the testing dataset directory
test_dir = 'Datasets\Test'

# Get the labels from the testing folder names
labels = os.listdir(test_dir)

print('Labels:', labels)
# Create the testing dataset using ImageDataGenerator
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128,128),
        batch_size=15,
        class_mode='categorical',
        shuffle=False)
        # Get the predicted classes for the testing dataset
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Get the true classes for the testing dataset
true_classes = test_generator.classes

# Create the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Print the confusion matrix
print('Confusion matrix:\n', cm)

# Print the accuracy
accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
print('Accuracy:', accuracy)

# Print the labels
print('Labels:', labels)
tp = np.diag(cm)
tn = np.sum(cm) - np.sum(tp)
print(f'True Positives: {tp}')
print(f'True Negatives: {tn}')

#Specificity = True Negative / (True Negative + False Positive)
#Precision = True Positive / (True Positive + False Positive)
#Recall = True Positive / (True Positive + False Negative)

# Calculate specificity, precision and recall
specificity = cm[0,0]/(cm[0,0]+cm[0,1])
precision = cm[1,1]/(cm[1,1]+cm[0,1])
recall = cm[1,1]/(cm[1,1]+cm[1,0])

print("Specificity: ", specificity)
print("Precision: ", precision)
print("Recall: ", recall)
