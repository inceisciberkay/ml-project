import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.xception import preprocess_input
import gc
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class RemoveGarbageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image_array = preprocess_input(np.array(image))
    return image_array

def read_all_images_and_labels(folder_paths, label_mapping):
    images = []
    labels = []
    count = 1
    for folder_path, label in zip(folder_paths, label_mapping):
        for image_path in glob.glob(os.path.join(folder_path, '*.png')):
            image = cv2.imread(image_path)
            preprocessed_image = preprocess_image(image)
            images.append(preprocessed_image)
            labels.append(label)
            print(count)
            count += 1

    return np.array(images), np.array(labels)

def image_paths_and_labels(folder_paths, label_mapping):
    all_image_paths = []
    all_labels = []
    for folder_path, label in zip(folder_paths, label_mapping):
        image_paths = glob.glob(os.path.join(folder_path, '*.png'))
        all_image_paths.extend(image_paths)
        all_labels.extend([label] * len(image_paths))

    return all_image_paths, all_labels

def image_generator(file_paths, labels, batch_size):
    while True:
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            batch_images = []
            for image_path, label in zip(batch_paths, batch_labels):
                image = cv2.imread(image_path)
                preprocessed_image = preprocess_image(image)
                batch_images.append(preprocessed_image)

            yield np.array(batch_images), np.array(batch_labels)

# Data paths
folder_paths_positive = ['dataset/images2/fractured']
folder_paths_negative = ['dataset/images2/nonfractured']
labels_positive = 1
labels_negative = 0

# Combine positive and negative class paths and labels
folder_paths = folder_paths_positive + folder_paths_negative
labels = [labels_positive] * len(folder_paths_positive) + [labels_negative] * len(folder_paths_negative)

# Split the data into training, validation, and test sets
image_paths, y = image_paths_and_labels(folder_paths, labels)

# ten_percent_size = int(1 * len(image_paths))
# selected_indices = np.random.choice(len(image_paths), ten_percent_size, replace=False)
# selected_image_paths = [image_paths[i] for i in selected_indices]
# selected_labels = [y[i] for i in selected_indices]
selected_image_paths = image_paths
selected_labels = y
X_train, X_temp, y_train, y_temp = train_test_split(selected_image_paths, selected_labels, test_size=0.3, random_state=1)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=1)
# X_train, X_temp, y_train, y_temp = train_test_split(selected_image_paths, selected_labels, test_size=0.7, random_state=1)
# X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.9, random_state=1)
# X_test, temp, y_test, temp = train_test_split(X_temp, y_temp, test_size=0.9, random_state=1)


def xception():
   # Base Xception model
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg') #133 layers

    # Freeze the base_model
    base_model.trainable = False # try
    fine_tune_at = len(base_model.layers) - 100
    # Freeze the pre-trained layers
    for layer in base_model.layers[:fine_tune_at]: # try
        layer.trainable = False

    # base_model.trainable = True # try

    # Remove last layers
    x = base_model.layers[fine_tune_at].output # try
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x) #try
    x = MaxPooling2D()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) #try
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) #try
    x = MaxPooling2D()(x)

    # help me to add flatten laye
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)  # Add dropout with a rate of 0.5 (adjust as needed)
    x = Dense(128, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    # x = BatchNormalization()(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # Create the model for fine-tuning
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary(show_trainable=True) # try

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Display the model summary
    model.summary()
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

    # Generators
    batch_size = 4
    train_generator = image_generator(X_train, y_train, batch_size=batch_size)
    valid_generator = image_generator(X_valid, y_valid, batch_size=batch_size)
    test_generator = image_generator(X_test, y_test, batch_size=batch_size)

    # Training
    hist = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=valid_generator,
        validation_steps=len(X_valid) // batch_size,
        epochs=15,
        callbacks=[RemoveGarbageCallback(), tensorboard_callback, earlystop, reduce_lr]
    )

    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(X_test) // batch_size)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    #  # Plot training and validation loss
    # plt.plot(hist.history['loss'], label='Training Loss')
    # plt.plot(hist.history['val_loss'], label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # # Plot training and validation accuracy
    # plt.plot(hist.history['accuracy'], label='Training Accuracy')
    # plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    y_pred = model.predict(test_generator, steps=len(X_test) // batch_size)
    # Convert probabilities to binary predictions (0 or 1)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Get true labels
    y_true = np.array(y_test[:len(y_pred_binary)])  # Make sure the true labels match the length of predictions

    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    print("Confusion Matrix:")
    print(cm)

xception()
