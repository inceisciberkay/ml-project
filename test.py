import numpy as np
import glob
import gc
import os
from PIL import Image
import tensorflow as tf
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.applications import Xception
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.layers import BatchNormalization

class RemoveGarbageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

def preprocess_image(image):
    image = image.resize((224, 224))
    image = image.convert('RGB')
    image_array = np.array(image) / 255.0
    return image_array

def read_all_images_and_labels(folder_paths, label_mapping):
    images = []
    labels = []
    count = 1
    for folder_path, label in zip(folder_paths, label_mapping):
        for image_path in glob.glob(os.path.join(folder_path, '*.png')):
            image = Image.open(image_path)
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
                image = Image.open(image_path)
                preprocessed_image = preprocess_image(image)
                batch_images.append(preprocessed_image)

            yield np.array(batch_images), np.array(batch_labels)

# Data paths
folder_paths_positive = ['split/fracture/images']
folder_paths_negative = ['split/nonfracture/images']
labels_positive = 1
labels_negative = 0

# Combine positive and negative class paths and labels
folder_paths = folder_paths_positive + folder_paths_negative
labels = [labels_positive] * len(folder_paths_positive) + [labels_negative] * len(folder_paths_negative)

# Split the data into training, validation, and test sets
image_paths, y = image_paths_and_labels(folder_paths, labels)
X_train, X_temp, y_train, y_temp = train_test_split(image_paths, y, test_size=0.3, random_state=1)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=1)


def xception():
   # Base Xception model
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Remove last layers
    x = base_model.output
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # Create the model for fine-tuning
    model = Model(inputs=base_model.input, outputs=predictions)
    

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


    # Display the model summary
    model.summary()


    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

    # Generators
    batch_size = 32
    train_generator = image_generator(X_train, y_train, batch_size=batch_size)
    valid_generator = image_generator(X_valid, y_valid, batch_size=batch_size)
    test_generator = image_generator(X_test, y_test, batch_size=batch_size)

    # Training
    hist = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=valid_generator,
        validation_steps=len(X_valid) // batch_size,
        epochs=20,
        callbacks=[RemoveGarbageCallback(), tensorboard_callback, earlystop, reduce_lr]
    )

    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(X_test) // batch_size)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

xception()