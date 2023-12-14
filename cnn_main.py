import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import gc
from keras.applications.xception import Xception
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from dataset_preparer import get_dataset, IMG_SIZE

def plot(history):
    # Plot training and validation loss
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # # Plot training and validation accuracy
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

class RemoveGarbageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

IMG_SHAPE = IMG_SIZE + (3,)

train_ds = get_dataset('train')
validation_ds = get_dataset('validation')
test_ds = get_dataset('test')
in_memory_test_ds = get_dataset('test', in_memory=True)

# CASE 1
print('=============================== CASE 1 ===============================')
# Freeze the model and just train the top
base_model = Xception(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
# print(base_model.summary())
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)   # run base_model in inference mode
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)  # Add dropout with a rate of 0.5 (adjust as needed)
x = Dense(128, activation='relu')(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(1)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=0.1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Training
hist = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=15,
    callbacks=[RemoveGarbageCallback(), tensorboard_callback, earlystop, reduce_lr]
)

# Testing
test_loss, test_accuracy = model.evaluate(test_ds)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
plot(hist.history)

# Print confusion matrix
y_pred = model.predict(test_ds)
y_pred_binary = (y_pred > 0.5).astype(int)
_, y_true = in_memory_test_ds

cm = confusion_matrix(y_true, y_pred_binary)
print("Confusion Matrix:")
print(cm)

# CASE 2
print('=============================== CASE 2 ===============================')
# Freeze the model, train the top, unfreeze the model, retrain everything
# Unfreeze the model
model.trainable = True

# recompiling the model
model.compile(optimizer=Adam(learning_rate=0.1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Train fine-tuned model
hist = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=15,
    callbacks=[RemoveGarbageCallback(), tensorboard_callback, earlystop, reduce_lr]
)

# Testing
test_loss, test_accuracy = model.evaluate(test_ds)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
plot(hist.history)

# Print confusion matrix
y_pred = model.predict(test_ds)
y_pred_binary = (y_pred > 0.5).astype(int)
_, y_true = in_memory_test_ds

cm = confusion_matrix(y_true, y_pred_binary)
print("Confusion Matrix:")
print(cm)

# CASE 3
print('=============================== CASE 3 ===============================')
# Freeze some portion of base layers, remove remaining base layers, add convolutional layers on top
base_model = Xception(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

fine_tune_at = len(base_model.layers) - 100
# Freeze the pre-trained layers
for layer in base_model.layers[:fine_tune_at]: # try
    layer.trainable = False

# Remove last layers
x = base_model.layers[fine_tune_at].output # try
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x) #try
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) #try
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) #try
x = MaxPooling2D()(x)

x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)  # Add dropout with a rate of 0.5 (adjust as needed)
x = Dense(128, activation='relu')(x)
x = Dense(16, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Training
hist = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=15,
    callbacks=[RemoveGarbageCallback(), tensorboard_callback, earlystop, reduce_lr]
)

# Testing
test_loss, test_accuracy = model.evaluate(test_ds)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
plot(hist.history)

# Print confusion matrix
y_pred = model.predict(test_ds)
y_pred_binary = (y_pred > 0.5).astype(int)
_, y_true = in_memory_test_ds

cm = confusion_matrix(y_true, y_pred_binary)
print("Confusion Matrix:")
print(cm)