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

class RemoveGarbageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

IMG_SHAPE = IMG_SIZE + (3,)

train_ds = get_dataset('train')
validation_ds = get_dataset('validation')
test_ds = get_dataset('test')

# fine_tune_at = len(base_model.layers) - 100
# # Freeze the pre-trained layers
# for layer in base_model.layers[:fine_tune_at]: # try
#     layer.trainable = False

# # Remove last layers
# x = base_model.layers[fine_tune_at].output # try
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x) #try
# x = MaxPooling2D()(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) #try
# x = MaxPooling2D()(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) #try
# x = MaxPooling2D()(x)

# # help me to add flatten laye
# x = GlobalAveragePooling2D()(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.2)(x)  # Add dropout with a rate of 0.5 (adjust as needed)
# x = Dense(128, activation='relu')(x)
# x = Dense(16, activation='relu')(x)

# Create the base model from the pre-trained model Xception
base_model = Xception(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
print(base_model.summary())

# Freeze the base_model
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)   # run base_model in inference mode
x = GlobalAveragePooling2D()(x)
outputs = Dense(1)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=Adam(), loss=BinaryCrossentropy(from_logits=True), metrics=[BinaryAccuracy()])

# Display the model summary
model.summary()
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

# Training
hist = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=15,
    callbacks=[RemoveGarbageCallback(), tensorboard_callback, earlystop, reduce_lr]
)

# Fine Tuning

# Unfreeze the last layers of base model
count_unfreezed_layers = 100
for layer in base_model.layers[-count_unfreezed_layers:]:
    layer.trainable = True

# recompiling the model
model.compile(optimizer=Adam(1e-5),  # Very low learning rate
              loss=BinaryCrossentropy(from_logits=True),
              metrics=[BinaryAccuracy()])

# Train fine-tuned model
hist = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=20,
    callbacks=[RemoveGarbageCallback(), tensorboard_callback, earlystop, reduce_lr]
)

test_loss, test_accuracy = model.evaluate(test_ds)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    # Plot training and validation loss
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

y_pred = model.predict(test_ds)
# Convert probabilities to binary predictions (0 or 1)
y_pred_binary = (y_pred > 0.5).astype(int)

in_memory_test_ds = get_dataset('test', in_memory=True)
_, y_true = in_memory_test_ds

# Print confusion matrix
cm = confusion_matrix(y_true, y_pred_binary)
print("Confusion Matrix:")
print(cm)