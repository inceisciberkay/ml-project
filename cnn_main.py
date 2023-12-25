import tensorflow as tf
import matplotlib.pyplot as plt
import gc
from keras.applications.xception import Xception
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy, TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from dataset_preparer import get_dataset, IMG_SIZE
from evaluation_utils import plot_confusion_matrix, calculate_evaluation_metrics

def plot_history(history):
    # Plot training and validation loss
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.plot(history['binary_accuracy'], label='Training Accuracy')
    plt.plot(history['val_binary_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def print_evaluations(title, TP, FP, FN, TN):
    accuracy, precision, recall, f1_score, f2_score = calculate_evaluation_metrics(TP, FP, FN, TN)
    with open('cnn_results.txt', "a+") as file:
        result = f"""\
{title}
TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}
Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
F1 Score: {f1_score}
F2 Score: {f2_score}

"""
        file.write(result)


# Create callbacks
class RemoveGarbageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)

# Get dataset
IMG_SHAPE = IMG_SIZE + (3,)
train_ds = get_dataset('train')
validation_ds = get_dataset('validation')
test_ds = get_dataset('test')

# # CASE 1
print('=============================== CASE 1 ===============================')
# Freeze the model and just train the top
base_model = Xception(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)   # run base_model in inference mode
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)  # Add dropout with a rate of 0.5 (adjust as needed)
x = Dense(128, activation='relu')(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=1e-4), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy(), 
                                                                                      TruePositives(), 
                                                                                      FalsePositives(), 
                                                                                      FalseNegatives(),
                                                                                      TrueNegatives()])

# Train the model
hist = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=15,
    callbacks=[RemoveGarbageCallback(), earlystop, reduce_lr]
)

# Evaluate the model
loss, acc, TP, FP, FN, TN = model.evaluate(validation_ds)
plot_history(hist.history)
plot_confusion_matrix(TP, FP, FN, TN)
print_evaluations('CASE 1', TP, FP, FN, TN)

# # CASE 2
print('=============================== CASE 2 ===============================')
# Freeze the model, train the top, unfreeze the model, retrain everything
# Unfreeze the model
model.trainable = True

# recompiling the model
model.compile(optimizer=Adam(learning_rate=1e-5), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy(), 
                                                                                      TruePositives(), 
                                                                                      FalsePositives(), 
                                                                                      FalseNegatives(),
                                                                                      TrueNegatives()])

# Train fine-tuned model
hist = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=15,
    callbacks=[RemoveGarbageCallback(), earlystop, reduce_lr]
)

# Evaluate the model
loss, acc, TP, FP, FN, TN = model.evaluate(validation_ds)
plot_history(hist.history)
plot_confusion_matrix(TP, FP, FN, TN)
print_evaluations('CASE 2', TP, FP, FN, TN)

# CASE 3
print('=============================== CASE 3 ===============================')
# Freeze some portion of base layers, remove remaining base layers, add convolutional layers on top
base_model = Xception(input_shape=IMG_SHAPE, include_top=False, weights='imagenet', pooling='avg')

base_model.trainable = False
fine_tune_at = len(base_model.layers) - 100

# Remove last layers
x = base_model.layers[fine_tune_at].output
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D()(x)

x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=1e-4), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy(), 
                                                                                      TruePositives(), 
                                                                                      FalsePositives(), 
                                                                                      FalseNegatives(),
                                                                                      TrueNegatives()])

# Training
hist = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=15,
    callbacks=[RemoveGarbageCallback(), earlystop, reduce_lr]
)

# Evaluate the model
loss, acc, TP, FP, FN, TN = model.evaluate(validation_ds)
plot_history(hist.history)
plot_confusion_matrix(TP, FP, FN, TN)
print_evaluations('CASE 3', TP, FP, FN, TN)