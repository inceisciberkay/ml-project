import tensorflow as tf
from keras.applications.xception import Xception
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from dataset_preparer import get_dataset, IMG_SIZE

IMG_SHAPE = IMG_SIZE + (3,)

train_ds = get_dataset('train')
validation_ds = get_dataset('validation')
test_ds = get_dataset('test')

# Create the base model from the pre-trained model MobileNet V2
base_model = Xception(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False

global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(1)

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)  # input is already preprocessed
x = global_average_layer(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate=base_learning_rate),
              loss=BinaryCrossentropy(from_logits=True),
              metrics=[BinaryAccuracy(threshold=0, name='accuracy')])

initial_epochs = 10
loss0, accuracy0 = model.evaluate(validation_ds)

history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=validation_ds)