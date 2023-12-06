import tensorflow as tf
import numpy as np

train_images_dir = './dataset/images/train'
validation_images_dir = './dataset/images/validation'
test_images_dir = './dataset/images/test'

class_names = ['nonfractured', 'fractured']

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# data augmentation layer
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

rescale = tf.keras.applications.xception.preprocess_input   # between -1 and 1

def process_dataset_generator(ds_generator, augment=False):
    # Apply normalization to all datasets
    ds_generator = ds_generator.map(lambda x, y: (rescale(x), y), 
                num_parallel_calls=tf.data.AUTOTUNE)
    
    # Use data augmentation only on the training set
    if augment:
        ds_generator = ds_generator.map(lambda x, y: (data_augmentation(x, training=True), y), 
                    num_parallel_calls=tf.data.AUTOTUNE)
    
    # Use buffered prefetching on all datasets
    return ds_generator.prefetch(buffer_size=tf.data.AUTOTUNE)

def consume_dataset_generator(dataset_generator):
  images = np.concatenate(list(dataset_generator.map(lambda x, y: x)))
  images = images.reshape(images.shape[0], -1)

  labels = np.concatenate(list(dataset_generator.map(lambda x, y: y)))
  labels = labels.reshape((-1,))

  return images, labels

# return either processed dataset generator or processed in memory dataset
def get_dataset(split, in_memory=False):
  if split != 'train' and split != 'validation' and split != 'test':
     print('Invalid split name. Possible splits are: train, validation, test')
     exit(1)

  if split == 'train':
    dataset_generator = tf.keras.utils.image_dataset_from_directory(train_images_dir, label_mode='binary', class_names=class_names, image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=42)
  elif split == 'validation':
    dataset_generator = tf.keras.utils.image_dataset_from_directory(validation_images_dir, label_mode='binary', class_names=class_names, image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=42)
  else:
    dataset_generator = tf.keras.utils.image_dataset_from_directory(test_images_dir, label_mode='binary', class_names=class_names, image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=42)

  dataset_generator = process_dataset_generator(dataset_generator, augment=(split == 'train'))

  if in_memory:
    return consume_dataset_generator(dataset_generator)
  
  return dataset_generator
