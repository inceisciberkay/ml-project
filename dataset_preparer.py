import tensorflow as tf
import numpy as np
import random

SEED = 42
def seed_everything():
  np.random.seed(SEED)
  tf.random.set_seed(SEED)
  random.seed(SEED)

train_images_dir = './dataset/images/train'
validation_images_dir = './dataset/images/validation'
test_images_dir = './dataset/images/test'

class_names = ['nonfractured', 'fractured']

BATCH_SIZE = 4
IMG_SIZE = (224, 224)

rescale = tf.keras.applications.xception.preprocess_input   # between -1 and 1

def process_dataset_generator(ds_generator):
    # Apply normalization to all datasets
    ds_generator = ds_generator.map(lambda x, y: (rescale(x), y), 
                num_parallel_calls=tf.data.AUTOTUNE)
    
    # Use buffered prefetching on all datasets
    return ds_generator.prefetch(buffer_size=tf.data.AUTOTUNE)

def consume_dataset_generator(dataset_generator):
  images = np.concatenate(list(dataset_generator.map(lambda x, y: tf.reduce_mean(x, axis=3))))  # collapse third dimension
  images = images.reshape(images.shape[0], -1)

  labels = np.concatenate(list(dataset_generator.map(lambda x, y: y)))
  labels = labels.reshape((-1,))

  return images, labels

# return either processed dataset generator or processed in memory dataset
def get_dataset(split, in_memory=False):
  if split != 'train' and split != 'validation' and split != 'test':
     print('Invalid split name. Possible splits are: train, validation, test')
     exit(1)
    
  seed_everything()

  if split == 'train':
    dataset_generator = tf.keras.utils.image_dataset_from_directory(train_images_dir, label_mode='binary', class_names=class_names, image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED)
  elif split == 'validation':
    dataset_generator = tf.keras.utils.image_dataset_from_directory(validation_images_dir, label_mode='binary', class_names=class_names, image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED)
  else:
    dataset_generator = tf.keras.utils.image_dataset_from_directory(test_images_dir, label_mode='binary', class_names=class_names, image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED)

  dataset_generator = process_dataset_generator(dataset_generator)

  if in_memory:
    return consume_dataset_generator(dataset_generator)
  
  return dataset_generator
