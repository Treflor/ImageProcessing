# ImageProcessing

Tools used
  -Tensorflow
  -Google Colab
 
# Import dependencies and modules

import tensorflow as tf
import os
import numpy as np
import zipfile
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, Model
from __future__ import division, absolute_import, print_function, unicode_literals
from tensorflow.keras.utils import Sequence
try:
  %tensorflow_version 2.x
except Exvception:
  pass
  
 # Check tensorflow version
 
tf.__version__ 


# Download and unzipping the dataset

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL, fname="flower_photos.tgz", extract=True)
base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

# Create data generator

img_size=224
batch_size =64

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    base_dir,
    target_size = (img_size,img_size),
    batch_size=batch_size,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    base_dir,
    target_size = (img_size,img_size),
    batch_size=batch_size,
    subset='validation'
)

# Check shapes

for image_batch , label_batch in train_gen:
  break
print('Shepe of the image batch : ')
print(image_batch.shape)
print('Shepe of the label batch : ')
print(label_batch.shape)

# Create labels.txt

print(train_gen.class_indices)

labels = "\n".join(sorted(train_gen.class_indices.keys()))
with open('labels.txt','w') as f:
  f.write(labels)
  
!cat labels.txt


# Create the base model from the pre-trained model MobileNet V2
img_shape = (img_size, img_size, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                              include_top=False, 
                                              weights='imagenet')
                                              
 #
 base_model.trainable = false
 
 #
 model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(32,3, activation ='relu'),
  tf.keras.layers.Dropout(0,2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(5, activation = 'softmax')
])

#
model.compile(optimizer=tf.keras.optimizer.Adam(),
              loss='categorical_crossentropy',
              matrics=['accuracy'])
              
#
model.summary()

#
print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

#
epochs = 10

history = model.fit_generator(train_gen,
                              epochs=epochs,
                              validation_data=val_gen)
                              
#
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#
base_model.trainable = True

#
print('Number of layers in the base model : ', len(base_model.layers))
fine_tune_at = 100

for layer in base.mode.layers[:fine_tune_at]:
  layer.trainable = False

#
model.compile(loss='categorical_crossentropy',
              optimizer = tf.keras.optimizer.Adam(1e-5),
              matrics=['accuracy'])
model.summary()            
              
#
print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

#
history_fine_tune =model.fit_generator(
    train_gen,
    epochs=5,
    validation_data = val_gen
)

#
saved_model_dir = 'save/fine_tuning'
tf.saved_model.save(model,saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('model.tflite','wb') as f:
  f.write(tflite_model)
  
# Download model.tflite & labels.txt
from google.colab import files

files.download('model.tflite')
files.download('labels.txt')


#
acc = history_fine_tune.history['accuracy']
val_acc = history_fine_tune.history['val_accuracy']

loss = history_fine_tune.history['loss']
val_loss = history_fine_tune.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()





 





