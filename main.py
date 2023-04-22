import glob
import cv2
import os
from os import path
import tensorflow as tf
import os
import random

# os.chdir("cat-data/cat-data/NonCats")
# for file in os.listdir():
#     name,ext = path.splitext(file)
#     if ext == '.png': # [jpeg, png]
#         dst= '{}.jpg'.format(name)
#         os.rename(file, dst)

# cnt = 0
# for img in glob.glob("cat-data/cat-data/Cats/*.jpg"):
#     n = cv2.imread(img, cv2.IMREAD_COLOR)
#     img_resize = cv2.resize(n, (224, 224), interpolation=cv2.INTER_AREA)
#     cv2.imwrite('Data/Cats/cat_' + str(cnt) + '.jpg', img_resize)
#     cnt += 1

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
from tensorflow import keras
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3

epochs = 30
data_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
#data_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2,
                              #rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, vertical_flip=False)
train_generator = data_gen.flow_from_directory('Data', target_size=(224, 224), batch_size=32, class_mode='binary', subset='training', shuffle=True)
valid_generator = data_gen.flow_from_directory('Data', target_size=(224, 224), batch_size=32, class_mode='binary', subset='validation', shuffle=True)

train_labels = train_generator.classes
total_size = train_labels.shape[0]
noncat_count = np.count_nonzero(train_labels==0)
cat_count = np.count_nonzero(train_labels==1)
class_weight = {0: cat_count/total_size, 1: noncat_count/total_size}

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable=False
x = base_model.output
x = Flatten()(x)
x = Dense(100, activation='gelu')(x)
x = Dense(100, activation='gelu')(x)
x = Dense(100, activation='gelu')(x)
x = Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', mode='max', patience=5, verbose=1, min_delta=0)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-7)
model_checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
step_size_train = train_generator.n//train_generator.batch_size
step_size_valid = valid_generator.n//valid_generator.batch_size
model.fit(train_generator, steps_per_epoch=step_size_train, epochs=epochs, validation_data=valid_generator,
          validation_steps=step_size_valid, callbacks=[early_stopping, reduce_lr, model_checkpoint], class_weight=class_weight)