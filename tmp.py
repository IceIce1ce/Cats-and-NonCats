import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Concatenate
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
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3, EfficientNetV2S
from tensorflow.keras.preprocessing import image_dataset_from_directory

epochs = 30
data_augmentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(224, 224, 3)),
                                         tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
                                         tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
                                         tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1)])
train_generator = image_dataset_from_directory('Gray_Data', image_size=(224, 224), batch_size=4, subset='training', shuffle=True,
                                               validation_split=0.2, seed=42, crop_to_aspect_ratio=True, interpolation='lanczos5')
train_generator = train_generator.map(lambda x, y: (data_augmentation(x), y))
valid_generator = image_dataset_from_directory('Gray_Data', image_size=(224, 224), batch_size=4, subset='validation', shuffle=True,
                                               validation_split=0.2, seed=42, crop_to_aspect_ratio=True, interpolation='lanczos5')

base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
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
model.fit(train_generator, epochs=epochs, validation_data=valid_generator, callbacks=[early_stopping, reduce_lr, model_checkpoint])