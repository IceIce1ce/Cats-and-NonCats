import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import classification_report

load_model = tf.keras.models.load_model('model.h5')
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('test', target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)
y_pred = load_model.predict(test_generator)
y_true = test_generator.classes
f1 = f1_score(y_true, y_pred > 0.5)
print('F1 score:', f1)
y_pred = np.round(y_pred)
target_names = ['Class 0', 'Class 1']
print(classification_report(y_true, y_pred, target_names=target_names))

# import os
# os.getcwd()
# collection = "test/NonCats"
# cnt = 1550
# for i, filename in enumerate(os.listdir(collection)):
#     os.rename("test/NonCats/" + filename, "test/NonCats/noncat_" + str(cnt) + ".jpg")
#     cnt += 1