import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

# from keras.models import load_model
# model = load_model('saved_models/keras_traffic_signs_trained_model.h5')

NUM_CLASSES = 7
WIDTH=52
HEIGHT=52

epochs = 100
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_traffic_signs_trained_model.h5'

data = pd.read_csv('data.csv')

train, test, class_train, class_test = train_test_split(data['Filename'], data['ClassId'], test_size=0.3, random_state=42)

images = []
for index, value in train.iteritems():
    image = preprocessing.image.load_img(value, target_size=(WIDTH, HEIGHT))
    images.append(preprocessing.image.img_to_array(image))

train = np.array(images)

images = []
for index, value in test.iteritems():
    image = preprocessing.image.load_img(value, target_size=(WIDTH, HEIGHT))
    images.append(preprocessing.image.img_to_array(image))

test = np.array(images)

class_train = keras.utils.to_categorical(class_train, NUM_CLASSES)
class_test = keras.utils.to_categorical(class_test, NUM_CLASSES)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(52, 52, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

train /=255
test /=255

# model.fit(train, class_train, batch_size=32, epochs=epochs, validation_data=(test, class_test), shuffle=True)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

# img_test = preprocessing.image.array_to_img(random.choice(test))
# # img_test = preprocessing.image.load_img('data/meta/14.png',target_size=(WIDTH, HEIGHT))
# plt.imshow(img_test)
# plt.show()

# img_test = np.expand_dims(img_test, axis=0)

# predictions = model.predict(img_test)
# print(predictions)

scores = model.evaluate(test, class_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
