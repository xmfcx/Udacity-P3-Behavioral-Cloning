import csv
import cv2
import numpy as np
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle

# read augmented csv into rows
rows = []
with open('data/new_csv.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        rows.append(row)

print(len(rows))

# split rows into train and validation
train_rows, validation_rows = train_test_split(rows, test_size=0.2)


# generator that will return image and label in required batch size
def generator(rows, batch_size: int):
    while 1:
        shuffle(rows)
        for offset in range(0, len(rows), batch_size):
            batch_rows = rows[offset:offset + batch_size]
            images, angles = [], []
            for row in batch_rows:
                images.append(cv2.imread(row[0]))
                angles.append(float(row[1]))

            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)


input_shape = (32, 128, 3)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
model.add(Conv2D(36, (4, 4), activation='relu', strides=(1, 1)))
model.add(Conv2D(48, (5, 5), activation='relu', strides=(1, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.summary()

adam = Adam(lr=0.001)
model.compile(loss='mse',
              optimizer=adam,
              metrics=['mse', 'accuracy'])

batch_size_train = 1000
batch_size_validate = batch_size_train / 4

train_generator = generator(train_rows, batch_size=batch_size_train)
validation_generator = generator(validation_rows, batch_size=batch_size_train)

model.fit_generator(train_generator,
                    epochs=3,
                    steps_per_epoch=batch_size_train,
                    validation_data=validation_generator,
                    validation_steps=batch_size_validate)

model.save('model.h5')

k.clear_session()
exit()
