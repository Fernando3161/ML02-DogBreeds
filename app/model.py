import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.preprocessing import image
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import adam_v2

def create_model():
    labels_all = pd.read_csv("dog_dataset/labels.csv")
    breeds_all = labels_all["breed"]
    CLASS_NAMES = ["scottish_deerhound", "maltese_dog", "bernese_mountain_dog"]
    labels = labels_all[(labels_all["breed"].isin(CLASS_NAMES))]
    labels = labels.reset_index()

    # Creating numpy matrix with zeros
    X_data = np.zeros((len(labels), 224, 224, 3), dtype='float32')
    # One hot encoding
    Y_data = label_binarize(labels['breed'], classes = CLASS_NAMES)

    for i in tqdm(range(len(labels))):
        img = image.load_img('dog_dataset/train/%s.jpg' % labels['id'][i], target_size=(224, 224))
        img = image.img_to_array(img)
        x = np.expand_dims(img.copy(), axis=0)
        X_data[i] = x / 255.0

    # Building the Model
    model = Sequential()

    model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu', input_shape = (224,224,3)))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', kernel_regularizer = 'l2'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters = 16, kernel_size = (7,7), activation ='relu', kernel_regularizer = 'l2'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters = 8, kernel_size = (5,5), activation ='relu', kernel_regularizer = 'l2'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation = "relu", kernel_regularizer = 'l2'))
    model.add(Dense(64, activation = "relu", kernel_regularizer = 'l2'))
    model.add(Dense(len(CLASS_NAMES), activation = "softmax"))

    model.compile(loss = 'categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Splitting the data set into training and testing data sets
    X_train_and_val, X_test, Y_train_and_val, Y_test = train_test_split(X_data, Y_data, test_size = 0.1)
    # Splitting the training data set into training and validation data sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_and_val, Y_train_and_val, test_size = 0.2)

    # Training the model
    epochs = 100
    batch_size = 128

    history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
                        validation_data = (X_val, Y_val))

    model.save("dog_breed.h5")
