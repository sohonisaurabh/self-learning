import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import cv2
import numpy as np

def load_lines(path):
    lines = []
    with open(path + "/driving_log.csv") as datafile:
        reader = csv.reader(datafile)
        for line in reader:
            lines.append(line)
    return lines

def get_data_generator(batch_size, lines):
    offset = 0
    while 1:
        images = []
        steering_angles = []
        shuffled_lines = shuffle(lines)
        line_batch = shuffled_lines[offset : offset + batch_size]
        #line_batch = lines[offset : offset + batch_size]
        for line in line_batch:
            center_image = cv2.imread(line[0])
            steering_angle_center = float(line[3])
            images.append(center_image)
            steering_angles.append(steering_angle_center)
        offset += batch_size
        yield (np.array(images), np.array(steering_angles))

def get_data_without_generator(path, lines):
    images = []
    steering_angles = []
    steering_offset = 0.25
    print(path + "/" + lines[0][0].split("\\")[-1])
    for line in lines:
        center_image = cv2.imread(path + "/IMG/" + line[0].split("\\")[-1])
        left_image = cv2.imread(path + "/IMG/" + line[1].split("\\")[-1])
        right_image = cv2.imread(path + "/IMG/" + line[2].split("\\")[-1])
        steering_angle_center = float(line[3])
        steering_angle_left = steering_angle_center + steering_offset
        steering_angle_right = steering_angle_center - steering_offset
        images.extend([center_image, left_image, right_image])
        steering_angles.extend([steering_angle_center, steering_angle_left, steering_angle_right])
    return np.array(images), np.array(steering_angles)
def train_model(training_generator, validation_generator, batch_size=50, epochs=10):
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.fit(training_generator, validation_generator, validation_split=0.3, shuffle=True, nb_epoch=20)
    #model.fit_generator(training_generator, nb_epoch=epochs, samples_per_epoch=batch_size)
    model.save("CarND-Behavioral-Cloning-P3/model-fit.h5")

def train_model_lenet(training_generator, validation_generator, batch_size=50, epochs=10):
    model = Sequential()
    model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape=(160, 320, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(16, 5, 5, border_mode='valid', input_shape=(156, 316, 6)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten(input_shape=(152, 312, 16)))
    model.add(Dense(120))
    model.add(Activation("relu"))
    model.add(Dense(84))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.fit(training_generator, validation_generator, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    model.save("CarND-Behavioral-Cloning-P3/model-lenet.h5")
    
def main():
    batch_size = 50
    data_path = "recorded-data"
    lines = load_lines(data_path) 
    training_set_lines, validation_set_lines = train_test_split(lines, test_size=0.3)
    number_of_samples = len(training_set_lines)
    epochs = int(number_of_samples/batch_size)

    #training_generator = get_data_generator(batch_size, training_set_lines)
    #validation_generator = get_data_generator(batch_size, validation_set_lines)
    #train_model(training_generator, validation_generator, batch_size=batch_size, epochs=epochs)

    training_images, steering_angles = get_data_without_generator(data_path, lines)
    #train_model(training_images, steering_angles)
    train_model_lenet(training_images, steering_angles, epochs=4)

main()
