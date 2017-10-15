import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Convolution2D, Flatten, Dense
from keras.layers.pooling import MaxPooling2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# CONSTANTS
EPOCHS = 2
BATCH_SIZE = 32
STEERING_OFFSET_CORRECTION = 0.2 # TODO: tune this parameter for the left/right sterring correction

# DATASETS TO USE FOR TRAINING
DATASETS = [
    './data_starter/',
    './data_3_laps/',
    './data_dirt_road_entrance/',
    './data_sharp_right/'
]


def fix_path(newpath, old_filename):
    parts = old_filename.split('/')
    return newpath + 'IMG/' + parts[-1]


samples = []
for dataset_path in DATASETS:
    with open(dataset_path + 'driving_log.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # check if our csv has headers and reject any such rows
            if (line[0] == 'center'):
                continue

            # limit how many lines we do for debugging purposes
            # if (len(lines) > 5):
            #     continue

            # print(line)
            steering_center = float(line[3])
            steering_left = steering_center + STEERING_OFFSET_CORRECTION
            steering_right = steering_center - STEERING_OFFSET_CORRECTION

            # in order to properly shuffle all the samples, I need to store all the combinations of center/left/right camera images, as well as the augmentation copies. The boolean represents whether this particular sample should be flipped later.
            samples.append([fix_path(dataset_path, line[0]), steering_center, True])
            samples.append([fix_path(dataset_path, line[1]), steering_left, True])
            samples.append([fix_path(dataset_path, line[2]), steering_right, True])
            samples.append([fix_path(dataset_path, line[0]), steering_center, False])
            samples.append([fix_path(dataset_path, line[1]), steering_left, False])
            samples.append([fix_path(dataset_path, line[2]), steering_right, False])


print('Dataset samples to be processed: {}'.format(len(samples)))

# SPLIT DATA INTO TRAINING AND VALIDATION
# Note: not exacly a perfect split since each 'line' represents a triad of images (left, center, right)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# NON-GENERATOR VERSION OF THE CODE
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------
# images = []
# steering_angles=[]


# # for line in lines[0:100:]:
# for line in lines:
#     # parse the three images from each csv row
#     image_center = cv2.imread(line[0])
#     image_left = cv2.imread(line[1])
#     image_right = cv2.imread(line[2])
#     images.extend([image_center, image_left, image_right])

#     # parse the center camera steering angle and calculate corrected approximate angles for the left and right cameras
#     steering_center = float(line[3])
#     steering_left = steering_center+correction
#     steering_right = steering_center-correction
#     steering_angles.extend([steering_center, steering_left, steering_right])


# # DATA AUGMENTATION: flip all the images and steering angles
# augmented_images, augmented_steering_angles = [], []
# for image, steering_angle in zip(images, steering_angles):
#     augmented_images.append(image)
#     augmented_steering_angles.append(steering_angle)
#     augmented_images.append(cv2.flip(image, 1))
#     augmented_steering_angles.append(steering_angle*-1.0)

# X_train = np.array(augmented_images)
# y_train = np.array(augmented_steering_angles)

# print('X_train shape: {}'.format(X_train.shape))
# print('y_train shape: {}'.format(y_train.shape))
# -------------------------------------------
# -------------------------------------------
# -------------------------------------------

# SAMPLE BATCH GENERATOR
# for each sample, takes the tuple of image filename, steering angle and whether the sample should be flipped or not
def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0])
                steering_angle = batch_sample[1]
                if (batch_sample[2] == True):
                    image = cv2.flip(image, 1)
                    steering_angle = steering_angle * -1.0
                images.append(image)
                steering_angles.append(steering_angle)

            X_train = np.array(images)
            y_train = np.array(steering_angles)

            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


# MODEL DEFINITION
# this is the first model I used before implementing a more powerful model
# model = Sequential()
# model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=(160, 320, 3)))
# model.add(Lambda(lambda x: x / 255.0 - 0.5))
# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))


# MODEL DEFINITION
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


# COMPILE, TRAIN, SAVE THE MODEL
model.compile(loss='mse', optimizer='adam')

# use the following with the non-generator code to train the model
# history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=EPOCHS, verbose=1)

# train the model using the generator function
history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data =
    validation_generator,
    nb_val_samples = len(validation_samples),
    nb_epoch=EPOCHS, verbose=1)

# save the model!
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss_visualization.jpg')

exit()

