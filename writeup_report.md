# **Behavioral Cloning**

## Kimon Roufas

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./writeup/center_lane_driving.jpg "Center Lane Driving"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./writeup/center_lane_driving_flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 recording of my vehicle drivign autonomously at least once around

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

I did not modify the drive.py file from the original given.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I build the NVIDIA model with 5 CNN layers, a flatten layer, three fully-connected layers and then a final output layer. I did not modify NVIDIA's model. See model.py lines 151 - 163

The model includes RELU activation on the CNN layers to introduce nonlinearity (lines 154 - 158), and the data is normalized in the model using a Keras lambda layer (line 153).

#### 2. Attempts to reduce overfitting in the model

I considered adding dropout layers but opted to collecte more data instead. This helped combat overfitting. Had it not worked, I would have added dropout.

I did plot the training loss vs validation loss and was able to observe overfitting initially.

[The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 167).

I completed the objective of training a working model using 2 epochs, and then when preparing to submit I decided to train for 10 epochs to squeeze as much as possible out of the data. This however also tends to overfit the model, but in this case I was satisfied since I knew the model also worked after 2 epochs.

I used the left and right cameras as well. I added and subtracted a fixed correction angle to simulate the effect of the car driving closer to each edge and the desire for the car to turn towards the center.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the started data set and added to it with additional center lane driving. I had to gather some additional center driving in the dirt road area. I did not collect recovery data, instead I relied on the effect of the left/right cameras. I did however crop the data to eliminate information not relevant to the steering angle decision (this also sped up training).

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for picking a model architecture was simply to pick a known powerful model (the NVIDIA model) and then collect enough training data that with the additional flip augmentation would avoid overfitting.

The chosen model was appropriate because it had been shown to work already. I was concerned about overfitting such a powerful model but decided to rely on gathering the appropriate amount of data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I didn't modify the model, rather I collected more data and further augmented it by flipping each image and multiplying the steering angle by -1. I also used the left and right cameras.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I collected more center line data and relied on the left and right cameras to have the effect of training the desire to return the car to the center of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 151 - 163) consisted of the NVIDIA convolution neural network with the following layers (including cropping and normalization):

* Starting input shape: 160 x 320 x 3
* Crop the top 70 and bottom 25 rows of image pixels
* Normalize to the range of -0.5 to 0.5
* Convolution 24 deep, 5x5 kernel
* Convolution 36 deep, 5x5 kernel
* Convolution 48 deep, 5x5 kernel
* Convolution 64 deep, 3x3 kernel
* Convolution 64 deep, 3x3 kernel
* Flatten
* Fully-connected, 100 wide
* Fully-connected, 50 wide
* Fully-connected, 10 wide
* Output layer, 1 wide

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I captured over 3 laps of very precise center lane driving at a reasonably low speed. In order to collect very good center driving I used a little-known accessibility feature in MacOS allowing me to steer just by touching my Mac's trackpad with three fingers instead of click and dragging. This gave me very fine steering control. In addition, I didn't drive too fast :)

Here is an example image of center lane driving:

![alt text][image2]

I decided that it was good enough to rely on the left and right cameras to provide trainable material to the CNN with steering angles pointed towards the center of the road. This worked well. I did not record any recovery driving.

Initially, my model drove the car onto the dirt road. To fix this I recorded some additional center line driving starting just before the dirt road entrance to just after the entrance.

In one spot, the car doesn't turn quite as much as I would have preferred it to. This is because of bias in the training set. Most of the data contains smaller steering angles. I could have collected additional data on the sharp turns, but since the car stayed on the road, I decided to leave it as is.

I wrote my data handling code in such a way that I could add and remove training sets (e.g. the dirt road entrance section, etc.) easily. This was very useful to save time managing data since I wanted to be able to easily remove data that was not contributing.

I did record some less precise center line steering but realized that it was not helping with recovery because it also added just as much bad steering away from the center as well. Thus, I removed this part of the dataset.

For sections that required a bit more data, I planned to bias the full dataset by replicating the same data additional times. This however was never needed since my car drives around the track continuously already.

I did include data augmentation by flipping the images and angles thinking that this would strengthen the model's flexibility without having to collect additional data painstakingly (or drive in reverse around the track) For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image7]

After the collection process, I had 63,609 number of data points including all three cameras. Post-processing this generated 127,218 data samples.

I finally randomly shuffled the data and put 20% of it into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I tested training on 2 epochs and 10 epochs. 2 worked just fine. The training loss was lower than the validation loss, but only by a very small amount.
