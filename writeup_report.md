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

I build the NVIDIA model with 5 CNN layers, a flatten layer, three fully-connected layers and then a final output layer. I modified NVIDIA's model by adding dropout layers after each convolution layer. See model.py lines 154 - 170

The model includes RELU activation on the CNN layers to introduce nonlinearity (lines 157 - 165), and the data is normalized in the model using a Keras lambda layer (line 156).

#### 2. Attempts to reduce overfitting in the model

I added dropout layers to prevent overfitting and make the model more robust. Without dropout, I did see some overfitting initially by observing a plot of training loss compared to validation loss.

The model was trained and validated on combinations of data sets to ensure that the model was not overfitting (code line 20-25). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. I eliminated the unecessary data sets.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 174).

I completed the objective of training a working model using 2 epochs. During the development I did train 10 epochs but realized that it wasn't necessary and was just wasting time. More epochs would also have the tendency to overfit the model, which I wanted to avoid.

I used the left and right cameras as well. I added and subtracted a fixed correction angle to simulate the effect of the car driving closer to each edge and the desire for the car to turn towards the center.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the starter data set provided and added to it with additional center lane driving. I had to gather some additional center driving in the dirt road area and the area with a sharp right turn.

I did not collect recovery data, instead I relied on the effect of the left/right cameras. I did however crop the data to eliminate information not relevant to the steering angle decision (this also sped up training).

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for picking a model architecture was simply to pick a known powerful model (the NVIDIA model) and then modify it with dropout layers to avoid overfitting.

The chosen model was appropriate because it had been generally shown to work for others developing self-driving capabilities.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation sets. Initially, I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I added dropout layers and collected more data. I augmented it by flipping each image and multiplying the steering angle by -1. I also used the left and right cameras.

The final step was to run the simulator to see how well the car was driving around track 1. Before combating overfitting, there were a few spots where the vehicle fell off the track... but with the additional data and dropout layers, all was fixed. Early on I had decided to approach the training by relying on the left/right cameras instead of collecting recovery data. This worked well.

By the end of the process, the vehicle was able to drive autonomously around the track quite near the center, without any weaving left or right, and, of course, without departing the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 151 - 163) consisted of the NVIDIA convolution neural network with the following layers (including cropping and normalization):

* Starting input shape: 160 x 320 x 3
* Crop the top 70 and bottom 25 rows of image pixels
* Normalize to the range of -0.5 to 0.5
* Convolution 24 deep, 5x5 kernel
* Dropout, keep probability 75%
* Convolution 36 deep, 5x5 kernel
* Dropout, keep probability 75%
* Convolution 48 deep, 5x5 kernel
* Dropout, keep probability 50%
* Convolution 64 deep, 3x3 kernel
* Dropout, keep probability 50%
* Convolution 64 deep, 3x3 kernel
* Flatten
* Fully-connected, 100 wide
* Fully-connected, 50 wide
* Fully-connected, 10 wide
* Output layer, 1 wide

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I captured over 3 laps of very precise center lane driving at a reasonably low speed. In order to collect very good center driving I used a little-known accessibility feature in MacOS allowing me to steer just by touching my Mac's trackpad with three fingers instead of click and dragging (which is very difficult to maintain for a prolonged period). This gave me very fine steering control. In addition, I didn't drive too fast :)

Here is an example image of center lane driving:

![alt text][image2]

I decided that it was good enough to rely on the left and right cameras to provide trainable material to my model with steering angles pointed towards the center of the road. This worked well. I did not record any recovery driving.

Initially, my model drove the car onto the dirt road. To fix this I recorded some additional center line driving starting just before the dirt road entrance to just after the entrance.

I had a similar problem on a tight right turn. Again, additional center lane data helped out.

I wrote my data handling code in such a way that I could add and remove training sets (e.g. the dirt road entrance section, etc.) easily. This was very useful to save time managing data since I wanted to be able to easily remove data that was not contributing.

Experimentally, I recorded some less precise center line steering but realized that it was not helping with recovery. I think that it basically  added just as much bad steering away from the center as well as back towards the center. Thus, I removed this data subset.

I expected and planned for using multiple subsets of the same data to bias towards the good behaviors I wanted to see in the difficult areas, but in the end this was not needed.

I did include data augmentation by flipping the images and angles thinking that this would strengthen the model's flexibility without having to collect additional data painstakingly (or drive in reverse around the track) For example, here is an image along with its flipped version:

![alt text][image2]
![alt text][image7]

After all was said and done with data collection, use of all three cameras, and augmentation, I had generated 130,776 data samples.

I randomly shuffled the data and put 20% of it into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I tested training on 2 epochs and 10 epochs. 2 worked just fine and was much quicker during model development and testing. The training loss and validation loss values were comperable.
