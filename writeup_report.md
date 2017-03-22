# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://i.imgur.com/oNs2tFy.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results
* pre_processor.py pre-process and save raw dataset
* process-yo.ipynb to showcase a single image pre-processing

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
##### Layers:
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

##### Size summary:
Layer (type)         |        Output Shape        |    Param 
------------ | -------------|-------------|
lambda_1 (Lambda)    |       (None, 32, 128, 3)   |    0
conv2d_1 (Conv2D)    |       (None, 14, 62, 24)   |    1824
conv2d_2 (Conv2D)    |       (None, 11, 59, 36)   |    13860     
conv2d_3 (Conv2D)    |       (None, 7, 28, 48)    |    43248     
conv2d_4 (Conv2D)    |       (None, 3, 13, 64)    |    27712     
conv2d_5 (Conv2D)    |       (None, 1, 6, 64)     |    36928     
flatten_1 (Flatten)  |       (None, 384)          |    0         
dense_1 (Dense)      |       (None, 120)          |    46200     
dense_2 (Dense)      |       (None, 84)           |    10164     
dense_3 (Dense)      |       (None, 1)            |    85        

##### Model Origin - Inspiration(?):
This is a modified NVIDIA End to End Driving CNN from 
https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

##### Layers Explanation:
My version of this network is implemented in model.py line 44.

The network starts with a lambda layer for input normalization into (0, 1) scale.

Then 5 layers of convolution at decreasing kernel sizes and increasing feature layers.

All convolution layers activated with ReLU because non-linearity is a must.

Then they are then flattened into a fully connected layer for further associative linear relationships 
over high order features extracted from above convolution layers.

3 Dense layers achieve this higher order information squeezing in the end only a single output of steering angle.

I could add drop-out layers bur I didn't because I have already pre-processed the inputs diverse enough.
I didn't need them.

I could add max-pooling layers but I didn't add them neither because
I already go over my conv2d layers with strides,
I don't much have room to further reduction in dimensions.

#### 2. Attempts to reduce overfitting in the model

I have expressed my views on drop-out layers above.

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py  line 23). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 57).

#### 4. Appropriate training data

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to mimic nvidia's way with well preprocessed input set.

After all, we are trying to achieve same thing they did.

The main difference in my solution is that I preprocessed all images and saved, 
the generator only pulled the already done images.

I split my image and steering angle data into a training and validation set in 4 to 1 ratio.
 
Set training batch size to 1000 , validation batch size to 250.

The final step was to run the simulator to see how well the car was driving around track one.

There were no spots where the vehicle fell off the track!

I even set the vehicle to full throttle whole road and it didn't go out of the road at all.

The interesting part is that I generated around 24 000 images for training 
and only trained the image 3 epochs with 1000 images each time.

This means that even only 3000 well picked training images are enough for perfect driving on this road.

I could even further reduce this probably but haven't tried.

At the end of the process, the vehicle is able to drive autonomously around the track 
without leaving the road at maximum speed like forever.

#### 2. Final Model Architecture

Already stated above.

#### 3. Creation of the Training Set & Training Process

I didn't record my own driving at all because I believe that even the given data should 
be more than enough to perfectly drive on the road if manipulated well enough.

##### 3.1. Work of pre_processor.py:

**_Summary:_ pre_processor.py does a static processing prior to the model training. 
So we don't lose time while feeding the network with data while training**

There are too many low to non-steering inputs. I randomly ignored 1/3 of non-steery stuff. pre_processor_py line 88 to 92

My inputs: 3 camera images (center, left, right), steering angle

First I put these 3 images into this preprocessing pipeline:

![alt text][image1]

###### 3.1.1. Pre-Processing steps and reasons:
1. Crop
    * Get rid of top sky, trees and bottom steering wheel because we don't need them.
2. Resize
    * Resize to 128x32 because it is faster to train and has enough detail.
3. Add Gaussian Noise
    * Random noise makes data prone to external noises and makes it generalize better, 
    like vaccinating the immune system with weakened microbes.
4. Histogram equalization
    * This maximizes the contrast and makes network prone to possible lighting conditions or shadows.
5. Gaussian Blur
    * This smoothens the image and prevents sharpness,
    we don't want singular pixel spikes, rather smooth multi-pixel spikes.
6. Normalize
    * As always, helps heavily images  treated equally, stretches them into same color space.

Now we have nice 3 images. And I added +- 0.2 steering into left/right 
camera images because they look from opposite angles at the road.

Then I flipped them vertically and multiplied steering angles by -1 and now we have total 6 images.

And I saved them into "data/IMG_Augmented" folder.

Also made a "new_csv.csv" file to use them in actual model with generators.

All images are 128x32 size.

##### 3.2. Work of model.py

Then to use these I read from "new_csv.csv" and use those lines in generators.

I kept model.py so simple, it's there.

I randomly shuffled the data set and put 20% of the data into a validation set. 

I generally have around 24500 input images and it varies according to the 1/3 chance of dropping low steering data.

training batch size = 1000 , validation batch size = 250.

Only 3 epochs!

This means I managed pre-processing part so well that only 3000 images were enough to generalize the first track.

And it can go there at max speed like forever.

#### 4. Time
My hardware: i7-6700k, 16GB RAM, Nvidia GTX Titan X
pre_processor.py takes less than 15 seconds on 24108 images that came from simulator.
model.py:
Epoch 1/3 1000/1000 - 141s - loss: 0.0069 - mean_squared_error: 0.0069 - acc: 0.1182 - val_loss: 0.0276 - val_mean_squared_error: 0.0276 - val_acc: 0.1280
Epoch 2/3 1000/1000 - 193s - loss: 3.3157e-04 - mean_squared_error: 3.3157e-04 - acc: 0.1182 - val_loss: 0.0255 - val_mean_squared_error: 0.0255 - val_acc: 0.1283
Epoch 3/3 1000/1000 - 142s - loss: 3.0129e-04 - mean_squared_error: 3.0129e-04 - acc: 0.1182 - val_loss: 0.0250 - val_mean_squared_error: 0.0250 - val_acc: 0.1282
Total 7.93 minutes of training.

Definitely could be less but haven't tried yet.
