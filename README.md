# beh_cloning
Udacity project N3
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./original.jpg 
[image2]: ./image_rgb.jpg
[image3]: ./image_gaus.jpg
[image4]: ./Recovery_from_left.jpg
[image5]: ./Recovery_from_right.jpg
[image6]: ./image_flip.jpg


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

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
The model I have chosen is from NVIDIA as suggested by Udacity(https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). This model takes (60,266,3) shape as an input which I have adjusted to fit (160,320,3). 
The model I used is leveraging keras library to build the network. 

a) The first layer consists of a lambda layer which is to normalize the image data(model.py line 83) 

b) then comes a cropping layer which resizes the images to  75x25

c) I used convolutional layers with 5x5 filter sizes with depths between 32, 64 and 128 (model.py lines 83-102)

d) convolutional layers with 3x3 filter sizes with depths between 64 and 32

e) a flatten layer followed by fully connected layers with sizes 100,50,10,1 with 2 dropout layers in between with 25% dropout rates

As an activation function I have used ELU, which is to introduce nonlinearity and cope with overfitting.

A generator was implemented to reduce memory usage(model.py line 26-66)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 108). Also, as mentioned above, ELU helps to reduce the overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 72-75 and 127). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track and moreover, to stay in the middle of the road.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually, it was the default 0.001 rate from the tf.keras library. Since with the other parameters I was able to construct a good model, there was no need for manual adjustments of the learning rate.
Here are the final parameters I have used:

No of epochs= 5

Optimizer Used- Adam

Learning Rate- Default 0.001

Validation Data split- 0.15

Generator batch size= 32

Correction factor- 0.2

Loss Function Used- MSE(Mean Squared Error) 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

Firstly, I used the the dataset provided by Udacity. I have used cv2 to load the images in RGB as in drive.py it is processed in RGB format.

Since we have a steering angle associated with three images we introduce a correction factor for left and right images as the steering angle is captured by the center angle.

My correction factor is 0.2.

For the left images I increase the steering angle by 0.2 and for the right images I decrease the steering angle by 0.2.

In addition, I used shuffling so that the order in which images come doesn't matters to the model.
To create more variety in the datasetm, I did augmentation. 

Augmenting the data- I have decided to flip the image horizontally and adjust steering angle accordingly, I used cv2 to flip the images.

After flipping I multiply the steering angle by a factor of -1 to get the steering angle for the flipped image. 

With this approach I was able to generate 6 images corresponding to one entry in .csv file

### Model Architecture and Training Strategy
#### 4. Solution Design Approach

The overall strategy for deriving a model architecture that will produce a model with the lowest MSE and no overfitting.

My first step was to use a convolution neural network model similar to the one in NVIDIA I thought this model might be appropriate because it addresses the similar task.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
To combat the overfitting, I have added the dropout layer, then I have reduced the number of epochs and added more training data. I have also driven the car backwards for 1 lane to generate more data and add some diversity in the images.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I have added more data and increased the number of epochs from 2 to 5.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 5. Final Model Architecture

The final model architecture (model.py lines 78-124) is described here & above in 2.

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
activation_1 (Activation)    (None, 31, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
activation_2 (Activation)    (None, 14, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
activation_3 (Activation)    (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
activation_4 (Activation)    (None, 3, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
activation_5 (Activation)    (None, 1, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
activation_6 (Activation)    (None, 100)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
activation_7 (Activation)    (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
activation_8 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11  
      
=================================================================

Total params: 348,219

Trainable params: 348,219

Non-trainable params: 0
_________________________________________________________________


 

#### 6. Creation of the Training Set & Training Process

To capture good driving behavior, I first used the udacity provided data. Here is an example image of center lane driving:

![alt text][image1]

This image was then converted to RGB as you can see here:

![alt text][image2]

Then I have applied Guassian blurring to eliminate the noise in the image. 

Here is how the image looked afterwards:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive in the center. 
Here are the images of recovery from the right & left:

![alt text][image4]

![alt text][image5]

Then I repeated this process by driving backwards in order to get more data points. 

To augment the dataset, I also flipped images and angles, as described above. For example, here is an image that has then been flipped:

![alt text][image6]


After the collection process, I had 8036 number of data points. 


I finally randomly shuffled the data set and put 15% of the data into a test set. 

I used this training data for training the model. The validation/test set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the small difference of training and val_loss, as well as driving behaviour. I used adam optimizer so that manually training the learning rate wasn't necessary.
