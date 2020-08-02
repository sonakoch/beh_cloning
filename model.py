
#importing libraries
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers import Lambda, Cropping2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from scipy import ndimage
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

lines =[]
with open('./data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines.pop(0)

#creating the generator to generate image samples
def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while 1: 
        shuffle(samples) #shuffling the total images
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                    for i in range(0,3):
                        
                        name = './data/IMG/'+batch_sample[i].split('/')[-1]
                        center_image = cv2.imread(name)
                        center_image = cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
                        #center_image=cv2.GaussianBlur(center_image,(3,3),0)
                        center_angle = float(batch_sample[3]) #getting the steering angle measurement
                        images.append(center_image)
                        
                        #Correction for left and right images
                        
                        if(i==0):
                            angles.append(center_angle)
                        elif(i==1):
                            angles.append(center_angle+0.2)
                        elif(i==2):
                            angles.append(center_angle-0.2)
                        
                        # Code for data augmentation
                        # We take the image and just flip it and negate the measurement
                        
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            angles.append(center_angle*-1)
                        elif(i==1):
                            angles.append((center_angle+0.2)*-1)
                        elif(i==2):
                            angles.append((center_angle-0.2)*-1)
                        #we have made 6 images from one image    
                        
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train) 

# generate training and test sets using the generator function
train_samples, test_samples = train_test_split(lines,test_size=0.15)
train_generator = generator(train_samples, batch_size=32)
test_generator = generator(test_samples, batch_size=32)



model = Sequential()
#here we use lambda and cropping within the model to preprocess the data, where lambda normalizes and cropping2d crops the images
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

model.add(Cropping2D(cropping=((70,25),(0,0))))           

#these are convolutional layers with elu activation
model.add(Convolution2D(32,5,5,subsample=(2,2)))
model.add(Activation('elu'))

model.add(Convolution2D(64,5,5,subsample=(2,2)))
model.add(Activation('elu'))

model.add(Convolution2D(128,5,5,subsample=(2,2)))
model.add(Activation('elu'))

model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))


model.add(Convolution2D(32,3,3))
model.add(Activation('elu'))

#flattening layer
model.add(Flatten())

#fully connected layer
model.add(Dense(100))
model.add(Activation('elu'))

#for overfitting we add dropout with 25% 
model.add(Dropout(0.25))

#layer 7- fully connected layer 1
model.add(Dense(50))
model.add(Activation('elu'))

#for overfitting we add dropout with 25% 
model.add(Dropout(0.25))


#layer 8- fully connected layer 1
model.add(Dense(10))
model.add(Activation('elu'))

#layer 9- fully connected layer
model.add(Dense(1)) 

#as an optimizer we use adam with a default lr=0.001
#we use mean squared error as our loss function
model.compile(loss='mse',optimizer='adam')
from workspace_utils import active_session
#with active_session():
 #   model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=test_generator,   nb_val_samples=len(test_samples), nb_epoch=4, verbose=1)
#printing model summary
#model.summary()

#saving the model
with active_session():
    history_keras=model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=test_generator,   nb_val_samples=len(test_samples), nb_epoch=5, verbose=1)
model.save('model.h5')

print(history_keras.history.keys())
# summarize history for accuracy
plt.plot(history_keras.history['accuracy'])
plt.plot(history_keras.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()
# summarize history for loss
plt.plot(history_keras.history['loss'])
plt.plot(history_keras.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
plt.show()