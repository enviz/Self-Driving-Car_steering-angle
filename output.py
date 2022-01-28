#let us import all the necessary packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import pi
import cv2
import scipy.misc
import tensorflow as tf

DATA_FOLDER = "driving_dataset/"
DATA_FILE = os.path.join(DATA_FOLDER, "data.txt")

x = []
y = []

train_batch_pointer = 0
test_batch_pointer = 0



with open(DATA_FILE) as f:
    for line in f:
        image_name, angle = line.split()
        
        image_path = os.path.join(DATA_FOLDER, image_name)
        x.append(image_path)
        
        angle_radians = float(angle) * (pi / 180)  #converting angle into radians
        y.append(angle_radians)
y = np.array(y)
print(str(len(x))+" "+str(len(y)))




#using 70-30 split of train and test data
split_ratio = int(len(x) * 0.7)

train_x = x[:split_ratio]
train_y = y[:split_ratio]

test_x = x[split_ratio:]
test_y = y[split_ratio:]

print("Split ratio")
print('-'*50)
print('Train dataset:',len(train_x)/len(x)*100,'%\n','size:',len(train_x))

print('Test dataset:',len(test_x)/len(x)*100,'%\n','size:',len(test_x))


def loadTrainBatch(batch_size):
    global train_batch_pointer
    x_result = []
    y_result = []
    for i in range(batch_size):
        read_image = cv2.imread(train_x[(train_batch_pointer + i) % len(train_x)]) 
        read_image_road = read_image[-150:] 
        read_image_resize = cv2.resize(read_image_road, (200, 66)) 
        read_image_final = read_image_resize/255.0
        x_result.append(read_image_final) #finally appending the image pixel matrix
        
        y_result.append(train_y[(train_batch_pointer + i) % len(train_y)]) #appending corresponding labels
        
    train_batch_pointer += batch_size
        
    return x_result, y_result



def loadTestBatch(batch_size):
    global test_batch_pointer
    x_result = []
    y_result = []
    for i in range(batch_size):
        read_image = cv2.imread(test_x[(test_batch_pointer + i) % len(test_x)]) 
        read_image_road = read_image[-150:] 
        read_image_resize = cv2.resize(read_image_road, (200, 66)) 
        read_image_final = read_image_resize/255.0
        x_result.append(read_image_final) 
        
        y_result.append(test_y[(test_batch_pointer + i) % len(test_y)]) 
        
    test_batch_pointer += batch_size
        
    return x_result, y_result

def weight_variable(shape):
    initial = tf.truncated_normal(shape = shape, stddev = 0.1)
    return tf.Variable(initial) 

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(previous_input, filter_input, strides,name):
    print(name,filter_input.shape,filter_input.shape[-1])
    return tf.nn.conv2d(previous_input, filter_input, strides = [1, strides, strides, 1], padding = "VALID")

def Dense(X, size, name):
    w = weight_variable(shape=size)
    b = weight_variable(shape=[size[-1]])
    
    dense = tf.matmul(X, w) + b
    print(name, size, size[-1])
    ## Applying activation

    
    h_fc = tf.nn.relu(dense)
    
    
    return h_fc


def flatten(X, size):
    return tf.reshape(X, [-1, size])



x_input = tf.placeholder(tf.float32, shape = [None, 66, 200, 3], name = "Plc_1")
y_true = tf.placeholder(tf.float32, name = "Plc_2")

input_image = x_input

keep_prob = tf.placeholder(tf.float32)

#Convolution Layers
#First convolution layer
W_Conv1 = weight_variable([5,5,3,24])
B_Conv1 = bias_variable([24])
Conv1 = tf.nn.relu(conv2d(input_image, W_Conv1, 2,name='conv2d_1') + B_Conv1)
#tf.nn.conv2d(previous_input, filter_input, strides = [1, strides, strides, 1], padding = "VALID")

#Second convolution layer
W_Conv2 = weight_variable([5,5,24,36])
B_Conv2 = bias_variable([36])
Conv2 = tf.nn.relu(conv2d(Conv1, W_Conv2, 2,name='conv2d_2') + B_Conv2)


#Third convolution layer
W_Conv3 = weight_variable([5,5,36,48])
B_Conv3 = bias_variable([48])
Conv3 = tf.nn.relu(conv2d(Conv2, W_Conv3, 2,name='conv2d_3') + B_Conv3)


#Fourth convolution layer
W_Conv4 = weight_variable([3,3,48,64])
B_Conv4 = bias_variable([64])
Conv4 = tf.nn.relu(conv2d(Conv3, W_Conv4, 1,name='conv2d_4') + B_Conv4)



#Fifth convolution layer
W_Conv5 = weight_variable([3,3,64,64])
B_Conv5 = bias_variable([64])
Conv5 = tf.nn.relu(conv2d(Conv4, W_Conv5, 1,name='conv2d_5') + B_Conv5)





# Flatten layer

h_conv5_flatten = flatten(Conv5, size=1152)


# Dense layer 1
h_fc1 = Dense(h_conv5_flatten, (1152, 1164), name='dense1')
# Dropout 1
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Dense Layer 2
h_fc2 = Dense(h_fc1_drop, (1164, 100), name='dense2')
# Dropout 2
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Dense Layer 3
h_fc3 = Dense(h_fc2_drop, (100, 50), name='dense3')
# Dropout 3
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

# Dense Layer 4
h_fc4 = Dense(h_fc3_drop, (50, 10), name='dense4')

# Dropout 4
h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)


# Output
W_fc5 = weight_variable(shape = [10, 1])
b_fc5 = bias_variable(shape = [1])
y = tf.matmul(h_fc4_drop, W_fc5) + b_fc5
y_predicted = y






sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save_model/self_driving_car_model_new.ckpt")

print('press q to quit')






img = cv2.imread('steering_wheel_image.png', 0)
rows, cols = img.shape

i = 0

while(cv2.waitKey(20) != ord("q")):
    full_image = cv2.imread(test_x[i])
    cv2.imshow('Frame Window', full_image)
    image = ((cv2.resize(full_image[-150:], (200, 66)) / 255.0).reshape((1, 66, 200, 3)))
    degrees = sess.run(y_predicted, feed_dict = {x_input: image, keep_prob: 1.0})[0][0] *180 / pi
    print("Predicted degrees: "+str(degrees))
    M = cv2.getRotationMatrix2D((cols/2,rows/2), -degrees, 1)
    dst = cv2.warpAffine(src = img, M = M, dsize = (cols, rows))
    cv2.imshow("Steering Wheel", dst)
    i += 1

cv2.destroyAllWindows()


