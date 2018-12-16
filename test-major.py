import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import bernoulli
import gzip
import itertools
import numpy as np
import os
import tensorflow as tf
import time
from skimage import io, color
from skimage import exposure
from PIL import Image,ImageOps
import itertools
import PIL


#data loading

test=[]
ytest=[]
n=1
h=180
w=160

path='data/'
for i in range(0,503):
  print(i)
  path1=path+str(i)
  if os.path.exists(path1):
      list3=os.listdir(path1)
      for elem in list3:
          img_path=path1+'/'+elem
          img1=io.imread(img_path)
          img1=color.rgb2gray(img1)
          img=np.reshape(img1,h*w*n)
          test.append(img)
          ytest.append(i)

data_test_images=np.array(test)

ytest=np.array(ytest)

# Dimensions of the data

in_height = h
in_width = w
num_channels = 1

image_size_flat = in_height * in_width*n
image_shape = (in_height, in_width)

# Number of classes
num_classes = 503

# Placeholder variables

# Placeholder variable for the input images
x = tf.placeholder(tf.float32, shape=[None, image_size_flat])

# Reshape 'x'
x_image = tf.reshape(x, shape=[-1, in_height, in_width, num_channels])

# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, num_classes])
y_true_cls = tf.argmax(y_true, dimension=1)


# Class labels are One-Hot coded, meaning that each label is a vector with 10 elements,all of which are zero except one element
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot

# Call function dense_to_one_hot()
data_train_labels  = dense_to_one_hot(labels_dense=ytrain, num_classes=num_classes)
data_test_labels  = dense_to_one_hot(labels_dense=ytest, num_classes=num_classes)



# Convolutional Layer 0

filter_height_0 = 1
filter_width_0 = 1
num_filters_0 = 3


# Convolutional Layer 1

filter_height_1 = 5
filter_width_1 = 3
num_filters_1 = 12

# Convolutional Layer 2

filter_height_2 = 5 
filter_width_2 = 3
num_filters_2 = 24

# Convolutional Layer 3

filter_height_3 = 5
filter_width_3 = 3
num_filters_3 = 36

# Convolutional Layer 4

filter_height_4 = 5
filter_width_4 = 3
num_filters_4 = 64

# Convolutional Layer 5

filter_height_5 = 5
filter_width_5 = 3
num_filters_5 = 128

# Fully-connected layer
# Number of neurons in fully-connected layer
fc_size1 = 1000
fc_size2 = 700

# Non-trainable filters
def get_filter0(): 
    filter=np.zeros((16,3,3),dtype=float)
    x=np.array([0,0,0,1,2,2,2,1])
    y=np.array([0,1,2,2,2,1,0,0])
    for i in range(0,8):
        filter[i][x[i]][y[i]]=-1
        filter[i][1][1]=1
    for i in range(8,16):
        filter[i][x[i]][y[i]]=1
        filter[i][1][1]=-1
    filter=np.swapaxes(filter,0,1)
    filter=np.swapaxes(filter,1,2)
    filter=filter.reshape(3,3,1,16)
    return filter

 
def get_filter1(gap):   
    x=np.array([0,0,0,1,2,2,2,1])
    y=np.array([0,1,2,2,2,1,0,0])
    filter=np.zeros((8,3,3),dtype=float)

    for i in range(0,8):
        filter[i][x[i]][y[i]]=1
        filter[i][x[(i+gap)%8]][y[(i+gap)%8]]=-1

    filter=np.swapaxes(filter,0,1)
    filter=np.swapaxes(filter,1,2)
    filter = filter.reshape(3, 3, 1, 8)
    return filter


def get_filter2(gap):   
    x=np.array([0,0,0,1,2,3,4,4,4,3,2,1])
    y=np.array([0,1,2,2,2,2,2,1,0,0,0,0])
    filter=np.zeros((12,5,3),dtype=float)

    for i in range(0,12):
        filter[i][x[i]][y[i]]=1
        filter[i][x[(i+gap)%12]][y[(i+gap)%12]]=-1

    filter=np.swapaxes(filter,0,1)
    filter=np.swapaxes(filter,1,2)
    filter = filter.reshape(5, 3, 1, 12)
    return filter

# Non-trainable filters initialized with distribution of Bernoulli as in article and then it's non-trainable
def new_weights_non_trainable(h,
                              w,
                              num_input,
                              num_output,
                              sparsity=0.5):
    
    # Number of elements
    num_elements = h * w * num_input * num_output
    
    # Create an array with n number of elements
    array = np.arange(num_elements)
    
    # Random shuffle it
    np.random.shuffle(array)
    
    # Fill with 0
    weight = np.zeros([num_elements])
    
    # Get number of elements in array that need be non-zero
    ind = int(sparsity * num_elements + 0.5)
    
    # Get it piece as indexes for weight matrix
    index = array[:ind]
  
    for i in index:
        # Fill those indexes with bernoulli distribution
        # Method rvs = random variates
        weight[i] = bernoulli.rvs(0.5)*2-1

    # Reshape weights array for matrix that we need
    weights = weight.reshape(h, w, num_input, num_output)

    return weights

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# Think of it as one block

def new_LBC_layer(input,              # The previous layer
                  filter_height,      # Height of each filter
                  filter_width,       # Width of each filter
                  in_channels,        # Num. channels in prev. layer
                  out_channels,       # # Number of filters
                  use_pooling=True):  # Use 2x2 max-pooling
    
    # The out_channels of the previous layer are the in_channels of the next layer

    # Shape of the filter-weights for the convolution
    # This format is determined by the TensorFlow API
    # shape = [filter_height, filter_width, in_channels, out_channels]

    # Non-trainable filters
    anchor_weights = tf.Variable(new_weights_non_trainable(h=filter_height,
                                                           w=filter_width,
                                                           num_input=in_channels,
                                                           num_output=out_channels).astype(np.float32),
                                                           trainable=False)
    
    # Difference Maps
    difference_maps = tf.nn.conv2d(input=input,
                                   filter=anchor_weights,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
    
    # Non-linear unit is ReLU, as in article
    
    # Bit Maps
    bit_maps = tf.nn.relu(difference_maps)
    
    # Set of learnable linear weights is a convolution with 1x1 kernels,
    # without bias and without non-linear unit

    shape = [1, 1, out_channels, 1]
    
    weights = new_weights(shape)
    
    
    # Feature Maps
    feature_maps = tf.nn.conv2d(input=bit_maps, 
                                filter=weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

    # Use pooling to down-sample the image resolution
    if use_pooling:
        feature_maps = tf.nn.max_pool(value=feature_maps,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # We will plot them later
    return feature_maps, anchor_weights, weights, difference_maps, bit_maps


def new_SB_layer(input,              # The previous layer
                  filter_height,      # Height of each filter
                  filter_width,       # Width of each filter
                  in_channels,        # Num. channels in prev. layer
                  out_channels,       # # Number of filters
                  use_pooling=True):  # Use 2x2 max-pooling
    
    # The out_channels of the previous layer are the in_channels of the next layer

    # Shape of the filter-weights for the convolution
    # This format is determined by the TensorFlow API
    # shape = [filter_height, filter_width, in_channels, out_channels]

    # Non-trainable filters
    filter=np.zeros((filter_height,filter_width,in_channels,out_channels),dtype=float)

    if out_channels==num_channels:
        shape = [1, 1, out_channels, 1]
        weights = new_weights(shape)
    
        feature_maps = tf.nn.conv2d(input=input, 
                                filter=weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

        return feature_maps, weights

    elif out_channels==8:
       filter=get_filter1(gap=1)
       if in_channels==3:
          filter=np.concatenate((filter,filter,filter),axis=2)

    elif out_channels==12:
       filter=get_filter2(gap=1)
       if in_channels==3:
          filter=np.concatenate((filter,filter,filter),axis=2)

    elif out_channels==16:
       filter1=get_filter1(gap=2)
       filter2=get_filter1(gap=4)
       filter=np.concatenate((filter1,filter2),axis=3)
       #filter=get_filter0()       

    elif out_channels==24:
       filter1=get_filter2(gap=2)
       filter2=get_filter2(gap=3)
       filter=np.concatenate((filter1,filter2),axis=3)

    elif out_channels==32:
       filter1=get_filter1(gap=1)
       filter2=get_filter1(gap=2)
       filter3=get_filter1(gap=3)
       filter4=get_filter1(gap=4)
       filter=np.concatenate((filter1,filter2,filter3,filter4),axis=3)

    elif out_channels==36:
       filter1=get_filter2(gap=4)
       filter2=get_filter2(gap=5)
       filter3=get_filter2(gap=6)
       filter=np.concatenate((filter1,filter2,filter3),axis=3)

    else: 
       filter=new_weights_non_trainable(h=filter_height,w=filter_width,num_input=in_channels,num_output=out_channels)


    anchor_weights = tf.Variable(filter.astype(np.float32),trainable=False)
    
    # Difference Maps
    difference_maps = tf.nn.conv2d(input=input,
                                   filter=anchor_weights,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
    
    # Non-linear unit is ReLU, as in article
    
    # Bit Maps
    bit_maps = tf.nn.relu(difference_maps)
    
    # Set of learnable linear weights is a convolution with 1x1 kernels,
    # without bias and without non-linear unit

    shape = [1, 1, out_channels, 1]
    
    weights = new_weights(shape)
    
    
    # Feature Maps
    feature_maps = tf.nn.conv2d(input=bit_maps, 
                                filter=weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

    # Use pooling to down-sample the image resolution
    if use_pooling:
        feature_maps = tf.nn.max_pool(value=feature_maps,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    return feature_maps, anchor_weights, weights, difference_maps, bit_maps

'''
# Convolutional layer 0 
feature_maps_conv_0, weights_conv_0 = new_SB_layer(input=x_image,
                                                                           filter_height=filter_height_0,
                                                                           filter_width=filter_width_0,
                                                                           in_channels=num_channels,
                                                                           out_channels=num_filters_0,
                                                                           use_pooling=False)

'''

# Convolutional layer 1 
feature_maps_conv_1, anchor_weights_conv_1, weights_conv_1, difference_maps1, bit_maps1 = new_SB_layer(input=x_image,
                                                                           filter_height=filter_height_1,
                                                                           filter_width=filter_width_1,
                                                                           in_channels=num_channels,
                                                                           out_channels=num_filters_1,
                                                                           use_pooling=False)


# Convolutional layer 2
feature_maps_conv_2, anchor_weights_conv_2, weights_conv_2, difference_maps2, bit_maps2 = new_SB_layer(input=feature_maps_conv_1,
                                                                           filter_height=filter_height_2,
                                                                           filter_width=filter_width_2,
                                                                           in_channels=1,
                                                                           out_channels=num_filters_2,
                                                                           use_pooling=False)

# Convolutional layer 3
feature_maps_conv_3, anchor_weights_conv_3, weights_conv_3, difference_maps3, bit_maps3 = new_SB_layer(input=feature_maps_conv_2,
                                                                           filter_height=filter_height_3,
                                                                           filter_width=filter_width_3,
                                                                           in_channels=1,
                                                                           out_channels=num_filters_3,
                                                                           use_pooling=False)

# Convolutional layer 4
feature_maps_conv_4, anchor_weights_conv_4, weights_conv_4, difference_maps4, bit_maps4 = new_SB_layer(input=feature_maps_conv_3,
                                                                           filter_height=filter_height_4,
                                                                           filter_width=filter_width_4,
                                                                           in_channels=1,
                                                                           out_channels=64,
                                                                           use_pooling=True)

# Convolutional layer 5
feature_maps_conv_5, anchor_weights_conv_5, weights_conv_5, difference_maps5, bit_maps5 = new_SB_layer(input=feature_maps_conv_4,
                                                                           filter_height=filter_height_5,
                                                                           filter_width=filter_width_5,
                                                                           in_channels=1,
                                                                           out_channels=128,
                                                                           use_pooling=True)


def flatten_layer(layer):
    # Get the shape of the input layer
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

layer_flat, num_features = flatten_layer(feature_maps_conv_5)

#layer_flat
#num_features

def new_fc_layer(input,          # The previous layer
                 num_inputs,     # Num. inputs from prev. layer
                 num_outputs,    # Num. outputs
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


dropout=0.5
# Fully-Connected Layer 1
layer_fc_1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=1000,
                         use_relu=True)

fc1 = tf.layers.dropout(layer_fc_1, rate=dropout, training=True)

# Fully-Connected Layer 2
layer_fc_2 = new_fc_layer(input=fc1,
                         num_inputs=1000,
                         num_outputs=700,
                         use_relu=True)

fc2 = tf.layers.dropout(layer_fc_2, rate=dropout, training=True)


# Fully-Connected Layer 3 (Classes)
layer_fc_3 = new_fc_layer(input=fc2,
                         num_inputs=700,
                         num_outputs=num_classes,
                         use_relu=False)


# Normalization of class-number output
y_pred = tf.nn.softmax(layer_fc_3)

# The class-number is the index of the largest element
y_pred_cls = tf.argmax(y_pred, dimension=1)


# Cross-entropy 
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc_3,
                                                        labels=y_true)

# Regularisation
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 0.01 


# Average of the cross-entropy
loss=tf.Variable(0.0)
loss = tf.reduce_mean(cross_entropy) +reg_constant*sum(reg_losses)

# Optimization method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# Measures of performance
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy=tf.Variable(0.0)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



saver=tf.train.Saver() 

# TensorFlow session
#session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
session=tf.Session()

# Directory used for the checkpoints6
save_dir = 'Checkpoints/major_sbl5x3/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
# Base-filename for the checkpoints
save_path = os.path.join(save_dir, 'ROI')

# First try to restore the latest checkpoint. This may fail and raise an exception e.g. if such a checkpoint does not exist, or if you have changed the TensorFlow graph


print("Trying to restore last checkpoint...\n")

# Find the latest checkpoint - if any
last_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

# Try and load the data in the checkpoint

saver.restore(session, save_dir+'model_knuckle.ckpt')

print("\nRestored checkpoint from" + last_checkpoint_path)

test_batch_size=200



def test_feature():
    
    num_test = len(data_test_labels)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    feature_extracted = np.zeros(shape=(num_test,num_classes), dtype=np.float)
    i = 0
    while i < num_test:
        
        j = min(i + test_batch_size, num_test)

        images = data_test_images[i:j, :]
        #images = np.array(data_test_images[i:j])
        labels = data_test_labels[i:j, :]

        # Create a feed-dict with these images and labels
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        feature_extracted[i:j,:] = session.run(layer_fc_3, feed_dict=feed_dict)

        ans=session.run(layer_fc_3,feed_dict=feed_dict)
        print(ans.shape)
            
        # Set the start-index for the next batch to the
        # end-index of the current batch
        i = j
    
    cls_true = np.argmax(data_test_labels,axis=1)
 
    return feature_extracted,cls_true


test_feature,test_labels=test_feature()
print(test_feature.shape)

np.savetxt('major_test_sbl5x3.txt',test_feature)
np.savetxt('major_test_labels.txt',test_labels)

# Close session

session.close()


