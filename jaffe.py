'''
An example based on the MNIST architecture using the JAFFE Dataset
'''
import sys
import os
import time
import fnmatch

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color
import numpy as np

import theano
import theano.tensor as T
import lasagne

# Split List Helper Function
def splitDataset(l):
	return l[1:len(l)/3], l[len(l)/3:(len(l)/3)*2], l[(len(l)/3)*2:len(l)]

# Load Images
def loadImages(filepath, imageNames):
	print 'Loading image datasets...'
	image_set = []
	label_set = []
	
	for imgFile in imageNames:
		imgFile = str(imgFile).replace('-', '.')
		for file in os.listdir(filepath):
			if imgFile in file:
#				image = color.rgb2gray(mpimg.imread(filepath + file, True))
				image = mpimg.imread(filepath + file, True)
				image = image.reshape(-1, 4, 256, 256)
				image_set.append(image)
				label_set.append(getID(file[3:5]))
	return image_set, label_set

# Load Semantic Labels
def loadLabels(filepath):
	print 'Loading dataset labels...'
	data = np.genfromtxt(label_path, dtype=[('index','i8'),('happy','f8'),
						('sad','f8'),('surprised','f8'),
						('angry','f8'),('disgust','f8'),
						('fear','f8'),('filename','S6')])
	return data

# Returns numeric representation of label
'''
0 - NE: Neutral		4 - AN: Angry
1 - HA: Happy		5 - DI: Disgust
2 - SA: Sad			6 - FE: Fear
3 - SU: Surprised
'''
def getID(label):
	return ['NE', 'HA', 'SA', 'SU', 'AN', 'DI', 'FE'].index(label)

def getLabel(id):
	return ['Neutral', 'Happy', 'Sad', 'Surprised', 'Angry', 'Disgust', 'Fear'][id]

# Plot the Image
def plotImage(image, label):
	image = plt.imshow(image, cmap='Greys_r')
	plt.title(label)
	plt.show()

# Build Multilayer Perceptron
def buildMLP(input_var=None):
	# Input Layer - Unspecified Batchsize, 1 channel, 256 by 256 Image
	l_in = lasagne.layers.InputLayer(shape=(None, 4, 256, 256), input_var=input_var)

	# Dropout Layer - Dropout weights by 20%
	l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

	# Fully Connected Layer - 800 Units Using Linear Rectifier Initialized with Glorot Scheme
	l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units=800, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

	# Dropout Layer - Dropout weights by 50%
	l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

	# 800-Unit Rectified Linear Layer
	l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units=800, nonlinearity=lasagne.nonlinearities.rectify)

	# Dropout Layer - Dropout weights by 50%
	l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

	# Softmax Unit 10 Layers
	l_out = lasagne.layers.DenseLayer(l_hid2_drop, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
	
	return l_out

# Helper funciton for iterating over dataset batches
'''
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
'''

# Configure Filepaths
image_path = '/storage/home/yjo5006/wang/yjo5006/Datasets/jaffe/'
label_path = 'semantic_ratings.txt'
num_epochs=500

# Load Raw Dataset and Split Into Train and Test
labels = loadLabels(label_path)
X, y = loadImages(image_path, labels['filename'])
X_train, X_val, X_test = splitDataset(X)
y_train, y_val, y_test = splitDataset(y)

#index = 21
#plotImage(X[index], getLabel(y[index]))
#plotImage(X[15], getLabel(y[15]))

# Initialize Theano Parameters
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

# Initialize Multilayer Perceptron Model
print 'Building model and compiling functions...'
network = buildMLP(input_var)

# Define loss expression function for minization
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

# Initialize Update Expression for Training
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

# Initialize loss expresion for validation/testing process
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()

# Initialize expression for classification accuracy
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

# Compile training function based on minibatch procedure
train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

# Compile validation loss and accuracy procedures
val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

# Initialize Training Loop
print 'Starting training...'
for epoch in range(num_epochs):
	# Complete full pass over training data
	train_err = 0
	train_batches = 0
	start_time = time.time()
	for i in range(0, len(X_train)):
		inpt = X_train[i]
		out = np.array([y_train[i]])
		train_err += train_fn(inpt, out)
		train_batches += 1

	# And a full pass over the validation data:
	val_err = 0
	val_acc = 0
	val_batches = 0
	for i in range(0, len(X_val)):
		inpt = X_val[i]
		out = np.array([y_val[i]])
		err, acc = val_fn(inpt, out)
		val_err += err
		val_acc += acc
		val_batches += 1

	# Then we print the results for this epoch:
	print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
	print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
	print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
	print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

