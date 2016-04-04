import os
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

###############################################################################
# Setting up

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

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

# Load Images
def loadImages(filepath, imageNames):
	print 'Loading image datasets...'
	image_set = []
	label_set = []
	
	for imgFile in imageNames:
		imgFile = str(imgFile).replace('-', '.')
		for file in os.listdir(filepath):
			if imgFile in file:
				image = color.rgb2gray(mpimg.imread(filepath + file, True))
				image_set.append(image.flatten())
				label_set.append(getID(file[3:5]))
	return image_set, label_set

# Plot the Image
def plotImage(image, label):
	image = plt.imshow(image, cmap='Greys_r')
	plt.title(label)
	plt.show()

# Configure Filepaths
image_path = '/storage/home/yjo5006/wang/yjo5006/Datasets/jaffe/'
label_path = 'semantic_ratings.txt'

# Load Data
labels = loadLabels(label_path)
X, Y = loadImages(image_path, labels['filename'])
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

# plotImage(X[0], 'Test Image')
# print X.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

###############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.1
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(X_train, Y_train)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train, Y_train)

###############################################################################
# Evaluation

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))

print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))

###############################################################################
# Plotting

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((256, 256)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
