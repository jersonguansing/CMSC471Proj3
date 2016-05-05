## Jerson Guansing
## Project 3
## CMSC 471
import os
import sys
from skimage import io
from skimage.transform import resize
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

imgType = ["Smile", "Hat", "Hash", "Heart", "Dollar"]

## open the image file and convert it to a matrix
def img_to_matrix(filename):
	imageFile = io.imread(filename)
	imageFile = resize(imageFile, (100, 100))
	img = []
	for row in imageFile:
		img += map(list, row)
	img = np.array(img)
	return img

## flatten the img array from an rgb pixel(3 values) to just bw (1 value)
def flatten_image(img):
	imgData = []
	# 100 x 100 = 10,000 pixels
	for row in img:
		current = 0
		## get the average of each pixel (r,g,b)
		for col in row:
			current = current + col
		current = current / (len(row))
		imgData.append(current)
	## shrink the image by getting the vertical average
	#imgData = shrink_image(imgData)
	## exponential
	#imgData = [i ** 7 for i in imgData]
	return imgData

def shrink_image(img):
	imgData = []
	# 100 x 100 = 10,000 pixels. Shrink it to a 100 length array
	rowCount = 0
	for row in img:
		current = 0
		if rowCount < 100:
			imgData.append(row)
		else:
			imgData[rowCount % 100] = imgData[rowCount % 100] + row
		rowCount += 1
	## get the average value of combining all the rows into 1
	imgData = [round(i / 100.0, 2) for i in imgData]
	return imgData
	
def get_train_set():
	img_dir = "./Images/Training/"
	train_sets = [img_dir+ f for f in os.listdir(img_dir)]
	data = []
	for train_set in train_sets:
		subfolder = str(train_set) + "/"
		images = [subfolder+ f for f in os.listdir(subfolder)]
		current_set = []
		for image in images:
			img = img_to_matrix(image)
			img = flatten_image(img)
			current_set.append(img)
		data.append(current_set)
	## plot the data -- UNCOMMENT the line below to produce a graph
	#plotData(data)
	return data

## get a graph representation of the images
def plotData(data):
	plots = []
	nextOne = 0
	colors = ["red", "yellow", "green", "blue", "black"]
	for sets in data:
		nextOne += 1
		if nextOne > (len(colors) - 1):
			nextOne = 0
		for record in sets:
			X = np.arange(0, len(record), 1)
			plots.append(plt.scatter(X, record, color=colors[nextOne] ))
	plt.legend((plots[0*len(data[0])],plots[1*len(data[1])],plots[2*len(data[2])],plots[3*len(data[3])],plots[4*len(data[4])]),("Smiley", "Hat", "Pound", "Heart", "Dollar"))
	plt.show()

## test set will only include images not used in the training set
def test_set(clf):
	img_dir = "./Images/Testing/"
	train_sets = [img_dir+ f for f in os.listdir(img_dir)]
	sets, correct, incorrect = 0, 0, 0
	for train_set in train_sets:
		subfolder = str(train_set) + "/"
		images = [subfolder+ f for f in os.listdir(subfolder)]
		for image in images:
			img = img_to_matrix(image)
			img = flatten_image(img)
			prediction = clf.predict([img])
			outResult = ""
			if sets == prediction[0]:
				correct += 1
				outResult = ""
			else:
				incorrect += 1
				outResult = "X"
			print(image + "	" + str(sets) +  "	" + str(prediction) + "	" + imgType[prediction[0]] + "	" + outResult)
		sets += 1
	print("Correct: " + str(correct))
	print("Incorrect: " + str(incorrect))
	print("Accuracy: " + str(round((correct * 100) / float(correct + incorrect), 2) ) + "%")

def main(argv):
	# get the training set
	classifiers = ["rbf", "linear", "poly", "kneighbor"]
	clf = [ svm.SVC(kernel="rbf"), svm.SVC(kernel="linear"), svm.SVC(kernel="poly", degree=3), KNeighborsClassifier() ]
	# K neighbors is the default because it is giving the highest accuracy
	whichClf = 3
	if len(argv) >= 3:
		whichClf = [i for i in range(0, len(classifiers)) if classifiers[i] == argv[2]]
		if len(whichClf) > 0:
			whichClf = whichClf[0]
		else:
			whichClf = 3
	trainFile = classifiers[whichClf] + ".pkl"
	train1 = clf[whichClf]
	if os.path.isfile(trainFile):
		train1 = joblib.load(trainFile) 
	else:
		# get the images as matrices
		data = get_train_set()
		X = []
		y = []
		for i in range (0, 5):
			# data is a 3d array so need to convert to a 2d array
			X += data[i]
			# set the label number for each sample
			y += [i for j in range (0, len(data[i])) ]
		X = np.array(X)
		train1.fit(X, y)
		joblib.dump(train1, trainFile)
	## test the train set with the images in the testing set -- UNCOMMENT to run with the test set
	#test_set(train1)
	
	## get the image to test
	if len(argv) < 2:
		print("The program takes a single argument, the filepath to a jpg.")
	else:
		if not os.path.isfile(argv[1]):
			print("Could not find the file " + argv[1])
		else:
			## convert the image to an array
			testImage = img_to_matrix(argv[1])
			testImage = flatten_image(testImage)
			# compare the image file with the training set
			prediction = train1.predict([testImage])
			print(imgType[prediction[0]])

# the main driver of the program
main(sys.argv)
