# CMSC471Proj3
# Machine Learning - Image Classification

## Required Packages (pip install)
#### 1. skimage
#### 2. sklearn
#### 3. numpy
#### 4. matplotlib

## How to run
##### Need to have Python 3+ (used 3.5).
##### The program expects one (1) command line argument for the target image file.
##### Use this command to run the program: Proj3.py &lt; filename &gt;
##### By default, it will use the K-Neighbors since it has the highest accuracy of the classifiers tested.
##### Optionally, you can specify the classifier you want to use: Proj3.py &lt; filename &gt;  &lt; linear | rbf | poly | kneighbor &gt;
##### You do not have to specify a classifier if you just want to test with K-Neighbors, and just use this command: Proj3.py &lt; filename &gt;
##### The program will then check if the training set file for the corresponding classifier exists.
##### If it does not, it will generate a pickled training set.
##### Once generated, the target image file will be tested.
##### This is useful for checking subsequent image files.
##### It is safe to delete all *.pkl* files, but deleting means they need to be created again when running a test for the particular classifier.
 
