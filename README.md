# Kaggle's Galaxy Zoo - The Galaxy Challenge

This is the Python & Theano code I used to make my submission to Kaggle's
Galaxy Zoo - The Galaxy Challenge

For this competition, I use a (deep) neural network architecture with a
convolutional neural network (CNN) followed by a multilayer perceptron (MLP).
In specific, I use CNN as first layers to process the galaxy images, and to
extract useful and learnable features, which are then fed to MLP. The whole
neural network are trained with back-propagation with MSE as cost function.

My score on the private leaderboard is RMSE = 0.10494, ranking 60th out of 331. [With an average of few different NN's, I obtain 0.10379, which is my best private score]. With a few epoch [e.g., 5~10] through the whole training set, one can easily obtain RMSE = 0.11xxx. With a few more, you can get to 0.10xxx, which seems
the limit of my approach and hard to break! I would appreciate if anyone tell me how to break it, anything regarding to the network architecture or parameter configurations etc.

Update: The winning solution for this competition is here
http://benanne.github.io/2014/04/05/galaxy-zoo.html
Convnet is again the winner for such task!!

The code itself contains lots of comments. So, you'd better see there for details.

## Requirements

You should have theano/numpy/pandas/cv2 installed in python.

## Instructions

* download data from competition website: http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/
* put these code and the following folders/files in the same dir
 - images_traing_rev1
 - images_test_rev1
 - training_solutions_rev1.csv
 - central_pixel_benchmark.csv
* run train_CNN.py