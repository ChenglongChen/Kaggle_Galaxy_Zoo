"""
Chenglong Chen
Mar 18 - Apr 4, 2014

c.chenglong@gmail.com
http://weibo.com/denny2cd
https://github.com/ChenglongChen
http://www.kaggle.com/users/102203/yr
-------------------------------------------------------------------------------
Description

This is the Python & Theano code I used to make my submission to Kaggle's
Galaxy Zoo - The Galaxy Challenge

My score on the private leaderboard is RMSE = 0.10494, ranking 60th out of 331.

With a few epoch [e.g., 5~10] through the whole training set, one can easily
obtain RMSE = 0.11xxx. With a few more, you can get to 0.10xxx, which seems
the limit of my approach and hard to break!
I would appreciate if anyone tell me how to break it, anything regarding to the
network architecture or parameter configurations etc.
-------------------------------------------------------------------------------
Get started
1) ensure you have theano/numpy/pandas/cv2 installed
2) put these code and the following folders/files in the same dir
 - images_traing_rev1
 - images_test_rev1
 - training_solutions_rev1.csv
 - central_pixel_benchmark.csv
3) run train_CNN.py
-------------------------------------------------------------------------------
Methodology

For this competition, I use a (deep) neural network architecture with a
convolutional neural network (CNN) followed by a multilayer perceptron (MLP).
In specific, I use CNN as first layers to process the galaxy images, and to
extract useful and learnable features, which are then fed to MLP. The whole
neural network are trained with back-propagation.

To address the goal of this competition and especially the evaluation metric
being RMSE, I modified the cost function of MLP to be MSE instead of negative
log-likelihood (NLL).

Due to the change of the cost function to MSE, one have to compute the gradient
with respect to that and subsititue the new gradient to back-propagation (BP).
However, thanks to the automatic differentiation feature of Theano, you don't
have to worry about the pain of hand claculating the gradient yourself. All you
have to do with Theano is: 
1) define the cost function you want to optimize
2) use theano.tensor.grad(cost, params) to compute the gradient of cost with
   respect to (wrt) params
3) done

Regarding the probability constraints illustrated in the Galaxy Zoo decision
tree:

http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/details/the-galaxy-zoo-decision-tree

I also try to encode this info into cost function. [See the LogisticRegression 
layer in layer_factory module for detail]
And thanks again to theano's automatic differentiation, I don't have to bother
the derivation of the gradients.
-------------------------------------------------------------------------------
References

1) The code I wrote is heavily based on those from the tutorial of using Theano
   for Deep Learning. You may want to have a look at them, which are at:

     http://www.deeplearning.net/tutorial/

   Of those, the most relevant part are the sections convering:
     - Logistic Regression
     - Multilayer perceptron
     - Deep Convolutional Network

2) Code for dropout is based on Misha Denil's implementation at:

     https://github.com/mdenil/dropout/blob/master

   Credit to Misha Denil for that.
-------------------------------------------------------------------------------
Required packages

 - theano
 - numpy
 - pandas
 - cv2
-------------------------------------------------------------------------------
Warning

Since the whole image set is too large to fit in my laptop memory, I thus use
a naive strategry: read in a mini-batch images and train on them, then read in
and train on the next mini-batch, and so on. [You can hack it to support "a few"
mini-batch instead of just one]

I currently can't afford a GPU card. The code provided here has only been 
tested through CPU.

I have tried my best to optimize the code, but it might still be slow and can
be improved. I would appreciate if someone can tell me how.

"""


#### Put all the required packages here
import cPickle
import os
import sys
import time
import copy

import numpy as np
import pandas as pd

from collections import OrderedDict
import theano
import theano.tensor as T
from theano.ifelse import ifelse

from layer_factory import *
from helper_function import *
        
       
        
#####################
## Make submission ##
#####################
# You have to make a few new dirs:
# ./submission/with probability constraint/csv
# ./submission/with probability constraint/pkl
# ./submission/without probability constraint/csv
# ./submission/without probability constraint/pkl
def makeSubmission(test_model, test_image_files, length_test, batch_size,
                   rotationAngles, cropImageSizes, finalImageSize, channel,
                   mean_image, substract_each_image_mean,
                   epoch, learning_rate, best_validation_RMSE,
                   csv_flag=False, pkl_flag=True, prob_constraint_on=True,
                   augmentation_method="on_the_fly"):

    
    print("... make submission for epoch {}".format(epoch))

    n_test_batches = length_test / batch_size
    test_index = np.arange(length_test)
    test_target_batch = np.zeros((batch_size, 37), dtype=theano.config.floatX)
    test_pred = []
    print "... {} batches in total".format(n_test_batches)
    for test_batch_index in xrange(n_test_batches):
        print "       batch: {}".format(test_batch_index+1)
        # make prediction for each combination of rotation angle and crop size
        this_batch_pred = np.zeros((batch_size, 37), dtype=theano.config.floatX)
        for rotationAngle in rotationAngles:
            for cropImageSize in cropImageSizes:
                test_images_batch = load_train_valid_images(
                                     test_image_files, test_index,
                                     test_batch_index, batch_size,
                                     rotationAngle, cropImageSize, finalImageSize,channel,
                                     substract_each_image_mean, augmentation_method) 
                # substract mean
                test_images_batch -= mean_image
                this_batch_pred += test_model(test_images_batch, test_target_batch)
        # then average them to get the final prediction
        this_batch_pred /= len(cropImageSizes)*len(rotationAngles)
        test_pred.append(this_batch_pred)
        
    test_pred = np.vstack(test_pred)

    # write submission as csv file
    if csv_flag == True:
        sub = pd.read_csv('central_pixel_benchmark.csv')
        sub.loc[:length_test-1,1:] = test_pred
        if prob_constraint_on != None:
            csvFileName = './submission/with probability constraint/csv/submission_CNN' + \
                          '_Epoch{}_lr{}_RMSE{}.csv'.format(epoch, learning_rate, np.round(best_validation_RMSE, 5))
        else:
            csvFileName = './submission/without probability constraint/csv/submission_CNN' + \
                          '_Epoch{}_lr{}_RMSE{}.csv'.format(epoch, learning_rate, np.round(best_validation_RMSE, 5))
                        
        sub.to_csv(csvFileName, index=False)
        print("... Done for csv format")

    # save in pkl format
    if pkl_flag == True:
        if prob_constraint_on != None:
            pklFileName = './submission/with probability constraint/pkl/submission_CNN' + \
                          '_Epoch{}_lr{}_RMSE{}.pkl'.format(epoch, learning_rate, np.round(best_validation_RMSE, 5))
        else:
            pklFileName = './submission/without probability constraint/pkl/submission_CNN' + \
                          '_Epoch{}_lr{}_RMSE{}.pkl'.format(epoch, learning_rate, np.round(best_validation_RMSE, 5))

        with open(pklFileName, 'w') as f:
            cPickle.dump((length_test, test_pred), f)
        print("... Done for pkl format")

    #return(test_pred)


###############
## Train CNN ##
###############
def trainChunkCNN(length_train, train_ratio, length_test, 
                  rotationAngles=[0.0], cropImageSizes=[150], finalImageSize=60, channel=3,
                  batch_size=175, n_epochs=200, squared_filter_length_limit=15.0,
                  initial_learning_rate=1.0, learning_rate_decay=0.998, mom_params=None,
                  weight_decay=0.0005, CNN_params=None, MLP_params=None,
                  validation_frequency=10, make_submission_frequency=10,
                  log_file_name='validation.txt', csv_flag=False, pkl_flag=True,
                  random_seed=2014, augmentation_method="on_the_fly"):
                      
    assert type(rotationAngles) == list
    assert type(cropImageSizes) == list

    # extract the params for CNN
    CNN_kernels = CNN_params["kernel"]
    CNN_poolings = CNN_params["pooling"]
    CNN_activations = CNN_params["activation"]
    
    assert len(CNN_kernels) == len(CNN_poolings)
    assert len(CNN_kernels) == len(CNN_activations)
    
    # extract the params for MLP
    MLP_layer_sizes = MLP_params["layer_size"]
    MLP_activations = MLP_params["activation"]
    MLP_use_bias = MLP_params["use_bias"]
    MLP_dropout_rates = MLP_params["dropout_rate"]
    MLP_prob_constraint_on = MLP_params["prob_constraint_on"]
    
    assert (len(MLP_layer_sizes) -1) == len(MLP_activations)
    assert (len(MLP_layer_sizes) -1) == len(MLP_dropout_rates)
    
    # extract the params for momentum
    mom_start = mom_params["start"]
    mom_end = mom_params["end"]
    mom_epoch_interval = mom_params["interval"]
    
    # number of training images
    trainNum = np.floor(length_train*train_ratio)
    n_train_batches = np.int32(np.floor(trainNum/batch_size))
    n_valid_batches = np.int32(np.floor((length_train-trainNum)/batch_size))
    # NOTE: due to n_train_batches = np.int32(np.floor(trainNum/batch_size))
    # the actual size of training images is not trainNum, but rather
    # n_train_batches*batch_size. 
    # The rest images after n_train_batches*batch_size (if there is any) will
    # not be used for training since they are of size samller than batch_size
    # and can not make up a whole mini-batch. Similar idea applies to the
    # validation set. However, there is no waste of testing images, since we
    # are asked to predict for each of them!! So, the batch_size is so chosen
    # to be a factor of the size of the whole testing set, i.e., 79975. Since
    # 79975 = 5 * 5 * 7 * 457, so we can choose 25/35/175 as batch size.
    # Of course, this is a waste of sources for training and validation, but
    # simplify the use of function train_model, validate_model, and test_model.
    # 
    # I would appreciate anyone tell me how to deal with varaiable batch-size.


    ###########################
    # Read training solutions #
    ###########################
    solution = pd.read_csv('training_solutions_rev1.csv')
    solution = np.asarray(solution.values, dtype=theano.config.floatX)
    # access the Galaxy ID for training set. Note that we have to cast it to int32
    train_Galaxy_ID = np.int32(solution[:solution.shape[0], 0])
    # ignore the first column which corresponds to the Galaxy ID 
    solution = solution[:solution.shape[0], 1:]    
    if augmentation_method == "pre_computed":
        # folder that contain the augmentated training data that are pre-computed
        train_folder = "images_training_rev1_augmented"
    elif augmentation_method == "on_the_fly":
        # folder that contains the original training data, the augmentated data
        # will be computed on the fly. This can be slow
        train_folder = "images_training_rev1"
    else:
        raise ValueError("augmentation_method must be either \"pre_computed\" or \"on_the_fly\"!")
        
    train_image_files = ["{}/{}.jpg".format(train_folder, ID) for ID in train_Galaxy_ID]

    # access the Galaxy ID for testing set
    benchmark = pd.read_csv('central_pixel_benchmark.csv')
    benchmark = np.asarray(benchmark.values, dtype=theano.config.floatX)
    # access the Galaxy ID for training set. Note that we have to cast it to int32
    testing_Galaxy_ID = np.int32(benchmark[:benchmark.shape[0], 0])
    if augmentation_method == "pre_computed":
        # folder that contain the augmentated testing data that are pre-computed
        test_folder = "images_test_rev1_augmented"
    elif augmentation_method == "on_the_fly":
        # folder that contains the original testing data, the augmentated data
        # will be computed on the fly. This can be slow
        test_folder = "images_test_rev1"
    else:
        raise ValueError("augmentation_method must be either \"pre_computed\" or \"on_the_fly\"!")
        
    test_image_files = ["{}/{}.jpg".format(test_folder, ID) for ID in testing_Galaxy_ID]
    
    ###################################
    # Creat training & validation set #
    ###################################
    # split the images into training and validation set
    # to ensure reproduciable results
    shuffle_rng = np.random.RandomState(random_seed)
    random_index = np.arange(length_train)    
    shuffle_rng.shuffle(random_index)
    train_index = random_index[:trainNum]
    valid_index = random_index[trainNum:]

    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    train_set_x = T.matrix('train_set_x')
    valid_set_x = T.matrix('valid_set_x')
    test_set_x = T.matrix('test_set_x')  

    y = T.matrix('y')  # the labels are presented as 1D vector of
    train_set_y = T.matrix('train_set_y')
    valid_set_y = T.matrix('valid_set_y')
    test_set_y = T.matrix('test_set_y')
    
    epoch = T.scalar()
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
                                             dtype=theano.config.floatX))
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    rng = np.random.RandomState(random_seed)
    
    #### Construct CNN layer
    # Reshape matrix of rasterized images of shape (batch_size,channel*finalImageSize*finalImageSize)
    # to a 4D tensor, compatible with our ConvPoolLayer
    CNN_input = x.reshape((batch_size, channel, finalImageSize, finalImageSize))
    CNN_layer = CNN(rng, input=CNN_input, batch_size=batch_size,
                    channel=channel, image_size=finalImageSize,
                    kernels=CNN_kernels, poolings=CNN_poolings,
                    activations=CNN_activations)
                    
    #### construct MLP layer
    MLP_input = CNN_layer.output.flatten(2)
    MLP_layer = MLP(rng, input=MLP_input, layer_sizes=MLP_layer_sizes,
                    dropout_rates=MLP_dropout_rates, activations=MLP_activations,
                    use_bias=MLP_use_bias, prob_constraint_on=MLP_prob_constraint_on)

    # the cost we minimize during training is the MSE of the model
    cost = MLP_layer.training_MSE(y)
    # grab all the params
    params = CNN_layer.params + MLP_layer.params
    
    # create a list of gradients for all model parameters
    gparams = T.grad(cost, params)
    
    # ... and allocate mmeory for momentum'd versions of the gradient
    gparams_mom = []
    for param in params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)
        
    # Compute momentum for the current epoch
    mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
            mom_end)

    # Update the step direction using momentum
    updates = OrderedDict()
    for gparam_mom, gparam, param in zip(gparams_mom, gparams, params):
        # Misha Denil's update for momentum version of gradient
        #updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam
        
        # My hack according to Appendix A.1 of Hinton's dropout paper
        # since (1. - mom) is small, we need large learning_rate to compensate
        #updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam
        
        # NOTE: In the ImageNet paper, they use a slight different update rule
        # including weight decay (detailed in Section 5).
        # We here use this method
        updates[gparam_mom] = mom * gparam_mom - learning_rate * (weight_decay*param + gparam)
        
    # ... and take a step along that direction
    for param, gparam_mom in zip(params, gparams_mom):
        # Misha Denil's update for the params
        #stepped_param = param - learning_rate * updates[gparam_mom]
    
        # My hack according to Appendix A.1 of Hinton's dropout paper
        stepped_param = param + updates[gparam_mom]

        # This is a silly hack to constrain the norms of the cols of the weight
        # matrices.  This just checks if there are two dimensions to the
        # parameter and constrains it if so... maybe this is a bit silly but it
        # should work for now.
        if param.get_value(borrow=True).ndim == 2:
            # This is the original code that constrains the weight matrix's rows
            # however, as reported in https://github.com/mdenil/dropout/issues/5
            # and in the original paper of Hinton's, it is the COLUMNs that should
            # be constrained
            #squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
            #scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
            
            # we hack it a little bit according to this
            # https://github.com/BVLC/caffe/issues/109
            #col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            #desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
            #scale = desired_norms / (1e-7 + col_norms)
            #updates[param] = stepped_param * scale
            
            # we use weight decay
            updates[param] = stepped_param
        else:
            updates[param] = stepped_param
            
    # create a function to train the model
    train_model = theano.function(inputs=[epoch, train_set_x, train_set_y],
                                  outputs=cost,
                                  updates=updates,
                                  givens={x: train_set_x, y: train_set_y},
                                  on_unused_input='ignore')

    # create a function to compute the MSE of the model
    validate_model = theano.function(inputs=[valid_set_x, valid_set_y],
                                     outputs=MLP_layer.y_pred,
                                     givens={x: valid_set_x, y: valid_set_y},
                                     on_unused_input='ignore')
    
    # create a function to grab the prediction made by the model
    test_model = theano.function(inputs=[test_set_x, test_set_y],
                                 outputs=MLP_layer.y_pred,
                                 givens={x: test_set_x, y: test_set_y},
                                 on_unused_input='ignore')
              
    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    update_learning_rate = {learning_rate: learning_rate * learning_rate_decay}
    decay_learning_rate = theano.function(inputs=[],
                                          outputs=learning_rate,
                                          updates=update_learning_rate)

    ###############
    # TRAIN MODEL #
    ###############
    #
    # early-stopping parameters
    # Actually, we don't need this, since we will make a submission every
    # make_submission_frequency epoch. You can manually choose the one with the 
    # best/smallest RMSE on the validation set. 
    patience = 50000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    
    best_params = None
    best_validation_RMSE = np.inf
    all_validation_RMSE = []
    best_iter = 0
    start_time = time.clock()

    epoch_counter = 0
    done_looping = False
    log_file = open(log_file_name, 'wb')
    
    # compute mean image
    print '... computing mean image'
    substract_each_image_mean = False
    mean_image = compute_mean_image(train_image_files, train_index, batch_size,
                                    rotationAngles, cropImageSizes, finalImageSize, channel,
                                    substract_each_image_mean, augmentation_method)

    print '... start training'
    while (epoch_counter < n_epochs) and (not done_looping):
        print "... training for epoch {}".format(epoch_counter+1)
        epoch_counter = epoch_counter + 1
        for train_batch_index in xrange(n_train_batches):
            
            iter = (epoch_counter - 1) * n_train_batches + train_batch_index
            
            #######################################################
            # load training images and targets for this batch_index
            #######################################################
            train_target_batch = load_train_valid_target(solution, train_index,
                                                         train_batch_index, batch_size)
            # for this mini-batch index, we train with the distorted image
            # with different combination of rotation angle and crop size
            for rotationAngle in rotationAngles:
                for cropImageSize in cropImageSizes:
                    print "       epoch: {0: >4} | batch: {1: >3} | rotationAngle: {2: >3} | cropImageSize: {3: >3}".format(
                          epoch_counter, train_batch_index+1, rotationAngle, cropImageSize)
                    train_images_batch = load_train_valid_images(
                                          train_image_files, train_index,
                                          train_batch_index, batch_size,
                                          rotationAngle, cropImageSize, finalImageSize, channel,
                                          substract_each_image_mean, augmentation_method)
                    # substract mean
                    train_images_batch -= mean_image
                    
                    this_train_MSE = train_model(epoch_counter, train_images_batch, train_target_batch)
            
            if (iter + 1) % validation_frequency == 0:
                print "    computing RMSE on validation set with {} batches in total".format(n_valid_batches)
                this_validation_MSE = 0
                for valid_batch_index in xrange(n_valid_batches):
                    print "       batch: {}".format(valid_batch_index+1)
                    valid_target_batch = load_train_valid_target(solution, valid_index,
                                                                 valid_batch_index , batch_size)
                    # for this mini-batch index, we make prediction with the distorted image
                    # with different combination of rotation angle and crop size
                    # make prediction for each combination of crop size and rotation angle
                    this_batch_pred = np.zeros((batch_size, 37), dtype=theano.config.floatX)
                    for rotationAngle in rotationAngles:
                        for cropImageSize in cropImageSizes:
                            valid_images_batch = load_train_valid_images(
                                                  train_image_files, valid_index,
                                                  valid_batch_index , batch_size,
                                                  rotationAngle, cropImageSize, finalImageSize, channel,
                                                  substract_each_image_mean, augmentation_method)
                            # substract mean
                            valid_images_batch -= mean_image
                            
                            this_batch_pred += validate_model(valid_images_batch, valid_target_batch)
                    # then average them to get the final prediction
                    this_batch_pred /= len(rotationAngles)*len(cropImageSizes)
                    # we accumulate the squared errors for all the n_valid_batches                                              
                    this_validation_MSE += np.sum((valid_target_batch - this_batch_pred)**2)
                # compute MSE on the whole validation set
                this_validation_MSE /= (n_valid_batches*batch_size*37)
                # now compute RMSE on the whole validation set
                this_validation_RMSE = np.sqrt(this_validation_MSE)

                print("    epoch {}, batch {}/{}, valid RMSE {}, lr={}{}".format(
                      epoch_counter, train_batch_index + 1, n_train_batches,
                      this_validation_RMSE,
                      np.round(learning_rate.get_value(borrow=True),5),
                      " **" if this_validation_RMSE < best_validation_RMSE else ""))
                # save the this validation RMSE
                all_validation_RMSE.append(this_validation_RMSE)       
                log_file.write("{}\n".format(this_validation_RMSE))
                log_file.flush()
                # plot it
                
                
                # if we got the best validation score until now
                if this_validation_RMSE < best_validation_RMSE:

                    #improve patience if loss improvement is good enough
                    if this_validation_RMSE < best_validation_RMSE * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score, best parameters, and iteration number
                    best_validation_RMSE = this_validation_RMSE
                    best_params = copy.deepcopy(params)
                    best_iter = iter
                    
            if patience <= iter:
                done_looping = True
                break
            
        print "Done for epoch {} with best validation RMSE of {} obtained at iteration {}".format(
              epoch_counter, best_validation_RMSE, best_iter + 1)

        
        # decay learing rate after this epoch
        new_learning_rate = decay_learning_rate()
          
        # make submission for every make_submission_frequency epoch
        if epoch_counter % make_submission_frequency == 0 and best_params != None:
            # we first save the params for this current iteration
            old_params = copy.deepcopy(params)
            # use the best_params so far for testing on testing set
            for param_i, best_param_i in zip(params, best_params):
                param_i.set_value(best_param_i.get_value(borrow=True))
                #print(param_i.get_value())  # for debugging
                
            makeSubmission(test_model, test_image_files, length_test, batch_size,
                           rotationAngles, cropImageSizes, finalImageSize, channel,
                           mean_image, substract_each_image_mean,
                           epoch_counter, learning_rate.get_value(borrow=True),
                           best_validation_RMSE,
                           csv_flag, pkl_flag, MLP_prob_constraint_on, augmentation_method)
            
            # after testing, we reset the params as the params of this current iteration
            for param_i, old_param_i in zip(params, old_params):
                param_i.set_value(old_param_i.get_value(borrow=True))
                #print(param_i.get_value())  # for debugging


    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation RMSE of %f obtained at iteration %i' %
          (best_validation_RMSE, best_iter + 1))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # make the last submission
    print "... make the last submission"
    # use the final best_params for testing on testing set
    for param_i, best_param_i in zip(params, best_params):
        param_i.set_value(best_param_i.get_value(borrow=True))
    
    makeSubmission(test_model, test_image_files, length_test, batch_size,
                   rotationAngles, cropImageSizes, finalImageSize, channel,
                   mean_image, substract_each_image_mean,
                   epoch_counter, learning_rate.get_value(borrow=True),
                   best_validation_RMSE,
                   csv_flag, pkl_flag, MLP_prob_constraint_on, augmentation_method)


##########
## Main ##
##########
if __name__ == '__main__':

    # random seed to ensure reproduciable results
    random_seed = 2014
    # flag for debugging
    debug_on = True
    
    # method to generate the augmented data and can take the following two values:
    # "on_the_fly": augmented data will be computed on the fly. This can be (very) slow...
    # "pre_computed": use pre-computed augmented data.
    augmentation_method = "on_the_fly"
    # If you want to use the second method, you might need to hack a little bit
    # in the trainChunkCNN function mostly about the variable
    # train_folder/test_floder, and the file name of precomputed image, i.e.,
    # imgf in the load_train_valid_images function which is in the
    # helper_function_rotation module. To ease your life, I'd suggest you to
    # follow the naming standard in the matlab code I provided, with which I
    # generated my augmented data. In that case, everything should work fine.
    
    
    
    ############################
    ## Set up for the dataset ##
    ############################
    # number of the whole training images (including both training and validation images)
    # NOTE: you can choose a number smaller than 61578, which is the total number of the provided training
    # images. In that case, only that many images will be used for training. This is useful for debugging 
    length_train = 61578
    length_train_debug = np.int(np.floor(1750/0.89)) # for debugging
    # of this many images are used for training CNN, the rest are used as validation set
    train_ratio = 0.9
    train_ratio_debug = 0.9

    # number of the testing images
    # NOTE: you can choose a number smaller than 79975, which is the total number of the provided testing
    # images. In that case, only the prediction of that many images will be made. This is useful for debugging
    length_test = 79975
    length_test_debug = 175 # for debugging
    

    ##########################
    ## Set up for the image ##
    ##########################
    #### We use the following strategy to generate distorted image.
    # 1) Due to the high level symmetric characteristics of the galaxy images,
    #    we rotate the image with varying rotationAngles to generate reasonable
    #    and target-preserving images.
    # 2) Forthermore, for each rotationAngles, we crop the central part of the
    #    whole image with different cropImageSizes and then resize them to the 
    #    same size of finalImageSize x finalImageSize. This is the final image
    #    that is fed to CNN
    # NOTE: Since most of the galaxy have a shape of circle or elipse, we thus
    # do NOT use flipping or flipping followed by rotation to get even more
    # distorted images. Of course, one can do that, and better results might be
    # expected. But we are running out of time here. 

    
    # a list of angles to be used to generate rotation distorted images
    angleStep = 45
    rotationAngles = np.arange(0, 180, angleStep).tolist()
    rotationAngles = [ 0 ]
    # we process the image by first cropping the central part and then resizing
    # it to the size of finalImageSize x finalImageSize before being fed to CNN
    finalImageSize = 44
    # a list of the size of the central part to be cropped from the whole image
    cropImageSizes = (np.int32(np.arange(2.0, 4.0, 0.5)*finalImageSize)).tolist()
    cropImageSizes = [ 128 ]
    # use channel = 3 for RGB color image and channel = 1 for grayscale image
    # NOTE: channel can only be {1, 3}
    channel = 1

    
    #################################################
    ## Set up for the Neural Network Architecture  ##
    #################################################

    #########
    ## CNN ##
    #########
    # kernels for CNN
    CNN_kernels = [{"num": 32, "size": 5},
                   {"num": 32, "size": 5}]
    # pooling method for CNN
    CNN_poolings = [{"type": "max", "size": 2},
                    {"type": "max", "size": 2}] # no pooling for this layer
    # activation functions for CNN
    CNN_activations = [ ReLU, ReLU ]    
    # grab all the params for CNN
    CNN_params = {"kernel": CNN_kernels,
                  "pooling": CNN_poolings,
                  "activation": CNN_activations}
    #########
    ## MLP ##
    #########
    # compute the image size outputed by CNN
    output_imageSize = compImageSizeAfterCNN(finalImageSize, CNN_kernels, CNN_poolings)
    assert output_imageSize % 1 == 0
    n_in = output_imageSize * output_imageSize * CNN_kernels[-1]["num"]
    MLP_layer_sizes = [ n_in, 1024, 1024, 37 ]
    # the activation of the last layer is actually unused since we always use sigmoid function
    # NOTE: When using the probability constraints, it seems ReLU doesn't work
    # well and the RMSE decreases very slow. However, for fast training, we still
    # use ReLU for all the CNN layers, but change the activations of the first
    # two MLP layers to Tanh
    MLP_activations = [ ReLU, ReLU, Sigmoid ]
    MLP_use_bias = True
    # dropout or not?
    MLP_dropout = False
    # When not using dropout, we SHOULD set the dropout rates to all zeros
    MLP_dropout_rates = [ 0.5, 0.5, 0.5 ] if MLP_dropout else [ 0.0, 0.0, 0.0 ]    
    # use probability constraints or not?
    MLP_prob_constraint_on = "down"
    # grab all the params for MLP
    MLP_params = {"layer_size": MLP_layer_sizes,
                  "activation": MLP_activations,
                  "use_bias": MLP_use_bias,
                  "dropout_rate": MLP_dropout_rates,
                  "prob_constraint_on": MLP_prob_constraint_on}
                  
                  
    #############################################
    ## Set up for the training & testing phase ##
    #############################################
    
    # total number of epoch for training
    n_epochs = 2000
    n_epochs_debug = 2000 # for debugging
    # for every this many MINI-BATCHes we check the RMSE on the validation set
    validation_frequency = 10 # mini-batches
    validation_frequency_debug = 10
    # for every this many epoch we make submission using the testing set
    make_submission_frequency = 5 # epoch
    # batch size for training
    # Since 79975 = 5 * 5 * 7 * 457, so we can choose 25/35/175 as batch size
    # See the reasons detailed in function trainChunkCNN 
    batch_size = 175
    batch_size_debug = 175
    
    
    #### the learning rate
    initial_learning_rate = 0.1
    learning_rate_decay = 0.998
    weight_decay = 0.0005
    squared_filter_length_limit = 15.0
    
    
    #### the params for momentum
    mom_start = 0.5
    mom_end = 0.9
    # for epoch in [0, mom_epoch_interval], the momentum increases linearly
    # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
    mom_epoch_interval = 10
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}
                  

    # the log file to save the RMSE on the validation set              
    log_file_name = './validation_RMSE.txt'

    # save prediction in csv format?
    # we have some problems with pandas in the Workstation, so we set csv_flag as False
    csv_flag = False
    csv_flag_debug = False
    # or save it in pkl format?
    pkl_flag = True
    pkl_flag_debug = False


    ####################
    ## Start training ##
    ####################

    if debug_on == True:
        trainChunkCNN(
        length_train=length_train_debug,
        train_ratio=train_ratio_debug,
        length_test=length_test_debug,
        rotationAngles=rotationAngles,
        cropImageSizes=cropImageSizes,
        finalImageSize=finalImageSize,        
        channel=channel,
        batch_size=batch_size_debug,
        n_epochs=n_epochs_debug, 
        squared_filter_length_limit=squared_filter_length_limit,
        initial_learning_rate=initial_learning_rate,
        learning_rate_decay=learning_rate_decay,
        mom_params=mom_params,
        weight_decay=weight_decay,
        CNN_params=CNN_params,
        MLP_params=MLP_params,
        validation_frequency=validation_frequency_debug,
        make_submission_frequency=make_submission_frequency,
        log_file_name=log_file_name,
        csv_flag=csv_flag_debug,
        pkl_flag=pkl_flag_debug,
        random_seed=random_seed,
        augmentation_method=augmentation_method)
    else:
        trainChunkCNN(
        length_train=length_train,
        train_ratio=train_ratio,
        length_test=length_test,
        rotationAngles=rotationAngles,
        cropImageSizes=cropImageSizes,
        finalImageSize=finalImageSize,        
        channel=channel,
        batch_size=batch_size,
        n_epochs=n_epochs, 
        squared_filter_length_limit=squared_filter_length_limit,
        initial_learning_rate=initial_learning_rate,
        learning_rate_decay=learning_rate_decay,
        mom_params=mom_params,
        weight_decay=weight_decay,
        CNN_params=CNN_params,
        MLP_params=MLP_params,
        validation_frequency=validation_frequency,
        make_submission_frequency=make_submission_frequency,
        log_file_name=log_file_name,
        csv_flag=csv_flag,
        pkl_flag=pkl_flag,
        random_seed=random_seed,
        augmentation_method=augmentation_method)
