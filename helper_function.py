import numpy as np
import cv2
import theano

##########################################
## Helper functions dealing with images ##
##########################################
    
def compute_mean_image(image_files, train_index, batch_size,
                       rotationAngles, cropImageSizes, finalImageSize,channel,
                       substract_each_image_mean=False, augmentation_method="on_the_fly"):
    """This function computes the mean image for a bunch training images"""
    
    # image_files contains all the file names of the images, within which only
    # train_index are used for training, while the rest are used to construct
    # a validation set.
    
    # we use load_train_valid_images(image_files, train_index, batch_index, batch_size,
    #                        patchSize, imageSize, channel, substract_each_image_mean=False)
    # to load the whole training images in a mini-batch manner to avoid memory issue
    
    # number of training images
    trainNum = len(train_index)
    n_train_batches = np.int32(np.floor(trainNum/batch_size))
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
    mean_image = np.zeros((channel*(finalImageSize**2),), dtype=theano.config.floatX)
    print "       {} batches in total".format(n_train_batches)
    for train_batch_index in xrange(n_train_batches):
        print "       batch: {}".format(train_batch_index+1)
        for rotationAngle in rotationAngles:
            for cropImageSize in cropImageSizes:
                train_images_batch = load_train_valid_images(
                                      image_files, train_index,
                                      train_batch_index, batch_size,
                                      rotationAngle, cropImageSize, finalImageSize, channel,
                          		      substract_each_image_mean, augmentation_method)
                # accumalate this mini-batch of images
                mean_image += np.sum(train_images_batch, axis=0)
        
    # now we compute the mean image for the whole training images
    # note the actual training image size is n_train_batches*batch_size
    # And further with different cropImageSizes and rotationAngles, the agumented 
    # image set is of size: n_train_batches*batch_size*len(cropImageSizes)*len(rotationAngles)
    mean_image /= (n_train_batches*batch_size*len(cropImageSizes)*len(rotationAngles))
    
    return(mean_image)


def load_train_valid_images(image_files, train_index, batch_index, batch_size,
                            rotationAngle, cropImageSize, finalImageSize, channel,
                            substract_each_image_mean=False, augmentation_method="on_the_fly"):
    """Since the whole training image set contains 61578 images, and is too large to fit in 
    the memory, we use this function to load a mini-batch images at a time, and use that
    mini-batch to train the CNN. Note that this 
    
    :type image_files: list
    :param image_files: contains all the file names of the whole training images
    NOTE: image_files contains both training set and validation set
    
    :type train_index: array
    :param train_index: contains all the indices for the training set
    
    :type batch_index: int
    :param batch_index: the index of this batch
    
    :type batch_size: int
    :param batch_size: size of each mini-batch
    
    """
    
    # the index of this mini-batch
    train_batch_index = train_index[batch_index * batch_size: (batch_index + 1) * batch_size]
    train_image_files = [image_files[i] for i in train_batch_index]

    images_train = np.zeros((batch_size, channel*(finalImageSize**2)),
                            dtype=theano.config.floatX)
    for i,imgf in enumerate(train_image_files):
        # read JPEG image, see:
        # http://docs.opencv.org/modules/highgui/doc/
        # reading_and_writing_images_and_video.html?highlight=imread#imread
        #
        # flags>0 for 3-channel color image
        # flags=0 for grayscale image
        flags = (0 if channel == 1 else 3)
        if augmentation_method == "pre_computed":
            s = imgf.split("/")
            f, n = s[0], s[1]
            imgf = f + "/angle{}".format(np.int32(rotationAngle)) + "/" + n
            imgf = imgf[:-4] + "_rotationAngle{}_cropImageSize{}.jpg".format(rotationAngle, cropImageSize)
            finalImage = cv2.imread(imgf, flags)
            
        elif augmentation_method == "on_the_fly":
            originalImage = cv2.imread(imgf, flags)
            # use only the central part of size patchSize x patchSize
            height = originalImage.shape[0]
            width = originalImage.shape[1]
            center = (height/2.0, width/2.0)
            # about geometric transform using opencv-python, see
            # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/
            # py_imgproc/py_geometric_transformations/py_geometric_transformations.html
            #image_counter = 0
            #for angle in rotationAngles:
            # we do not scale it, so we set scale = 1
            rotationMatrix = cv2.getRotationMatrix2D(center, rotationAngle, scale=1)
            # keep the output image with the same size as the input
            rotatedImage = cv2.warpAffine(originalImage, rotationMatrix, (height, width))
            # see,
            # http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#warpaffine
            #print center_coord
            # crop the central part of size patchSize
            #for cropSize in cropImageSizes:
            croppedImage = cv2.getRectSubPix(rotatedImage, (cropImageSize, cropImageSize), center)
            # resize the central part to size imageSize
            finalImage = cv2.resize(croppedImage, (finalImageSize, finalImageSize))
                                
        # whether we substract mean of each image (and each channel for channel = 3)
        if(substract_each_image_mean == True):
            meanImage = np.mean(np.mean(finalImage, axis=0), axis=0)
            finalImage -= meanImage
        # swap axes, so dim = (imageSize, imageSize, 3) becomes dim = (3, imageSize, imageSize)
        if(channel == 3):
            finalImage = np.swapaxes(np.swapaxes(finalImage, 1, 2), 0, 1)
        # reshape it into 1-D rasterized image
        finalImage = np.reshape(finalImage, channel*(finalImageSize**2))
        images_train[i] = finalImage
            
    return(images_train)
    

def load_train_valid_target(solution, train_index, batch_index, batch_size):

    train_batch_index = train_index[batch_index * batch_size: (batch_index + 1) * batch_size]

    solution_train = solution[train_batch_index]

    return(solution_train)
    

def compImageSizeAfterCNN(imageSize, CNN_kernels, CNN_poolings, reversed=False):
    """This function computes the image size before/after CNN for a given image size
    """
    filter_num = len(CNN_kernels)
    if reversed == True:
        for i in np.arange(filter_num)[::-1]:
            imageSize = (imageSize*CNN_poolings[i]["size"] + CNN_kernels[i]["size"] - 1)
        print("Image size before CNN is {}.".format(imageSize))
    else:
        for i in np.arange(filter_num):
            imageSize = (imageSize - CNN_kernels[i]["size"] + 1)/CNN_poolings[i]["size"]
        print("Image size after CNN is {}.".format(imageSize))
    return(imageSize)