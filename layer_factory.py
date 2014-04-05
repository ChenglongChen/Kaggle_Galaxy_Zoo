import numpy as np


import theano
import theano.tensor as T
import theano.sandbox
import theano.tensor.shared_randomstreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

##################################
## Various activation functions ##
##################################
#### rectified linear unit
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
#### sigmoid
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
#### tanh
def Tanh(x):
    y = T.tanh(x)
    return(y)
#### softmax
def SoftMax(x):
    y = T.nnet.softmax(x)
    return(y)
    
    
############################################
## Definition of LogisticRegression class ##
############################################
class LogisticRegression(object):
    """Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. 
    
    Instead of classification, we here use LR for the purpose of regression,
    since the targets in this case are continous and in the range [0,1].
    Secondly, they have some kind of probability interpration, especially for 
    class 1 and class 6, with the following facts hold
    class 1.1 + class 1.2 + class 1.3 = 1
    class 6.1 + class 6.2 = 1
    """

    #### initialization
    def __init__(self, input, n_in, n_out,
                 W=None, b=None, prob_constraint_on=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
    
        :type prob_constraint_on: boolean
        :param prob_constraint_on: whether we use the probability constraints or not

        """

        # initialize weight matrix W
        if W is None:
            self.W = theano.shared(
                    value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W')
        else:
            self.W = W

        # initialize bias b
        if b is None:
            self.b = theano.shared(
                    value=np.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        # compute prediction
        # the linear output
        lin_output = T.dot(input, self.W) + self.b
        
        if prob_constraint_on == None:
            #### we do not use those probability constraints
            self.y_pred = Sigmoid(lin_output)

        elif prob_constraint_on == "top":
            #### We first predict the probability of each class using softmax.
            # We then weight those probabilities by multiplying them by the
            # probability of their parent in the Galaxy Zoo Decision Tree.
        
            # class 1
            prob_Class1 = SoftMax(lin_output[:,0:3])
            
            # class 2
            prob_Class2 = SoftMax(lin_output[:,3:5])
            # weight these probabilities using the probability of class 1.2
            prob_Class2 *= T.shape_padright(prob_Class1[:,1])
            
            # class 3
            prob_Class3 = SoftMax(lin_output[:,5:7])
            # weight these probabilities using the probability of class 2.2
            prob_Class3 *= T.shape_padright(prob_Class2[:,1])
            
            # class 4
            prob_Class4 = SoftMax(lin_output[:,7:9])
            # weight these probabilities using the probability of class 2.2
            prob_Class4 *= T.shape_padright(prob_Class2[:,1])
            
            # class 5
            prob_Class5 = SoftMax(lin_output[:,9:13])
            # weight these probabilities using the probability of class 2.2
            prob_Class5 *= T.shape_padright(prob_Class2[:,1])
            
            # class 6
            prob_Class6 = SoftMax(lin_output[:,13:15])
            
            # class 7
            prob_Class7 = SoftMax(lin_output[:,15:18])
            # weight these probabilities using the probability of class 1.1
            prob_Class7 *= T.shape_padright(prob_Class1[:,0])
            
            # class 8
            prob_Class8 = SoftMax(lin_output[:,18:25])
            # weight these probabilities using the probability of class 6.1
            prob_Class8 *= T.shape_padright(prob_Class6[:,0])
            
            # class 9
            prob_Class9 = SoftMax(lin_output[:,25:28])
            # weight these probabilities using the probability of class 2.1
            prob_Class9 *= T.shape_padright(prob_Class2[:,0])
            
            # class 10
            prob_Class10 = SoftMax(lin_output[:,28:31])
            # weight these probabilities using the probability of class 4.1
            prob_Class10 *= T.shape_padright(prob_Class4[:,0])
            
            # class 11
            prob_Class11 = SoftMax(lin_output[:,31:37])
            # weight these probabilities using the probability of class 4.1
            prob_Class11 *= T.shape_padright(prob_Class4[:,0])
 
            # concatenate all the probabilities into a single tensor variable
            self.y_pred = T.concatenate(
                            [prob_Class1, prob_Class2, prob_Class3, prob_Class4,
                             prob_Class5, prob_Class6, prob_Class7, prob_Class8,
                             prob_Class9, prob_Class10, prob_Class11], axis=1)
        elif prob_constraint_on == "down":
            #### we use those probability constraints
            
            # the following probabilities should sum up to 1, so we use SoftMax
            # to predict all of them
            ind1 = [2, 8, 15, 16, 17, 25, 26, 27, 31, 32, 33, 34, 35, 36]
            p1 = SoftMax(lin_output[:,ind1])
            prob_Class1_3 = p1[:,0]
            prob_Class4_2 = p1[:,1]
            prob_Class7 = p1[:,2:5]
            prob_Class9 = p1[:,5:8]
            prob_Class11 = p1[:,8:14]
            
            prob_Class4_1 = T.sum(prob_Class11, axis=1)
            prob_Class2_1 = T.sum(prob_Class9, axis=1)
            prob_Class2_2 = prob_Class4_1 + prob_Class4_2
            prob_Class1_1 = T.sum(prob_Class7, axis=1)
            prob_Class1_2 = prob_Class2_1 + prob_Class2_2
            prob_Class1 = T.concatenate(
                                        [T.shape_padright(prob_Class1_1),
                                         T.shape_padright(prob_Class1_2),
                                         T.shape_padright(prob_Class1_3)], axis=1)
            prob_Class2 = T.concatenate(
                                        [T.shape_padright(prob_Class2_1),
                                         T.shape_padright(prob_Class2_2)], axis=1)
            prob_Class4 = T.concatenate(
                                        [T.shape_padright(prob_Class4_1),
                                         T.shape_padright(prob_Class4_2)], axis=1)
            
            # the following probabilities should sum up to 1, so we use SoftMax
            # to predict all of them
            ind2 = [14, 18, 19, 20, 21, 24, 23, 24]                             
            p2 = SoftMax(lin_output[:,ind2])
            prob_Class6_2 = p2[:,0]
            prob_Class8 = p2[:,1:8]
            prob_Class6_1 = T.sum(prob_Class8, axis=1)
            prob_Class6 = T.concatenate(
                                        [T.shape_padright(prob_Class6_1),
                                         T.shape_padright(prob_Class6_2)], axis=1)
            
            # for the following probabilities, we resort to the same strategy in
            # the "top" option
            # class 3
            prob_Class3 = SoftMax(lin_output[:,5:7])
            # weight these probabilities using the probability of class 2.2
            prob_Class3 *= T.shape_padright(prob_Class2[:,1])
            
            # class 5
            prob_Class5 = SoftMax(lin_output[:,9:13])
            # weight these probabilities using the probability of class 2.2
            prob_Class5 *= T.shape_padright(prob_Class2[:,1])
                             
            # class 10
            prob_Class10 = SoftMax(lin_output[:,28:31])
            # weight these probabilities using the probability of class 4.1
            prob_Class10 *= T.shape_padright(prob_Class4[:,0])
            
            # concatenate all the probabilities into a single tensor variable
            self.y_pred = T.concatenate(
                            [prob_Class1, prob_Class2, prob_Class3, prob_Class4,
                             prob_Class5, prob_Class6, prob_Class7, prob_Class8,
                             prob_Class9, prob_Class10, prob_Class11], axis=1)
        
                                     
        # parameters of the model
        self.params = [self.W, self.b]

    #### the loss function in this case is (R)MSE
    def MSE(self, y):
        return T.mean((self.y_pred - y)**2)
        
    #### the cross-entropy
    def CrossEntropy(self, y):
        return -T.mean(y*T.log(self.y_pred))
        
    #### KL divergence
    def KL(self, y):
        return T.mean(y*T.log(y/self.y_pred) + (1-y)*T.log((1-y)/(1-self.y_pred)))
        


#####################################
## Definition of HiddenLayer class ##
#####################################
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation=Tanh, use_bias=True,
                 W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.activation = activation

        if W is None:
            W_values = np.asarray(0.01 * rng.standard_normal(
                       size=(n_in, n_out)), dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = W
        
        if b is None:
            if activation == ReLU:
                # for ReLU, we initialize bias as constant 1 as suggested in
                # the dropout and ImageNet paper
                b_values = np.ones((n_out,), dtype=theano.config.floatX)
            else:
                b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


###############################################
## Helper function to compute dropout output ##
###############################################
#### Credit to Misha Denil
def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
            
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


############################################
## Definition of DropoutHiddenLayer class ##
############################################
#### Credit to Misha Denil
class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, use_bias, dropout_rate, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)
        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


#############################
## Definition of MLP class ##
#############################
#### Credit to Misha Denil
class MLP(object):
    """A multilayer perceptron with all the trappings required to do dropout
    training.

    """
    def __init__(self, rng, input, layer_sizes, dropout_rates,
                 activations=None, use_bias=True, prob_constraint_on=True):
        """For training without dropout, you should set dropout_rates to all
        zeros. Otherwise, with non-zero dropout_rate, dropout is included in
        the training, with dropout probability for each hidden layer in the MLP
        specified by dropout_rate.
        """
        # Set up all the hidden layers
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        # we build two parallel layers
        # - training_layers for training with/without dropout
        # - testing_layers for testing the performance
        self.training_layers = []
        self.testing_layers = []
        
        # dropout the input
        next_training_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        next_testing_layer_input = input
        
        layer_counter = 0
        for n_in, n_out in weight_matrix_sizes[:-1]:
            
            # setup the training layer
            next_training_layer = DropoutHiddenLayer(rng=rng,
                    input=next_training_layer_input,
                    n_in=n_in, n_out=n_out,
                    activation=activations[layer_counter],
                    use_bias=use_bias,
                    dropout_rate=dropout_rates[layer_counter])
            self.training_layers.append(next_training_layer)
            next_training_layer_input = next_training_layer.output

            # setup the testing layer
            # Reuse the paramters from the dropout layer here, in a different
            # path through the graph.
            next_testing_layer = HiddenLayer(rng=rng,
                    input=next_testing_layer_input,
                    n_in=n_in, n_out=n_out,
                    activation=activations[layer_counter],
                    use_bias=use_bias,
                    # for testing, we SHOULD scale the weight matrix W with (1-p)
                    W=next_training_layer.W * (1 - dropout_rates[layer_counter]),
                    b=next_training_layer.b)
            self.testing_layers.append(next_testing_layer)
            next_testing_layer_input = next_testing_layer.output
            
            layer_counter += 1
        
        # Set up the output layer for training layers
        n_in, n_out = weight_matrix_sizes[-1]
        training_output_layer = LogisticRegression(
                input=next_training_layer_input,
                n_in=n_in, n_out=n_out,
                prob_constraint_on=prob_constraint_on)
        self.training_layers.append(training_output_layer)

        # Set up the output layer for testing layers
        # Again, reuse paramters in the dropout output.
        testing_output_layer = LogisticRegression(
            input=next_testing_layer_input,
            n_in=n_in, n_out=n_out,
            # for testing, we SHOULD scale the weight matrix W with (1-p)
            W=training_output_layer.W * (1 - dropout_rates[-1]),
            b=training_output_layer.b,
            prob_constraint_on=prob_constraint_on)
        self.testing_layers.append(testing_output_layer)

        # Use the MSE of the logistic regression layer as the objective
        # In training phase, we use the MSE of the logistic regression layer
        # which is on top of the dropout_layers
        self.training_MSE = self.training_layers[-1].MSE
        # In validation/testing phase, we use the MSE of the logistic regression layer
        # which is on top of the normal_layers
        self.testing_MSE = self.testing_layers[-1].MSE
       
        # NOTE: for prediction, we use all the weights, thus we should use
        # the normal layers instead of the dropout layers
        self.y_pred = self.testing_layers[-1].y_pred
        
        # Grab all the parameters together.
        self.params = [ param for layer in self.training_layers for param in layer.params ]
        # The above is Double Iteration in List Comprehension
        # See the discussion in
        # http://stackoverflow.com/questions/17657720/python-list-comprehension-double-for
        # In regular for-loop format, we have
        # for layer in self.dropout_layers:
        #     for param in layer.params:
        #         put param in the resulting list
        
       
#######################################
## Definition of ConvPoolLayer class ##
#######################################
class ConvPoolLayer(object):
    """ A convolutional followed by pooling layer """

    def __init__(self, rng, input, filter_shape, image_shape,
                 pooltype="max", poolsize=(2, 2), activation=Tanh):


        assert image_shape[1] == filter_shape[1]
        self.input = input

        W_values = np.asarray(0.01 * rng.standard_normal(
                   size=filter_shape), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        if activation == ReLU:
            # for ReLU, we initialize bias as constant 1 as suggested in
            # the dropout and ImageNet paper
            b_values = np.ones((filter_shape[0],), dtype=theano.config.floatX)
        else:
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                               filter_shape=filter_shape, image_shape=image_shape)
        if pooltype == "max":
            # downsample each feature map individually, using maxpooling
            pooled_out = downsample.max_pool_2d(input=conv_out,
                                                ds=poolsize,
                                                ignore_border=True)
        elif pooltype == "mean":
            # For mean/average pooling, see
            # https://groups.google.com/forum/#!msg/theano-users/MiyDStm1W0c/L1dcRAz1Y9kJ
            # and for images2neibs, see
            # http://deeplearning.net/software/theano/library/sandbox/neighbours.html
        
            # no module theano.sandbox.neighbours found? will look into this in the future
            pooled_out = theano.sandbox.neighbours.images2neibs(ten4=conv_out,
                                                                neib_shape=poolsize,
                                                                ignore_border=True)
            pooled_out = pooled_out.mean(axis=-1)
        elif pooltype == "none":
            pooled_out = conv_out

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


#############################
## Definition of CNN class ##
#############################
class CNN(object):
    """A CNN architecture

    """
    def __init__(self, rng, input, batch_size, channel, image_size, kernels, poolings, activations=None):
                     
        self.layers = []
        #next_CNN_n_in = channel
        next_CNN_layer_input = input
        #batch_size = input.shape[0]
        next_CNN_input_size = channel
        #image_size = input.shape[2]
        for i in xrange(len(kernels)):
            image_shape = (batch_size, next_CNN_input_size, image_size, image_size)
            filter_shape = (kernels[i]["num"], next_CNN_input_size,
                            kernels[i]["size"], kernels[i]["size"])
            poolsize = (poolings[i]["size"], poolings[i]["size"])
            next_CNN_layer = ConvPoolLayer(rng,
                                           input=next_CNN_layer_input,
                                           image_shape=image_shape,
                                           filter_shape=filter_shape,
                                           pooltype=poolings[i]["type"],
                                           poolsize=poolsize,
                                           activation=activations[i])
            self.layers.append(next_CNN_layer)
            
            next_CNN_layer_input = next_CNN_layer.output
            image_size = (image_size - kernels[i]["size"] + 1)/poolings[i]["size"]
            assert image_size % 1 == 0
            next_CNN_input_size = kernels[i]["num"]
        
        self.output = self.layers[-1].output
        self.params = [ param for layer in self.layers for param in layer.params ]
        # The above is Double Iteration in List Comprehension as used in MLP class