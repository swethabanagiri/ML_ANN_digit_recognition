import numpy as np
import pickle, random, pdb
from PIL import Image
import os
def sigmoid(a):
    return 1/(1+np.exp(-a))
def load(model_file):
    """
    Loads the network from the model_file
    :param model_file: file onto which the network is saved
    :return: the network
    """
    return pickle.load(open(model_file))

class NeuralNetwork(object):
    """
    Implementation of an Artificial Neural Network
    """
    def __init__(self, input_dim, hidden_size, output_dim, learning_rate=0.01, reg_lambda=0.01):
        """
        Initialize the network with input, output sizes, weights, biases, learning_rate and regularization parameters
        :param input_dim: input dim
        :param hidden_size: number of hidden units
        :param output_dim: output dim
        :param learning_rate: learning rate alpha
        :param reg_lambda: regularization rate lambda
        :return: None
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(self.hidden_size, self.input_dim) * 0.01 # Weight matrix for input to hidden
        
        self.Why = np.random.randn(self.output_dim, self.hidden_size) * 0.01 # Weight matrix for hidden to output
        self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
        self.by = np.zeros((self.output_dim, 1)) # output bias
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        #print self.Wxh.shape,self.bh.shape,self.Why.shape,self.by.shape
    def _feed_forward(self, X):
        """
        Performs forward pass of the ANN
        :param X: input
        :return: hidden activations, softmax probabilities for the output
        """
        
        # Add code to calculate h_a and probs
        X=X.reshape(self.input_dim,1)
        h_a=sigmoid(np.dot(self.Wxh,X)+self.bh)
        ys=np.dot(self.Why,h_a)+self.by
        #print np.exp(a),np.exp(a).sum()
        probs=np.exp(ys)/np.sum(np.exp(ys))
        #print "x",X.shape
        #print "h_a",h_a.shape
        #print "ys",ys.shape
        #print "probs",probs.shape
        #exit(0)
        #print probs
        
        #exit(0)
        return h_a, probs

    def _regularize_weights(self, dWhy, dWxh, Why, Wxh):
        """
        Add regularization terms to the weights
        :param dWhy: weight derivative from hidden to output
        :param dWxh: weight derivative from input to hidden
        :param Why: weights from hidden to output
        :param Wxh: weights from input to hidden
        :return: dWhy, dWxh
        """
        # Add code to calculate the regularized weight derivatives
        dWxh+=self.reg_lambda*Wxh
        dWhy+=self.reg_lambda*Why
        return dWhy, dWxh

    def _update_parameter(self, dWxh, dbh, dWhy, dby):
        """
        Update the weights and biases during gradient descent
        :param dWxh: weight derivative from input to hidden
        :param dbh: bias derivative from input to hidden
        :param dWhy: weight derivative from hidden to output
        :param dby: bias derivative from hidden to output
        :return: None
        """
        # Add code to update all the weights and biases here
        self.Wxh+=self.learning_rate*dWxh
        self.Why+=self.learning_rate*dWhy
        self.by+=self.learning_rate*dby
        self.bh+=self.learning_rate*dbh

    def _back_propagation(self, X, t, h_a, probs):
        """
        Implementation of the backpropagation algorithm
        :param X: input
        :param t: target
        :param h_a: hidden activation from forward pass
        :param probs: softmax probabilities of output from forward pass
        :return: dWxh, dWhy, dbh, dby
        """
        # Add code to compute the derivatives and return
        t=t.reshape(self.output_dim,1)
        #print t.shape,probs.shape
        dy=(t-probs)
        X=X.reshape(self.input_dim,1)
        dh=np.dot(self.Why.T,dy)* h_a*(1-h_a)       
        dWxh=np.dot(dh,X.T)
        dWhy=np.dot(dy,h_a.T)
        dby=np.sum(dy)
        dbh=np.sum(dh)
        return dWxh, dWhy, dbh, dby

    def _calc_smooth_loss(self, loss, len_examples, regularizer_type=None):
        """
        Calculate the smoothened loss over the set of examples
        :param loss: loss calculated for a sample
        :param len_examples: total number of samples in training + validation set
        :param regularizer_type: type of regularizer like L1, L2, Dropout
        :return: smooth loss
        """
        if regularizer_type == 'L2':
            # Add regulatization term to loss
            loss = loss + 5 / (2*len_examples) * sum(self.Wxh ** 2)
            return 1./len_examples * loss
        else:
            return 1./len_examples * loss

    def train(self, inputs, targets, validation_data, num_epochs, regularizer_type=None):
        """
        Trains the network by performing forward pass followed by backpropagation
        :param inputs: list of training inputs
        :param targets: list of corresponding training targets
        :param validation_data: tuple of (X,y) where X and y are inputs and targets respectively
        :param num_epochs: number of epochs for training the model
        :param regularizer_type: type of regularizer like L1, L2, Dropout
        :return: None
        """
        for k in xrange(num_epochs):
            loss = 0
            for i in xrange(len(inputs)):
                # Forward pass
                #print len(inputs[i])
                h_a,probs=self._feed_forward(inputs[i])
                # Backpropagation
                #print "In back",probs
                dWxh, dWhy, dbh, dby=self._back_propagation(inputs[i],targets[i],h_a, probs)
                # Perform the parameter update with gradient descent
                self._update_parameter(dWxh, dbh, dWhy, dby)
                    

            # validation using the validation data

            validation_inputs = validation_data[0]
            validation_targets = validation_data[1]

            #print 'Validation'
            loss=0;
            for i in xrange(len(validation_inputs)):
                # Forward pass
                h_a,probs=self._feed_forward(validation_inputs[i])
                loss+=np.log(probs[np.argmax(probs)])
                # Backpropagation
                dWxh, dWhy, dbh, dby=self._back_propagation(validation_inputs[i],validation_targets[i],h_a, probs)
                # Perform the parameter update with gradient descent
                #self._update_parameter(dWxh, dbh, dWhy, dby)

                # Backpropagation

                if regularizer_type == 'L2':
                    dWhy,dWxh=self._regularize_weights( dWhy, dWxh, self.Why,self.Wxh)
                self._update_parameter(dWxh, dbh, dWhy, dby)

                # Perform the parameter update with gradient descent
                

            if k%1 == 0:
                print "Epoch " + str(k) + " : Loss = " + str(self._calc_smooth_loss(loss, len(inputs), regularizer_type))


    def predict(self, X):
        """
        Given an input X, emi
        :param X: test input
        :return: the output class
        """
        # Implement the forward pass and return the output class (argmax of the softmax outputs)
        xs=np.array(X).reshape((self.input_dim,1))
        h_a = np.tanh(np.dot(self.Wxh,xs)).reshape((self.hidden_size,1)) + self.bh
        ys = np.dot(self.Why,h_a) + self.by
        probs = np.exp(ys)/np.sum(np.exp(ys))
        #print probs
        return np.argmax(probs)
    
    def save(self, model_file):
        """
        Saves the network to a file
        :param model_file: name of the file where the network should be saved
        :return: None
        """
        pickle.dump(self, open(model_file, 'wb'))

if __name__ == "__main__":
    """
    Toy problem where input = target
    """
    nn = NeuralNetwork(4,8,4)
    inputs = []
    targets = []
    for i in range(1000):
        num = random.randint(0,3)
        inp = np.zeros((4,))
        inp[num] = 1
        inputs.append(inp)
        targets.append(inp)

    nn.train(inputs[:800], targets[:800], (inputs[800:], targets[800:]), 10)
    print nn.predict([1,0,0,0])
    print nn.predict([0,1,0,0])
    print nn.predict([0,0,1,0])
    print nn.predict([0,0,0,1])


