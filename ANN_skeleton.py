import numpy as np
import pickle, random

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
    def __init__(self, input_dim, hidden_size, output_dim, learning_rate=0.001, reg_lambda=0.01):
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

    def _feed_forward(self, X):
        """
        Performs forward pass of the ANN
        :param X: input
        :return: hidden activations, softmax probabilities for the output
        """
        
        xs=np.array(X).reshape((self.input_dim,1))
        h_a = np.tanh(np.dot(self.Wxh,xs)).reshape((self.hidden_size,1)) + self.bh
        ys = np.dot(self.Why,h_a) + self.by
        probs = np.exp(ys)/np.sum(np.exp(ys))
        #print "xs",xs.shape
        #print "h_a",h_a.shape
        #print "ys",ys.shape
        #print "probs",probs.shape
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
        dWhy += self.reg_lambda * Why
        dWxh += self.reg_lambda * Wxh
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
        self.Wxh += -self.learning_rate *dWxh
        self.bh += -self.learning_rate * dbh
        self.Why += -self.learning_rate *dWhy
        self.by += -self.learning_rate * dby
        # Add code to update all the weights and biases here

    def _back_propagation(self, X, t, h_a, probs):
        """
        Implementation of the backpropagation algorithm
        :param X: input
        :param t: target
        :param h_a: hidden activation from forward pass
        :param probs: softmax probabilities of output from forward pass
        :return: dWxh, dWhy, dbh, dby
        """
        
        #
        # Add code to compute the derivatives and return
        dy = np.copy(probs)
        dy[np.where(t==1)[0][0]] -= 1
        X=X.reshape(self.input_dim,1)
        dWhy = np.dot(dy,h_a.T)
        dby = np.sum(dy,axis=0,keepdims=True)
        dh = np.dot(dWhy.T,dy) * (1-np.power(h_a,2))
        dWxh = np.dot(dh,X.T)
        dbh = np.sum(dh,axis=0)  
        return dWxh, dWhy, dbh, dby,probs

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

            return 1./len_examples * loss
        else:
            return 1./len_examples * loss

    def train(self, inputs, targets, validation_data, num_epochs, regularizer_type=None):
        """(inputs[:800], targets[:800], (inputs[800:], targets[800:]), 10
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
                X=inputs[i]
                h,p=self._feed_forward(X)
                inp = np.zeros((self.output_dim,))
                inp[targets[i]] = 1


                # Backpropagation
                d1,d3,d2,d4,probs=self._back_propagation(X,inp,h,p)

                # Perform the parameter update with gradient descent
                self._update_parameter(d1,d2,d3,d4)
                


            # validation using the validation data

            validation_inputs = validation_data[0]
            validation_targets = validation_data[1]

            #print 'Validation'

            for i in xrange(len(validation_inputs)):
                # Forward pass
                X=validation_inputs[i]
                h,p=self._feed_forward(X)
                inp = np.zeros((self.output_dim,))
                inp[validation_targets[i]] = 1

                # Backpropagation
                d1,d3,d2,d4,probs=self._back_propagation(X,inp,h,p)
                loss+=np.log(probs[np.argmax(probs)])

                if regularizer_type == 'L2':
                    d3,d1 = self._regularize_weights(d3,d1,self.Why,self.Wxh)
                    # Add code for regularization of weights

                # Perform the parameter update with gradient descent
                self._update_parameter(d1,d2,d3,d4)
                


            if k%1 == 0:
                print "Epoch " + str(k) + " : Loss = " + str(self._calc_smooth_loss(loss, len(inputs), regularizer_type))


    def predict(self, X):
        """
        Given an input X, emi
        :param X: test input
        :return: the output class
        """
        xs=np.array(X).reshape((self.input_dim,1))
        h_a = np.tanh(np.dot(self.Wxh,xs)).reshape((self.hidden_size,1)) + self.bh
        ys = np.dot(self.Why,h_a) + self.by
        probs = np.exp(ys)/np.sum(np.exp(ys))
        #print probs
        return np.argmax(probs)
        
        # Implement the forward pass and return the output class (argmax of the softmax outputs)

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
        targets.append(num)

    nn.train(inputs[:800], targets[:800], (inputs[800:], targets[800:]), 10)
    print nn.predict([1,0,0,0])
    print nn.predict([0,1,0,0])
    print nn.predict([0,0,1,0])
    print nn.predict([0,0,0,1])
