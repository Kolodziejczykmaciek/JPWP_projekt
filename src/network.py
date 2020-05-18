import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #initilizes randomly with Gaussian distr.
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])] #initilizes randomly with Gaussian distr.


    def feedforward(self, a): # caltulates the output of the networkwith given input a
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, num_train_units, mini_batch_size, eta, #Trainng network
            test_data=None):

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(num_train_units):
            random.shuffle(training_data)  #mix training data

            mini_batches = [
                training_data[k:k+mini_batch_size]  #take ten data sets
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:     #w jednym mini_batchu jest 10 iteracji, 10 próbek
                self.update_mini_batch(mini_batch, eta) # to sie wykona 5k razy
            if test_data:
                print("Number of training unit {}. Correctly classified digits: {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Unit {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):   #updates weights and biases, applying drad desct
                                                    #using backpropagation
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw              #equetion 20 update rule
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb               #equation 21 update rule
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y): # function to compute complicated gradient
        """Returns a tuple (nabla_b, nabla_w) representing the
        gradient for the cost function. nabla_b and
        nabla_w are layer-by-layer lists of numpy arrays."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data): #returns number of correctly classified digits
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
