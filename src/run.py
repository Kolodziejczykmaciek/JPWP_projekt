import data_load
import network

training_data, validation_data, test_data = data_load.load_data_wrapper() #get and parse data
net = network.Network([784, 30, 10]) #build up a network with 784 inputs, 30 hidden and 10 output neurons
net.SGD(training_data, 30, 10, 3.0, test_data=test_data) #train the network with stochastic gradient descent

