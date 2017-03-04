
from __future__ import print_function
import random
import csv
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

class Network:
    def __init__( self, learning_rate=0.001, max_steps=15, hidden1=256, hidden2=256, batch_size=100, input_data_dir=None, log_dir=None, data_length=mnist.train.num_examples, data_width=784, n_classes=10 ):
        # Parameters
        self.learning_rate = learning_rate
        self.training_epochs = max_steps
        self.batch_size = batch_size
        self.display_step = 1
        self.input_data_dir = input_data_dir 

        # Network Parameters
        self.n_hidden_1 = hidden1 # 1st layer number of features
        self.n_hidden_2 = hidden2  # 2nd layer number of features
        self.n_input = data_width # MNIST data input (img shape: 28*28)
        self.data_length = data_length # Number of total data points
        self.n_classes = n_classes # MNIST total classes (0-9 digits)

        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])

        # Initialize weights and biases dicts
        self.weights = {}
        self.biases = {}

    # Create model
    def multilayer_perceptron( self, x, weights, biases ):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer
    
    # Read the data from the csv
    def read_data( self, dir ):
        print( "reading data..." )
        # List of lists that will hold data, [0] holds inputs, [1] holds labels
        self.training_data = [[],[]]
        with open(dir+"/data.csv", newline='') as csvfile:
            reader = csv.reader(csvfile) 
            for row in reader:
                # Get first through second to last items from each line as floats
                self.training_data[0].append(list(map(float, row[0:-1])))
                # create label array, should be all 0 except correct answer, which is 1
                labels = [0]*self.n_classes
                labels[int(row[-1])] = 1
                self.training_data[1].append(labels) 

    # Divide data up into batches
    def make_batches( self, num_batches ):
        batch_inputs = [] 
        batch_labels = []
        # define random seed and shuffle both arrays, should shuffle same way because seed is the same
        random.seed()
        random.shuffle( self.training_data[0] )
        random.shuffle( self.training_data[1] )
        for i in range( num_batches ):
                # make batches
                batch_inputs.append( self.training_data[0][i*self.batch_size:(i+1)*self.batch_size] )
                batch_labels.append( self.training_data[1][i*self.batch_size:(i+1)*self.batch_size])
        return batch_inputs, batch_labels 

    def train( self ):
        data = self.read_data( self.input_data_dir )
        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Construct model
        pred = self.multilayer_perceptron(self.x, self.weights, self.biases)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            print( "STARTING SESSION" )
            sess.run(init)
            # Training cycle
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                num_batches = int(self.data_length/self.batch_size)
                # Make batches
                batch_inputs, batch_labels = self.make_batches( num_batches )
                # Loop over all batches
                for i in range(num_batches):
                    # Get inputs and labels for current batch
                    batch_x, batch_y = [batch_inputs[i], batch_labels[i]]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x,
                                                          self.y: batch_y})
                    # Compute average loss
                    avg_cost += c / num_batches
                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
            print("Optimization Finished!")

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({self.x: self.training_data[0], self.y: self.training_data[1]}))
            sess.close()

#if __name__ == "__main__":
#    n = Network()
#    n.train()
