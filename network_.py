
from __future__ import print_function

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

    def next_batch( batch_size ):
        pass
        # WRITE THIS SHITTTTTTT        

    def train( self ):
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
            sess.run(init)

            # Training cycle
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                total_batch = int(self.data_length/self.batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(self.batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x,
                                                          self.y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
            print("Optimization Finished!")

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({self.x: mnist.test.images, self.y: mnist.test.labels}))
            sess.close()

#if __name__ == "__main__":
#    n = Network()
#    n.train()
