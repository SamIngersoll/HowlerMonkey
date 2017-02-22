import tensorflow as tf
import argparse
import os.path
import sys
import time
import network
import numpy as np

class NetworkFeeder:
    def __init__(self, learning_rate, max_steps, hidden1, hidden2, batch_size, input_data_dir, log_dir, data_rows, data_columns):
        #will need to do input data
        self.FLAGS = tf.app.flags.FLAGS
        tf.app.flags.DEFINE_float('learning_rate', learning_rate,"""Initial learning rate.""")
        tf.app.flags.DEFINE_integer('max_steps', max_steps,"""Number of steps to run trainer.""")
        tf.app.flags.DEFINE_integer('hidden1', hidden1,"""Number of units in hidden layer 1.""")
        tf.app.flags.DEFINE_integer('hidden2', hidden2,"""Number of units in hidden layer 2.""")
        tf.app.flags.DEFINE_integer('batch_size', batch_size,"""Size of each batch.""")
        tf.app.flags.DEFINE_string('input_data_dir', input_data_dir,"""Directory to the input data.""")
        tf.app.flags.DEFINE_string('log_dir', log_dir,"""Directry to put the log.""")
        tf.app.flags.DEFINE_integer('data_rows', data_rows,"""Row Size.""")
        tf.app.flags.DEFINE_integer('data_columns', data_columns,"""Column Size.""")
        tf.app.flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data ' 'for unit testing.')
        tf.app.run(main=self.main)

    def placeholder_inputs(self, batch_size):
        data_placeholder = tf.placeholder(tf.float32, shape=(batch_size, self.FLAGS.data_rows * self.FLAGS.data_columns))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
        return data_placeholder, labels_placeholder


    def fill_feed_dict(self, data_set, data_pl, labels_pl):
        data_feed, labels_feed = data_set.next_batch(self.FLAGS.batch_size, self.FLAGS.fake_data)
        feed_dict = {
            data_pl: data_feed,
            labels_pl: labels_feed,
        }
        return feed_dict


    def do_eval(self, sess, eval_correct, data_placeholder, labels_placeholder, data_set):
        true_count = 0
        steps_per_epoch = data_set.num_examples // self.FLAGS.batch_size
        num_examples = steps_per_epoch * self.FLAGS.batch_size
        for step in range(steps_per_epoch):
            feed_dict = self.fill_feed_dict(data_set, data_placeholder, labels_placeholder)
            true_count += sess.run(eval_correct, feed_dict=feed_dict)
        precision = float(true_count) / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))


    def run_training(self):
#        data_sets = input_data.read_data_sets(self.FLAGS.input_data_dir, self.FLAGS.fake_data)

        data_train = tf.contrib.learn.datasets.base.load_csv_with_header(filename=self.FLAGS.input_data_dir+"/data.csv", target_dtype=np.int, features_dtype=np.float32)

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            # Generate placeholders for the data and labels.
            data_placeholder, labels_placeholder = self.placeholder_inputs(
                self.FLAGS.batch_size)

            # Build a Graph that computes predictions from the inference model.
            logits = network.inference(data_placeholder, self.FLAGS.data_rows, self.FLAGS.data_columns,
                                     self.FLAGS.hidden1,
                                     self.FLAGS.hidden2)

            # Add to the Graph the Ops for loss calculation.
            loss = network.loss(logits, labels_placeholder)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = network.training(loss, self.FLAGS.learning_rate)

            # Add the Op to compare the logits to the labels during evaluation.
            eval_correct = network.evaluation(logits, labels_placeholder)

            # Build the summary Tensor based on the TF collection of Summaries.
            summary = tf.summary.merge_all()

            # Add the variable initializer Op.
            init = tf.global_variables_initializer()

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.summary.FileWriter(self.FLAGS.log_dir, sess.graph)

            # And then after everything is built:

            # Run the Op to initialize the variables.
            sess.run(init)

            # Start the training loop.
            for step in range(self.FLAGS.max_steps):
              start_time = time.time()

              # Fill dictionary
              feed_dict = self.fill_feed_dict(data_sets.train,
                                         data_placeholder,
                                         labels_placeholder)

              # Run one step of the model.  The return values are the activations
              # from the `train_op` (which is discarded) and the `loss` Op.  To
              # inspect the values of your Ops or variables, you may include them
              # in the list passed to sess.run() and the value tensors will be
              # returned in the tuple from the call.
              _, loss_value = sess.run([train_op, loss],
                                       feed_dict=feed_dict)

              duration = time.time() - start_time

              # Write the summaries and print an overview fairly often.
              if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

              # Save a checkpoint and evaluate the model periodically.
              if (step + 1) % 1000 == 0 or (step + 1) == self.FLAGS.max_steps:
                checkpoint_file = os.path.join(self.FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                self.do_eval(sess,
                        eval_correct,
                        data_placeholder,
                        labels_placeholder,
                        data_sets.train)
                print('Validation Data Eval:')
                self.do_eval(sess,
                        eval_correct,
                        data_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                self.do_eval(sess,
                        eval_correct,
                        data_placeholder,
                        labels_placeholder,
                        data_sets.test)

    def main(self, _):
        print("\n","\n",self.FLAGS.log_dir)
        if tf.gfile.Exists(self.FLAGS.log_dir):
            tf.gfile.DeleteRecursively(self.FLAGS.log_dir)
        tf.gfile.MakeDirs(self.FLAGS.log_dir)
        self.run_training()
