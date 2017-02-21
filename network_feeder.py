import tensorflow as tf
#from tensorboard import Tensorboard
import argparse
import os.path
import sys
import time
import network
FLAGS = None
input_data = None

class NetworkFeeder:
    def __init__( self, learning_rate, max_steps, hidden1, hidden2, batch_size, input_data_dir, log_dir, data_rows, data_columns):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--learning_rate',
            type=float,
            default=learning_rate,
            help='Initial learning rate.'
        )
        parser.add_argument(
          '--max_steps',
          type=int,
          default=max_steps,
          help='Number of steps to run trainer.'
        )
        parser.add_argument(
          '--hidden1',
          type=int,
          default=hidden1,
          help='Number of units in hidden layer 1.'
        )
        parser.add_argument(
          '--hidden2',
          type=int,
          default=hidden2,
          help='Number of units in hidden layer 2.'
        )
        parser.add_argument(
          '--batch_size',
          type=int,
          default=batch_size,
          help='Batch size.  Must divide evenly into the dataset sizes.'
        )
        parser.add_argument(
          '--input_data_dir',
          type=str,
          default=input_data_dir,
          help='Directory to put the input data.'
        )
        parser.add_argument(
          '--log_dir',
          type=str,
          default=log_dir,
          help='Directory to put the log data.'
        )
        parser.add_argument(
          '--data_rows',
          type=int,
          default=data_rows,
          help='Row size.',
        )
        parser.add_argument(
          '--data_columns',
          type=int,
          default=data_columns,
          help='Column Size.'
        )
        print( "\n--------------------------------------" )
        FLAGS, unparsed = parser.parse_known_args()
        print( FLAGS )
        print()
        tf.app.run(main=func, argv=[sys.argv[0]] + unparsed)

def placeholder_inputs(batch_size):
  data_placeholder = tf.placeholder(tf.float32, shape=(batch_size, network.DATA_SIZE))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return data_placeholder, labels_placeholder


def fill_feed_dict(data_set, data_pl, labels_pl):
  data_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
  feed_dict = {
      data_pl: data_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess, eval_correct, data_placeholder, labels_placeholder, data_set):
  true_count = 0
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set, data_placeholder, labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
  data_rows = FLAGS.data_rows
  data_columns = FLAGS.data_columns
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the data and labels.
    data_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = network.inference(data_placeholder, data_rows, data_columns,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = network.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = network.training(loss, FLAGS.learning_rate)

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
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in range(FLAGS.max_steps):
      start_time = time.time()

      # Fill dictionary
      feed_dict = fill_feed_dict(data_sets.train,
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
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                data_placeholder,
                labels_placeholder,
                data_sets.train)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                data_placeholder,
                labels_placeholder,
                data_sets.validation)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                data_placeholder,
                labels_placeholder,
                data_sets.test)


def func(_):
    print( "\nMAIN" )
    print( FLAGS )
    print()
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()
