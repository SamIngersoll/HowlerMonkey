import tensorflow as tf
import datetime
import os

class Tensorboard:
    def __init__( self, max_queue=10, flush_secs=12):
        self.log_dir = os.getcwd()+"/logs"
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir,
                                             max_queue=max_queue,
                                             flush_secs=flush_secs,
                                             graph_def=None)

# To run tensorboard: tensorboard --logdir=path.

    def log( self, name, value, datetime ):
        epoch = datetime.toordinal()
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        self.writer.add_summary( summary, global_step=epoch )
        self.writer.flush()
