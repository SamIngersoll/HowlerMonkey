import tensorflow as tf
import datetime
import os
import time

class FileWriter:
    def __init__( self, log_dir, max_queue=10, flush_secs=12):
        # self.log_dir = os.getcwd()+"/logs/"+time.asctime()
        self.log_dir = log_dir
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
        # self.writer.flush()
