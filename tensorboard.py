import tensorflow as tf

class Tensorboard:
    def __init__(self, log_dir='./logs', max_queue=10, flush_secs=120):
        self.log_dir = log_dir
        self.merged = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter(self.log_dir,
                                             max_queue=max_queue,
                                             flush_secs=flush_secs,
                                             graph_def=None)
