#!/usr/bin/env python3

# write pbtxt for graph (with placeholder input) in prep for freeze_graph.sh

import argparse
import model
import tensorflow as tf

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run', type=str, required=True, help='model')
parser.add_argument('--no-use-skip-connections', action='store_true')
parser.add_argument('--base-filter-size', type=int, default=16)
parser.add_argument('--no-use-batch-norm', action='store_true')
opts = parser.parse_args()

# feed data through an explicit placeholder for frozen version
imgs = tf.placeholder(dtype=tf.uint8, shape=(1, 1024, 768, 3), name='input_imgs')

# restore model
with tf.variable_scope("train_test_model") as scope:  # clumsy :/
  model = model.Model(imgs,
                      is_training=False,
                      use_skip_connections=not opts.no_use_skip_connections,
                      base_filter_size=opts.base_filter_size,
                      use_batch_norm=not opts.no_use_batch_norm)
sess = tf.Session()
model.restore(sess, "ckpts/%s" % opts.run)

# save graph def
tf.train.write_graph(sess.graph_def, ".", "bnn_graph.predict.pbtxt")
