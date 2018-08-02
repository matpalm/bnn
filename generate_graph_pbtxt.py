#!/usr/bin/env python3

# write pbtxt for graph (with placeholder input) in prep for freeze_graph.sh

import argparse
import model
import tensorflow as tf

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--no-use-skip-connections', action='store_true')
parser.add_argument('--base-filter-size', type=int, default=8)
parser.add_argument('--no-use-batch-norm', action='store_true')
parser.add_argument('--width', type=int, default=768, help='input image width')
parser.add_argument('--height', type=int, default=1024, help='input image height')
opts = parser.parse_args()

# feed data through an explicit placeholder for frozen version
imgs = tf.placeholder(dtype=tf.float32, shape=(1, opts.height, opts.width, 3), name='input_imgs')

# restore model
model = model.Model(imgs,
                    is_training=False,
                    use_skip_connections=not opts.no_use_skip_connections,
                    base_filter_size=opts.base_filter_size,
                    use_batch_norm=not opts.no_use_batch_norm)
sess = tf.Session()

# save graph def
tf.train.write_graph(sess.graph_def, ".", "bnn_graph.predict.pbtxt")
