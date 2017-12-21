#!/usr/bin/env python
from __future__ import print_function

import argparse
from sklearn.metrics import confusion_matrix
import data
import model
import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--base-dir', type=str, default="images/test2",
                    help="sdcsdc")
#parser.add_argument('--train-data', type=str, default="train.jsons",
#                    help="json file for training data")
# parser.add_argument('--test-data', type=str, default="test.jsons",
#                     help="json file for training data")
#parser.add_argument('--run', type=str, required=True)
parser.add_argument('--img-shape', type=str, default='384')
parser.add_argument('--steps', type=int, default=10,
                    help='number of x100 training + test steps')
parser.add_argument('--num-filters', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--learning-rate', type=float, default=0.005)
opts = parser.parse_args()
print("opts %s" % opts, file=sys.stderr)
  
np.set_printoptions(precision=2, threshold=10000, suppress=True, linewidth=10000)

imgs, xys_bitmaps = data.img_xys_iterator(base_dir=opts.base_dir,
                                          batch_size=1,
                                          img_shape=opts.img_shape)

model = model.Model(imgs)

global_step = tf.train.create_global_step()

print("xys_bitmaps", xys_bitmaps.shape)
print("logits", model.logits.shape)

loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=xys_bitmaps,
                                            logits=model.logits))
#loss = tf.reduce_mean(tf.nn.l2_loss(xys_bitmaps - model.output))

optimiser = tf.train.AdamOptimizer(learning_rate=opts.learning_rate)
train_op = optimiser.minimize(loss)

#train_op = slim.learning.create_train_op(total_loss = loss, # + regularisation_loss,
#                                         optimizer = optimiser,
#                                         summarize_gradients = False)

sess_config = tf.ConfigProto()
# sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=sess_config)

sess.run(tf.global_variables_initializer())

for idx in range(200):
  print("---------------------------------", idx)
  
  i, bm, l, o = sess.run([imgs, xys_bitmaps, model.logits, model.output],
                  feed_dict={model.is_training: False})

#  print("i", i.shape, np.min(i), np.mean(i), np.max(i))

#  print("bm", bm.shape, np.min(bm), np.mean(bm), np.max(bm))
#  print(bm.squeeze())

  print("l", l.shape, np.min(l), np.mean(l), np.max(l))
  print(l.squeeze())
  
  print("o", o.shape, np.min(o), np.mean(o), np.max(o))
  print(o.squeeze())
  data.bitmap_to_pil_image(o[0]).save("o_%03d.png" % idx)

  for _ in range(10):
    _, l = sess.run([train_op, loss],
                    feed_dict={model.is_training: True})
    print(l)






