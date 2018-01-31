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
import util as u

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-image-dir', type=str, default="images/sample_originals/train")
parser.add_argument('--test-image-dir', type=str, default="images/sample_originals/test")
parser.add_argument('--patch-fraction', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--run', type=str, required=True, help="run dir for tb")
parser.add_argument('--no-use-skip-connections', action='store_true')
parser.add_argument('--base-filter-size', type=int, default=16)
opts = parser.parse_args()
print("opts %s" % opts, file=sys.stderr)
  
np.set_printoptions(precision=2, threshold=10000, suppress=True, linewidth=10000)

# Build readers for train and test data.
train_imgs, train_xys_bitmaps = data.img_xys_iterator(base_dir=opts.train_image_dir,
                                                      batch_size=opts.batch_size,
                                                      patch_fraction=opts.patch_fraction,
                                                      distort=True, repeat=True)
test_imgs, test_xys_bitmaps = data.img_xys_iterator(base_dir=opts.test_image_dir,
                                                    batch_size=1,
                                                    patch_fraction=1,
                                                    distort=False, repeat=True)
print(test_imgs.get_shape())
print(test_xys_bitmaps.get_shape())
#train_imgs = tf.reshape(train_imgs, (opts.batch_size, 1024/opts.patch_fraction, 768/opts.patch_fraction, 3))  
#test_imgs = tf.reshape(test_imgs, (1, 1024, 768, 3))

# Build training and test model with same params.
# TODO: opts for skip and base filters
with tf.variable_scope("train_test_model") as scope:
  print("patch train model...")
  train_model = model.Model(train_imgs,
                            use_skip_connections=not opts.no_use_skip_connections,
                            base_filter_size=opts.base_filter_size)
  scope.reuse_variables()
  print("full res test model...")  
  test_model = model.Model(test_imgs,
                           use_skip_connections=not opts.no_use_skip_connections,
                           base_filter_size=opts.base_filter_size)

global_step = tf.train.get_or_create_global_step()

# Setup loss and optimiser.
labels = tf.to_float(train_xys_bitmaps)

xent_loss = tf.reduce_mean(
  tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                          logits=train_model.logits))

dice_loss = tf.reduce_mean(u.dice_loss(labels,
                                       tf.sigmoid(train_model.logits),
                                       batch_size=opts.batch_size, # clumsy :/
                                       smoothing=1e-2))

optimiser = tf.train.AdamOptimizer(learning_rate=opts.learning_rate)

xent_weight = 1.000
dice_weight = 0.000  # TODO: hmm. no success with this yet...  (1e-4 would put it ~= xent)
total_loss = xent_weight * xent_loss + dice_weight * dice_loss

# TODO: include this
# regularisation_loss = tf.add_n(tf.losses.get_regularization_losses())  

train_op = slim.learning.create_train_op(total_loss = total_loss, # + regularisation_loss,
                                         optimizer = optimiser,
                                         summarize_gradients = False)

# Create session.
sess_config = tf.ConfigProto()
# sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())

# Setup summaries.
summaries = [
  tf.summary.scalar('xent_loss', xent_loss),
  tf.summary.scalar('dice_loss', dice_loss),
]
summaries_op = tf.summary.merge(summaries)
train_summaries_writer = tf.summary.FileWriter("tb/%s/training" % opts.run, sess.graph)
test_summaries_writer = tf.summary.FileWriter("tb/%s/test" % opts.run, sess.graph)

import time

for idx in range(10000):
  
  # train a bit.
  start_time = time.time()
  for _ in range(100):
    _, xl, dl = sess.run([train_op, xent_loss, dice_loss],
                         feed_dict={train_model.is_training: True})
  training_time = time.time() - start_time
  print("idx %d\txent_loss %f\tdice_loss %f\ttime %f" % (idx, xl, dl, training_time))
    
  # train / test summaries
  # includes loss summaries as well as a hand rolled debug image
  # ...train
  i, bm, o, loss_summaries, step = sess.run([train_imgs, train_xys_bitmaps, train_model.output,
                                             summaries_op, global_step],
                                            feed_dict={train_model.is_training: False})
  debug_img_summary = u.PILImageToTFSummary(u.debug_img(i, bm, o))
  train_summaries_writer.add_summary(loss_summaries, step)
  train_summaries_writer.add_summary(debug_img_summary, step)  

  # ... test
  i, bm, logits, o, loss_summaries, step = sess.run([test_imgs, test_xys_bitmaps,
                                                     test_model.logits, test_model.output,
                                                     summaries_op, global_step],
                                                    feed_dict={test_model.is_training: False})
  print("logits", logits.shape, np.min(logits), np.max(logits))
  debug_img_summary = u.PILImageToTFSummary(u.debug_img(i, bm, o))
  test_summaries_writer.add_summary(loss_summaries, step)
  test_summaries_writer.add_summary(debug_img_summary, step)

  # save checkpoint
  train_model.save(sess, "ckpts/%s" % opts.run)
