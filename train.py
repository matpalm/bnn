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
parser.add_argument('--flip-left-right', action='store_true')
parser.add_argument('--steps', type=int, default=1000, help='number of times to do x100 steps, then summaries')
opts = parser.parse_args()
print("opts %s" % opts, file=sys.stderr)
  
np.set_printoptions(precision=2, threshold=10000, suppress=True, linewidth=10000)

# Build readers for train and test data.
train_imgs, train_xys_bitmaps = data.img_xys_iterator(base_dir=opts.train_image_dir,
                                                      batch_size=opts.batch_size,
                                                      patch_fraction=opts.patch_fraction,
                                                      distort_rgb=True,
                                                      flip_left_right=opts.flip_left_right,
                                                      repeat=True)
test_imgs, test_xys_bitmaps = data.img_xys_iterator(base_dir=opts.test_image_dir,
                                                    batch_size=1,
                                                    patch_fraction=1,
                                                    distort_rgb=False,
                                                    flip_left_right=False,
                                                    repeat=True)
print(test_imgs.get_shape())
print(test_xys_bitmaps.get_shape())

# Build training and test model with same params.
# TODO: opts for skip and base filters
with tf.variable_scope("train_test_model") as scope:
  print("patch train model...")
  train_model = model.Model(train_imgs,
                            use_skip_connections=not opts.no_use_skip_connections,
                            base_filter_size=opts.base_filter_size)
  train_model.calculate_losses_wrt(labels=train_xys_bitmaps,
                                   batch_size=opts.batch_size)

  scope.reuse_variables()
  print("full res test model...")  
  test_model = model.Model(test_imgs,
                           use_skip_connections=not opts.no_use_skip_connections,
                           base_filter_size=opts.base_filter_size)
  test_model.calculate_losses_wrt(labels=test_xys_bitmaps,
                                  batch_size=opts.batch_size)

global_step = tf.train.get_or_create_global_step()

optimiser = tf.train.AdamOptimizer(learning_rate=opts.learning_rate)

# TODO: reinclude reg loss
# regularisation_loss = tf.add_n(tf.losses.get_regularization_losses())  
train_op = slim.learning.create_train_op(total_loss = train_model.loss, # + regularisation_loss,
                                         optimizer = optimiser,
                                         summarize_gradients = False)

# Create session.
sess_config = tf.ConfigProto()
# sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())

# Setup summary writers. (Will create explicit summaries to write)
train_summaries_writer = tf.summary.FileWriter("tb/%s/training" % opts.run, sess.graph)
test_summaries_writer = tf.summary.FileWriter("tb/%s/test" % opts.run, sess.graph)

import time

for idx in range(opts.steps):
  
  # train a bit.
  start_time = time.time()
  for _ in range(100):
    _, xl, dl = sess.run([train_op, train_model.xent_loss, train_model.dice_loss],
                         feed_dict={train_model.is_training: True})
  training_time = time.time() - start_time
  print("idx %d\txent_loss %f\tdice_loss %f\ttime %f" % (idx, xl, dl, training_time))
    
  # train / test summaries
  # includes loss summaries as well as a hand rolled debug image
  # ...train
  i, bm, logits, o, xl, dl, step = sess.run([train_imgs, train_xys_bitmaps,
                                             train_model.logits, train_model.output,
                                             train_model.xent_loss, train_model.dice_loss,
                                             global_step],
                                            feed_dict={train_model.is_training: False})
  print("train logits", logits.shape, np.min(logits), np.max(logits))
  train_summaries_writer.add_summary(u.explicit_loss_summary(xl, dl), step)
  debug_img_summary = u.pil_image_to_tf_summary(u.debug_img(i, bm, o))
  train_summaries_writer.add_summary(debug_img_summary, step)  
  train_summaries_writer.flush()

  # ... test
  i, bm, logits, o, xl, dl, step = sess.run([test_imgs, test_xys_bitmaps,
                                             test_model.logits, test_model.output,
                                             test_model.xent_loss, test_model.dice_loss,
                                             global_step],
                                            feed_dict={test_model.is_training: False})
  print("test logits", logits.shape, np.min(logits), np.max(logits))
  test_summaries_writer.add_summary(u.explicit_loss_summary(xl, dl), step)
  debug_img_summary = u.pil_image_to_tf_summary(u.debug_img(i, bm, o))
  test_summaries_writer.add_summary(debug_img_summary, step)
  test_summaries_writer.flush()

  # save checkpoint
  train_model.save(sess, "ckpts/%s" % opts.run)
