#!/usr/bin/env python3

import argparse
from sklearn.metrics import confusion_matrix
import data
import model
import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import util as u
import time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-image-dir', type=str, default="sample_data/training/", help="training images")
parser.add_argument('--test-image-dir', type=str, default="sample_data/test/", help="test images")
parser.add_argument('--label-dir', type=str, default="sample_data/labels/", help="labels for train/test")
parser.add_argument('--patch-width-height', type=int, default=None,
                    help="what size square patches to sample. None => no patch, i.e. use full res image"
                         " (in which case --width & --height are required)")
parser.add_argument('--batch-size', type=int, default=32, help=' ')
parser.add_argument('--learning-rate', type=float, default=0.001, help=' ')
parser.add_argument('--run', type=str, required=True, help="run dir for tb & ckpts")
parser.add_argument('--no-use-skip-connections', action='store_true', help='set to disable skip connections')
parser.add_argument('--no-use-batch-norm', action='store_true', help='set to disable batch norm')
parser.add_argument('--base-filter-size', type=int, default=8, help=' ')
parser.add_argument('--flip-left-right', action='store_true', help='randomly flip training egs left/right')
parser.add_argument('--random-rotate', action='store_true', help='randomly rotate training images')
parser.add_argument('--steps', type=int, default=100000, help='max number of steps (test, summaries every --train-steps)')
parser.add_argument('--train-steps', type=int, default=100, help='number training steps between test and summaries')
parser.add_argument('--secs', type=int, default=None, help='If set, max number of seconds to run.')
opts = parser.parse_args()
print("opts %s" % opts, file=sys.stderr)

np.set_printoptions(precision=2, threshold=10000, suppress=True, linewidth=10000)

# Build readers for train and test data.
train_imgs, train_xys_bitmaps = data.img_xys_iterator(image_dir=opts.train_image_dir,
                                                      label_dir=opts.label_dir,
                                                      batch_size=opts.batch_size,
                                                      patch_width_height=opts.patch_width_height,
                                                      distort_rgb=True,
                                                      flip_left_right=opts.flip_left_right,
                                                      random_rotation=opts.random_rotate,
                                                      repeat=True)
test_imgs, test_xys_bitmaps = data.img_xys_iterator(image_dir=opts.test_image_dir,
                                                    label_dir=opts.label_dir,
                                                    batch_size=opts.batch_size,
                                                    patch_width_height=None,  # i.e. no patchs
                                                    distort_rgb=False,
                                                    flip_left_right=False,
                                                    random_rotation=False,
                                                    repeat=True)
print("Train image `shape:", train_imgs.get_shape())
print("Train bmp shape:", train_xys_bitmaps.get_shape())
print("Test image `shape:", test_imgs.get_shape())
print("test bmp shape:", test_xys_bitmaps.get_shape())

# Build training and test model with same params.
# TODO: opts for skip and base filters
print("patch train model...")
train_model = model.Model(train_imgs,
                          is_training=True,
                          use_skip_connections=not opts.no_use_skip_connections,
                          base_filter_size=opts.base_filter_size,
                          use_batch_norm=not opts.no_use_batch_norm)
train_model.calculate_losses_wrt(labels=train_xys_bitmaps)

print("full res test model...")
test_model = model.Model(test_imgs,
                         is_training=False,
                         use_skip_connections=not opts.no_use_skip_connections,
                         base_filter_size=opts.base_filter_size,
                         use_batch_norm=not opts.no_use_batch_norm)
test_model.calculate_losses_wrt(labels=test_xys_bitmaps)

global_step = tf.train.get_or_create_global_step()

optimiser = tf.train.AdamOptimizer(learning_rate=opts.learning_rate)

# TODO: reinclude reg loss
# regularisation_loss = tf.add_n(tf.losses.get_regularization_losses())
train_op = slim.learning.create_train_op(total_loss = train_model.xent_loss, # + regularisation_loss,
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

start_time = time.time()
for idx in range(opts.steps // opts.train_steps):
  # train a bit.
  for _ in range(opts.train_steps):
    _, xl = sess.run([train_op, train_model.xent_loss])
  print("idx %d/%d\ttime %d\txent_loss %f" % (idx, opts.steps // opts.train_steps,
                                              int(time.time()-start_time),
                                              xl))

  # train / test summaries
  # includes loss summaries as well as a hand rolled debug image
  # ...train
  i, bm, logits, o, xl, step = sess.run([train_imgs, train_xys_bitmaps,
                                         train_model.logits, train_model.output,
                                         train_model.xent_loss,
                                         global_step])
  train_summaries_writer.add_summary(u.explicit_summaries({"xent": xl}), step)
  debug_img_summary = u.pil_image_to_tf_summary(u.debug_img(i, bm, o))
  train_summaries_writer.add_summary(debug_img_summary, step)
  train_summaries_writer.flush()

  # ... test
  i, bm, o, xl, step = sess.run([test_imgs, test_xys_bitmaps, test_model.output,
                                 test_model.xent_loss,
                                 global_step])

  set_comparison = u.SetComparison()
  for idx in range(bm.shape[0]):
    true_centroids = u.centroids_of_connected_components(bm[idx])
    predicted_centroids = u.centroids_of_connected_components(o[idx])
    set_comparison.compare_sets(true_centroids, predicted_centroids)
  precision, recall, f1 = set_comparison.precision_recall_f1()
  tag_values = {"xent": xl, "precision": precision, "recall": recall, "f1": f1}
  test_summaries_writer.add_summary(u.explicit_summaries(tag_values), step)

  debug_img_summary = u.pil_image_to_tf_summary(u.debug_img(i, bm, o))
  test_summaries_writer.add_summary(debug_img_summary, step)
  test_summaries_writer.flush()

  # save checkpoint
  train_model.save(sess, "ckpts/%s" % opts.run)

  # check max time to run (if set)
  if opts.secs is not None:
    run_time = time.time() - start_time
    remaining_time = opts.secs - run_time
    print("run_time %s remaining_time %s" % (u.hms(run_time), u.hms(remaining_time)))
    if remaining_time < 0: exit()
