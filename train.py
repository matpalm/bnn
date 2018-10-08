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
import test
import time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-image-dir', type=str, default="sample_data/training/", help="training images")
parser.add_argument('--test-image-dir', type=str, default="sample_data/test/", help="test images")
parser.add_argument('--label-dir', type=str, default="sample_data/labels/", help="labels for train/test")
parser.add_argument('--patch-width-height', type=int, default=None,
                    help="what size square patches to sample. None => no patch, i.e. use full res image")
parser.add_argument('--batch-size', type=int, default=32, help=' ')
parser.add_argument('--learning-rate', type=float, default=0.001, help=' ')
parser.add_argument('--pos-weight', type=float, default=1.0, help='positive class weight in loss. 1.0 = balanced')
parser.add_argument('--run', type=str, required=True, help="run dir for tb & ckpts")
parser.add_argument('--no-use-skip-connections', action='store_true', help='set to disable skip connections')
parser.add_argument('--no-use-batch-norm', action='store_true', help='set to disable batch norm')
parser.add_argument('--base-filter-size', type=int, default=8, help=' ')
parser.add_argument('--flip-left-right', action='store_true', help='randomly flip training egs left/right')
parser.add_argument('--random-rotate', action='store_true', help='randomly rotate training images')
parser.add_argument('--steps', type=int, default=100000, help='max number of training steps (summaries every --train-steps)')
parser.add_argument('--train-steps', type=int, default=100, help='number training steps between test and summaries')
parser.add_argument('--secs', type=int, default=None, help='If set, max number of seconds to run')
parser.add_argument('--width', type=int, default=None, help='test input image width')
parser.add_argument('--height', type=int, default=None, help='test input image height')
opts = parser.parse_args()
print("opts %s" % opts, file=sys.stderr)

np.set_printoptions(precision=2, threshold=10000, suppress=True, linewidth=10000)

# Build readers / model for training
train_imgs, train_xys_bitmaps = data.img_xys_iterator(image_dir=opts.train_image_dir,
                                                      label_dir=opts.label_dir,
                                                      batch_size=opts.batch_size,
                                                      patch_width_height=opts.patch_width_height,
                                                      distort_rgb=True,
                                                      flip_left_right=opts.flip_left_right,
                                                      random_rotation=opts.random_rotate,
                                                      repeat=True,
                                                      width=opts.width, height=opts.height)

print("patch train model...")
train_model = model.Model(train_imgs,
                          is_training=True,
                          use_skip_connections=not opts.no_use_skip_connections,
                          base_filter_size=opts.base_filter_size,
                          use_batch_norm=not opts.no_use_batch_norm)
train_model.calculate_losses_wrt(labels=train_xys_bitmaps,
                                 pos_weight=opts.pos_weight)

print("full res test model...")
tester = test.ModelTester(opts.test_image_dir, opts.label_dir,
                          opts.batch_size, opts.width, opts.height,
                          opts.no_use_skip_connections, opts.base_filter_size,
                          opts.no_use_batch_norm)

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
done = False
while not done:

  # train a bit.
  for _ in range(opts.train_steps):
    sess.run(train_op)

  # fetch global_step & xent_loss
  step, xl = sess.run([global_step, train_model.xent_loss])

  # report one liner
  print("step %d/%d\ttime %d\txent_loss %f" % (step, opts.steps,
                                               int(time.time()-start_time),
                                               xl))

  # train / test summaries
  # includes loss summaries as well as a hand rolled debug image

  # ...train
  i, bm, logits, o, xl = sess.run([train_imgs, train_xys_bitmaps,
                                   train_model.logits, train_model.output,
                                   train_model.xent_loss])
  train_summaries_writer.add_summary(u.explicit_summaries({"xent": xl}), step)
  debug_img_summary = u.pil_image_to_tf_summary(u.debug_img(i[0], bm[0], o[0]))
  train_summaries_writer.add_summary(debug_img_summary, step)
  train_summaries_writer.flush()

  # save checkpoint (to be reloaded by test)
  # TODO: this is clumsy; need to refactor test to use current session instead
  #       of loading entirely new one... will do for now.
  train_model.save(sess, "ckpts/%s" % opts.run)

  # ... test
  stats = tester.test(opts.run)
  tag_values = {k: stats[k] for k in ['precision', 'recall', 'f1']}
  test_summaries_writer.add_summary(u.explicit_summaries(tag_values), step)
  debug_img_summary = u.pil_image_to_tf_summary(stats['debug_img'])
  test_summaries_writer.add_summary(debug_img_summary, step)
  test_summaries_writer.flush()

  # check if done by steps or time
  if step >= opts.steps:
    done = True
  if opts.secs is not None:
    run_time = time.time() - start_time
    remaining_time = opts.secs - run_time
    print("run_time %s remaining_time %s" % (u.hms(run_time), u.hms(remaining_time)))
    if remaining_time < 0:
      done = True
