#!/usr/bin/env python3

from sklearn.metrics import confusion_matrix
import argparse
import data
import datetime
import json
import model
import numpy as np
import os
import sys
import tensorflow as tf
import test
import time
import util as u

np.set_printoptions(precision=2, threshold=10000, suppress=True, linewidth=10000)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-image-dir', type=str, default="sample_data/training/", help="training images")
parser.add_argument('--test-image-dir', type=str, default="sample_data/test/", help="test images")
parser.add_argument('--label-dir', type=str, default="sample_data/labels/", help="labels for train/test")
parser.add_argument('--label-db', type=str, default="label.201802_sample.db",
                    help="label_db for test P/R/F1 stats")
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
parser.add_argument('--width', type=int, default=768,
                    help='test image width (assumed training image width if --patch-width-height not set)')
parser.add_argument('--height', type=int, default=1024,
                    help='test image height (assumed training height if --patch-width-height not set)')
parser.add_argument('--connected-components-threshold', type=float, default=0.05)
opts = parser.parse_args()
print("opts %s" % opts, file=sys.stderr)

# prep ckpt dir (and save training_opts for restoring model later)
ckpt_dir = "ckpts/%s" % opts.run
if not os.path.exists(ckpt_dir):
  os.makedirs(ckpt_dir)
with open("%s/opts.json" % ckpt_dir, "w") as f:
  f.write(json.dumps(vars(opts)))

#from tensorflow.python import debug as tf_debug
#tf.keras.backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))

# Build readers / model for training
# training can be either patch based, or full resolution
train_imgs_xys_bitmaps = data.img_xys_iterator(image_dir=opts.train_image_dir,
                                               label_dir=opts.label_dir,
                                               batch_size=opts.batch_size,
                                               patch_width_height=opts.patch_width_height,
                                               distort_rgb=True,
                                               flip_left_right=opts.flip_left_right,
                                               random_rotation=opts.random_rotate,
                                               repeat=True,
                                               width=None if opts.patch_width_height else opts.width,
                                               height=None if opts.patch_width_height else opts.height)

# TODO: could we do all these calcs in test.pr_stats (rather than iterating twice) ??
# test images are always full res
test_imgs_xys_bitmaps = data.img_xys_iterator(image_dir=opts.test_image_dir,
                                              label_dir=opts.label_dir,
                                              batch_size=opts.batch_size,
                                              patch_width_height=None,
                                              distort_rgb=False,
                                              flip_left_right=False,
                                              random_rotation=False,
                                              repeat=False,
                                              width=opts.width, height=opts.height)

num_test_files = len(os.listdir(opts.test_image_dir))
num_test_steps = num_test_files // opts.batch_size
print("num_test_files=", num_test_files, "batch_size=", opts.batch_size, "=> num_test_steps=", num_test_steps)

# training model.
train_model = model.construct_model(width=opts.patch_width_height or opts.width,
                                    height=opts.patch_width_height or opts.height,
                                    use_skip_connections=not opts.no_use_skip_connections,
                                    base_filter_size=opts.base_filter_size,
                                    use_batch_norm=not opts.no_use_batch_norm)
model.compile_model(train_model,
                    learning_rate=opts.learning_rate,
                    pos_weight=opts.pos_weight)
print("TRAIN MODEL")
print(train_model.summary())

# test model.
test_model =  model.construct_model(width=opts.width,
                                    height=opts.height,
                                    use_skip_connections=not opts.no_use_skip_connections,
                                    base_filter_size=opts.base_filter_size,
                                    use_batch_norm=not opts.no_use_batch_norm)
model.compile_model(test_model,
                    learning_rate=opts.learning_rate,
                    pos_weight=opts.pos_weight)
print("TEST MODEL")
print(test_model.summary())

# Setup summary writers. (Will create explicit summaries to write)
# TODO: include keras default callback
train_summaries_writer = tf.summary.FileWriter("tb/%s/training" % opts.run, None)
test_summaries_writer = tf.summary.FileWriter("tb/%s/test" % opts.run, None)

start_time = time.time()
done = False
step = 0

while not done:

  # train a bit.
  history = train_model.fit(train_imgs_xys_bitmaps,
                            epochs=1, verbose=1,
                            steps_per_epoch=opts.train_steps)
  train_loss = history.history['loss'][0]

  # do eval using test model
  # TODO: switch to sharing layers between these two over this explicit get/set_weights
  test_model.set_weights(train_model.get_weights())
  test_loss = test_model.evaluate(test_imgs_xys_bitmaps,
                                  verbose=1,
                                  steps=num_test_steps)

  # train / test summaries
  # includes loss summaries as well as a hand rolled debug image

  # ...train
  train_summaries_writer.add_summary(u.explicit_summaries({"xent": train_loss}), step)
#  debug_img_summary = u.pil_image_to_tf_summary(u.debug_img(i[0], bm[0], o[0]))
#  train_summaries_writer.add_summary(debug_img_summary, step)
  train_summaries_writer.flush()

  # save model
  dts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  save_filename = "%s/%s" % (ckpt_dir, dts)
  train_model.save_weights(save_filename)

  # ... test
  stats = test.pr_stats(opts.run, opts.test_image_dir, opts.label_db, opts.connected_components_threshold)
  tag_values = {k: stats[k] for k in ['precision', 'recall', 'f1']}
  test_summaries_writer.add_summary(u.explicit_summaries({"xent": test_loss}), step)
  test_summaries_writer.add_summary(u.explicit_summaries(tag_values), step)
  for idx, img in enumerate(stats['debug_imgs']):
    debug_img_summary = u.pil_image_to_tf_summary(img, tag="debug_img_%d" % idx)
    test_summaries_writer.add_summary(debug_img_summary, step)
  test_summaries_writer.flush()

  # report one liner
  log = []
  log.append("step %d/%d" % (step, opts.steps))
  log.append("time %d" % int(time.time()-start_time))
  log.append("train_loss %f" % train_loss)
  log.append("test_loss %s" % test_loss)
  log.append("test stats { p:%0.2f, r:%0.2f, f1:%0.2f }" % tuple([stats[k] for k in ['precision', 'recall', 'f1']]))
  print("\t".join(log))

  # check if done by steps or time
  step += 1  # TODO: fetch global_step from keras model (?)
  if step >= opts.steps:
    done = True
  if opts.secs is not None:
    run_time = time.time() - start_time
    remaining_time = opts.secs - run_time
    print("run_time %s remaining_time %s" % (u.hms(run_time), u.hms(remaining_time)))
    if remaining_time < 0:
      done = True
