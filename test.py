#!/usr/bin/env python3

# given a directory of images and labels output overall P/R/F1 for entire set

import data
import model
import numpy as np
import random
import tensorflow as tf
import util as u

class ModelTester(object):

  def __init__(self, image_dir, label_dir, batch_size, width, height,
               no_use_skip_connections, base_filter_size, no_use_batch_norm):

    # test data reader
    iter_items = data.img_xys_iterator(image_dir=image_dir,
                                       label_dir=label_dir,
                                       batch_size=batch_size,
                                       patch_width_height=None,  # i.e. no patchs
                                       distort_rgb=False,
                                       flip_left_right=False,
                                       random_rotation=False,
                                       repeat=False,
                                       width=width,
                                       height=height,
                                       one_shot=False)
    self.iter_init_op, (self.test_imgs, self.test_xys_bitmaps) = iter_items

    # build the model
    self.model = model.Model(self.test_imgs,
                             is_training=False,
                             use_skip_connections=not no_use_skip_connections,
                             base_filter_size=base_filter_size,
                             use_batch_norm=not no_use_batch_norm)

    # define loss ops for calculating xent
    self.model.calculate_losses_wrt(labels=self.test_xys_bitmaps)

  def test(self, run):
    with tf.Session() as sess:
      self.model.restore(sess, "ckpts/%s" % run)
      sess.run(self.iter_init_op)

      set_comparison = u.SetComparison()
      num_imgs = 0
      xent_losses = []
      debug_img = None  # created on first call
      while True:
        try:
          if debug_img is None:
            # fetch imgs as well to create debug_img
            imgs, true_bitmaps, predicted_bitmaps, xent_loss = sess.run([self.test_imgs,
                                                                         self.test_xys_bitmaps,
                                                                         self.model.output,
                                                                         self.model.xent_loss])
            # choose a random element from batch
            idx = random.randint(0, true_bitmaps.shape[0]-1)
            debug_img = u.debug_img(imgs[idx], true_bitmaps[idx], predicted_bitmaps[idx])
          else:
            true_bitmaps, predicted_bitmaps, xent_loss = sess.run([self.test_xys_bitmaps,
                                                                   self.model.output,
                                                                   self.model.xent_loss])

          xent_losses.append(xent_loss)
          iterator_batch_size = true_bitmaps.shape[0]
          num_imgs += iterator_batch_size

          for idx in range(iterator_batch_size):
            # this is dumb; should do against label db!
            true_centroids = u.centroids_of_connected_components(true_bitmaps[idx])
            predicted_centroids = u.centroids_of_connected_components(predicted_bitmaps[idx])
            tp, fn, fp = set_comparison.compare_sets(true_centroids, predicted_centroids)
        except tf.errors.OutOfRangeError:
          # end of iterator
          break

    precision, recall, f1 = set_comparison.precision_recall_f1()

    return {"num_imgs": num_imgs,
            "debug_img": debug_img,  # for tensorboard
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "xent": np.mean(xent_losses)}


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--image-dir', type=str, required=True)
  parser.add_argument('--label-dir', type=str, required=True)
  parser.add_argument('--run', type=str, required=True, help='model')
  parser.add_argument('--batch-size', type=int, default=1)
  parser.add_argument('--no-use-skip-connections', action='store_true')
  parser.add_argument('--no-use-batch-norm', action='store_true')
  parser.add_argument('--base-filter-size', type=int, default=8)
  parser.add_argument('--width', type=int, default=768, help='input image width')
  parser.add_argument('--height', type=int, default=1024, help='input image height')
  opts = parser.parse_args()
  print(opts)

  tester = ModelTester(image_dir=opts.image_dir,
                       label_dir=opts.label_dir,
                       batch_size=opts.batch_size,
                       width=opts.width,
                       height=opts.height,
                       no_use_skip_connections=opts.no_use_skip_connections,
                       base_filter_size=opts.base_filter_size,
                       no_use_batch_norm=opts.no_use_batch_norm)

  stats = tester.test(opts.run)

  print("final stats over %d images: precision %0.2f recall %0.2f f1 %0.2f" \
        % (stats['num_imgs'], stats['precision'], stats['recall'], stats['f1']))
  print("mean batch xent_loss %f" % stats['xent'])
