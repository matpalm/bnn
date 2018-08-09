#!/usr/bin/env python3

# given a directory of images and labels output overall P/R/F1 for entire set

import argparse
import data
import model
import tensorflow as tf
import util as u

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

# test data reader
test_imgs, test_xys_bitmaps = data.img_xys_iterator(image_dir=opts.image_dir,
                                                    label_dir=opts.label_dir,
                                                    batch_size=opts.batch_size,
                                                    patch_width_height=None,  # i.e. no patchs
                                                    distort_rgb=False,
                                                    flip_left_right=False,
                                                    random_rotation=False,
                                                    repeat=False,
                                                    width=opts.width,
                                                    height=opts.height)
model = model.Model(test_imgs,
                    is_training=False,
                    use_skip_connections=not opts.no_use_skip_connections,
                    base_filter_size=opts.base_filter_size,
                    use_batch_norm=not opts.no_use_batch_norm)

sess = tf.Session()
model.restore(sess, "ckpts/%s" % opts.run)

set_comparison = u.SetComparison()
num_imgs = 0
while True:
  try:
    true_bitmaps, predicted_bitmaps = sess.run([test_xys_bitmaps, model.output])
    iterator_batch_size = true_bitmaps.shape[0]  # note: final one may be < opts.batch_size
    num_imgs += iterator_batch_size
    for idx in range(iterator_batch_size):
      true_centroids = u.centroids_of_connected_components(true_bitmaps[idx])  # this is dumb; should do against label db!
      predicted_centroids = u.centroids_of_connected_components(predicted_bitmaps[idx])
      tp, fn, fp = set_comparison.compare_sets(true_centroids, predicted_centroids)
  except tf.errors.OutOfRangeError:
    # end of iterator
    break

print("final stats over %d images: precision %0.2f recall %0.2f f1 %0.2f" % (num_imgs, *set_comparison.precision_recall_f1()))
