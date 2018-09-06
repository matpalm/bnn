#!/usr/bin/env python3

# source /opt/movidius/virtualenv-python/bin/activate

import mvnc.mvncapi as mvnc
import numpy as np
from PIL import Image
from scipy.special import expit
import util as u

devices = mvnc.enumerate_devices()
if len(devices) == 0:
  raise Exception("no compute stick?")
device = mvnc.Device(devices[0])
device.open()

binary_graph = open('graph.mv', 'rb' ).read()
graph = mvnc.Graph('g')
input_fifo, output_fifo = graph.allocate_with_fifos(device, binary_graph)

def run_on_ncs(input):
  graph.queue_inference_with_fifo_elem(input_fifo, output_fifo,
                                       np.float32(input), None)
  output, _user_object = output_fifo.read_elem()
  return output

img = np.array(Image.open('sample_data/training_512_patches/20180205_143023.jpg'))
img = img.astype(np.float32)
img = (img / 127.5) - 1.0  # -1.0 -> 1.0  # see data.py

prediction = run_on_ncs(img)
prediction = prediction[:127*127].reshape((127,127,1))  # workaround for output bug
prediction = expit(prediction)  # logits -> sigmoid
print(prediction.shape)

centroids = u.centroids_of_connected_components(prediction,
                                                rescale=1.0/0.25)
debug_img = u.red_dots(rgb=img, centroids=centroids)
debug_img.save("from_ncs.png")

input_fifo.destroy()
output_fifo.destroy()
graph.destroy()
device.close()
device.destroy()
