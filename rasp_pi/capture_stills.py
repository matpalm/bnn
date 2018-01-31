# capture stills
# controlled / respawned by systemd  (see capture.service)
# started/stopped as service by root cron at 5am/9pm
# see raspberrypi.org/learning/getting-started-with-picamera/worksheet
# and also https://picamera.readthedocs.io/en/release-1.13/recipes1.html

from picamera import PiCamera
import datetime
import os
import time

while True:
  dts = datetime.datetime.now()
  YMD ="%04d%02d%02d" % (dts.year, dts.month, dts.day)
  HMS ="%02d%02d%02d.jpg" % (dts.hour, dts.minute, dts.second)
  
  output_dir = "/home/pi/capture/images/%s" % YMD
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  output_filename = "%s/%s_%s" % (output_dir, YMD, HMS)

  with PiCamera() as camera:
    camera.resolution = (1024, 768)
    time.sleep(2)  # wait for auto gains / balances to settle.
    camera.capture(output_filename)
  
  time.sleep(8)  # bring it up to ~10s for entire loop.
