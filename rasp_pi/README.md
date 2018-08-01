# rasp pi setup

we use `systemctl` to install, and keep running, a `capture_stills.py` script.
images are output to `~pi/capture/images/YMD/YMD_HMS.jpg`

```
cd ~pi
git clone https://github.com/matpalm/bnn.git
cd bnn/rasp_pi/
sudo cp capture.service /lib/systemd/system/
sudo systemctl enable capture.service
sudo systemctl start capture.service
```

optionally have service started / stopped at 5am / 9pm by
root crontab (no use capturing in the dark)

```
0 5 * * * systemctl start capture.service
0 21 * * * systemctl stop capture.service
```

## components for rasp pi

* Raspberry Pi Camera Board v2.1
* Adafruit 6V 6W Solar panel
* Adafruit USB / DC / Solar Lithium Ion/Polymer charger - v2
* Adafruit PowerBoost 500 Charger
* 2500 mAh LiPo battery
