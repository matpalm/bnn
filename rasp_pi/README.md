# rasp pi setup

images are captured by `capture_stills.py` script

script monitored (start at boot, respawn) etc by systemd

```
sudo cp capture.service /lib/systemd/system/
sudo systemctl enable capture.service
```

service started / stopped at 5am / 9pm by root crontab (no use capturing in the dark)

```
0 5 * * * systemctl start capture.service
0 21 * * * systemctl stop capture.service
```

