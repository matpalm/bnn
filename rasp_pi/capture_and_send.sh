#!/bin/bash
# A script for transmitting images for bees lucky enough to have internet
# requires imagemagick to compute brightness
mkdir -p /home/pi/images
while true; do
	IMGNAME=/home/pi/images/img$(date +"%Y%m%d%H%M%S").jpg
	echo "capturing image $IMGNAME"
	raspistill -w 640 -h 480 -q 100 -o $IMGNAME
        #use convert to calculate average brighness of image
        brightfloat=`convert $IMGNAME -colorspace Gray -format "%[mean]" info:`
        brightint=${brightfloat%.*}
        # If it is not bright, stop sending images
        if [ $brightint -lt "10000" ]; then
            echo ""$IMGNAME its dark now
            sleep 600
        else
            scp $IMGNAME user@xxx.xxx.xxx.xxx:images
        sleep 5 
        fi
	rm -f $IMGNAME
done
