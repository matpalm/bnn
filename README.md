# BNN v2

unet style image translation from image of hive entrance to bitmap of location of center of bees.

trained in a semi supervised way on a desktop gpu and deployed to run in real time on the hive using
either a [raspberry pi](https://www.raspberrypi.org/) using a [neural compute stick](https://developer.movidius.com/)
or a [je vois embedded smart camera](http://jevois.org/)

see [this blog post](http://matpalm.com/blog/counting_bees/) for more info..


here's an example of predicting bee position on some held out data. the majority of examples trained had ~10 bees per image.

![rgb_labels_predictions.png](rgb_labels_predictions.png)

the ability to locate each bee means you can summarise with a count. note the spike around 4pm when the bees at this time of year come back to base.

![counts_over_days.png](counts_over_days.png)
