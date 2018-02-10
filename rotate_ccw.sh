#!/usr/bin/env bash
set -e
mkdir r 
ls *jpg | perl -ne'chomp;print "convert -rotate -90 $_ r/$_\n"' | parallel -j20
mv r/* .
rmdir r
