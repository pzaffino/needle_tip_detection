#!/bin/bash
thr=+0

parallel --bar -j $thr --header : ./Slicer --no-splash --python-script ~/Dropbox/MachineLearning/script.py -k {i} ::: i `seq 0 1 70`
 
