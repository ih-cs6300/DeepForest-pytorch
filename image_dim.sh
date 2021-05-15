#!/bin/bash
# list the dimensions of each image file
FILES=./annotations/*.tif
for f in $FILES
do
  echo "$f:" 
  identify -quiet -format '%w %h' $f
  echo
  echo
done
