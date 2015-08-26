#!/bin/bash

# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
DIR=../../models/Models_PT_ImageNet/
cd $DIR

FILE=imagenet_models.tgz
URL=http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/$FILE
CHECKSUM=8b1d4b9da0593fc70ef403284f810adc

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading pretrained ImageNet models (1G)..."

wget $URL -O $FILE

echo "Unzipping..."

tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
