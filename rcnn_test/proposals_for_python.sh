#!/bin/bash

BASEDIR=$(dirname $0)
echo $BASEDIR

echo $BASEDIR/lib
cd $BASEDIR
export LD_LIBRARY_PATH=./lib/

if [ $# -ne 1 ]
then
    echo "Error, you must specify the image path!"
    exit
fi

echo "----- Using the C++ port to calculate the proposals!"

./bin/single_proposals.exe $1

echo "----- Existing the shell program"
