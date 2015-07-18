#!/bin/bash

# generate the symbolic link to the image data set
# and the label data set


root_dir=`pwd`

for ((i=1;i<=26;i++))
do
    if [ $i -eq 18 ]; then
        continue
    fi
    echo "Establishing the $i th category"
    rm $root_dir/test/$i/images
    rm $root_dir/test/$i/Label
    ln -s $root_dir/train/$i/images/ $root_dir/test/$i/images
    ln -s $root_dir/train/$i/Label/ $root_dir/test/$i/Label
done
