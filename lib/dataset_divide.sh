#!/bin/bash

# ----------------------------------------------------------
# Written by Tingwu Wang
# Copyright by Sensetime CO. Beijing
# 
# function:
# 	it is a shell script to divide the original dataset into
# 	a training data set and a test data set
#
# The original filetree:
#	clothesDataset/1/images
# The original filetree:
#	clothesDataset/train/1/images
#	clothesDataset/test/1/images
# ----------------------------------------------------------

# a shuffle function to shuffle the image data set to
# the training set and data set
if [ $# -ne 2 ]
then
	echo 'Not enough input arguments'
	echo 'Usage: ./dataset_divide $PATH_of_Dataset $PATH_of_OutputDir'
	exit
fi

data_path=$1
output_path=$2
current_path=`pwd`
percentage_of_test=0.1
cloth_ext=".clothInfo"

echo 'processing the file'
echo "The top level data set files is $data_path"

# check for the file directory
if [ -d $data_path ]
then 
	echo 'The data file exists'
else
	echo 'The data file doesnt exist, error'
fi

if [ -d $output_path ]
then 
	echo 'The file already exist'
else
	echo "Make a new output directory in the below directory,"
	echo "$current_path/$output_path"
	mkdir -p $current_path/$output_path
fi

# making the output directory
train_data_path=$output_path/train
test_data_path=$output_path/test
echo "The training set will be put in $train_data_path"
echo "The testing set will be put in $test_data_path"
mkdir -p $output_path

# operating on each subdirectory
for ((i=1;i<=26;i++))
do
	# the 18th class is lost
	echo "Now working on the $i th subdirectory"
	echo "Currently at $data_path/$i"
	subdirectory=$data_path/$i

	output_subdirectory_train=$output_path/train/$i
	output_subdirectory_test=$output_path/test/$i
	output_train_Mappingtxt=$output_subdirectory_train/GUIDMapping.txt
	output_test_Mappingtxt=$output_subdirectory_test/GUIDMapping.txt

	echo $output_test_Mappingtxt
	mkdir -p $output_subdirectory_test
	mkdir -p $output_subdirectory_train

	
	echo "Making the data output path at $output_subdirectory_train and $output_subdirectory_test"

	# get the number of image
	number_image=`cat $subdirectory/GUIDMapping.txt | wc -l`
	echo "Getting $number_image files in the $i th subdirectory "

	# get the number of image in the test dataset
	test_number=`echo "$percentage_of_test * $number_image" | bc`
	test_number=`printf "%.f" $test_number`
	mkdir -p $output_subdirectory_train/images
	mkdir -p $output_subdirectory_test/images
	mkdir -p $output_subdirectory_train/Label
	mkdir -p $output_subdirectory_test/Label
	rm -f $output_test_Mappingtxt
	rm -f $output_train_Mappingtxt

	for ((line_number=1;line_number<=$number_image;line_number++))
	do
		# get the image infomation and label information
		line_text=`awk "NR==$line_number" \
			$subdirectory/GUIDMapping.txt \
			| sed 's/.$//' \
			| tr '\\' '/a' 2> /dev/null` 
		# get the image name
		jpg_position=`echo "$line_text" | grep -b -o '\.jpg'`
		echo "the postion is $jpg_position"
		if [ $? -ne 0 ] # the file is not a jpg
		then
			jpg_position=`echo "$line_text" | grep -b -o '\.png' | \
				awk 'BEGIN {FS=":"}{print $1}'`
		else
			jpg_position=`echo "$line_text" | grep -b -o '\.jpg' | \
				awk 'BEGIN {FS=":"}{print $1}'`
		fi
		# there are jpg and png image at the same time
		image_tail=`expr 4 + $jpg_position`
		image_name=`expr substr $line_text 1 $image_tail`

		label_head=`expr 2 + $image_tail`
		label_tail=`expr length $line_text`
		label_tail=`expr $label_tail`

		label_name=`expr substr $line_text $label_head $label_tail`
		label_name=`echo $label_name$cloth_ext`
		remainder=`expr $line_number % 10`
		echo $label_name
		echo $image_name
		if [ `expr $line_number % 10` -eq 0 ]
		then
			echo "Currently processing the $line_number th picture in the $i th class..."
		fi
		if [ $remainder -ne 1 ] # it is the train data
		then
			cp $subdirectory/$image_name \
				$output_subdirectory_train/$image_name
			if [ $? -eq 0 ]
			then
				echo "Oops, we are having a trouble here"
				continue
			fi
			cp $subdirectory/Label/$label_name\
				$output_subdirectory_train/Label/$label_name
			if [ $? -eq 0 ]
			then
				echo "Oops, we are having a trouble here"
				continue
			fi

			echo $line_text >> $output_train_Mappingtxt
		else
			# it is the test data
			cp $subdirectory/$image_name \
				$output_subdirectory_test/$image_name
			if [ $? -eq 0 ]
			then
				echo "Oops, we are having a trouble here"
				continue
			fi
			cp $subdirectory/Label/$label_name\
				$output_subdirectory_test/Label/$label_name
			if [ $? -eq 0 ]
			then
				echo "Oops, we are having a trouble here"
				continue
			fi
			echo $line_text >> $output_test_Mappingtxt
		fi
	done


done
