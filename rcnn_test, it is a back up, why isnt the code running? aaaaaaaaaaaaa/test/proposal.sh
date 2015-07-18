#!/bin/bash

# ----------------------------------------------------------
# Written by Tingwu Wang
# Copyright by Sensetime CO. Beijing
# 
# function:
# 	it is a shell script to generate the box proposal by 
#   using the proposal.exe in the proposal.cpp files
#
# The original filetree:
#	clothesDataset/train/1/
# The original filetree:
#	clothesDataset/train/1/proposals
# ----------------------------------------------------------

# There are the training set and the test set, operating on
# them simulataneously operating on each subdirectory
root_directory='/home/user/rccn_for_cloth/data/clothesDataset'
cd /home/user/rccn_for_cloth/rcnn_test

for ((i=1;i<=26;i++))
do

	echo "Now working on the $i th subdirectory"

	test_root_directory=$root_directory/test/$i
	train_root_directory=$root_directory/train/$i

	# making the directory for the proposals
	mkdir -p $test_root_directory/proposals
	mkdir -p $train_root_directory/proposals

	# using the cpp function to generate proposals
	echo "Now calculating the image proposals at $test_root_directory"
	#./bin/proposal.exe $test_root_directory
	echo "Now calculating the image proposals at $train_root_directory"
	./bin/proposal.exe $train_root_directory

done
