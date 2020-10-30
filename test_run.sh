#!/bin/bash

source activate medaka_env

set -e
set -u
set -o pipefail

basedir=$PWD
in_dir=$basedir/test_data
out_dir=$basedir/analyses/test_data
rm -fr $out_dir
mkdir -p $out_dir


#Analyse video 1, tiffs that need to be normalised and cropped
wells=( WE00037 ) 
loops=( LO001 )
for well in "${wells[@]}"; do

	start=$SECONDS
	
	for loop in "${loops[@]}"; do

		mkdir -p $out_dir/$loop/$well
		$basedir/./segment_heart.py -i $in_dir -w $well -l $loop -o $out_dir/$loop/$well --crop -t tiff

	done

	duration=$(( SECONDS - start ))

	echo "$duration secs elapsed"
done

#Analyse video 2, jpegs
wells=( WE00001 ) 
loops=( LO001 )
for well in "${wells[@]}"; do

	start=$SECONDS
	
	for loop in "${loops[@]}"; do

		mkdir -p $out_dir/$loop/$well
		$basedir/./segment_heart.py -i $in_dir -w $well -l $loop -o $out_dir/$loop/$well --no-crop -t jpeg

	done

	duration=$(( SECONDS - start ))

	echo "$duration secs elapsed"
done

