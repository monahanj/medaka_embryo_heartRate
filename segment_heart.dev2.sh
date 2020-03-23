#!/bin/bash

source activate medaka_env

set -e
set -u
set -o pipefail

basedir=$PWD
in_dir=$basedir/data/00_test_data/170414181030_OlE_ICF2_HR_28C_2x
out_dir=$basedir/analyses/00_dev/tiff
lsf=$basedir/lsf_out/analysis_dev2
rm -fr $lsf $out_dir
mkdir -p $lsf $out_dir

#wells=( WE00043 ) 
wells=( WE00038 ) 
#wells=( WE00045 WE00047 WE00048 )
#wells=( WE00047 )
#wells=( WE00045 ) #will pass
#wells=( WE00048 ) #will fail
#loops=( LO001 LO002 )
loops=( LO002 )
for well in "${wells[@]}"; do

	echo $well

	start=$SECONDS
	
	for loop in "${loops[@]}"; do

		lsf_err=$lsf/${well}_${loop}_err
		lsf_out=$lsf/${well}_${loop}_out

		rm -fr $out_dir/$loop/$well
		mkdir -p $out_dir/$loop/$well
		$basedir/./segment_heart.dev2.py -i $in_dir -w $well -l $loop -o $out_dir/$loop/$well -c True #> $out_dir/${well}.${loop}.report2.txt 
	done

	duration=$(( SECONDS - start ))

	echo "$duration secs elapsed"
done

