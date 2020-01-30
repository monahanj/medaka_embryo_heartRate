#!/bin/bash

source activate medaka_env

set -e
set -u
set -o pipefail

basedir=$PWD
in_dir=$basedir/170414181030_OlE_ICF2_HR_28C_2x
out_dir=$basedir/analysis_dev2
lsf=$basedir/lsf_dev2
rm -fr $lsf $out_dir
mkdir -p $lsf $out_dir

wells=( WE00043 ) 
#wells=( WE00045 WE00047 WE00048 )
#wells=( WE00047 )
#wells=( WE00045 ) #will pass
#wells=( WE00048 ) #will fail
#loops=( LO001 LO002 )
loops=( LO001 )
for well in "${wells[@]}"; do

	echo $well
	
	for loop in "${loops[@]}"; do

		lsf_err=$lsf/${well}_${loop}_err
		lsf_out=$lsf/${well}_${loop}_out
#		$basedir/./segment_heart.dev.py  -i $in_dir -w $well -l $loop -o $out_dir/$well/$loop 
		bsub -e $lsf_err -o $lsf_out -n 20 -M 20000 "$basedir/./segment_heart.dev2.py  -i $in_dir -w $well -l $loop -o $out_dir/$well/$loop > $out_dir/${well}.${loop}.report.txt "

	done
done

