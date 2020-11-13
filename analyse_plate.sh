#!/bin/bash

source activate medaka_env

set -e
set -u
set -o pipefail

basedir=$PWD
data=$basedir/data
out_dir=$basedir/analyses
rm -fr $out_dir
mkdir -p $out_dir

plates=(
	200824_Imaging
)

for plate in "${plates[@]}"; do

	echo $plate 

	in_dir=$data/$plate

	#Determine the imaging sessions
	sessions=($(find $in_dir -mindepth 1 -maxdepth 1 -type d))

	for session in "${sessions[@]}"; do

		expt=$(basename $session)

		echo $expt

		rm -fr $lsf/$plate/$expt
		mkdir -p $lsf/$plate/$expt

		#Determine if frames are jpegs or tiffs from the subdirectory name
		if ls $session/*Tiff* &> /dev/null; then

			frame_type=tiff

		elif ls $session/*Jpeg* &> /dev/null; then

			frame_type=jpeg
		fi

		#Iterate over wells in a plate, maximum of 96 wells
		for i in 0{1..9} {10..96}; do

			well=WE000$i

			#Up to 4 timepoints per well
			for j in 0{1..4}; do
	
				loop=LO0$j 

				#Check if well was videoed
				#File naming can vary
				if ls $session/*/*$well*$loop* &> /dev/null; then 

					$basedir/./segment_heart.py -i $session -w $well -l $loop -o $out_dir/$plate/$expt/$loop/$well -t $frame_type --no-crop 

				elif ls $session/*/*$loop*$well* &> /dev/null; then 
					$basedir/./segment_heart.py -i $session -w $well -l $loop -o $out_dir/$plate/$expt/$loop/$well -t $frame_type --no-crop 

				fi
			done
		done	
	done
done

