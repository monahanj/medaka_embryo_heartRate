Medaka heart rate analysis
==========================

## Analysis Overview
1. Python script segments beating heart from videos of medaka fish embryos.

2. Then performs Fourier Transform on each pixel in segmented area.

3. Kernel Density Estimation determines the heart rate from the most common Fourier peak.

## Installation 
It is advisable that you create an "environment" using the .yml file e.g. with [Conda](https://docs.conda.io/en/latest).

This installs all the python dependencies.

Code will run on Linux and MacOS, I'd recommend installing an Ubuntu Virtual Machine if you have a Windows PC such as with [VirtualBox](https://www.virtualbox.org/).


## Usage
```
usage: python segment_heart.py [-h] -i INDIR [-t FRAME_FORMAT] -w WELL [-l LOOP] [--crop] [--no-crop] [-f FPS] -o OUT

-i Need to specify directory with a tiff or jpeg subdirectory
-w Specify the well e.g. WE00001
-l specify acquistion loop e.g. LO001
-t specifies if frames are tiff or jpeg
--crop or --no-crop to specify whether or not frames need to be cropped
-f can optionally specify fps, otherise worked out in script
-o Output directory for analyses
```

Python script run by the shell script that supplies parameters.

Example data in `test_data` directory.

example run:
```
source activate medaka_env #activates analysis env 

python3 segment_heart.py -i test_data -w WE00001 -l LO001 --no-crop -t jpeg -o analyses/test_data
```

This would analyse `jpegs` from well `WE00001`, loop `LO001` in the `test_data` directory.

Results will be in `analyses/test_data`.

