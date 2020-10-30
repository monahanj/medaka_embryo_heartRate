Medaka heart rate analysis
==========================

1. Python script segments beating heart from videos of medaka fish embryos.

2. Then performs Fourier Transform on each pixel in segmented area.

3. Kernel Density Estimation determines the heart rate from the most common Fourier peak.

```
usage: python segment_heart.py [-h] -i INDIR [-t FRAME_FORMAT] -w WELL [-l LOOP] [--crop] [--no-crop] [-f FPS] -o OUT

-i Need to specify directory with a tiff or jpeg subdirectory
-w Specify the well e.g. WE00001
-l specify acquistion loop e.g. LO001
--crop or --no-crop to specify whether or not frames need to be cropped
-f can optionally specify fps, otherise worked out in script
-o Output directory for analyses
```

Python script run by the shell script that supplies parameters.

Example data in `test_data` directory.

example run:
```
python segment_heart.py -i test_data -w WE00001 -l LO001 --no-crop -t jpeg -o analyses/test_data
```

This would analyse `jpegs` from well `WE00001`, loop `LO001`.

It is advisable that you create an "environment" using the .yml file e.g. with [Conda](https://docs.conda.io/en/latest)

This installs all the python dependencies.

