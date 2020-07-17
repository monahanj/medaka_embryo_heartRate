#!/usr/bin/env python3

import argparse
import pandas as pd
import os
import glob2
import errno

import numpy as np
import cv2

import skimage
from skimage.util import compare_images, img_as_ubyte, img_as_float
from skimage.filters import threshold_triangle, threshold_otsu, threshold_yen, roberts, sobel, scharr
from skimage.measure import label#, find_contours 
from skimage.feature import peak_local_max, canny
from skimage import color

import scipy
from scipy import stats
from scipy import ndimage as ndi
from scipy.signal import find_peaks, peak_prominences, welch
from scipy.interpolate import CubicSpline

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from collections import Counter

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

np.set_printoptions(threshold=np.inf)

#Go through frames in stack to determine changes between seqeuential frames (i.e. the heart beat)
#Determine absolute differences between current and previous frame
#Use this to determine range of movement (i.e. the heart region)

parser = argparse.ArgumentParser(description='Read in medaka heart video frames')
parser.add_argument('-i','--indir', action="store", dest='indir', help='Directory containing frames', default=False, required = True)
parser.add_argument('-t','--format', action="store", dest='frame_format', help='Frame format', default='tiff', required = False)
parser.add_argument('-w','--well', action="store", dest='well', help='Well Id', default=False, required = True)
parser.add_argument('-l','--loop', action="store", dest='loop', help='Well frame acquistion loop', default=None, required = False)
parser.add_argument('--crop', dest='crop', action='store_true')
parser.add_argument('--no-crop', dest='crop', action='store_false')
parser.add_argument('-f','--fps', action="store", dest='fps', help='Frames per second', default=False, required = False)
parser.add_argument('-o','--out', action="store", dest='out', help='Output', default=False, required = True)
parser.set_defaults(feature=True)
args = parser.parse_args()

indir = args.indir
frame_format = args.frame_format
well_number = args.well
loop = args.loop
crop = args.crop
out_dir = args.out

#Make output dir if doesn't exist
try:
	os.makedirs(out_dir)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise

#All images in subdirs, one level below if tiff, 2 below if jpeg

#If multiple loops
if args.loop:

	loop = args.loop

	#If tiff
	if frame_format == "tiff":
		well_frames = glob2.glob(indir + '/*/' + well_number + '*' + loop + '*.tif') + glob2.glob(indir + '/*/' + well_number + '*' + loop + '*.tiff')
	#If jpeg
	elif frame_format == "jpeg":
		well_frames = glob2.glob(indir + '/*/' + well_number + '*' + loop + '*.jpg') + glob2.glob(indir + '/*/' + well_number + '*' + loop + '*.jpeg')   
else:
	#If tiff
	if frame_format == "tiff":
		well_frames = glob2.glob(indir + '/*/*' + well_number + '*.tif') + glob2.glob(indir + '/*/*' + well_number + '*.tiff')
	#If jpeg
	elif frame_format == "jpeg":
		well_frames = glob2.glob(indir + '/*/*' + well_number + '*.jpg') + glob2.glob(indir + '/*/*' + well_number + '*.jpeg')   

# Improve contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

#Kernel for image smoothing
kernel = np.ones((5,5),np.uint8)

##########################
##	Functions	##
##########################
#Normalise across frames to harmonise intensities (& possibly remove flickering)
def normVideo(frames):

	for i in range(len(frames)):

		frame = frames[i]

		if frame is not None:

			if i == 0:
				filtered_frames = np.asarray(frame)
			else:
				filtered_frames = np.dstack((filtered_frames, frame))

	norm_frames = [] 
	##Divide by max to try and remove flickering between frames
	for i in range(len(frames)):

		frame = frames[i]

		if frame is not None:
			norm_frame = np.uint8(frame / np.max(filtered_frames) * 255)

		#If empty frame
		else:
			norm_frame = None

		norm_frames.append(norm_frame)

	return(norm_frames)

#Detect eyes from frame
def detectEyes(frame):

	#Tresholding to detect eyes 
	frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	eye_mask = cv2.inRange(frame_grey, 0, 50)

	#Opening
	eye_mask = cv2.morphologyEx(eye_mask, cv2.MORPH_OPEN, kernel)  

	return(eye_mask)

#Pre-process frame
def processFrame(frame):
	"""Image pre-processing and illumination normalisation"""

	#Convert RGB to LAB colour
	lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

	#Split the LAB image into different channels
	l, a, b = cv2.split(lab)

	# Improve contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
	#Apply CLAHE to L-channel
	cl = clahe.apply(l)

	#Merge the CLAHE enhanced L-channel with the A and B channel
	limg = cv2.merge((cl,a,b))

	#Convert LAB back to RGB colour 
	out_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

	#Convert to greyscale
	frame_grey = cv2.cvtColor(out_frame, cv2.COLOR_BGR2GRAY)
#	frame_grey = cl 

	#Convert CLAHE-normalised greyscale frame back to BGR
	out_frame = cv2.cvtColor(frame_grey, cv2.COLOR_GRAY2BGR)

	#Blur the CLAHE frame
	#Blurring kernel numbers must be odd integers
	blurred_frame = cv2.GaussianBlur(frame_grey, (9, 9), 0) 

	return out_frame, frame_grey, blurred_frame
 
def maskFrame(frame, mask):

	"""Add constant value in green channel to region of frame from the mask.""" 

	# split source frame into B,G,R channels
	b,g,r = cv2.split(frame)

	# add a constant to G (green) channel to highlight the masked area
	g = cv2.add(g, 50, dst = g, mask = mask, dtype = cv2.CV_8U)
	masked_frame = cv2.merge((b, g, r))

	return masked_frame

def filterMask(mask, min_area = 300):

	#Contour mask 
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	#Filter contours based on their area
	filtered_contours = []
	for i in range(len(contours)):
		contour = contours[i]

		#Filter contours by their area
		if cv2.contourArea(contour) >= min_area:
			filtered_contours.append(contour)

	contours = filtered_contours

	#Create blank mask
	rows, cols = mask.shape
	filtered_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)

	#Draw and fill-in filtered contours on blank mask
	cv2.drawContours(filtered_mask, contours, -1, 255, thickness = -1)

	return filtered_mask

#Differences between two frames
def diffFrame(frame, frame2_blur, frame1_blur, min_area = 300):
	"""Calculate the abs diff between 2 frames and returns frame2 masked with the filtered differences."""

	#Absolute differnce between frames
	abs_diff = cv2.absdiff(frame2_blur, frame1_blur)

	#Triangle thresholding on differences
	triangle = threshold_triangle(abs_diff)
	thresh = abs_diff > triangle
	thresh = thresh.astype(np.uint8)

	#Opening to remove noise
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

	#Find contours in mask and filter them based on their area
	mask = filterMask(mask = thresh, min_area = min_area)

	#Mask frame
	masked_frame = maskFrame(frame, mask)

	#Return the masked frame, the filtered mask and the absolute differences for the 2 frames
	#return masked_frame, mask, thresh
	return masked_frame, thresh

# Forward or reverse rolling window of width w with step size ws
def rolling_diff(index, frames, win_size = 5, direction = "forward", min_area = 300):
	"""
	Implement rolling window 
	* win_size INT
		Window size (default = 5)
	"""

	if direction == "forward":
		
		if (index + win_size) > len(frames):
			window_indices = list(range(index, len(frames)))
		else:
			window_indices = list(range(index, index + win_size))

		frame0 = frames[window_indices[0]]
		_, _, old_blur = processFrame(frame0)

		#Determine absolute differences between current and previous frame
		#Frame[i] vs. frame[i + 1] ... [i + 4]

		#Generate blank images for masking
		rows, cols, _ = frame0.shape
		abs_diffs = np.zeros(shape=[rows, cols], dtype=np.uint8)

		for i in window_indices[1:]:

			frame = frames[i]

			if frame is not None:
				_, _, frame_blur = processFrame(frame)
				_, triangle_thresh = diffFrame(frame, frame_blur, old_blur)
				abs_diffs = cv2.add(abs_diffs, triangle_thresh)

	elif direction == "reverse":

		if index >= win_size:
			window_indices = list(range(index, index - win_size, -1))[::-1]
		else:
			window_indices = list(range(0, index + 1))

		frame = frames[window_indices[-1]]
		_, _, frame_blur = processFrame(frame)

		#Determine absolute differences between current and previous frame
		#Frame[i] vs frame[i - 1] .... [(i - 3]

		#Generate blank images for masking
		rows, cols, _ = frame.shape
		abs_diffs = np.zeros(shape=[rows, cols], dtype=np.uint8)

		for i in window_indices[:-1]:

			frame0 = frames[i]

			if frame0 is not None:
				_, _, old_blur = processFrame(frame0)
				_, triangle_thresh = diffFrame(frame, frame_blur, old_blur)
				abs_diffs = cv2.add(abs_diffs, triangle_thresh)

	#Filter mask by area
	#Opening to remove noise
	thresh = cv2.morphologyEx(abs_diffs, cv2.MORPH_OPEN, kernel)

	#Filter based on their area
	thresh = filterMask(mask = thresh, min_area = min_area)

	#Mask frame
	masked_frame = maskFrame(frame, thresh)

	return(masked_frame,abs_diffs)

#Detrend heart signal and normalise
def detrendSignal(interpolated_signal, time_domain):

	p = np.polyfit(time_domain - time_domain[0], interpolated_signal(time_domain), 1)

	dat_notrend = interpolated_signal(time_domain) - np.polyval(p, time_domain - time_domain[0])

	std = dat_notrend.std()  # Standard deviation
	var = std ** 2  # Variance

	#Normalise Signal
	normalised_signal = dat_notrend / std 

	#Generate new Cubic Spline based on normalised data
	norm_cs = CubicSpline(time_domain, normalised_signal)

	return(norm_cs)

#Perform a Fourier Transform on interpolated signal from heart region
def fourierHR(interpolated_signal, time_domain, heart_range = (0.5, 6)):

	"""
	When 3 or less peaks detected is Fourier, 
	the true heart-rate is usually the first one.
	The second peak may be higher in some circumstances 
	if BOTH chambers were segmented.
	Fourier seems to detect frequencies coreesponding to 
	1 beat, 2 beats and/or 3 beats in this situation. 
	"""

	#Fast fourier transform
	fourier = np.fft.fft(interpolated_signal(time_domain))
	# Power Spectral Density
	psd = np.abs(fourier) ** 2

	N = interpolated_signal(time_domain).size
	timestep =  np.mean(np.diff(time_domain))
	freqs = np.fft.fftfreq(N, d=timestep)

	#one-sided Fourier spectra
	psd = psd[freqs > 0]
	freqs = freqs[freqs > 0]

	#Calculate ylims for xrange 0.5 to 6 Hz
	heart_indices = np.where(np.logical_and(freqs >= heart_range[0], freqs <= heart_range[1]))[0]

	#Peak Calling on Fourier Transform data
	peaks, _ = find_peaks(psd)

	#Filter out peaks lower than 1
#	peaks = peaks[psd[peaks] >= 1]
	peaks = peaks[psd[peaks] >= 0.75]

	n_peaks = len(peaks)
	if n_peaks >= 1:
		#Determine the peak within the heart range
		max_peak = max(psd[peaks])

		#Filter peaks based on ratio to largest peak
		filtered_peaks = peaks[psd[peaks] >= (max_peak * 0.25)]

		#Calculate heart rate in beats per minute (bpm) from the results of the Fourier Analysis
		n_filtered = len(filtered_peaks)
		if n_filtered > 0:

			beat_indices = list(set(filtered_peaks) & set(heart_indices))
			beat_psd = psd[beat_indices]
			beat_freqs = freqs[beat_indices]

			if 0 < len(beat_indices) < 4:
				beat_freq = beat_freqs[0] 
				beat_power = beat_psd[0]

				bpm = beat_freq * 60 
				peak_coord  = (beat_freq, beat_power)

			else:
				bpm = None
				peak_coord = None
				
		else:
			bpm = None
			peak_coord = None

	#Flat Fourier
	else:
		bpm = None
		peak_coord = None

	return(psd, freqs, peak_coord, bpm)

#Plot Fourier Transform
def plotFourier(psd, freqs, peak, bpm, heart_range, figure_loc = 211):

	#Prepare label for plot
	if bpm is not None:
		bpm_label = "Heart rate = " + str(int(bpm)) + " bpm"
	else:
		bpm_label = "Heart rate = NA"

	ax = plt.subplot(figure_loc)
 
	#Plot frequency of Fourier Power Spectra
	_ = ax.plot(freqs,psd)

	#Plot frequency peak if given
	if peak is not None:
		#Put x on freq that correpsonds to heart rate
		_ = ax.plot(peak[0], peak[1],"x")
		#Dotted line to peak
		_ = ax.vlines(x = peak[0], ymin = 0, ymax = peak[1], linestyles = "dashed")
		_ = ax.hlines(y = peak[1], xmin = 0, xmax = peak[0], linestyles = "dashed")
		#Annotate with BPM
		_ = ax.annotate(bpm_label, xy=peak, xytext=(peak[0] + 0.5 , peak[1] + (peak[1] * 0.05)), arrowprops=dict(facecolor='black', shrink=0.05))

	# Only plot within heart range (in Hertz) if necessary
	if heart_range is not None:
		_ = ax.set_xlim(heart_range)

	_ = ax.set_ylim(top = max(psd) + (max(psd) * 0.2))

	#Y-axis label
	_ = ax.set_ylabel('Power Spectra')

	return(ax)

#Perform Welch's Power method on interpolated signal from heart region
def welchHR(interpolated_signal, time_domain, heart_range = (1, 6)):

	Fs = round(1/  np.mean(np.diff(time_domain)))
	window = np.hanning(3*Fs)

	#Welch's Power Method
	freqs, p_density = welch(x = interpolated_signal(time_domain), window = window, fs = Fs, nfft = np.power(2,14), return_onesided=True, detrend="constant")

	p_final = 10*np.log10(p_density)

	#Calculate ylims for xrange 1 to 6 Hz
	heart_indices = np.where(np.logical_and(freqs >= heart_range[0], freqs <= heart_range[1]))
	heart_freqs = freqs[heart_indices]
	heart_psd = p_final[heart_indices]

	#Determine the peak within the range
	heart_peak = np.argmax(heart_psd)

	p_min = np.amin(heart_psd)
	p_max = heart_range[heart_peak]
	ylims = (p_min -1, p_max +1)

	freq_peaks, _ = find_peaks(p_final)

	#Peak prominence
	freq_prominences = peak_prominences(p_final, freq_peaks)
	prominent_peaks = [idx for idx, prominence in enumerate(freq_prominences[0])]
	
	peaks_values = freqs[freq_peaks[prominent_peaks]]
	prominent_values = p_final[freq_peaks[prominent_peaks]]

	#prominent_peaks2, _ = find_peaks(p_final, prominence=12)
	#peaks_values2 = f[prominent_peaks2]
	#prominent_values2 = p_final[prominent_peaks2]

	bpm = freqs[heart_freq][heart_peak] * 60

	return(bpm)

#TODO
#Wavelet Analysis
#def waveletHR(normalised_interpolated_signal, time_domain, heart_range = (0.5, 6)):

	#import pywt
	#import pycwt as wavelet

	#out_fig = out_dir + "/bpm_trace.detrended.png"
	#plt.plot(time_domain, normalised_interpolated_signal)
	#plt.savefig(out_fig)
	#plt.close()

#	return(bpm)

#Calculate RMSSD
#Root mean square of successive differences
def getRMSSD(beat_times):

	#Root mean square of successive differences of successive beats
	#first calculating each successive time difference between heartbeats in ms. 
	#Each of value is squared and the result is averaged before the square root of the total is obtained

	#Calculate beat-to-beat times
	#(time between peaks)
	peak2peak = [t2 - t1 for t1, t2 in zip(beat_times, beat_times[1:])]
                
	#Square peak2peak
	peak2peak_sq = [i**2 for i in peak2peak]

	#Average squared peak2peak
	mean_peak2peak_sq = np.mean(peak2peak_sq)
                
	#Square root of Average squared peak2peak
	rmssd = np.sqrt(mean_peak2peak_sq)

	return(rmssd)

print("Reading in frames\n")

#Generate html report from images and videos using jinja2
#def htmlReport():

#	return(html_report)

#Read all images for a given well
#>0 returns a 3 channel RGB image. This results in conversion to 8 bit.
#0 returns a greyscale image. Also results in conversion to 8 bit.
#<0 returns the image as is. This will return a 16 bit image.
imgs = {}
imgs_meta = {}  
sizes = {}
crop_params = {}
for frame in well_frames:

	fname = os.path.basename(frame)
	fname = fname.split(".")[0]
	fname = fname.split("---")
	well = fname[0]
	frame_details = fname[1].split("--")

	plate_pos = frame_details[0]
	loop = frame_details[2]

	if frame_format == "tiff":

		#Tiff names....
		#Extract info from tiff file names and create a pandas df
		#WE00048---D001--PO01--LO002--CO6--SL037--PX32500--PW0080--IN0010--TM280--X014463--Y038077--Z223252--T0016746390.tif
		frame_number = frame_details[4]
		time = frame_details[-1]
		#Drop 'T' from time stamp
		time = float(time[1:])

	elif frame_format == "jpeg":

		#Jpeg names....
		#Extract info from the file names and create a pandas df
		#WE00001---A001--PO01--LO001--CO6--SL049_00.01.00_00.00.03,692.jpg
		frame_number_time = frame_details[-1]
		frame_number = frame_number_time.split("_")[0]

		#Time in milliseconds
		time = frame.split("_")[-1]
		#Remove file extension
		time = time.split(".")[:-1]

		time = "".join(time)
		#Remove commas
		time = float(time.replace(",", ""))

	#name = (well, loop, frame_number)
	name = (plate_pos, loop, frame_number)
	#frame_details = [well] + frame_details
	frame_details = [plate_pos] + frame_details
	#well_id = (well, loop)
	well_id = (plate_pos, loop)

	#If frame is not empty, read it in
	if not os.stat(frame).st_size == 0:

		#Read image in colour
		img = cv2.imread(frame,1)
		imgs[name] = img

	#if image empty
	else:
		imgs[name] = None

	#Crop if necessary based on centre of embryo
	if crop is True:

		#Find circle i.e. the embryo in the yolk sac 
		img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img_blur = cv2.GaussianBlur(img_grey, (3, 3), 0)

		circles = cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT,1,150,param1=50,param2=30)

		if circles is not None:

			#Sort detected circles
			circles = sorted(circles[0],key=lambda x:x[2],reverse=True)
			#Only take largest circle to be embryo
			circle = np.uint16(np.around(circles[0]))

			#Circle coords
			centre_x = circle[0]
			centre_y = circle[1]
			radius = circle[2]
	
			x1 = centre_x - radius
			x2 = centre_x + radius
			y1 = centre_y - radius
			y2 = centre_y + radius
		
			#Round coords
			x1_test = 100 * round(x1 / 100)
			x2_test = 100 * round(x2 / 100)
			y1_test = 100 * round(y1 / 100)
			y2_test = 100 * round(y2 / 100)

			#If rounded figures are greater than x1 or y1, take 50 off it 
			if x1_test > x1:
				x1 = x1_test - 50 
			else:
				x1 = x1_test

			if y1_test > y1:
				y1 = y1_test - 50 
			else:
				y1 = y1_test

			#If rounded figures are less than x2 or y2, add 50 
			if x2_test < x2:
				x2 = x2_test + 50 
			else:
				x2 = x2_test

			if y2_test < y2:
				y2 = y2_test + 50 
			else:
				y2 = y2_test

			x1 = int(x1)
			y1 = int(y1)
			x2 = int(x2)
			y2 = int(y2)

		#tiff cropping parameters
		#crop_img = img[y1:y2, x1:x2]
		crop_size = (x2 - x1) * (y2 - y1)
		#crop_id = (well, loop, crop_size)
		crop_id = (plate_pos, loop, crop_size)
		crop_params[crop_id] = [x1, y1, x2, y2]

		try:
			sizes[well_id].append(crop_size)

		except KeyError:
			sizes[well_id] = [crop_size]

	#Make dict based on the file data fields
	try:
		imgs_meta['frame'].append(frame)
		imgs_meta['well'].append(plate_pos) #(frame_details[0])
		imgs_meta['loop'].append(loop) #(frame_details[3])
		imgs_meta['frame_number'].append(frame_number) #(frame_details[5])
		imgs_meta['time'].append(time)

	except KeyError:
		imgs_meta['frame'] = [frame]
		imgs_meta['well'] = [plate_pos] #[frame_details[0]]
		imgs_meta['loop'] = [loop] #[frame_details[3]]
		imgs_meta['frame_number'] = [frame_number] #[frame_details[5]]
		imgs_meta['time'] = [time]

#Save original frame with embryo highlighted with a circle
img_out = img.copy()

#Crop if a tiff
if crop is True:

	#Uniformly crop image per loop per well based on the circle radii + offset
	well_crop_params = {}
	for well_id in sizes.keys():

		size = sizes[well_id]
		dimension_counts = Counter(size)
		frame_size, _ = dimension_counts.most_common(1)[0]
		crop_id = well_id + (frame_size,)
		well_crop_params[well_id] = crop_params[crop_id]

	# Draw the center of the circle
	cv2.circle(img_out,(circle[0],circle[1]),2,(0,255,0),3)
	# Draw the circle
	cv2.circle(img_out,(circle[0],circle[1]),circle[2],(0,255,0),2)

out_fig = out_dir + "/embryo.original.png"
cv2.imwrite(out_fig,img_out)

# creating a dataframe from a dictionary 
imgs_meta = pd.DataFrame(imgs_meta)
#Sort by well, loop and frame 
imgs_meta = imgs_meta.sort_values(by=['well','loop','frame_number'], ascending=[True,True,True])
#Reindex pandas df
imgs_meta = imgs_meta.reset_index(drop=True)

#Frames per loop
#('WE00048', 'LO001', 'SL117')
#('D01', 'LO001', 'SL117')
sorted_frames = []
sorted_times = []
#for index,row in loop_meta.iterrows():
for index,row in imgs_meta.iterrows():

	#tiff = row['tiff']
	well = row['well']
	loop = row['loop']
	frame_number = row['frame_number']
	time = row['time']
	name = (well, loop, frame_number)
	img = imgs[name]

	sorted_times.append(time)

	well_id = (well, loop)

	#To deal with empty frames
	if img is not None:

		#Use cropping parameters to uniformly crop frames
		if crop is True:

			#Well and loop specific parameters for cropping frame
			#crop_params[crop_id] = [x1, y1, x2, y2]
			crop_values = well_crop_params[well_id]

			#crop_img = img[y1 : y2, x1: x2]
			crop_img = img[crop_values[1] : crop_values[3], crop_values[0] : crop_values[2]]

			sorted_frames.append(crop_img)

			height, width, layers = crop_img.shape
		else:
			sorted_frames.append(img)
			height, width, layers = img.shape

		size = (width,height)

	else:
                sorted_frames.append(None)

#Only process if less than 5% frames are empty
if sum(frame is None for frame in sorted_frames) < len(sorted_frames) * 0.05:

	#Determine frame rate from time-stamps if unspecified 
	if args.fps:
		fps = int(args.fps)
	else:
		#total time acquiring frames in seconds
		timestamp0 = sorted_times[0]
		timestamp_final = sorted_times[-1]
	
		total_time = (timestamp_final - timestamp0) / 1000 
		fps = int(len(sorted_times) / round(total_time))

	#Normalise intensities across frames by max pixel if tiff images
	if frame_format == "tiff":
		norm_frames = normVideo(sorted_frames)

	#jpegs already normalised
	else:
		norm_frames = sorted_frames.copy()

	#Write video
	vid_frames = [frame for frame in norm_frames if frame is not None]
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	#fourcc = cv2.VideoWriter_fourcc(*'avc1')
	height, width, layers = vid_frames[0].shape
	size = (width,height)
	out_vid = out_dir + "/embryo.mp4"
	out = cv2.VideoWriter(out_vid,fourcc, fps, size)
	for i in range(len(vid_frames)):
		out.write(vid_frames[i])
	out.release()

	embryo = []
	#Start from first non-empty frame
	start_frame = next(x for x, frame in enumerate(norm_frames) if frame is not None)
	frame0 = norm_frames[start_frame]
	embryo.append(frame0)

	#Generate blank images for masking
	rows, cols, _ = frame0.shape
#	eye_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)
	heart_roi = np.zeros(shape=[rows, cols], dtype=np.uint8)

	#Process frame0
	old_cl, old_grey, old_blur = processFrame(frame0)
	f0_grey = old_grey.copy()

	#Detect eyes in all frames
#	eye_masks = [detectEyes(frame) for frame in norm_frames if frame is not None]

	#Combine all individual eye masks  
#	for img in eye_masks:
#		eye_mask = cv2.add(eye_mask, img)

	#Numpy matrix: 
	#Coord 1 = row(s)
	#Coord 2 = col(s)

	#Detect heart region (and possibly blood vessels) 
	#by determining the differences across windows of frames
	j = start_frame + 1
	while j < len(norm_frames):

		frame = norm_frames[j]

		if frame is not None:

			#masked_frame, triangle_thresh = rolling_diff(j, norm_frames, win_size = 3, direction = "reverse", min_area = 250)
#			masked_frame, triangle_thresh = rolling_diff(j, norm_frames, win_size = 3, direction = "reverse", min_area = 150)
			masked_frame, triangle_thresh = rolling_diff(j, norm_frames, win_size = 2, direction = "reverse", min_area = 150)

			heart_roi = cv2.add(heart_roi, triangle_thresh)
			embryo.append(masked_frame)

		else:
			embryo.append(None)
		j += 1

	#Get indices of N most changeable pixels
	top_pixels = 250
	changeable_pixels = np.unravel_index(np.argsort(heart_roi.ravel())[-top_pixels:], heart_roi.shape)

	#Create boolean matrix the same size as the RoI image
	maxima = np.zeros((heart_roi.shape), dtype=bool)

	#Label pixels based on based on the top changeable pixels
	maxima[changeable_pixels] = True
	label_maxima = label(maxima)

	#Round of opening before thresholding
	heart_roi = cv2.morphologyEx(heart_roi, cv2.MORPH_OPEN, kernel)

	#Threshold heart RoI to generate mask
	yen = threshold_yen(heart_roi)
	heart_roi_clean = heart_roi > yen
	heart_roi_clean = heart_roi_clean.astype(np.uint8)

	#Filter mask based on area of contours
	#heart_roi_clean = filterMask(mask = heart_roi_clean, min_area = 500)
	#heart_roi_clean = filterMask(mask = heart_roi_clean, min_area = 300)
	heart_roi_clean = filterMask(mask = heart_roi_clean, min_area = 150)

	#Scikit image to opencv
	#cv_image = img_as_ubyte(any_skimage_image)
	#Opencv to Scikit image
	#image = img_as_float(any_opencv_image)

	#Find contours based on thresholded, summed absolute differences
	contours, _ = cv2.findContours(heart_roi_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	rows, cols = heart_roi_clean.shape

	out_fig = out_dir + "/embryo.frame_diff.png"
	plt.imshow(heart_roi)
	plt.savefig(out_fig,bbox_inches='tight')
	plt.close()

	#Filter contours based on which overlaps with the most changeable pixels
	final_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)
	img = np.zeros(shape=[rows, cols], dtype=np.uint8)
	mask_contours = []
	for i in range(len(contours)):

		#Contour to test
		test_contour = contours[i]

		rect = cv2.minAreaRect(test_contour)

		#Create blank mask
		contour_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)

		#Calculate overlap with the most changeable pixels i.e. max_opening
		#Draw and fill-in filtered contours on blank mask (contour has to be in a list)
		cv2.drawContours(contour_mask, [test_contour], -1, (255), thickness = -1)

		contour_pixels = (contour_mask / 255).sum()

		#Find overlap between contoured area and the N most changeable pixels
		overlap = np.logical_and(maxima, contour_mask)
		overlap_pixels = overlap.sum()

		#Calculate ratio between area of intersection and contour area 
		pixel_ratio = overlap_pixels / contour_pixels
#		print("contour pixels", contour_pixels)
#		print("overlap pixels", overlap_pixels)
#		print("pixel ratio", pixel_ratio)

		contour_area =  cv2.contourArea(test_contour)
		
		#Calculate minimum area parallelogram that encloses the contoured area
	        #centre, size, angle = cv2.minAreaRect(test_contour)
		#(x, y), (width, height), angle = cv2.minAreaRect(test_contour)
		rect = cv2.minAreaRect(test_contour)
		_, (l1, l2), _ = rect

		#Take the longer length to be the height
		if l1 >= l2:
			height = l1
			width = l2
		else:
			height = l2
			width = l1

		rect_area = width * height

		#Ratio of contoured area to the area of the minimal-sized rectangle enclosing it.
		area_ratio = contour_area / rect_area

		#Ration of width to height
		aspect_ratio = float(width) / float(height)

		#TODO
		#Take all regions that overlap with the the >=20% of the N most changeable pixels
#		if overlap_pixels >= (top_pixels * 0.2):
#		if overlap_pixels >= (top_pixels * 0.25):
		if overlap_pixels >= (top_pixels * 0.4):

			mask_contours.append(test_contour)
			final_mask = cv2.add(final_mask, contour_mask)

		#Compare ratio of areas for contour to rectangle
		#Determines how much the contoured area fills the parallelogram
#		area_ratio = contour_area / rect_area 

#		if area_ratio >= 0.5:
			
#			mask_contours.append(test_contour)	
#			final_mask = cv2.add(final_mask, contour_mask)

	#Check if heart region was detected, i.e. if sum(masked) > 0
	if final_mask.sum() > 0:  
		mask = final_mask
		#Overlay points on RoI
		overlay = color.label2rgb(label_maxima, image = mask, alpha=0.7, bg_label=0, bg_color=None, colors=[(1, 0, 0)])

		out_fig = out_dir + "/embryo_heart_roi.png"

		fig, ax = plt.subplots(2, 2,figsize=(15, 15))
		#First  frame
		#ax[0, 0].imshow(norm_frames[start_frame])
		ax[0, 0].imshow(f0_grey,cmap='gray')
		ax[0, 0].set_title('Embryo',fontsize=10)
		ax[0, 0].axis('off')
		#Summed Absolute Difference
		ax[1, 0].imshow(heart_roi)
		ax[1, 0].set_title('Summed Absolute\nDifferences', fontsize=10)
		ax[1, 0].axis('off')
		#Heart RoI
		ax[0, 1].imshow(heart_roi_clean)
		ax[0, 1].set_title('Thresholded Differences', fontsize=10)
		ax[0, 1].axis('off')
		#Heart RoI Contour
		ax[1, 1].imshow(overlay)
		ax[1, 1].set_title('Heart RoI with Pixel Maxima', fontsize=10)
		ax[1, 1].axis('off')

		plt.savefig(out_fig,bbox_inches='tight')
		plt.close()

		#Signal standard deviation
		#stds = {}
		#Coefficient of variation
		cvs = {}
		times = []

		timestamp0 = sorted_times[0]

		#Draw contours of heart
		for i in range(len(embryo)):

			raw_frame = sorted_frames[i]
			frame = embryo[i]

			if raw_frame is not None:
				masked_data = cv2.bitwise_and(raw_frame, raw_frame, mask=mask)
				masked_grey = cv2.cvtColor(masked_data, cv2.COLOR_BGR2GRAY)

				#Create vector of the signal within the region of the heart
				#Flatten array (matrix) into a vector
				heart_values = np.ndarray.flatten(masked_grey)
				#Remove zero elements
				heart_values = heart_values[np.nonzero(heart_values)]

				# Mean signal in region
				heart_mean = np.mean(heart_values)
				#Standard deviation for signal in region
				heart_std = np.std(heart_values)
				#Coefficient of variation
				heart_cv =  heart_std / heart_mean

#				cropped_frame = masked_data.copy()[y2 : y2 + h2, x2 : x2 + w2]

				# split source frame into B,G,R channels
				b,g,r = cv2.split(frame)

				# add a constant to B (blue) channel to highlight the heart
				b = cv2.add(b, 100, dst = b, mask = mask, dtype = cv2.CV_8U)

				# add a constant to R (red) channel to highlight the eyes
				#r = cv2.add(r, 100, dst = r, mask = eye_mask, dtype = cv2.CV_8U)

				masked_frame = cv2.merge((b, g, r))

				#################################

			#No signal in heart RoI if the frame is empty
			else:
				masked_frame = None
				heart_std = np.nan
				heart_cv = np.nan

			embryo[i] = masked_frame

			frame_num = i + 1
#			stds[frame_num] = heart_std
			cvs[frame_num] = heart_cv

			#Calculate time between frames
			#Time frame 1 = 0 secs
			if i == 0:
				time = 0
				time_elapsed = 0

				if masked_frame is not None:
					#Save first frame with the ROI highlighted
					out_fig =  out_dir + "/masked_frame.png"
					cv2.imwrite(out_fig,masked_frame)

			#Time frame i = (frame i - frame i-1) / 1000
			#Total Time frame i = (frame i - frame 0) / 1000
			else:
				timestamp  = sorted_times[i]
				old_timestamp  = sorted_times[i-1]
				#Time between frames in seconds
				time = (timestamp - old_timestamp ) / 1000
				#Total time elapsed in seconds
				time_elapsed = (timestamp - timestamp0) / 1000

			times.append(time_elapsed)

		#Write video
		out_vid = out_dir + "/embryo_changes.mp4"
		vid_frames = [i for i in embryo if i is not None]
		height, width, layers = vid_frames[0].shape
		size = (width,height)
		out2 = cv2.VideoWriter(out_vid,fourcc, fps, size)
		for i in range(len(vid_frames)):
			out2.write(vid_frames[i])
		out2.release()

		############################
		#Quality control heart rate estimate
		############################
		# Min and max bpm from Jakob paper
		#Only consider bpms (i.e. frequencies) less than 300 and greater than 60
		minBPM = 60 # 1 hz
		maxBPM = 300 # 5 hz

		times = np.asarray(times, dtype=float)
		#y = np.asarray(list(stds.values()), dtype=float)
		y = np.asarray(list(cvs.values()), dtype=float)

		#Get indices of na values
		na_values = np.isnan(y)
		empty_frames = [i for i, x in enumerate(na_values) if x]

		frame2frame = 1 / fps

		#Time domain
		increment = np.mean(np.diff(times)) / 6
		td = np.arange(times[0], times[-1] + increment, increment)

		#No filtering needed for interpolation if no empty frames
		if len(empty_frames) == 0:
			y_final = y.copy()
			cs = CubicSpline(times, y)

		#Filter out missing signal values before interpolation
		else:
			#Remove NaN values from signal and time domain for interpolation
			y_filtered = y.copy()
			y_filtered = np.delete(y_filtered, empty_frames)
			times_filtered = times.copy()
			times_filtered = np.delete(times_filtered, empty_frames)

			#Perform cubic spline interpolation to calculate in missing values
			cs = CubicSpline(times_filtered, y_filtered)

			#Replace NaNs with interpolated values
			y_final = y.copy()
			y_final[empty_frames] = cs(times[empty_frames])

		meanY = np.mean(y_final)

		#Write Signal to file
		out_signal = out_dir + "/medaka_heart.signal_CoV.txt"
		with open(out_signal, 'w') as output:
	
			output.write("Time" + "\t" + "Signal (CoV)" + "\n")

			for i in range(len(times)):
				time = times[i]
				signal = y_final[i]
				output.write(str(time) + "\t" + str(signal) + "\n")

		#Calculate slope
		#Presumably should be flat(ish) if good
		#Or fit line
		slope, intercept, r_value, p_value, std_err = stats.linregress(times, y_final)

		mad = stats.median_absolute_deviation(y_final)

#		if np.float64(p_value) < np.float_power(10, -8):
#			sig = "sig"
#		else:
#			sig = "no"

#		out_file = out_dir + "/signal.stats.txt"
#		with open(out_file, 'w') as output:

#			output.write("well\tmad\tslope\tp_value\tr_value\n")
#			output.write(well_number + "\t" + str(mad) + "\t" + str(slope) + "\t" + str(p_value) + "\t" + str(r_value) + "\t" + sig + "\n")

		#prominent_peaks = [idx for idx, prominence in enumerate(prominences[0]) if prominence > 0.2]

		#Peak Calling
		peaks, _ = find_peaks(cs(td), height = np.mean(cs(td)))
		#Peak prominence
#		prominences = peak_prominences(cs(td), peaks)
#		prominent_peaks = [idx for idx, prominence in enumerate(prominences[0])]


		out_fig = out_dir + "/bpm_trace.png"
		plt.plot(td, cs(td))
		plt.plot(td[peaks], cs(td)[peaks], "x")
		plt.ylabel('Heart intensity (CoV)')
		plt.xlabel('Time [sec]')
		plt.hlines(y = np.mean(cs(td)), xmin = td[0], xmax = td[-1], linestyles = "dashed")
		plt.savefig(out_fig,bbox_inches='tight')
		plt.close()

		#Filter out if linear regression captures signal trend well
		#(i.e. if p-value highly significant)
		if (np.float64(p_value) > np.float_power(10, -8)) or (mad <= 0.02) or (np.absolute(slope) <= 0.002):

			#Detrend and normalise cubic spline interpolated data
#			norm_cs = detrendSignal(cs,td)

			#Root mean square of successive differences
			#first calculating each successive time difference between heartbeats in ms. Then, each of the values is squared and the result is averaged before the square root of the total is obtained
#			rmssd = getRMSSD(peak_times)

			#Heart range in Hz
			heart_range = (0.5, 6)

			#Perform Fourier Analysis 
			psd, freqs, peak, bpm_fourier = fourierHR(cs, td, heart_range)
	
			if bpm_fourier is not None:	
				#Round heart rate
				bpm_fourier = np.around(bpm_fourier, decimals=2)

			#Plot full. one-sided Fourier Transform 
			ax = plotFourier(psd = psd, freqs = freqs, peak = None, bpm = bpm_fourier, heart_range = None, figure_loc = 211)
			ax.set_title("Power spectral density of HRV")

			#Plot one-sided Fourier within specified range
			ax = plotFourier(psd = psd, freqs = freqs, peak = peak, bpm = bpm_fourier, heart_range = heart_range, figure_loc = 212)
			plt.xlabel('Frequency (Hz)')
		
			out_fourier = out_dir + "/bpm_power_spectra.fourier.png"
			plt.savefig(out_fourier)#, bbox_inches='tight')
			plt.close()

			#Welch's Method for spectral analysis
#			bpm_welch =  welchHR(interpolated_signal = cs, time_domain = td, heart_range = (1, 6))
#			bpm_welch = np.around(bpm_welch, decimals=2)
#			ax = plotFourier(psd = psd, freqs = freqs, peak = None, bpm = bpm_fourier, heart_range = None, figure_loc = 211)

#			if bpm_welch < 300 and bpm_welch > 60:
#				bpm_label2 = "BPM = " +  str(int(bpm_welch))
#			else:
#				bpm_label2 = "BPM = " +  str(int(bpm_welch)) + " (unreliable)"	

#			out_fig3 = out_dir + "/bpm_power_spectra.welch.png"

#			fig, [ax1,ax2] = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
			#Plot all power spectra
#			ax1.semilogx(f, p_final)
			##ax1.semilogx(peaks_values, prominent_values, "x")
#			ax1.semilogx(peaks_values2, prominent_values2, "x")
#			ax1.set_ylabel('Power Spectrum (dB/Hz)')

#			ax2.plot(f, p_final)
#			ax2.plot(peaks_values2, prominent_values2, "x")
##			ax2.plot(f[heart_freq][heart_peak], p_final[heart_freq][heart_peak], "x") #Peak
#			ax2.set_xlim((0.75, 6))
#			ax2.set_ylim(ylims)        
#			ax2.vlines(x=f[heart_freq][heart_peak], ymin=ylims[0], ymax=p_final[heart_freq][heart_peak], linestyles = "dashed")
#			ax2.hlines(y=p_final[heart_freq][heart_peak], xmin=0.75, xmax=f[heart_freq][heart_peak], linestyles = "dashed")
#			ax2.set_title(bpm_label2, loc='right')

#			fig.suptitle("Power spectral density of HRV")
#			plt.xlabel('Frequency (Hz)')
#			plt.ylabel('Power Spectrum (dB/Hz)')

#			plt.savefig(out_fig3)
#			plt.close()

			#Write bpm estimates to file
			out_file = out_dir + "/heart_rate.txt"
			with open(out_file, 'w') as output:
	
				output.write("well\twell_id\tbpm\n")
				output.write(well_number + "\t" + well + "\t" +  str(bpm_fourier) + "\n")

#		else:
#			out_file = out_dir + "/heart_rate.txt"
			#Write bpm to file
#			with open(out_file, 'w') as output:

#				output.write("well\twell_id\tbpm\tnote\n")
#				output.write(well_number + "\t" + well + "\tNA\tsignal_issue\n")
#	else:
#		out_file = out_dir + "/heart_rate.txt"
		#Write bpm to file
#		with open(out_file, 'w') as output:

#			output.write("well\twell_id\tbpm\tnote\n")
#			output.write(well_number + "\t" + well + "\tNA\tno_heart_roi\n")


#else:
#	out_file = out_dir + "/heart_rate.txt"
	#Write bpm to file
#	with open(out_file, 'w') as output:

#		output.write("well\twell_id\tbpm\tnote\n")
#		output.write(well_number + "\t" + well + "\tNA\tempty_frames\n")

#Welch’s method [R145] computes an estimate of the power spectral density by dividing the data into overlapping segments, computing a modified periodogram for each segment and averaging the periodograms.
#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html


#Issues 
#Heart obscured and picks up blood vessels
#differences in illumination between frames (flickering)
