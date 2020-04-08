#!/usr/bin/env python3

import argparse
import pandas as pd
import os
import glob2
import errno

import numpy as np
from numpy.fft import fft, hfft, fftfreq, fftshift 
import cv2

import skimage
from skimage.util import compare_images, img_as_ubyte, img_as_float
from skimage.filters import threshold_triangle, threshold_otsu, threshold_yen, roberts, sobel, scharr
from skimage.measure import label#, find_contours 
from skimage.feature import peak_local_max, canny
from skimage import color

import scipy
from scipy import ndimage as ndi
from scipy.signal import find_peaks, peak_prominences#, find_peaks_cwt
from scipy.interpolate import CubicSpline

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

from collections import Counter

import pywt

np.set_printoptions(threshold=np.inf)

#Go through frames in stack to determine changes between seqeuential frames (i.e. the heart beat)
#Determine absolute differences between current and previous frame
#Use this to determine range of movement (i.e. the heart region)

parser = argparse.ArgumentParser(description='Read in medaka heart video frames')
parser.add_argument('-i','--indir', action="store", dest='indir', help='Directory containing frames', default=False, required = True)
parser.add_argument('-t','--format', action="store", dest='frame_format', help='Frame format', default='tiff', required = False)
parser.add_argument('-w','--well', action="store", dest='well', help='Well Id', default=False, required = True)
parser.add_argument('-l','--loop', action="store", dest='loop', help='Well frame acquistion loop', default=None, required = False)
parser.add_argument('-c','--crop', action="store", dest='crop', type=bool, help='Crop frame images?', default=True, required = False)
parser.add_argument('-f','--fps', action="store", dest='fps', help='Frames per second', default=False, required = False)
parser.add_argument('-o','--out', action="store", dest='out', help='Output', default=False, required = True)
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
	#Divide by max to try and remove flickering between frames
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

#Detect oil droplet(s) in embryo sac
def detectDroplet(frame):

	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#	plt.imshow(img,'gray')
#	plt.show()


	frame = img_as_float(frame)
	#edges = canny(frame, sigma=3)
	edges = sobel(frame)

	

#	plt.imshow(markers)
#	plt.axis('off')
#	plt.show()

	return(frame)

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
#	frame_grey = cv2.cvtColor(out_frame, cv2.COLOR_BGR2GRAY)
	frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Improve contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
#	cl = clahe.apply(frame_grey)

	#Convert CLAHE-normalised greyscale frame back to BGR
#	out_frame = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)

	#Blur the CLAHE frame
	#Blurring kernel numbers must be odd integers
#	blurred_frame = cv2.GaussianBlur(cl, (9, 9), 0) 
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

#Differences between two frames
def diffFrame(frame2, frame1):
	"""Calculate the abs diff between 2 frames and returns frame2 masked with the filtered differences."""

	min_area = 300

	#Absolute differnce between frames
	abs_diff = cv2.absdiff(frame2,frame1)
	#abs_diff = cv2.morphologyEx(abs_diff, cv2.MORPH_OPEN, kernel) 

	#Triangle thresholding on differences
	triangle = threshold_triangle(abs_diff)
	thresh = abs_diff > triangle
	thresh = thresh.astype(np.uint8)

	#Opening to remove noise
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        #Find contours based on thresholded frame
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	#Filter contours based on their area
	filtered_contours = []
	for i in range(len(contours)):
		contour = contours[i]

		#Filter contours by their area
		if cv2.contourArea(contour) >= min_area:
			filtered_contours.append(contour)

	#Create blank mask
	rows, cols = thresh.shape
	mask = np.zeros(shape=[rows, cols], dtype=np.uint8)

	#Draw and fill-in filtered contours on blank mask
	cv2.drawContours(mask, filtered_contours,-1, 255, thickness = -1)
	
	#Mask frame
	masked_frame = maskFrame(frame,mask)

	#Return the masked frame, the filtered mask and the absolute differences for the 2 frames
	#return masked_frame, mask, abs_diff, thresh, yen_thresh
#	return masked_frame, mask, thresh, yen_thresh
	return masked_frame, mask, thresh

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
				x1 = x1_test - 50 #100
			else:
				x1 = x1_test

			if y1_test > y1:
				y1 = y1_test - 50 #100
			else:
				y1 = y1_test

			#If rounded figures are less than x2 or y2, add 50 
			if x2_test < x2:
				x2 = x2_test + 50 #100
			else:
				x2 = x2_test

			if y2_test < y2:
				y2 = y2_test + 50 #100
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

	#Threshold based on size of cropped image (will add if necessary eventually)
	print("Cropping frames\n")
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

out_fig = out_dir + "/embryo.uncropped.png"
plt.imshow(img_out)
plt.axis('off')
plt.savefig(out_fig)
plt.close()

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

#Only process if less than 5 frames are empty
if sum(frame is not None for frame in sorted_frames) < 5:

	#Determine frame rate from time-stamps if unspecified 
	if args.fps:
		fps = int(args.fps)
	else:
		#total time acquiring frames in seconds
		timestamp0 = sorted_times[0]
		timestamp_final = sorted_times[-1]
	
		total_time = (timestamp_final - timestamp0) / 1000 
		fps = int(len(sorted_times) / round(total_time))

		#fps = 13 #will be 30 in final dataset 

	#Normalise intensities across frames if tiff images
	if frame_format == "tiff":
		#sorted_frames = normVideo(sorted_frames)
		norm_frames = normVideo(sorted_frames)
	#jpegs already normalised
	else:
		norm_frames = sorted_frames.copy()

	#Write video
	vid_frames = [frame for frame in norm_frames if frame is not None]
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	height, width, layers = vid_frames[0].shape
	size = (width,height)
	out_vid = out_dir + "/embryo.avi"
	out = cv2.VideoWriter(out_vid,fourcc, fps, size)
	for i in range(len(vid_frames)):
		out.write(vid_frames[i])
	out.release()

	embryo = []
	#Find first frame that exists
	#start_frame = next(x for x, frame in enumerate(sorted_frames) if frame is not None)
	#frame0 = sorted_frames[start_frame]
	start_frame = next(x for x, frame in enumerate(norm_frames) if frame is not None)
	frame0 = norm_frames[start_frame]
	embryo.append(frame0)

	#Generate blank images for masking
	rows, cols, _ = frame0.shape
	eye_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)
	heart_roi = np.zeros(shape=[rows, cols], dtype=np.uint8)
	heart_roi2 = np.zeros(shape=[rows, cols], dtype=np.uint8)

	#Process frame0
	old_cl, old_grey, old_blur = processFrame(frame0)

	#Detect eyes in all frames
	eye_masks = [detectEyes(frame) for frame in norm_frames if frame is not None]

	#droplet_masks = detectDroplet(frame0)

	#Combine all individual eye masks  
	for img in eye_masks:
		eye_mask = cv2.add(eye_mask, img)

	#Numpy matrix: 
	#Coord 1 = row(s)
	#Coord 2 = col(s)

	#Detect heart region (and possibly blood vessels)
	#Frame j vs. j - 1
	j = start_frame + 1
	while j < len(norm_frames):

#		frame = sorted_frames[j]
		frame = norm_frames[j]

		#Check if jth frame exists
		if frame is not None:
		
			frame_cl, frame_grey, frame_blur = processFrame(frame)

			#Only compare if both frames exist
			if frame0 is not None:

				#Determine absolute differences between current and previous frame
				#Frame j vs. j - 1
				masked_frame, opening, triangle_thresh = diffFrame(frame_blur,old_blur) 
				heart_roi = cv2.add(heart_roi, triangle_thresh) 
				embryo.append(masked_frame)

			else:
				embryo.append(None)

			# Update the data for the next comparison(s)
			old_blur = frame_blur.copy()

		#If frame empty
		else:
			embryo.append(None)

		j += 1

	#Get indices of N most changeable pixels
	changeable_pixels = np.unravel_index(np.argsort(heart_roi.ravel())[-250:], heart_roi.shape)

	#Create boolean matrix the same size as the RoI image
	maxima = np.zeros((heart_roi.shape), dtype=bool)

	#Label pixels based on based on the top changeable pixels
	maxima[changeable_pixels] = True
	label_maxima = label(maxima)

	#Threshold heart RoI to generate mask
	yen = threshold_yen(heart_roi)
	heart_roi_clean = heart_roi > yen
	heart_roi_clean = heart_roi_clean.astype(np.uint8)


	#Scikit image to opencv
	#cv_image = img_as_ubyte(any_skimage_image)
	#Opencv to Scikit image
	#image = img_as_float(any_opencv_image)

	#Find contours based on thresholded, summed absolute differences
	contours, hierarchy = cv2.findContours(heart_roi_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	#Filter contours based on which overlaps with the most changeable pixels
	overlap = 0
	for i in range(len(contours)):
		test_contour = contours[i]

		#Create blank mask
		rows, cols = heart_roi_clean.shape
		contour_mask = np.zeros(shape=[rows, cols], dtype=np.uint8)

		#Calculate overlap with the most changeable pixels i.e. max_opening
		#Draw and fill-in filtered contours on blank mask (contour has to be in a list)
		cv2.drawContours(contour_mask, [test_contour], -1, (255), thickness = -1)

		#Find intersection between contour and the top most changeable pixels
		intersection = np.logical_and(maxima, contour_mask)
		#Area of intersection = sum of pixels
		area_of_intersection = intersection.sum()

		#Calculate ratio between area of intersection and contour area 
		ratio = area_of_intersection / contour_mask.sum()	
		#ratio = area_of_intersection 

		if i == 0:
			mask = contour_mask
			overlap = area_of_intersection
			contour = test_contour

		elif ratio > overlap:
			mask = contour_mask
			overlap = area_of_intersection
			contour = test_contour

	#Create blank mask
	rows, cols = heart_roi_clean.shape
	mask_contour = np.zeros(shape=[rows, cols], dtype=np.uint8)
	#Draw contour without fill 
	cv2.drawContours(mask_contour, [contour], -1, (255), 3)

	#Overlay points on RoI
	overlay = color.label2rgb(label_maxima, image = mask, alpha=0.7, bg_label=0, bg_color=None, colors=[(1, 0, 0)])

	#Expand masked region
	mask2 = cv2.dilate(mask, kernel, iterations = 1)
	#Find contours based on thresholded, summed absolute differences
	contours2, hierarchy = cv2.findContours(mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	#Take largest contour to be heart
	contour2 = max(contours2, key = cv2.contourArea)

	#Calculate bounding rectangles
	(x1, y1, w1, h1) = cv2.boundingRect(contour)
	box1 = [x1,y1, x1+w1,y1+h1]
	(x2, y2, w2, h2) = cv2.boundingRect(contour2)
	#box2 = [x2,y2, x2+w2,y2+h2]

	#Plot heart RoI images
	out_fig = out_dir + "/embryo_heart_roi.png"

	fig, ax = plt.subplots(2, 2,figsize=(15, 15))
	#First  frame
	ax[0, 0].imshow(norm_frames[start_frame])
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

	print("Determining heart rate (bpm)\n") 

	#Signal standard deviation
	#stds = {}
	#Coefficient of variation
	cvs = {}
	times = []

	timestamp0 = sorted_times[0]

	heart_changes = []
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

			cropped_frame = masked_data.copy()[y2 : y2 + h2, x2 : x2 + w2]
#			cropped_frame = raw_frame.copy()[y : y+h, x : x +w]

			# split source frame into B,G,R channels
			b,g,r = cv2.split(frame)

			# add a constant to B (blue) channel to highlight the heart
			b = cv2.add(b, 100, dst = b, mask = mask, dtype = cv2.CV_8U)

			# add a constant to R (red) channel to highlight the eyes
			r = cv2.add(r, 100, dst = r, mask = eye_mask, dtype = cv2.CV_8U)

			masked_frame = cv2.merge((b, g, r))

			#Crop based on bounding rectangle
			cropped_mask = masked_frame.copy()[y2 : y2 + h2, x2 : x2 + w2]

			#Draw bounding rectangle
#			cv2.rectangle(masked_frame,(x1,y1),(x1 + w1,y1 + h1),(0,255,0),2)

			#################################

		#No signal in heart RoI if the frame is empty
		else:
			masked_frame = None
			cropped_mask = None
			heart_std = np.nan
			heart_cv = np.nan

		embryo[i] = masked_frame
		heart_changes.append(cropped_mask)

		frame_num = i + 1
#		sums[frame_num] = heart_total
#		stds[frame_num] = heart_std
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

	#times = np.arange(x.size) / fps

	#cv2.rectangle(heart_roi_clean,(x1,y1),(x1 + w1,y1 + h1),255,2)

	out_vid = out_dir + "/embryo_changes.avi"
	vid_frames = [i for i in embryo if i is not None]
	#height, width, layers = embryo[0].shape
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
	minBPM = 60
	maxBPM = 300

	times = np.asarray(times, dtype=float)
	#y = np.asarray(list(stds.values()), dtype=float)
	y = np.asarray(list(cvs.values()), dtype=float)

	#Get indices of na values
	na_values = np.isnan(y)
	empty_frames = [i for i, x in enumerate(na_values) if x]

	frame2frame = 1 / fps

	#No interpolation necessary if no empty frames
	elif len(empty_frames) == 0:
		y_final = y.copy()
		cs = CubicSpline(times, y)
	else:
		#Remove NaN values from signal and time domain
		y_filtered = y.copy()
		y_filtered = np.delete(y_filtered, empty_frames)
		times_filtered = times.copy()
		times_filtered = np.delete(times_filtered, empty_frames)

		#Perform cubic spline interpolation to calculate in missing values
		cs = CubicSpline(times_filtered, y_filtered)

		#plt.plot(times_filtered, cs(times_filtered), label="Interpolated Function")
		#plt.plot(times, y,'x', label="Real Values")
		#plt.plot(times[empty_frames], cs(times[empty_frames]), 'o', label='Interpolated missing value(s)')
		#plt.show()

		#Replace NaNs with interpolated values
		y_final = y.copy()
		
		print(empty_frames)
#       interpolated_value = cs(times[empty_frames[0]]
#		y_final = cs(times[empty_frames[0]] 

	meanY = np.mean(y_final)
	#xnorm = y_final - meanY

        #Fast fourier transform


	#Write Signal to file
	out_signal = out_dir + "/medaka_heart.signal_stdev.txt"
	with open(out_signal, 'w') as output:

		output.write("Time" + "\t" + "Signal (CoV)" + "\n")

		for i in range(len(times)):
			time = times[i]
			signal = y_final[i]
			output.write(str(time) + "\t" + str(signal) + "\n")

	#Find peaks in heart ROI signal, peaks only those above the mean stdev
	#Minimum distance of 2 between peaks
	peaks, _ = find_peaks(y_final, height = meanY)
	#peaks, _ = find_peaks(y)

	#Peak prominence
	prominences = peak_prominences(y_final, peaks)

	out_fig = out_dir + "/bpm_prominences.png"
	contour_heights = y[peaks] - prominences
	plt.plot(y_final)
	plt.plot(peaks, y_final[peaks], "x")
	plt.ylabel('Heart ROI intensity (CoV)')
	plt.vlines(x=peaks, ymin=contour_heights, ymax=y_final[peaks])
	plt.hlines(y = meanY, xmin = 0, xmax = len(y_final), linestyles = "dashed")
	plt.savefig(out_fig,bbox_inches='tight')
	plt.close()

#	print(prominences)
#	prominent_peaks = [idx for idx, prominence in enumerate(prominences[0]) if prominence > 0.2] 
	prominent_peaks = [idx for idx, prominence in enumerate(prominences[0])]

	#Only calculate if more than 4 beats are detected
	if len(peaks[prominent_peaks]) > 4:

		#peak_times = times[peaks]

		peak_times = times[peaks[prominent_peaks]]
		peak_signal = y[peaks[prominent_peaks]]

		beats = len(peaks[prominent_peaks])
		#beats per second
		bps = beats / times[-1]
		bpm = bps * 60
		bpm = np.around(bpm) #, decimals=1)

		out_fig = out_dir + "/bpm_trace.png"

		if bpm < 300 and bpm > 60:
			bpm_label = "BPM = " +  str(int(bpm))
		else:
			bpm_label = "BPM = " +  str(int(bpm)) + " (uneliable)"

		#Signal QC  

		out_file = out_dir + "/heart_rate.txt"
		#Write bpm to file
		with open(out_file, 'w') as output:
	
			output.write("well\twell_id\tbpm\n")

			output.write(well_number + "\t" + well + "\t" +  str(int(bpm)))

		plt.plot(times, y)
		plt.plot(peak_times, peak_signal, "x")
		plt.ylabel('Heart ROI intensity (CoV)')
		plt.xlabel('Time [sec]')

		#Label trace with bpm
		plt.title(bpm_label)
		plt.hlines(y = meanY, xmin = times[0], xmax = times[-1], linestyles = "dashed")
		plt.savefig(out_fig)
		plt.close()

		#Root mean square of successive differences
		#first calculating each successive time difference between heartbeats in ms. Then, each of the values is squared and the result is averaged before the square root of the total is obtained

		rmssd = getRMSSD(peak_times)

		#Calculate beat-to-beat times
		#(time between peaks)
		peak2peak = [t2 - t1 for t1, t2 in zip(peak_times, peak_times[1:])]

		#Square peak2peak
		peak2peak_sq = [i**2 for i in peak2peak]

		#Average squared peak2peak
		mean_peak2peak_sq = np.mean(peak2peak_sq)

		#Square root of Average squared peak2peak
		rmssd = np.sqrt(mean_peak2peak_sq)

		#for R–R interval time series 


#		RmssdThresh = 
#       	 % arrhythmia array (with 1)
#	        if RMSSD(b-3) > RmssdThresh
#       	      % 'unacceptable'
#	              arrhythmia(b-3) = 1;

#       	 else
#	            % 'acceptable';
#       	         arrhythmia(b-3) = 0;
	#TODO 
	#Fast fourier transform of cubic spline interpolation
	Fs = 1 / fps

#Exclude well if more than one empty frame
else:

	print("Too many empty frames")



#figure;
#Fs=round(1/mean(diff(td)));
#plot(td,data_interp);hold on;
#plot(td(min_locs),data_interp(min_locs),'rv','MarkerFaceColor','r','LineWidth',2)
#grid;ylim([7 8.2]);
#title(sprintf('HRV after interpolation'))

#[Psig,Fsig] = pwelch(data_interp, hanning(3*Fs), [], pow2(14), Fs,'onesided');
#figure;semilogx(Fsig,10*log10(Psig),'Color',[0, 0.4470, 0.7410],'LineWidth',2);
#xlabel('Frequency (Hz)','FontSize',12,'FontWeight','bold');ylabel('Power Spectrum (dB/Hz)','FontSize',12,'FontWeight','bold');
#grid
#title('Power spectral density of HRV')
#xlim([2 6]);


#	from scipy import signal

#	j = 0
	#Time between peaks
#	while j < len(peaks):

#		time1 = "foo" 
#		time2 = "bar"
#		j += 1
#	print(len(x))
#	print(fps)
#	freqs, times, Sxx = signal.spectrogram(x, fs=fps,nperseg = 20)
	#freqs, times, Sx = signal.spectrogram(x, fs=fps, window='hanning', nperseg=1024, noverlap=M - 100, detrend=False, scaling='spectrum')

#	print(times)
#	plt.pcolormesh(times, freqs, Sxx)
#	plt.ylabel('Frequency [Hz]')
#	plt.xlabel('Time [sec]')
#	plt.show()

#	f, ax = plt.subplots(figsize=(4.8, 2.4))
#	ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
#	ax.set_ylabel('Frequency [kHz]')
#	ax.set_xlabel('Time [s]');



#	Fs = fps
#	N = len(x)

#	L = N / rate

#	Ts = 1.0/Fs # sampling interval
#	n = len(x) # length of the signal
#	k = np.arange(n)
#	print(Ts)
#	t = np.arange(0,1,Ts) # time vector
#	print(t)
#	t2 = no.arange(times) / 10 # time vector
#	print(t2)

#	t3 = np.arange(0,n) * Ts
#	print(t3)

#	T = n/Fs
#	frq = k/T # two sides frequency range
#	frq = frq[range(int(n/2))] # one side frequency range
#	X = fft(x)/n # fft computing and normalization
#	X = X[range(int(n/2))]

#	plot(frq,abs(X),'r') # plotting the spectrum
#	xlabel('Freq (Hz)')
#	ylabel('|Y(freq)|')
#	plt.show()

	#ff = 5; # frequency of the signal
	#y = sin(2*pi*ff*t)
	#y = sin(2*pi*ff*t)
	#subplot(2,1,1)
#plot(t,y)
#xlabel('Time')
#ylabel('Amplitude')
#subplot(2,1,2)

#Fourier transform 


# Signal Processing
#Welch’s method [R145] computes an estimate of the power spectral density by dividing the data into overlapping segments, computing a modified periodogram for each segment and averaging the periodograms.
#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html


#Issues 
#Heart obscured and picks up blood vessels
#Empty frames
#Disconnected heart RoI sections
#differences in illumination between frames (flickering)
