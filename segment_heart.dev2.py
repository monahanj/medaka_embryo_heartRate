#!/usr/bin/env python3

import argparse
import pandas as pd
import os
import glob2
import numpy as np
from numpy.fft import fft, hfft, fftfreq, fftshift 
import cv2
#import imutils
import skimage
from skimage.metrics import structural_similarity
from skimage.util import compare_images, img_as_ubyte, img_as_float
from skimage.filters import threshold_triangle, threshold_local, threshold_otsu, roberts, sobel, scharr
from skimage.morphology import watershed
from skimage.feature import peak_local_max, canny
import scipy
from scipy import ndimage as ndi
from scipy.signal import find_peaks, peak_prominences
import tifffile
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from collections import Counter
import pywt
#import tiffstack2avi #convert tiff stack to avi video

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='Read in medaka heart video frames')
parser.add_argument('-i','--indir', action="store",dest='indir', help='Directory containing tiffs', default=False, required = True)
parser.add_argument('-w','--well', action="store",dest='well', help='Well Id', default=False, required = True)
parser.add_argument('-l','--loop', action="store",dest='loop', help='Well frame acquistion loop', default=False, required = True)
parser.add_argument('-o','--out', action="store",dest='out', help='Output', default=False, required = False)
args = parser.parse_args()

indir = args.indir
well_id = args.well
loop = args.loop
out_dir = args.out

#Make output dir if doesn't exist
try:
	os.makedirs(out_dir)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise

#well_loop1 = glob2.glob(indir + '/' + well_id + '*LO001*.tif')
#well_loop2 = glob2.glob(indir + '/' + well_id + '*LO002*.tif')
well_frames = glob2.glob(indir + '/' + well_id + '*' + loop + '*.tif')

# Improve contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

#Kernel for image smoothing
kernel = np.ones((5,5),np.uint8)

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
	#cl = clahe.apply(frame_grey)

	#Blur the CLAHE frame
	blurred_frame = cv2.GaussianBlur(cl, (21, 21), 0)
	#out_frame = cv2.GaussianBlur(old_grey, (5,5), 0)

	return out_frame, frame_grey, blurred_frame

#Detect embryo in well by identifying the largest circle detected by Hough circle algorithm
#def detectEmbryo(frame):

#	return coords

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

	#Put in otsu also?

	#Triangle thresholding on differences
	triangle = threshold_triangle(abs_diff)
	thresh = abs_diff > triangle
	thresh = thresh.astype(np.uint8)

	dilation = cv2.dilate(thresh,kernel,iterations = 2)
	closing = cv2.erode(dilation,kernel,iterations = 1)
	#Remove noise from thresholded differences by opening (erosion followed by dilation)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel)
#	closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel)

        #Find contours based on thresholded frame
#	contours, hierarchy = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	#Filter contours based on their area
	filtered_contours = []
	for i in range(len(contours)):
		contour = contours[i]

		#Filter contours by their area
		if cv2.contourArea(contour) >= min_area:
			filtered_contours.append(contour)

	#Create blank mask
	rows, cols = opening.shape
#	rows, cols = thresh.shape
	mask = np.zeros(shape=[rows, cols], dtype=np.uint8)

	#Draw and fill-in filtered contours on blank mask
	cv2.drawContours(mask, filtered_contours,-1, 255, thickness = -1)
	
	#Mask frame
	masked_frame = maskFrame(frame,mask)

	#Return the masked frame, the filtered mask and the absolute differences for the 2 frames
	return masked_frame, mask, abs_diff, thresh 

print("Reading in frames")

#Read all images for a given well
#>0 returns a 3 channel RGB image. This results in conversion to 8 bit.
#0 returns a greyscale image. Also results in conversion to 8 bit.
#<0 returns the image as is. This will return a 16 bit image.
imgs = {}
imgs_meta = {}  
sizes = {}
crop_params = {}
for tiff in well_frames:

	#Tiff names.... :(
	#Extract info from the (excessively) verbose tiff file names and create a pandas df
	#WE00048---D001--PO01--LO002--CO6--SL037--PX32500--PW0080--IN0010--TM280--X014463--Y038077--Z223252--T0016746390.tif
	tiff_name = os.path.basename(tiff)
	tiff_name = tiff_name.split(".")[0]
	tiff_name = tiff_name.split("---")
	well = tiff_name[0]
	frame_details = tiff_name[1].split("--")

	frame = frame_details[4]
	loop = frame_details[2]
	time = frame_details[-1]
	#Drop 'T' from time stamp
	time = float(time[1:])

	name = (well, loop, frame)
	frame_details = [well] + frame_details
	well_id = (well, loop)

	#If frame is not empty, read it in
	if not os.stat(tiff).st_size == 0:

		#Read image in colour
		img = cv2.imread(tiff,1)
		imgs[name] = img
#		img_out = img.copy()

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

	#If tiff empty
	else:
		print(tiff + " is empty!")

	#Make dict based on the file data fields
	try:
		imgs_meta['tiff'].append(tiff)
		imgs_meta['well'].append(frame_details[0])
#		imgs_meta['D'].append(frame_details[1])
#		imgs_meta['PO'].append(frame_details[2])
		imgs_meta['loop'].append(frame_details[3])
#		imgs_meta['CO'].append(frame_details[4])
		imgs_meta['frame'].append(frame_details[5])
#		imgs_meta['PX'].append(frame_details[6])
#		imgs_meta['PW'].append(frame_details[7])
#		imgs_meta['IN'].append(frame_details[8])
#		imgs_meta['TM'].append(frame_details[9])
#		imgs_meta['X'].append(frame_details[10])
#		imgs_meta['Y'].append(frame_details[11])
#		imgs_meta['Z'].append(frame_details[12])
		imgs_meta['time'].append(time)
#		imgs_meta['time'].append(frame_details[13])

	except KeyError:
		imgs_meta['tiff'] = [tiff]
		imgs_meta['well'] = [frame_details[0]]
#		imgs_meta['D'] = [frame_details[1]]
#		imgs_meta['PO'] = [frame_details[2]] 
		imgs_meta['loop'] = [frame_details[3]]
#		imgs_meta['CO'] = [frame_details[4]]
		imgs_meta['frame'] = [frame_details[5]]
#		imgs_meta['PX'] = [frame_details[6]]
#		imgs_meta['PW'] = [frame_details[7]]
#		imgs_meta['IN'] = [frame_details[8]] 
#		imgs_meta['TM'] = [frame_details[9]]
#		imgs_meta['X'] = [frame_details[10]]
#		imgs_meta['Y'] = [frame_details[11]]
#		imgs_meta['Z'] = [frame_details[12]]
		imgs_meta['time'] = [time]
#		imgs_meta['time'] = [frame_details[13]]

	crop_img = img[y1:y2, x1:x2]
	crop_size = (x2 - x1) * (y2 - y1)
	crop_id = (well, loop, crop_size)
	crop_params[crop_id] = [x1, y1, x2, y2] 

	try:
		sizes[well_id].append(crop_size)

	except KeyError:			
		sizes[well_id] = [crop_size]

#Save original frame with embryo highlighted with a circle
img_out = img.copy()
# Draw the center of the circle
cv2.circle(img_out,(circle[0],circle[1]),2,(0,255,0),3)
# Draw the circle
cv2.circle(img_out,(circle[0],circle[1]),circle[2],(0,255,0),2)
out_fig = out_dir + "/embryo.uncropped.png"
plt.imshow(img_out)
plt.savefig(out_fig)
plt.close()

#Threshold based on size of cropped image (will add if necessary eventually)

print("Cropping frames")
#Uniformly crop image per loop per well based on the circle radii + offset
well_crop_params = {}
for well_id in sizes.keys():

	size = sizes[well_id]
	dimension_counts = Counter(size)
	frame_size, _  = dimension_counts.most_common(1)[0]
	crop_id = well_id + (frame_size,)
	well_crop_params[well_id] = crop_params[crop_id]

# creating a dataframe from a dictionary 
imgs_meta = pd.DataFrame(imgs_meta)
#Sort by well, loop and frame 
imgs_meta = imgs_meta.sort_values(by=['well','loop','frame'], ascending=[True,True,True])
#Reindex pandas df
imgs_meta = imgs_meta.reset_index(drop=True)

loop_meta = imgs_meta[imgs_meta['loop'] == loop]
#Reindex pandas df
loop_meta = loop_meta.reset_index(drop=True)

#Frames per loop
#('WE00048', 'LO001', 'SL117')
loop_imgs = {}
loop_times = []
loop_tiffs = []
sorted_frames = []
sorted_times = []
for index,row in loop_meta.iterrows():

	tiff = row['tiff']
	well = row['well']
	loop = row['loop']
	frame = row['frame']
	time = row['time']
	name = (well, loop, frame)
	img = imgs[name]
	loop_tiffs.append(tiff)
	sorted_times.append(time)

	well_id = (well, loop)
	#Well and loop specific paramters for cropping frame
	#crop_params[crop_id] = [x1, y1, x2, y2]
	crop_values = well_crop_params[well_id]

	#crop_img = img[y1 : y2, x1: x2]
	crop_img = img[crop_values[1] : crop_values[3], crop_values[0] : crop_values[2]]

	loop_imgs[frame] = crop_img
	sorted_frames.append(crop_img)
	height, width, layers = crop_img.shape
	size = (width,height)

#image_test = tifffile.TiffSequence('170414181030_OlE_ICF2_HR_28C_2x/WE00048---D001--PO01--LO002*.tif', pattern='axes')
#print(image_test.shape)

#total time acquiring frames in seconds
#timestamp0 = sorted_times[0]
#timestamp_final = sorted_times[-1]

#total_time = (timestamp_final - timestamp0) / 1000 
#fps = len(sorted_times) / round(total_time)

fps = 13 #will be 30 in final dataset 
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_vid = out_dir + "/embryo.avi"
out = cv2.VideoWriter(out_vid,fourcc, fps, size)
for i in range(len(sorted_frames)):
	out.write(sorted_frames[i])
out.release()


print("Detecting heart RoI")

#Go through frames in stack to determine changes between seqeuential frames (i.e. the heart beat)
#Determine absolute differences between current and previous frame
#Use this to determine range of movement (i.e. the heart region)

embryo = []
frame0 = sorted_frames[0]
embryo.append(frame0)

#Process frame0
old_cl, old_grey, old_blur = processFrame(frame0)

rows, cols, _ = frame0.shape
heart_roi = np.zeros(shape=[rows, cols], dtype=np.uint8)

#Numpy matrix: 
#Coord 1 = row(s)
#Coord 2 = col(s)

#Detect heart region (and possibly blood vessels)
#Frame j vs. j - 1
j = 1
while j < len(sorted_frames):

	frame = sorted_frames[j]
	timestamp  = sorted_times[j]
	old_timestamp  = sorted_times[j-1]
	
	frame_cl, frame_grey, frame_blur = processFrame(frame)

	#Determine absolute differences between current and previous frame
	##Frame j vs. j - 1
#	masked_frame, opening, abs_diff, thresh = diffFrame(frame_blur,old_blur) 
	masked_frame, opening, abs_diff, thresh = diffFrame(frame_blur,old_blur) 
	heart_roi = cv2.add(heart_roi, thresh) 

#	images2 = [frame,frame_cl, frame_blur,abs_diff]
#	for i in range(len(images2)):
#		plt.subplot(2,2,i+1),plt.imshow(images2[i])
#		plt.xticks([]),plt.yticks([])
#	plt.savefig("/tmp/test.png")
#	plt.close()

	embryo.append(masked_frame)

	# Update the data for the next comparison(s)
#	old_grey = frame_grey.copy()
	old_blur = frame_blur.copy()

	j += 1

heart_roi_clean = cv2.threshold(heart_roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
#Fill in empty regions (if any) in heart mask
heart_roi_clean = img_as_ubyte(ndi.binary_fill_holes(heart_roi_clean))

#Scikit image to opencv
#cv_image = img_as_ubyte(any_skimage_image)
#Opencv to Scikit image
#image = img_as_float(any_opencv_image)

#elevation_map = sobel(heart_roi)

#Find contours based on thresholded, summed absolute differences
contours, hierarchy = cv2.findContours(heart_roi_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#Take largest contour to be heart
contour = max(contours, key = cv2.contourArea)

#Create blank mask
rows, cols = heart_roi_clean.shape
mask = np.zeros(shape=[rows, cols], dtype=np.uint8)
mask_contour = np.zeros(shape=[rows, cols], dtype=np.uint8)
#Draw and fill-in filtered contours on blank mask (contour has to be in a list)
cv2.drawContours(mask, [contour], -1, (255),thickness = -1)
#Draw contour without fill 
cv2.drawContours(mask_contour, [contour], -1, (255), 3)

#Expand masked region
mask2 = cv2.dilate(mask,kernel, iterations = 1)
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
roi_imgs = [heart_roi,heart_roi_clean, mask_contour]

fig, ax = plt.subplots(2, 2)
#First  frame
ax[0, 0].imshow(sorted_frames[0])
ax[0, 0].set_title('Embryo',fontsize=10)
ax[0, 0].axis('off')
#Summed Absolute Difference
ax[1, 0].imshow(heart_roi)
ax[1, 0].set_title('Summed Absolute\nDifferenceis', fontsize=10)
ax[1, 0].axis('off')
#Heart RoI
ax[0, 1].imshow(heart_roi_clean)
ax[0, 1].set_title('Thresholded Differences', fontsize=10)
ax[0, 1].axis('off')
#Heart RoI Contour
ax[1, 1].imshow(mask_contour)
ax[1, 1].set_title('Heart RoI Contour', fontsize=10)
ax[1, 1].axis('off')

#fig.delaxes(ax[1,1])

plt.savefig(out_fig,bbox_inches='tight')
plt.close()

from mpl_toolkits.mplot3d import axes3d
#out_fig = out_dir + "/test.png"
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_surface(heart_roi,cmap='viridis', edgecolor='none')
#plt.savefig(out_fig,bbox_inches='tight')
#plt.close()


print(heart_roi.ndim)
print(heart_roi.values)
#test = cv2.cvtColor(heart_roi, cv2.COLOR_BGR2GRAY)
#out_fig = out_dir + "/test.png"
#fig = plt.figure()
#plt.imshow(test)
#ax = plt.axes(projection='3d')
#ax.plot_surface(heart_roi,cmap='viridis', edgecolor='none')
#plt.savefig(out_fig,bbox_inches='tight')
#plt.close()
#print(heart_roi.dtype)
#print(type(heart_roi))



#XX,YY=np.meshgrid(xx,yy)
#ax3=figure.add_subplot(2,2,3,projection='3d')
#ax3.plot_surface(XX,YY,slab,rstride=4,cstride=4,cmap='viridis',alpha=0.8)



#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')



#for i in range(len(roi_imgs)):
#	plt.subplot(2,2,i+1),plt.imshow(roi_imgs[i])
#	plt.xticks([]),plt.yticks([])
#plt.savefig(out_fig)
#plt.close()

print(cv2.contourArea(contour))


print("Determining heart rate (bpm)") 
#Check that area of heart contour has reasonable size
if cv2.contourArea(contour) > 1000: 

	#sums =	{}
	stds = {}
	times = []

	timestamp0 = sorted_times[0]

#	distance = ndi.distance_transform_edt(heart_roi_clean)
#	local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),labels=heart_roi_clean)
#	markers = ndi.label(local_maxi)[0]
##	labels = watershed(-distance, markers, mask=heart_roi_clean)
#	labels = watershed(-distance, markers, mask=mask)

	heart = []
	heart_changes = []
	#Draw contours of heart
	for i in range(len(embryo)):

		raw_frame = sorted_frames[i]
		frame = embryo[i]
		masked_data = cv2.bitwise_and(raw_frame, raw_frame, mask=mask)
		masked_grey = cv2.cvtColor(masked_data, cv2.COLOR_BGR2GRAY)

		#raw_grey = cv2.cvtColor(raw_frame[y1 : y1 + h1, x1 : x1 + w1], cv2.COLOR_BGR2GRAY)
		# Improve contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
		#cl = clahe.apply(raw_grey)
		#cl2 = cv2.GaussianBlur(cl, (5, 5), 0)

		#Edge-detection
#		edges = canny(cl2)
#		edges2 = canny(cl2, sigma=3)
##		edges2 = canny(img_as_float(cl2))
##		edges2 = canny(img_as_float(cl2), sigma=3)
##		edge_roberts = roberts(img_as_float(cl2))
##		edge_sobel = sobel(img_as_float(cl2))
##		edge_scharr = scharr(img_as_float(cl2))

#		images2 = [cl,cl2,edges,edges2]
#		for i in range(len(images2)):
#			plt.subplot(2,2,i+1),plt.imshow(images2[i])
#			plt.xticks([]),plt.yticks([])
#		plt.show()

		#Create vector of the signal within the region of the heart
		#Flatten array (matrix) into a vector
		heart_values = np.ndarray.flatten(masked_grey)
		#Remove zero elements
		heart_values = heart_values[np.nonzero(heart_values)]

		#Sum the signal region
		heart_total = np.sum(heart_values)
		#Standard deviation for signal in region
		heart_std = np.std(heart_values)

#               images2 = [heart_roi,mask,masked_data,masked_grey]
#               for i in range(len(images2)):
#                       plt.subplot(2,2,i+1),plt.imshow(images2[i])
#                       plt.xticks([]),plt.yticks([])
#               plt.show()

		cropped_frame = masked_data.copy()[y2 : y2 + h2, x2 : x2 + w2]
#		cropped_frame = raw_frame.copy()[y : y+h, x : x +w]

		# split source frame into B,G,R channels
		b,g,r = cv2.split(frame)

		# add a constant to B (blue) channel to highlight the outline of the heart
		b = cv2.add(b, 100, dst = b, mask = mask, dtype = cv2.CV_8U)
		masked_frame = cv2.merge((b, g, r))

		#Crop based on bounding rectangle
		cropped_mask = masked_frame.copy()[y2 : y2 + h2, x2 : x2 + w2]

		#Draw bounding rectangle
#		cv2.rectangle(masked_frame,(x1,y1),(x1 + w1,y1 + h1),(0,255,0),2)

		embryo[i] = masked_frame
#		heart.append(cropped_frame)
		heart_changes.append(cropped_mask)

		#################################

		frame_num = i + 1
#		sums[frame_num] = heart_total
		stds[frame_num] = heart_std
#		mads[frame_num] = heart_mad

		#Calculate time between frames
		#Time frame 1 = 0 secs
		if i == 0:
			time = 0
			time_elapsed = 0

			#Create blank mask
#			test = raw_frame.copy()
#			inv_mask = cv2.bitwise_not(mask)
#			background = np.full(raw_frame.shape, 255, dtype=np.uint8)
#			bg = cv2.bitwise_or(background, background, mask = inv_mask)

#			fg = cv2.bitwise_and(raw_frame, raw_frame, mask = mask)

			#Save first frame with the ROI highlighted
			out_fig =  out_dir + "/masked_frame.png"
			cv2.imwrite(out_fig,masked_frame)
#			plt.show(masked_frame)
#			plt.show()
#			plt.savefig(out_fig)


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

	#cv2.rectangle(heart_roi_clean,(x1,y1),(x1 + w1,y1 + h1),255,2)

	out_vid = out_dir + "/heart_changes.avi"
	#Make videos
	height, width, layers = heart_changes[0].shape
	size = (width,height)
	out1 = cv2.VideoWriter(out_vid,fourcc, fps, size)
	for i in range(len(heart_changes)):
		out1.write(heart_changes[i])
	out1.release()

	out_vid = out_dir + "/embryo_changes.avi"
	height, width, layers = embryo[0].shape
	size = (width,height)
	out2 = cv2.VideoWriter(out_vid,fourcc, fps, size)
	for i in range(len(embryo)):
		out2.write(embryo[i])
	out2.release()

	#Wavelet Transformation
	#pywt.wavedec(sample, 'haar', 'smooth') 
	
	#label_objects, nb_labels = ndi.label(fill_coins)
	#>>> sizes = np.bincount(label_objects.ravel())
	#>>> mask_sizes = sizes > 20
	#>>> mask_sizes[0] = 0
	#>>> coins_cleaned = mask_sizes[label_objects]


	############################
	#Quality control heart rate estimate
	############################
	# Min and max bpm from Jakob paper
	#Only consider bpms (i.e. frequencies) less than 300 and greater than 60
	minBPM = 60
	maxBPM = 300

	times = np.asarray(times, dtype=float)
	x = np.asarray(list(stds.values()), dtype=float)
#	x2 = np.asarray(list(cvs.values()), dtype=float)
	#x2 = np.asarray(list(medians.values()), dtype=float)
	#x2 = np.asarray(list(mads.values()), dtype=float)
	meanX = np.mean(x)
	#meanX2 = np.mean(x2)
	#xnorm = x - meanX
	#xnorm2 = x2 - meanX2

	#Find peaks in heart ROI signal, peaks only those above the mean stdev
	#Minimum distance of 2 between peaks
	peaks, _ = find_peaks(x, height = meanX, distance = 2)
#	peaks2, _ = find_peaks(x2)


	#Peak prominence
	prominences = peak_prominences(x,peaks)
	out_fig = out_dir + "/bpm_prominences.png"
	contour_heights = x[peaks] - prominences
	plt.plot(x)
	plt.plot(peaks, x[peaks], "x")
	plt.vlines(x=peaks, ymin=contour_heights, ymax=x[peaks])
	plt.savefig(out_fig,bbox_inches='tight')
	plt.close()

	#print(contour_heights)
	print("Prominences")
	print(prominences)
	print("Peaks")
	print(peaks)
	print("Contours")
	print(contour_heights)

	#Only calculate if more than 4 beats are detected
	if len(peaks) > 4:

		#Root mean square of successive interbeat intervals between all successive heartbeats 
#    		#calculate RMSSD
		 
#        for b = 4:size(BeatToBeat,2)
#            % do RMSSD calculation
#            RMSSD(b-3) = sqrt(((BeatToBeat(b)-BeatToBeat(b-1))^2 ...
#                + (BeatToBeat(b-1)-BeatToBeat(b-2))^2 ...
#                + (BeatToBeat(b-2)-BeatToBeat(b-3))^2)/3);

#            % if RMSSD is higher than threshold its indicated in the
#            % arrhythmia array (with 1)
#            if RMSSD(b-3) > RmssdThresh
#                % 'unacceptable'
#                arrhythmia(b-3) = 1;

#            else
#                % 'acceptable';
#                arrhythmia(b-3) = 0;

  #          end
 #       end
#    end
#end



		#times = np.arange(x.size) / fps
		peak_times = times[peaks]
		peak_signal = x[peaks]

		#Find signal minima
		#Full heart beat will only include comlete minimum to maximum cycles
#		from scipy.signal import argrelextrema
		# for local minima
#		argrelextrema(x, np.less)

		# calculate beat-to-beat times
		#BeatToBeat = diff(timePeak);

		beats = len(peaks)
		#beats per second
		bps = beats / times[-1]
		bpm = bps * 60
		bpm = np.around(bpm) #, decimals=1)

		out_fig = out_dir + "/bpm_trace.png"

		if bpm < 300 and bpm > 60:
			bpm_label = "BPM = " +  str(int(bpm))
		else:
			bpm_label = "BPM estimate unreliable"

		plt.plot(times, x)
		plt.plot(peak_times, peak_signal, "x")
		plt.ylabel('Heart ROI intensity (stdev)')
		plt.xlabel('Time [sec]')
		#Label trace with bpm
		plt.title(bpm_label)
		plt.hlines(y = meanX, xmin = times[0], xmax = times[-1], linestyles = "dashed")
		#plt.show()
		plt.savefig(out_fig)
		plt.close()


print("QC for bpm estimate") 

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

############################

#image_sequence = tiff.imread(well_series)
#image_sequence.shape
#series1 = tiff.imread(well_series, series=1)
#series1.shape

#Fourier transform 


#FREQUENCYANALYSIS

#RMSSD = Root Mean Square of the Successive Differences between successive heartbeats


#flippedStdOfSegment = 2*mean(stdOfSegment)-stdOfSegment
#[bpm, RMSSD, arrhythmia, indPeak, timePeak] =... frequencyAnalysis( flippedStdOfSegment(startFFF:FFF), timeStamps(startFFF:FFF), peakDetectionThreshold, RmssdThresh);


#    % window and therefore add some zeros at the end of the signal if its
#    % to short, this would cause a step if the offset wasnt removed)
#    meanStd = mean(stdOfSegment{nrOfSegment});
#    FourierTransform = fft(stdOfSegment{nrOfSegment}-meanStd,1024);
#    twoSidedSpectrum = abs(FourierTransform/(frameRate*1024)); %signalLength);
#    singleSidedSpectrum = twoSidedSpectrum(1:size(twoSidedSpectrum,2)/2);
#    singleSidedSpectrum(2:end) = 2*singleSidedSpectrum(2:end);
#    % get corresponding frecuencies
#    frequencies = (0:size(singleSidedSpectrum,2)-1)/((size(singleSidedSpectrum,2)-1)*2)*frameRate*60;
    
#    % calculate features for classification
#    % get highest peak (+freq) within specified frequency range
#    [highestPeakInArea(nrOfSegment),bpm(nrOfSegment)] = max(singleSidedSpectrum(minFrequency:maxFrequency));
#%     % get range
#%     diffMaxMinFFT(nrOfSegment)= max(p1)-min(p1);
#%     % get main frequency
#%     [~,iMax] = max(singleSidedSpectrum);
#%     bpm(nrOfSegment) = frequencies(iMax);
#%     frequency(nrOfSegment) = bpm(nrOfSegment)/60;
    

#    %calculate frequency
#%     % with median
#%     medianBeatToBeat = median(BeatToBeat);
#%     frequency = 1/medianBeatToBeat;
#%     bpm = frequency*60;

#    % with mean
#    frequency = 1/mean(BeatToBeat);
#    bpm = frequency*60;
    
    
#%     %calculate std of beat-to-beat times
#%     stdBeatToBeat = std(BeatToBeat);
    
#    % calculate RMSSD for last 4 Beats
#    % only calculate if more than 4 beats are detected
#    if size(BeatToBeat,2) < 4
#        RMSSD = NaN;
#        arrhythmia = 2; % marker for 'not enough beats detected'
#    else
#        for b = 4:size(BeatToBeat,2)
#            % do RMSSD calculation
#            RMSSD(b-3) = sqrt(((BeatToBeat(b)-BeatToBeat(b-1))^2 ...
#                + (BeatToBeat(b-1)-BeatToBeat(b-2))^2 ...
#                + (BeatToBeat(b-2)-BeatToBeat(b-3))^2)/3);
            
#            % if RMSSD is higher than threshold its indicated in the
#            % arrhythmia array (with 1)
#            if RMSSD(b-3) > RmssdThresh
#                % 'unacceptable'
#                arrhythmia(b-3) = 1;
                
#            else
#                % 'acceptable';
#                arrhythmia(b-3) = 0;
               
  #          end
 #       end
#    end
#end

# Signal Processing
#Welch’s method [R145] computes an estimate of the power spectral density by dividing the data into overlapping segments, computing a modified periodogram for each segment and averaging the periodograms.
#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html


#Issues 
#Heart obscured and picks up blood vessels
#Empty frame
#Disconnected heart RoI sections
#differences in illumination between frames
