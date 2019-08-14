import os
import csv
import argparse
import _pickle as pickle
from shutil import copyfile

import cv2
import numpy as np

import scipy
import scipy.ndimage as nd
from skimage import measure
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import moviepy
from moviepy.editor import ImageSequenceClip

class CropInfo(object):
	def __init__(self, r, Ma):
		self.r  = r
		self.Ma = Ma

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('video')
	parser.add_argument('--new_bg')
	parser.add_argument('--temp_img_dir')
	parser.add_argument('--vid_len')
	args = parser.parse_args()
	if args.new_bg=='False':
		new_bg=False
	else:
		new_bg=True
	return args.video, new_bg, args.temp_img_dir, args.vid_len

def get_bg_avg(video_address, Ma, r):

	cap = cv2.VideoCapture(video_address)
	_, frame =cap.read()
	cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
	dim = frame.shape[:-1]
	depth = int(np.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT)/300))
	blank = []
	frameIdxs = np.linspace(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-100), num=depth)
	frameIdxs = [int(e) for e in frameIdxs]

	for idx,i in enumerate(frameIdxs):
		print('{} % Done computing background'.format(int(100*(idx/(len(frameIdxs))))))
		cap.set(cv2.CAP_PROP_POS_FRAMES, i)
		_, frame = cap.read()

		rows,cols = frame.shape[:2]
		frame = cv2.warpAffine(frame,Ma,(cols,rows))

		frame = frame[r[1]:r[3]+r[1], r[0]:r[2]+r[0],:]

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blank.append(frame[:,:])

	blank = np.asarray(blank)
	bg = np.mean(blank, axis=0)
	bg = bg.astype(np.uint8)
	return bg

def make_subbed_frames(video_address, bg, Ma, r, results_dir, area_LOW=100, area_HIGH=400, display_tracking=False):
	cap = cv2.VideoCapture(video_address)
	_, first_frame = cap.read()
	cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

	f = open(os.path.join(results_dir,'stats_trimmed.txt'), mode='w')
	fw = csv.writer(f, delimiter=',')

	instance_counter = np.zeros((first_frame.shape[0], first_frame.shape[1]))

	lastGood = []

	for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
		print(i,'\t',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
		ret, frame = cap.read()

		if ret:

			rows,cols = frame.shape[:2]
			frame = cv2.warpAffine(frame,Ma,(cols,rows))

			frame = frame[r[1]:r[3]+r[1], r[0]:r[2]+r[0],:]

			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			frame=frame-bg + 10
			ret,thresh = cv2.threshold(frame,20,255,cv2.THRESH_BINARY)
			imagem = cv2.bitwise_not(thresh)

			contours, hier = cv2.findContours(imagem.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			valids = [contour for contour in contours if (cv2.contourArea(contour) > area_LOW) and (cv2.contourArea(contour) < area_HIGH) and (cv2.arcLength(contour,True)/cv2.contourArea(contour)) < 0.8 ]

			if len(valids)>30:
				valids = lastGood

			imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

			row = []

			for c in valids:
				M = cv2.moments(c)
				if M['m00'] != 0:
					cx = np.float16(M['m10']/M['m00'])
					cy = np.float16(M['m01']/M['m00'])
					cv2.circle(imagem, (cx,cy), 50,(0,0,0),1)
					instance_counter[int(cy), int(cx)]+=1
					row.append(cx)
					row.append(cy)

			if display_tracking:
				cv2.imshow('traces', imagem)
				cv2.waitKey(0)

			fw.writerow(row)
			lastGood = valids
	f.close()
	np.save(os.path.join(results_dir, 'pixel_instances.npy'), instance_counter)

	return None

def pickROI(frame):

	im = frame.copy()
	im = cv2.resize(im,None, fx=0.5, fy=0.5)
	r = cv2.selectROI(im, fromCenter=False)
	cv2.destroyAllWindows()
	return [2*e for e in r]

def save_bg_mat(address, bg):
	np.save(address, bg)
	return None

def load_bg_mat(address):
	return np.load(address)

def check_for_data():
	if 'data' in os.listdir(os.getcwd()):
		pass
	else:
		os.mkdir(os.path.join(os.getcwd(), 'data'))

def make_new_bg_and_info(video_address, results_dir):

	cap = cv2.VideoCapture(video_address)
	_, frame1 = cap.read()
	rows,cols = frame1.shape[:2]

	print('\n--------------------------------------------------------------')
	print('Enter a value to rotate the frame by, and press ENTER to see it.')
	print('When the box is properly aligned, enter Y to continue')
	print('--------------------------------------------------------------')
	fixed = False
	rows,cols = frame1.shape[:2]
	lastVal = 0
	while fixed==False:

		Ma = cv2.getRotationMatrix2D((cols/2,rows/2),lastVal,1)
		testFrame = cv2.warpAffine(frame1,Ma,(cols,rows))
		cv2.imshow('Rotation',testFrame)
		cv2.waitKey(1000)

		rinput = input('ENTER ROTATION VAL OR Y HERE >> ')
		cv2.destroyAllWindows()

		if rinput=='Y':
			fixed = True
		else:
			lastVal = float(rinput)

	rotVal = lastVal

	Ma = cv2.getRotationMatrix2D((cols/2,rows/2),rotVal,1)
	cap.release()
	r = pickROI(testFrame)

	bg = get_bg_avg(video_address, Ma, r)
	save_bg_mat(os.path.join(results_dir, 'bg_mat.npy'), bg)
	bg = load_bg_mat(os.path.join(results_dir, 'bg_mat.npy'))

	info = CropInfo(r,Ma)
	with open(os.path.join(results_dir, 'CropInfo.obj'), 'wb') as fp:
		pickle.dump(info, fp)

	return Ma, r, bg

def get_old_bg_and_info(results_dir):
	with open(os.path.join(results_dir, 'CropInfo.obj'), 'rb') as fp:
		info = pickle.load(fp)
	Ma = info.Ma
	r = info.r
	bg = load_bg_mat(os.path.join(results_dir, 'bg_mat.npy'))
	return Ma, r, bg

def get_save_interval(video_len, n_frames):
	return int(int(n_frames)/(60*int(video_len)))

def make_trace_frames(results_dir, video_len, bg, temp_img_dir):

	for file in os.listdir(temp_img_dir):
		os.remove(os.path.join(temp_img_dir, file))

	f = open(os.path.join(results_dir,'stats_trimmed.txt'))
	fr = csv.reader(f, delimiter=',')
	rows = [row for row in fr]
	pair_log = []

	si = get_save_interval(video_len, len(rows))
	save_interval = max(1, si)

	for r in rows:
		pairs = []
		for idx in np.arange(0, len(r), 2):
			pairs.append([int(float(r[idx])), int(float(r[idx+1]))])
		pair_log.append(pairs)

	blank = 255*np.ones(shape=[bg.shape[0], bg.shape[1], 3])
	blank = blank.astype(np.uint8)

	for idx, row in enumerate(pair_log):
		print("Writing {} of {}".format(idx, len(pair_log)))
		overlay = blank.copy()
		for pt in row:
			cv2.circle(overlay, (pt[0],pt[1]), 1,(0,0,0),-1)
		alpha = 0.15
		image_new = cv2.addWeighted(overlay, alpha, blank, 1 - alpha, 0)
		j=str(idx)
		k = str(j.zfill(8))

		if idx%save_interval==0:
			cv2.imwrite(os.path.join(temp_img_dir ,'{}.jpg'.format(k)), image_new)
		blank = image_new.copy()

	return blank

def write_videofile(temp_img_dir, results_dir):
	clip = ImageSequenceClip(temp_img_dir, fps=60)
	clip.write_videofile(os.path.join(results_dir, '_output_video.mp4'), audio=False)
	return None

def make_density_plot(results_dir, sigma, plane=True, surface=True):

	pixel_instances = np.load(os.path.join(results_dir, 'pixel_instances.npy'))
	pixel_instances_small = cv2.resize(pixel_instances, None, fx=0.25, fy=0.25)
	planarPlotSmall = nd.filters.gaussian_filter(pixel_instances_small, sigma)
	planarPlotLarge = nd.filters.gaussian_filter(pixel_instances, sigma*2)
	if plane:
		np.save(os.path.join(results_dir, 'planar_density.npy'), 255*planarPlotLarge)
		cv2.imwrite(os.path.join(results_dir, '_planar_density.jpg'),255*planarPlotLarge)
	if surface:
		np.save(os.path.join(results_dir, 'planar_density_small.npy'), planarPlotSmall)
		def z_function(x, y):
			statistics = np.load(os.path.join(results_dir, 'planar_density_small.npy'))
			return statistics[x,y]
		y = np.arange(0,planarPlotSmall.shape[1],1)
		x = np.arange(0,planarPlotSmall.shape[0],1)
		X, Y = np.meshgrid(x, y)
		Z = z_function(X, Y)
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.view_init(elev=24, azim=-45)
		ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
		                cmap='winter', edgecolor='none')
		ax.set_title('Smoothed Density of Fly Location');
		fig.savefig(os.path.join(results_dir, '_surface_density.jpg'))

	return None

def cleanup(results_dir):
	try:
		os.mkdir(os.path.join(results_dir, 'figures'))
	except FileExistsError:
		pass
	fig_dir = os.path.join(results_dir, 'figures')
	list_of_files = os.listdir(results_dir)
	for file in list_of_files:
		if file.startswith('_'):
			copyfile(os.path.join(results_dir, file), os.path.join(fig_dir, file))
			os.remove(os.path.join(results_dir, file))
	return None

if __name__=='__main__':

	# Check to see if data directory exists, if not, make it
	check_for_data()

	# Parse in command line arguments
	video_address, new_bg, temp_img_dir, vid_len = parse_args()

	# Create folder where our results will be stored
	data_dir = os.path.join(os.getcwd(), 'data')
	video_session = video_address.split('.')[0].split('/')[-1]
	results_dir = os.path.join(data_dir, video_session)

	try:
		os.mkdir(results_dir)
	except FileExistsError:
		pass

	# If making a new background is necessary, initiate the sequence
	if new_bg:
		Ma, r, bg = make_new_bg_and_info(video_address, results_dir)
	else:
		Ma, r, bg = get_old_bg_and_info(results_dir)

	# Start main routine
	make_subbed_frames(video_address, bg, Ma, r, results_dir,
	 					area_LOW=100,
						area_HIGH=400,
						display_tracking=False)

	# Make last frame plot and video if desired
	lastFrame = make_trace_frames(results_dir, vid_len, bg, temp_img_dir)
	cv2.imwrite(os.path.join(results_dir, '_paths.jpg'), lastFrame)
	if int(vid_len)!=0:
		write_videofile(temp_img_dir, results_dir)

	make_density_plot(results_dir, 15)
	cleanup(results_dir)









#
