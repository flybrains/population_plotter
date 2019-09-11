import os
import csv
import numpy as np
import cv2
import argparse
import scipy
import itertools
import scipy.ndimage as nd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.cm as cm

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('program')
	parser.add_argument('paths_to_data_folders')
	parser.add_argument('--save_dir')
	parser.add_argument('--sigma')
	parser.add_argument('--bw')
	args = parser.parse_args()
	if args.paths_to_data_folders.startswith('['):
		paths = [e.strip() for e in args.paths_to_data_folders[1:-1].split(',')]
	else:
		paths = [args.paths_to_data_folders]

	return args.program, paths, args.save_dir, args.sigma, args.bw

class MplColorHelper:

	def __init__(self, cmap_name, start_val, stop_val):
		self.cmap_name = cmap_name
		self.cmap = plt.get_cmap(cmap_name)
		self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
		self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

	def get_rgb(self, val):
		return self.scalarMap.to_rgba(val)

def convert_log_to_coord_pairs(stats_path):
	f = open(stats_path)
	fr = csv.reader(f, delimiter=',')
	rows = [row for row in fr]
	pair_log = []

	for r in rows:
		pairs = []
		for idx in np.arange(0, len(r), 2):
			pairs.append([int(float(r[idx])), int(float(r[idx+1]))])
		pair_log.append(pairs)
	return pair_log

def make_activity_data_dir(data_dir):
	if 'activity_traces' in os.listdir(data_dir):
		pass
	else:
		os.mkdir(os.path.join(data_dir, 'activity_traces'))
	return os.path.join(data_dir, 'activity_traces')

def make_color_data_dir(data_dir):
	if 'colored_traces' in os.listdir(data_dir):
		pass
	else:
		os.mkdir(os.path.join(data_dir, 'colored_traces'))
	return os.path.join(data_dir, 'colored_traces')

def make_density_plot(instance_map, results_dir, sigma=30, bw=True):

	planarPlot = nd.filters.gaussian_filter(instance_map, sigma)
	if bw==True:
		plt.imsave(os.path.join(results_dir, '_heatmap.jpg'), planarPlot, cmap='gray')
	else:
		plt.imsave(os.path.join(results_dir, '_heatmap.jpg'), planarPlot)
	np.save(os.path.join(results_dir, 'planar_density.npy'), 255*planarPlot)
	return None

def get_frame_shape(path_to_data_folder):
	frame = cv2.imread(os.path.join(path_to_data_folder, 'figures/_paths.jpg'))
	return frame.shape

def make_multi_data_dir(data_dir):
	if 'heatmap' in os.listdir(data_dir):
		pass
	else:
		os.mkdir(os.path.join(data_dir, 'heatmap'))
	return os.path.join(data_dir, 'heatmap')

def get_distance_to_NN_from_prev_frame(pair_log):
	last_row = pair_log[0]
	row_averages = []

	for row in pair_log[1:]:

		row_dists = []

		r_burn = row.copy()
		lr_burn = last_row.copy()

		paired_items_row = []
		paired_items_lastRow = []

		for item in row:
			if item in last_row:
				row_dists.append(0)
				r_burn.remove(item)
				paired_items_row.append(item)
				lr_burn.remove(item)
				paired_items_lastRow.append(item)

		r_burn.sort(key=lambda x: x[0])
		lr_burn.sort(key=lambda x: x[0])


		if len(r_burn)==len(lr_burn):
			eq_dists = [np.linalg.norm(np.asarray(r_burn[i]) - np.asarray(lr_burn[i])) for i in range(len(r_burn))]
			row_dists.extend(eq_dists)

		else:
			if len(r_burn) < len(lr_burn):
				picker = r_burn
				choices = lr_burn
			else:
				picker = lr_burn
				choices = r_burn


			minDists = []
			for point in picker:
				intDists = [np.linalg.norm(np.asarray(point) - np.asarray(choices[i])) for i in range(len(choices))]

				minDists.append(np.min(intDists))

			row_dists.extend(minDists)

		row_averages.append(np.mean(row_dists))
		last_row = row
	return row_averages

def load_pixel_instances(pi_address):
	return np.load(pi_address)

def process_and_smooth(distance_metrics,path_to_folder):
	diffs = np.diff(distance_metrics)
	var = np.std(diffs)/4
	threshold = var + np.median(diffs)

	new_mvmt_data = []
	lastGood = dist_metrics[0]
	for idx, element in enumerate(dist_metrics):
		if element > threshold:
			new_mvmt_data.append(lastGood)
		else:
			new_mvmt_data.append(element)
			lastGood = element

	filtered = lowess(new_mvmt_data, np.arange(0,len(new_mvmt_data),1), frac=0.035, it=0)
	plt.figure(figsize=(16,6))
	plt.plot(new_mvmt_data)
	plt.plot(filtered[:,1])
	plt.savefig(os.path.join(path_to_folder, 'filter_comparison.jpg'))
	np.save(os.path.join(path_to_folder, 'raw_movement_data.npy'), np.asarray(dist_metrics))
	np.save(os.path.join(path_to_folder, 'filtered_movement_data.npy'), np.asarray(filtered[:,1]))
	np.savetxt(os.path.join(path_to_folder, 'raw_movement_data.csv'), np.asarray(dist_metrics), delimiter=',')
	np.savetxt(os.path.join(path_to_folder, 'filtered_movement_data.csv'), np.asarray(filtered[:,1]), delimiter=',')

	return None

if __name__=='__main__':

	program, paths_to_folders, save_dir, sigma, bw = parse_args()

	if len(paths_to_folders)==1:
		path_to_folder = paths_to_folders[0]

	if program=='activity':
		activity_data_dir = make_activity_data_dir(path_to_folder)
		stats_path = os.path.join(path_to_folder, 'stats_trimmed.txt')
		coord_pairs = convert_log_to_coord_pairs(stats_path)
		dist_metrics = get_distance_to_NN_from_prev_frame(coord_pairs)
		process_and_smooth(dist_metrics, activity_data_dir)
		print('Raw activity data and plots have been generated and stored at {}'.format(os.path.join(activity_data_dir)))

	if program=='color':
		color_data_dir = make_color_data_dir(path_to_folder)
		fShape = get_frame_shape(path_to_folder)
		stats_path = os.path.join(path_to_folder, 'stats_trimmed.txt')
		coord_pairs = convert_log_to_coord_pairs(stats_path)

		# Color Gradient Stuff
		length = len(coord_pairs)
		NUM_VALS = length
		COL = MplColorHelper('cool', 0, NUM_VALS)
		color_vals = [COL.get_rgb(e) for e in range(length)]
		color_vals = itertools.cycle(color_vals)

		canvas = np.zeros(fShape)

		for idx, row in enumerate(coord_pairs):
			col_val = next(color_vals)[:-1]
			for pt in row:
				cv2.circle(canvas, (pt[0],pt[1]), 1,(col_val[2],col_val[1],col_val[0]),-1)

		#cv2.imshow('c', canvas)
		cv2.waitKey(0)
		cv2.imwrite(os.path.join(color_data_dir,'_colored_traces.jpg'), canvas*255)

		print('A color-shifted trace has been generated and stored at {}'.format(os.path.join(color_data_dir,'_colored_traces.jpg')))

	if program=='heatmap':

		sigma = float(sigma)
		if bw=="True":
			bw = True
		else:
			bw=False

		activity_data_dir = make_multi_data_dir(save_dir)
		list_of_instance_frames = [load_pixel_instances(os.path.join(e, 'pixel_instances.npy')) for e in paths_to_folders]
		shapes = [e.shape for e in list_of_instance_frames]
		minx = 20000
		miny = 20000
		for shape in shapes:
			if shape[0]<minx:
				minx = shape[0]
			if shape[1]<miny:
				miny = shape[1]
		composite_instance_map = np.zeros((minx, miny))
		for frame in list_of_instance_frames:
			composite_instance_map += frame
		make_density_plot(composite_instance_map, activity_data_dir, sigma, bw=bw)
		print('A heatmap of {} trials has been generated and stored at {}'.format(len(list_of_instance_frames), activity_data_dir+'/_heatmap.jpg'))
