import numpy as np
import cv2
import os
import argparse
import scipy
import scipy.ndimage as nd
from skimage import measure
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def parse_args():
	scale_to = None
	parser = argparse.ArgumentParser()
	parser.add_argument('folder')
	parser.add_argument('--sigma', type=np.float32, default=15.)
	parser.add_argument('--downsample', type=int, default=4)
	parser.add_argument('--scale_to', nargs='*', help='<Required> Set flag', required=False)

	args = parser.parse_args()
	return args.folder, args.sigma, args.downsample, args.scale_to

def make_density_plot(instance_mat, sigma, downsample, max_scale):

	pixel_instances_small = cv2.resize(instance_mat, None, fx=float(1/downsample), fy=float(1/downsample))
	planarPlotSmall = nd.filters.gaussian_filter(pixel_instances_small, sigma)

	def z_function(x, y):
		return planarPlotSmall[x,y]

	y = np.arange(0,planarPlotSmall.shape[1],1)
	x = np.arange(0,planarPlotSmall.shape[0],1)
	X, Y = np.meshgrid(x, y)
	Z = z_function(X, Y)
	fig = plt.figure()

	ax = plt.axes(projection='3d')
	ax.set_zlim3d(0,int(max_scale))
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
	                cmap='winter', edgecolor='none')

	ax.set_title('Smoothed Density of Fly Location');
	plt.show()
	return None


def load_mat(folder):
	try:
		address = os.path.join(folder, 'pixel_instances.npy')
		mat = np.load(address)
	except FileNotFoundError:
		input('No instance matrix found. Make sure that the analysis has been properly run. Press ENTER to exit')
		exit()
	return mat

def get_max_scale(scale_to, downsample, sigma):
	maxs = []
	for folder in scale_to:
		instance_mat = load_mat(folder)
		pixel_instances_small = cv2.resize(instance_mat, None, fx=float(1/downsample), fy=float(1/downsample))
		planarPlotSmall = nd.filters.gaussian_filter(pixel_instances_small, sigma)
		maxs.append(np.max(planarPlotSmall))
	return max(maxs)

if __name__=='__main__':
	folder, sigma, downsample, scale_to = parse_args()
	max_scale = get_max_scale(scale_to, downsample, sigma)
	instance_mat = load_mat(folder)
	max_scale=200
	make_density_plot(instance_mat, sigma, downsample, max_scale)
