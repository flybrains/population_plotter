##################################################################################
PLOTTING UTILITIES
##################################################################################
All of the plotting tools here use either pixel instance maps or log files from
videos that have already been tracked.

##################################################################################
HEATMAPS
##################################################################################
To generate heatmaps of one video, or a composite of multiple videos, use the
heatmap program.
The usage is as follows:

  arguments:
	  paths_to_data_folders...Addresses of data folders from tracked videos.
				  Must be in [] separated by commas, NO SPACES. See ex. below.
	  --save_dir................Location to save the heatmaps
	  --sigma...................Variance parameter for plot. Larger means more
				  spread, smaller means closer to single points.
	  --bw......................Set to False for a colored plot, otherwise plots
			          are in black and white.

	Example:
	>> python plot_tools.py heatmap [/home/path/to/data/video1,/home/path/to/data/video2] \
							--save_dir /home/my/save/dir --sigma 15 --bw False


##################################################################################
COLORED TRACES THROUGH TIME
##################################################################################
To generate position traces for a single video in which the trace changes from blue
to pink as time passes, use the color_through_time program.
The usage is as follows:

	arguments:
  	  paths_to_data_folders.....Address of data folder from a tracked video.
																This can only process a single video, so do not
																use [ ] brackets and comma separation.

	Example:
	>> python plot_tools.py color_through_time /home/path/to/data/video1


##################################################################################
ACTIVITY THROUGH TIME
##################################################################################
To generate activity traces and raw data for a single video, use the
activity program.
The usage is as follows:

	arguments:
  	  path_to_data_folder.....Address of data folder from a tracked video.
			          This can only process a single video, so do not
			          use [ ] brackets and comma separation.

	Example:
	>> python plot_tools.py activity /home/path/to/data/video1
