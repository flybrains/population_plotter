##################################################################################
INSTALL DEPENDENCIES
##################################################################################
Install Python 3.6.7 and virtualenv
Add dependencies with
	$ pip install -r requirements.txt

##################################################################################
TRIMMING VIDEO TO STABLE PART
##################################################################################
Tracking will only work well if the frame does not shift mid-video.
To trim out the first part of the video where the frame is shifting, use the following command in Linux:

	$ ffmpeg -ss 00:00:00 -i input_file.mp4 -c copy output_name.mp4

Where 00:00:00 is replaced with the hours, minutes, and seconds of the starting point of the stable part of video
Example: 00:03:45 if the video is stable at 3m45s

Where input_file.mp4 and output_name.mp4 are the full path names of the input video and the desired name for the shortened video.
Example:
	input_file.mp4 = /home/patrick/Desktop/my_videos/myHourLongVideo.mp4
	output_name.mp4 = /home/patrick/Desktop/my_videos/myHourLongVideo_SHORTENED.mp4

FULL EXAMPLE COMMAND:
	$ ffmpeg -ss 00:03:45 -i /home/patrick/Desktop/my_videos/myHourLongVideo.mp4 -c copy 	/home/patrick/Desktop/my_videos/myHourLongVideo_SHORTENED.mp4

##################################################################################
RUNNING ANALYSIS
##################################################################################
To run analysis, run the command:
	python generate_traces.py [video_name] [--keyword arguments]
Which has a few required arguments. If you need to remember them while working, use the command:
	python generate_traces.py -h
For help.
The important arguments are:
* video [address of video] 
	Run with no flag, in first position after generate_traces.py
* --new_bg [True or False] 
	Must be run with --new_bg tag. 
	This indicates that it is a new video and you want to compute a new background. 
	If you are analyzing many IDENTICAL videos, this can be set to False to save time.
* --temp_img_dir [address of a place to store images]
	Must be run with --temp_img_dir tag.
	Many images must be saved in a buffer folder to make the figures. If your computer is low on space
	it is a good idea to use an external hard drive for this.
* --vid_len [integer specifying length in seconds of output video to make]
	Must be run with --vid_len tag.
	Enter 0 if you don't want a video


FULL EXAMPLE COMMAND:
	$ python generate_traces.py my_videos/ere.mp4 --new_bg True --temp_img_dir /media/my_external_drive/img_folder --vid_len 30

All figures will be made and stored in population_plotter/data/{vidname}/figures


