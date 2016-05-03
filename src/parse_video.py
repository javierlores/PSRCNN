#! /usr/bin/env python2.7

import argparse
import cv2
import scipy.misc
import sys
import datetime


DEFAULT_OUTPUT_PATH = "../data/images/"


def main():
    """ 
        The main logic function.

        Parameters
        ----------
        None
    """

    # Parse arguments
    args = parse_arguments()

    # Extract the arguments
    video_path = args.video_path
    output_path = args.output_path if args.output_path is not None else DEFAULT_OUTPUT_PATH

    # Ensure video passed in
    if video_path is None:
        print "Video path not passed in"
        sys.exit(1)

    # Read in the video
    video = read_video(video_path)

    # Save the video frames
    save_video_frames(video, output_path)


def parse_arguments():
    """ 
        This function retrieves the command line parameters passed into this script.

        Parameters
        ----------
        None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-vp', '--video-path', type=str, help='Specify the path to the video file')
    parser.add_argument('-op', '--output-path', type=str, help='Specify the path to write the files')
    args = parser.parse_args()

    return args


def read_video(video_path):
    """ 
        This function reads in a video

        Parameters
        ----------
        video_path: str
            the path to the video to be read
    """
    cam = cv2.VideoCapture(video_path)
    data = []
    while True:
        ret, img = cam.read()
        if (type(img) == type(None)):
            break
        if (0xFF & cv2.waitKey(5) == 27) or img.size == 0:
            break

        # Rotate image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data.append(img)
    return data


def save_video_frames(video, output_path):
    """ 
        This function takes in a video and saves the invidual frames.
        The names of the frames will be formatted as date-id in the format yymmdd-0001
        For example 010316-0000001

        Parameters
        ----------
        video: list of numpy imgs
            the video frames to be saved
        output_path: str
             the location to save the individual video frames
    """
    now = datetime.datetime.now()
    counter = 1
    for frame in video:
       file_name = now.strftime("%Y%m%d")[2:] + '_' + str(counter).zfill(7) + '.bmp'
       scipy.misc.imsave(output_path+file_name, frame)
       counter += 1


if __name__ == '__main__':
    main()
