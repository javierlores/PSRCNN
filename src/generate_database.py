#! /usr/bin/env python2.7

import argparse
import sys
import h5py
import os
import cv2
import numpy as np
import lmdb
import caffe


DEFAULT_OUTPUT_PATH = "../data/"
DEFAULT_DATA_PATH = "../data/"
DEFAULT_SIZE_INPUT = 33
DEFAULT_SIZE_LABEL = 21
DEFAULT_SCALE = 3
TRAIN_DEV_CHUNKSZ = 128
TRAIN_DEV_STRIDE = 14
TEST_CHUNKSZ = 2
TEST_STRIDE = 21


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
    set = args.set 
    data_path = args.data_path if args.data_path is not None else DEFAULT_DATA_PATH
    output_path = args.output_path if args.output_path is not None else DEFAULT_OUTPUT_PATH
    size_input = args.size_input if args.size_input is not None else DEFAULT_SIZE_INPUT
    size_label = args.size_label if args.size_label is not None else DEFAULT_SIZE_LABEL
    scale = args.scale if args.scale is not None else DEFAULT_SCALE

    # Ensure a set was passed in
    # Set paramters based on set
    if set == 'train' or set == 'dev':
        stride = args.stride if args.stride is not None else TRAIN_DEV_STRIDE
        chunksz = args.chunksz if args.chunksz is not None else TRAIN_DEV_CHUNKSZ
    elif set == 'test':
        stride = args.stride if args.stride is not None else TEST_STRIDE
        chunksz = args.chunksz if args.chunksz is not None else TEST_CHUNKSZ
    else:
        print 'Error: Invalid set'
        sys.exit(1)

    if args.k is None:
        print 'Error: Invalid train set size k'
        sys.exit(1)

    # Read the data
    set_filenames = read_set_filenames(data_path, set, args.k)

    # Extract the patches from the images
    patches, labels = extract_patches(set_filenames, args.rgb, size_input, size_label, scale, stride)

    # Write the data
    write_hdf5(output_path, set, patches, labels, chunksz)


def parse_arguments():
    """ 
        This function retrieves the command line parameters passed into this script.

        Parameters
        ----------
        None
    """

    parser = argparse.ArgumentParser()    
    parser.add_argument('--set', type=str, help='The set from which to generate the HDF5 file (train|dev|test)')
    parser.add_argument('-dp', '--data-path', type=str, help='The location where the set files and images resides')
    parser.add_argument('-op', '--output-path', type=str, help='The location to save the HDF5 file')
    parser.add_argument('--rgb', action='store_true', help='Set this to use the RGB channels of the image, default is YCbCr')
    parser.add_argument('--size-input', type=int, help='The dimensions of the image to input image')
    parser.add_argument('--size-label', type=int, help='The dimensions of the ground truth label')
    parser.add_argument('--scale', type=int, help='The scale to downsample the input image to')
    parser.add_argument('--stride', type=int, help='The stride for extracting the image patches')
    parser.add_argument('--chunksz', type=int, help='The chunk size for storing to HDF5')
    parser.add_argument('--k', type=int, help='Training set will have 2*k images while dev and test will have k images')
    args = parser.parse_args()

    return args


def read_set_filenames(data_path, set, k):
    """ 
        This function reads in the image set file names

        Parameters
        ----------
        data_path: str
            the path to the location of the directories containing the data
        set: (train|dev|test)
            the set whose image files will be read
        k: int
            K will determine the amount of images used for each set type where train will get 2*k, and dev and test will receive K
    """
    image_paths = []
    # Read in images for the the specified set
    count = 0
    for subdir, dirs, files in os.walk(data_path+'sets/'):
        for file in files:
            # Check if the file belongs to the desired set
            if set == file.split('.')[0]:

                # Read images
                with open(os.path.join(data_path+'sets/', file), 'r') as output_file:
                    for image_name in output_file:
                        image_paths.append(os.path.join(data_path+'images/', image_name.rstrip('\n')))
                        count+=1
                        if set == 'train':
                            if count > 2*k:
                                return image_paths
                        elif set == 'test' or set == 'dev':
                            if count > k:
                                return image_paths


def extract_patches(image_paths, use_rgb, size_input, size_label, scale, stride):
    """ 
        This function

        Parameters
        ----------
        image_paths: list
            A list of images from whom patches will be extracted
        use_rgb: boolean
            whether or not we should use the RGB or Y (luminance) channels
        size_input: int
            this will be the dimensions of the image inputted into the CNN
        size_label: int
            this will be the dimensions of the ground truth image of the image
        scale: int
            the scale to downsample the input image by
        stride: int
            the stride for extracting the patches
    """
    def modcrop(img, modulo):
        """ 
            This function takes an img and ensures it is is divisible by modulo

            Parameters
            ---------
            img:
                the image whose dimensions will be checked
            modulo:
                the modulo of for which the image dimensions should be divisble by
        """
        if img.shape[2] == 1:
            size = np.array(img.shape)
            size = size - size % modulo
            img = img[:size[0], :size[1]]
        else:
            tmpsize = np.array(img.shape)
            size = tmpsize[0:2]
            size = size - size % modulo
            img = img[:size[0], :size[1], :]
        return img


    data = []
    labels = []
    padding = abs(size_input-size_label)/2

    for image_path in image_paths:
        # Read in the image as BGR
        image = cv2.imread(image_path, 1)

        if image is not None:
            if not use_rgb:
                # Convert image to YCrCb
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

                # Use only the luminance (Y) channel
                image = image[:, :, 0:1]

            image = image.astype('float32')
            # Create the ground truth, and training input versions of the image
            # Because the training input will be downsampled from the ground truth by 'scale'
            # We call modcrop to ensure the ground truth label is divisible by 'scale'
            # Then we downsample the ground truth image by 'scale' to obtain our training input image
            image_label = modcrop(image, scale)
            height, width, depth = image_label.shape
            image_input = cv2.resize(image_label, (0,0), 1.0/scale, 1.0/scale, cv2.INTER_CUBIC)
            image_input = cv2.resize(image_input, (width, height), cv2.INTER_CUBIC)

            # The resize function removes the last dimensions if there is only 1
            # For proper usage we need that dimension therefore we add it back in here
            if len(image_input.shape) == 2:
                image_input = image_input.reshape(image_input.shape+(1,))

            # Here we extract the training image patches and ground truth patches
            for x in range(0, height-size_input, stride):
                for y in range(0, width-size_input, stride):
                    sub_image_input = image_input[x:x+size_input, y:y+size_input, :]
                    sub_image_label = image_label[x+padding:x+padding+size_label, y+padding:y+padding+size_label, :]

                    # Normalize images
                    sub_image_input = sub_image_input / 255.0
                    sub_image_label = sub_image_label / 255.0

                    data.append(sub_image_input)
                    labels.append(sub_image_label)

    # Convert data to np array
    data = np.array(data)
    labels = np.array(labels)

    # Randomly reorder data
    new_order = np.arange(data.shape[0])
    np.random.seed(0)  # set seed
    np.random.shuffle(new_order)
    data = data[new_order]
    labels = labels[new_order]

    return data, labels


def write_hdf5(output_path, set, data, labels, chunksz):
    """ 
        This function writes each datum and corresponding label to HDF5

        Parameters
        ----------
        output_path: str
            the path to save the HDF5 files
        set: (train|dev|test)
            the set for whom files will be written
        data: list
            the data to save to the HDF5 file
        labels: list
            the corresponding labels for teh data
        chunksz: int
            the batch size to store the data
    """
    # Reorder data and labels as number x channels x width x height
    # Data is currently as number x height x width x channels
    data = np.transpose(data, (0, 3, 2, 2))
    labels = np.transpose(labels, (0, 3, 2, 1))

    # Write the data and labels
    with h5py.File(output_path+set+'-prac.h5', 'w') as hf:
        hf.create_dataset('data', data.shape, chunks=(chunksz,)+data.shape[1:], dtype='f')
        hf.create_dataset('label', labels.shape, chunks=(chunksz,)+labels.shape[1:], dtype='f')

        for batch in range(0, data.shape[0], chunksz):
            hf['data'][batch:batch+chunksz, :, :, :] = data[batch:batch+chunksz, :, :, :]
            hf['label'][batch:batch+chunksz, :, :, :] = labels[batch:batch+chunksz, :, :, :]


if __name__ == '__main__':
    main()
