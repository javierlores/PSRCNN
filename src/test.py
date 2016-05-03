#! /usr/bin/env python2.7

import sys
import argparse
import cv2
import os
import numpy as np
import random
import scipy.misc
from SRCNN import SRCNN


DEFAULT_OUTPUT_PATH = "../data/"
DEFAULT_DATA_PATH = "../data/"
DEFAULT_SCALES = [2, 3]


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
    data_path = args.data_path if args.data_path is not None else DEFAULT_DATA_PATH
    output_path = args.output_path if args.output_path is not None else DEFAULT_OUTPUT_PATH

    if args.model_filters is not None:
        model_filters = args.model_filters
    else:
        print 'Error: missing model filter file' 
        sys.exit(1)

    # Read image set
    image_paths = read_filenames(data_path) 

    # Create model
    srcnn = SRCNN(model_filters)

    # Test images
    for scale in DEFAULT_SCALES:
        test(srcnn, image_paths, args.rgb, scale, output_path)


def parse_arguments():
    """ 
        This function retrieves the command line parameters passed into this script.

        Parameters
        ----------
        None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', type=str, help='The location where the set files and images resides')
    parser.add_argument('-op', '--output-path', type=str, help='The location where to save the results')
    parser.add_argument('--model-filters', type=str, help='The location where the saved filters reside')
    parser.add_argument('--rgb', action='store_true', help='If the RGB channels of the testing images should be used')
    args = parser.parse_args()

    return args
    

def read_filenames(data_path):
    """ 
        This function reads in the image set file names

        Parameters
        ----------
        data_path: str
            the path to the location of the directories containing the data
    """
    image_paths = []
    # Read in images for the the specified set
    for subdir, dirs, files in os.walk(data_path+'sets/'):
        for file in files:
            # Check if the file belongs to the desired set
            if 'test' == file.split('.')[0]:
                # Read images
                with open(os.path.join(data_path+'sets/', file), 'r') as output_file:
                    for image_name in output_file:
                        image_paths.append(os.path.join(data_path+'images/', image_name.rstrip('\n')))

    return image_paths


def test(srcnn, image_paths, use_rgb, scale, output_path):
    """ 
        This function takes an img and ensures it is

        Parameters
        ---------
        srcnn: class SRCNN
            the SRCNN network abstraction that will be tested
        image_paths: list
            the list of images to test the network on
        use_rgb: boolean
            whether or not to use RGB values for the images
        scale: int
            the scale to test the images on
        output_path: str
            the location to store the results
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


    def compute_psnr(img1, img2):
        """ 
            This function computes the Peak-Signal to Noise Ratio of two images

            Parameters
            ---------
            img1: np.array
                the first image
            img2: np.array
                the second image
        """
        img_diff = img1.astype('float32') - img2.astype('float32')
        img_diff = img_diff.flatten()
        img_diff = img_diff.reshape((img_diff.shape[0], 1))

        rmse = np.sqrt(np.mean(img_diff**2))
        psnr = 20*np.log10(255.0/rmse)

        return psnr


    for image_path in image_paths:
        # Read in the image as BGR
        image = cv2.imread(image_path, 1)

        if image is not None:
            if not use_rgb:
                # Convert image to YCrCb
                imageYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

                # Use only the luminance (Y) channel
                imageY = imageYCrCb[:, :, 0:1]
                imageCrCb = imageYCrCb[:, :, 1:]
                img_tmp = imageY
            else:
                img_tmp = image
                
            # Scale and normalize the ground truth image
            image_gnd = modcrop(img_tmp, scale)
            image_gnd = image_gnd.astype('float32')/255.0
            height, width, depth = image_gnd.shape

            # Resize each channel
            image_label = cv2.resize(image_gnd, (0,0), 1.0/scale, 1.0/scale, cv2.INTER_CUBIC)
            image_bicubic = cv2.resize(image_label, (width, height), cv2.INTER_CUBIC)

            # For proper usage we need that dimension therefore we add it back in here
            if not use_rgb:
                image_bicubic = image_bicubic.reshape(image_bicubic.shape+(1,))

            # Create the reconstructed image
            image_srcnn = srcnn.reconstruct(image_bicubic)

            # Unnormalize images
            image_gnd = image_gnd*255
            image_srcnn = image_srcnn*255
            image_bicubic = image_bicubic*255

            # Convert images to ints
            image_gnd = image_gnd.astype('uint8')
            image_srcnn = image_srcnn.astype('uint8')
            image_bicubic = image_bicubic.astype('uint8')

            # Compute PSNR
            psnr_bicubic = compute_psnr(image_gnd, image_bicubic)
            psnr_srcnn = compute_psnr(image_gnd, image_srcnn)

            # Extract the image name 
            filename = image_path.split('/')[-1].split('.')[0]

            # Show results
            with open(output_path+filename+'.txt', 'a') as file:
                file.write(str(scale)+'\n')
                file.write(image_path+'\n')
                file.write(str(psnr_bicubic)+'\n')
                file.write(str(psnr_srcnn)+'\n')

            # If not RGB we need to reshape, and convert back to RGB before saving
            if not use_rgb:
                # Upscale dimensions are appropriate
                height, width, depth = image.shape
                image_srcnn = cv2.resize(image_srcnn, (width, height), cv2.INTER_CUBIC)
                image_bicubic = cv2.resize(image_bicubic, (width, height), cv2.INTER_CUBIC)
                image_srcnn = image_srcnn.reshape(image_srcnn.shape+(1,))
                image_bicubic = image_bicubic.reshape(image_bicubic.shape+(1,))

                # Create new YCrCb SRCNN image
                new_image_srcnn = np.zeros(image.shape)
                new_image_srcnn[:, :, 0:1] = image_srcnn
                new_image_srcnn[:, :, 1:] = imageCrCb

                # Create new YCrCb Bicubic image
                new_image_bicubic = np.zeros(image.shape)
                new_image_bicubic[:, :, 0:1] = image_bicubic
                new_image_bicubic[:, :, 1:] = imageCrCb

                # Convert to BGR
                image_srcnn = cv2.cvtColor(new_image_srcnn.astype('uint8'), cv2.COLOR_YCrCb2BGR)
                image_bicubic = cv2.cvtColor(new_image_bicubic.astype('uint8'), cv2.COLOR_YCrCb2BGR)

            # Rotate to RGB for saving
            image_srcnn = cv2.cvtColor(image_srcnn, cv2.COLOR_BGR2RGB)
            image_bicubic = cv2.cvtColor(image_bicubic, cv2.COLOR_BGR2RGB)

            # Save the computed images
            scipy.misc.imsave(output_path+filename+'-bicubic.bmp', image_bicubic)
            scipy.misc.imsave(output_path+filename+'-srcnn.bmp', image_srcnn)


if __name__ == '__main__':
    main()
