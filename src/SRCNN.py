#! /usr/bin/env python2.7


import csv
import numpy as np
import scipy.signal


class SRCNN():
    """ This class represents the train CNN for super-resolution."""

    def __init__(self, filters_file):
        self.model = self.load_model(filters_file)


    def load_model(self, file_name):
        """ 
            This function reads in a csv file containing the weights of a CNN model

            Parameters
            ----------
            file_name: str
                the file containing the weights of the CNN model
        """

        model = {}

        # Read in the csv file
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')

            data = []
            for row in reader:
                data.append(row)

            for i in range(0, len(data), 3):
                # Read the layer information
                shape = np.array(data[i]).astype('int32')
                weights = np.array(data[i+1]).astype('float32').reshape(shape)
                biases = np.array(data[i+2]).astype('float32')

                # Store the layer information
                layer = {}
                layer['weights'] = weights
                layer['biases'] = biases
                model['conv'+str(i/3+1)] = layer

        return model


    def reconstruct(self, img):
        """ 
            This function takes and image and passes it through a CNN to reconstruction a high resolution version

            Parameters
            ----------
            img: numpy array
                the low resolution input image from which we will generate the high resolution image
        """
        height, width, channels = img.shape

        # First layer
        conv1_data = np.zeros((height, width, self.model['conv1']['weights'].shape[0]))
        for i in range(self.model['conv1']['weights'].shape[0]):   # Iterate over the filters
            for j in range(self.model['conv1']['weights'].shape[1]): # Iterate over the channels
                conv1_data[:, :, i] = conv1_data[:, :, i] + scipy.signal.convolve(img[:, :, j], self.model['conv1']['weights'][i, j, :, :], mode='same')
            conv1_data[:, :, i] = np.maximum(conv1_data[:, :, i]+self.model['conv1']['biases'][i], 0)

        # Second layer
        conv2_data = np.zeros((height, width, self.model['conv2']['weights'].shape[0]))
        for i in range(self.model['conv2']['weights'].shape[0]):   # Iterate over the filters
            for j in range(self.model['conv2']['weights'].shape[1]): # Iterate over the channels
                conv2_data[:, :, i] = conv2_data[:, :, i] + scipy.signal.convolve(conv1_data[:, :, j], self.model['conv2']['weights'][i, j, :, :], mode='same')
            conv2_data[:, :, i] = np.maximum(conv2_data[:, :, i]+self.model['conv2']['biases'][i], 0)

        # Third layer
        conv3_data = np.zeros((height, width, self.model['conv3']['weights'].shape[0]))
        for i in range(self.model['conv3']['weights'].shape[0]):   # Iterate over the filters
            for j in range(self.model['conv3']['weights'].shape[1]): # Iterate over the channels
                conv3_data[:, :, i] = conv3_data[:, :, i] + scipy.signal.convolve(conv2_data[:, :, j], self.model['conv3']['weights'][i, j, :, :], mode='same')
            conv3_data[:, :, i] = np.maximum(conv3_data[:, :, i]+self.model['conv3']['biases'][i])

        return conv3_data
