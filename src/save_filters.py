#! /usr/bin/env python2.7


import caffe
import numpy as np
import argparse
import csv


DEFAULT_NUM_LAYERS = 3

caffe.set_mode_gpu()


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
    model = args.model
    weights = args.weights
    output_path = args.output_path
    filename = args.filename

    # Load the caffe model
    net = caffe.Net(model, weights, caffe.TEST)

    # Extract the learned filters for the model
    filters = extract_filters(net, DEFAULT_NUM_LAYERS)

    # Save the filters
    write_to_file(output_path, filename, filters)


def parse_arguments():
    """ 
        This function retrieves the command line parameters passed into this script.

        Parameters
        ----------
        None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='The path to the model prototxt file')
    parser.add_argument('--weights', type=str, help='The path to the saved caffe model')
    parser.add_argument('--filename', type=str, help='The filename to save the filters as')
    parser.add_argument('--output-path', type=str, help='The path to save the filter file')
    args = parser.parse_args()

    return args


def extract_filters(net, num_layers):
    """ 
        This function extracts the weights and biases from a network

        Parameters
        ----------
        net: caffe.Net
            the caffe net
        num_layers: int
            the number of layers in the network
    """
    weights = []
    biases = []
    for i in range(1, num_layers+1):
        layer_weights = net.params['conv'+str(i)][0]
        layer_biases = net.params['conv'+str(i)][1]

        weights.append(layer_weights.data)
        biases.append(layer_biases.data)
        
    return weights, biases

    
def write_to_file(output_path, file_name, data):
    """ 
        This function writes to data to a file

        Parameters
        ----------
        ouput_path: str
            the location to write the data to
        file_name: str
            the name of the file to be written
        data: list
            the data to write
    """
    with open(output_path+file_name+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        weights, biases = data
        for layer_weights, layer_biases in zip(weights, biases):
            writer.writerow(layer_weights.shape)
            writer.writerow(layer_weights.flatten().tolist())
            writer.writerow(layer_biases.flatten().tolist())


if __name__ == '__main__':
    main()
