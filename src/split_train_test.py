#! /usr/bin/env python2.7


import argparse
import os
import numpy as np 


DEFAULT_DATA_PATH = '../../data/images/'
DEFAULT_OUTPUT_PATH = '../../data/sets/'


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

    # Read in the data
    data = read_data(data_path)

    # Split the data
    train_data, dev_data, test_data = split(data, 0.5, 0.25, 0.25)

    # Write the data to files
    export(output_path, train_data, dev_data, test_data)


def parse_arguments():
    """ 
        This function retrieves the command line parameters passed into this script.

        Parameters
        ----------
        None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', type=str, help='Specify the path to the dataset')
    parser.add_argument('-op', '--output-path', type=str, help='Specify the path to write the files')
    args = parser.parse_args()

    return args


def read_data(data_path):
    """ 
        This function reads in the data and corresponding labels.

        Parameters
        ----------
        data_path: str
            the path to the location of the directories containing the data
    """
    data = []
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            data.append(file)

    return np.array(data)


def split(data, train_per=0.8, dev_per=0.0, test_per=0.2):
    """ 
        This function splits the data into the training, development/validation, 
        and test sets based on the percentages passed in.

        Parameters
        ----------
        data: numpy array
            the data to be split
        train_per (optional): float
            the percentage of data to be allocated for the training set
        dev_per (optional):  float
            the percentage of data to be allocated for the development/validation set
        test_per (optional): float
            the percentage of data to be allocated for the test set
    """
    # Ensure proper percentages for each set are passed in
    if (train_per + dev_per + test_per) != 1:
        print "train, dev, and test splits should sum to one"
        return

    # Randomize data and labels
    new_order = np.arange(data.shape[0])
    np.random.seed(0)  # set seed
    np.random.shuffle(new_order)
    data = data[new_order]

    # If there is to be no development/validation set
    if dev_per == 0:
        dim = data.shape[0]
        split1 = int(dim*train_per)            # The train/test boundary
        train_data = data[0:split1]            # Split training set data
        test_data = data[split1:]              # Split test set data
        dev_data = np.array([])
    # If there is a development/validation set
    else:
        dim = data.shape[0]
        split1 = int(dim*train_per)            # The train/dev boundary
        split2 = int(dim*(train_per+dev_per))  # The dev/test boundary
        train_data = data[0:split1]            # Split training set data
        dev_data = data[split1:split2]         # Split development set data
        test_data = data[split2:]              # Split test set data

    return train_data, dev_data, test_data


def export(output_path, train_data, dev_data, test_data):
    """ 
        This function writes the training set, development set, and testing set to text files.

        Parameters
        ----------
        output_path: str
            the path to store the output files
        train_data: numpy array
            the data for the training set
        dev_data: numpy array
            the data for the dev set
        test_data: numpy array
            the data for the test set
    """
    # Export the train set
    for datum in train_data:
        file_name = output_path+'train.txt'

        with open(file_name, 'a') as file:
            file.write(datum+'\n')

    # Export the dev set
    for datum in dev_data:
        file_name = output_path+'dev.txt'

        with open(file_name, 'a') as file:
            file.write(datum+'\n')

    # Export the test set
    for datum in test_data:
        file_name = output_path+'test.txt'

        with open(file_name, 'a') as file:
            file.write(datum+'\n')


if __name__ == '__main__':
    main()
