#!/usr/bin/env python3

'''
    Parsers for training models
'''

import sys
import os
import time
import argparse

def create_ssnnet_parser():
    '''
        CLI parser for learning optimal sampling pattern
        
        Inputs:

        Outputs:
    '''
    # Create a new parser
    parser = argparse.ArgumentParser(description='SSNNet parser')

    # Name of experiment
    parser.add_argument('-e', '--experiment', action='store',
                        required=True, type=str, help='Experiment name')
    
    # Dropout rate
    parser.add_argument('-d', '--dropout', action='store', type=float,
                        default=0.5, help='Dropout')

    # Save directory
    parser.add_argument('-s', '--savedir', action='store', type=str,
                        default='./', help='Save directory')

    # Size of number of bands -- useful for learning for sparse reconstruction
    parser.add_argument('-f', '--nbands', action='store', type=int,
                        default=5, help='Number of bands')
    
    parser.add_argument('-k', '--filters', action='store', type=int,
                        default=8, help='Number of filters')

    # Training fraction
    parser.add_argument('-t', '--train', action='store', type=float,
                        default=0.5, help='Training fraction')

    # Learning rate
    parser.add_argument('-l', '--learning_rate', action='store',  type=float,
                        default=0.1, help='Learning rate')

    # Number of epochs
    parser.add_argument('-i', '--epochs', action='store', type=int,
                        default=10, help='Number of epochs')

    # Weight decay constant
    parser.add_argument('-w', '--decay', action='store', type=float,
                        default=0, help='Weight decay for optimizer')

    # Number of cubes to use for training
    parser.add_argument('-n', '--ndata', action='store', type=int,
                        default=50, help='Number of data points')
    
    # Log directory
    parser.add_argument('-g', '--logdir', action='store', type=str,
                        default='logs', help='Tensorboard log directory')
    
    # Pre-trained model parameters
    parser.add_argument('-c', '--compactness', action='store', type=float,
                        default=None, help='Superpixel compactness')
    
    # Now arse
    args = parser.parse_args()

    return args

def create_flownet_parser():
    '''
        CLI parser for learning optimal sampling pattern
        
        Inputs:

        Outputs:
    '''
    # Create a new parser
    parser = argparse.ArgumentParser(description='FlowNet parser')

    # Name of experiment
    parser.add_argument('-e', '--experiment', action='store',
                        required=True, type=str, help='Experiment name')
    
    # Dropout rate
    parser.add_argument('-d', '--dropout', action='store', type=float,
                        default=0.5, help='Dropout')

    # Save directory
    parser.add_argument('-s', '--savedir', action='store', type=str,
                        default='./', help='Save directory')

    # Size of number of bands -- useful for learning for sparse reconstruction
    parser.add_argument('-f', '--nbands', action='store', type=int,
                        default=5, help='Number of bands')

    # Training fraction
    parser.add_argument('-t', '--train', action='store', type=float,
                        default=0.5, help='Training fraction')

    # Learning rate
    parser.add_argument('-l', '--learning_rate', action='store',  type=float,
                        default=0.1, help='Learning rate')

    # Number of epochs
    parser.add_argument('-i', '--epochs', action='store', type=int,
                        default=10, help='Number of epochs')

    # Weight decay constant
    parser.add_argument('-w', '--decay', action='store', type=float,
                        default=0, help='Weight decay for optimizer')

    # Number of cubes to use for training
    parser.add_argument('-n', '--ndata', action='store', type=int,
                        default=50, help='Number of data points')
    
    # Log directory
    parser.add_argument('-g', '--logdir', action='store', type=str,
                        default='logs', help='Tensorboard log directory')
    
    # Pre-trained model parameters
    parser.add_argument('-p', '--pretrained', action='store', type=str,
                        default=None, help='Pretrained model path (None default)')
    
    # Now arse
    args = parser.parse_args()

    return args
    

def create_pancnn_parser():
    '''
        Create a commandline parser for configuring neuralnet training for
        spectral classification.

        Inputs:
            config_name: Configuration file that contains number of layers
                and number of neurons per layer.

        Outputs:
            args: Namespace object with following fields:
                experiment: Name of experiment. Required.
                nfilters: Number of filters to learn. This is the first
                    hidden dimension of network. Default is 10
                dropout: Dropout fraction. Default is 0.5
                savedir: Directory for saving final model
                train: Fraction of data to use for training. Default is 0.5
                lr: Learning rate. Default is 0.1
                epochs: Number of epochs to train for. Default is 10
    '''
    # Create a new parser
    parser = argparse.ArgumentParser(description='PanCNN parser')

    # Name of experiment
    parser.add_argument('-e', '--experiment', action='store',
                        required=True, type=str, help='Experiment name')
    
    # Model to use
    parser.add_argument('-m', '--model', action='store', required=True,
                        default='guided', type=str, help='Model')

    # Dropout rate
    parser.add_argument('-d', '--dropout', action='store', type=float,
                        default=0.5, help='Dropout')

    # Save directory
    parser.add_argument('-s', '--savedir', action='store', type=str,
                        default='./', help='Save directory')

    # Scaling of spatial dimensions
    parser.add_argument('-x', '--scaling', action='store', type=float,
                        default=4, help='Spatial scaling')

    # Size of number of bands -- useful for learning for sparse reconstruction
    parser.add_argument('-f', '--nbands', action='store', type=int,
                        default=5, help='Number of bands')

    # Number of initial convolutional filters
    parser.add_argument('-k', '--nfilters', action='store', type=int,
                        default=8, help='Number of filters')

    # Training fraction
    parser.add_argument('-t', '--train', action='store', type=float,
                        default=0.5, help='Training fraction')

    # Learning rate
    parser.add_argument('-l', '--learning_rate', action='store',  type=float,
                        default=0.1, help='Learning rate')

    # Number of epochs
    parser.add_argument('-i', '--epochs', action='store', type=int,
                        default=10, help='Number of epochs')

    # Weight decay constant
    parser.add_argument('-w', '--decay', action='store', type=float,
                        default=0, help='Weight decay for optimizer')
    
    # Pre-trained model parameters
    parser.add_argument('-o', '--pretrained', action='store', type=str,
                        default=None, help='Pretrained model path (None default)')

    # Number of cubes to use for training
    parser.add_argument('-n', '--ndata', action='store', type=int,
                        default=50, help='Number of data points')
    
    # Readout noise (in electrons)
    parser.add_argument('-r', '--readout_noise', action='store', type=float,
                        default=5, help='Readout noise in electrons')
    
    # Poisson noise (in electrons)
    parser.add_argument('-p', '--poisson_noise', action='store', type=int,
                        default=2000, help='Maximum poisson rate')
    
    # Log directory
    parser.add_argument('-g', '--logdir', action='store', type=str,
                        default='logs', help='Tensorboard log directory')

    # Now arse
    args = parser.parse_args()

    return args
