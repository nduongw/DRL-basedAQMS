import torch.nn as nn
import math
import os
import argparse
from config import Config

def weight_histograms_conv2d(writer, step, weights, layer_number):
    weights_shape = weights.shape
    num_kernels = weights_shape[0]
    for k in range(num_kernels):
        flattened_weights = weights[k].flatten()
        tag = f"layer_{layer_number}/kernel_{k}"
        writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')

def weight_histograms_linear(writer, step, weights, layer_number):
    flattened_weights = weights.flatten()
    tag = f"layer_{layer_number}"
    writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')

def weight_histograms(writer, step, model):
    print("Visualizing model weights...")
  # Iterate over all model layers
    for layer_number in range(len(model.layers)):
        # Get layer
        layer = model.layers[layer_number]
        # Compute weight histograms for appropriate layer
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight
            weight_histograms_conv2d(writer, step, weights, layer_number)
        elif isinstance(layer, nn.Linear):
            weights = layer.weight
            weight_histograms_linear(writer, step, weights, layer_number)


def createFolder():
    if not os.path.exists('models'):
        os.mkdir('models')
    
    if not os.path.exists('logs'):
        os.mkdir('logs')
        
    if not os.path.exists('runs'):
        os.mkdir('runs')
        
def createOption():
    parser = argparse.ArgumentParser()
    parser.add_argument('--storepath',type=str, help='Location to store runs of tensorboard', required=True)
    parser.add_argument('--model',type=str, help='Dense or CNN model', required=True)
    parser.add_argument('--modelpath',type=str, help='Name of saved model state dict', required=False)
    parser.add_argument('--rewardfunc',type=str, help='Rerward version which you want to use', required=False)
    parser.add_argument('--config', type=int, help='Map parameters for running', required=False)
    parser.add_argument('--zoom', type=int, help='Zoom observation', required=True)
    parser.add_argument('--pso', type=bool, help='Want to use PSO-Based algorithm or not', required=False)
    parser.add_argument('--testing', type=bool, help='Use this if you want to test your model', required=False)
    parser.add_argument('--usingmodel', type=bool, help='Use model or another algorithm to test', required=False)
    parser.add_argument('--sendingpercentage', type=float, help='Sending percentage to testing phase', required=False)
    # for map parameters
    parser.add_argument('--mapwidth', type=int, help='Map width', required=False)
    parser.add_argument('--mapheight', type=int, help='Map Height', required=False)
    parser.add_argument('--uncover', type=int, help='Time period when air quality doesn\'t change so much', required=False)
    parser.add_argument('--generation', type=int, help='Car generation amount', required=False)
    parser.add_argument('--coverrange', type=int, help='Cover range', required=False)
    parser.add_argument('--morningv', type=int, help='Car\'s velocity in the morning', required=False)
    parser.add_argument('--afternoonv', type=int, help='Car\'s velocity in the afternoon', required=False)
    parser.add_argument('--eveningv', type=int, help='Car\'s velocity in the evening', required=False)
    parser.add_argument('--clambda', type=int, help='Poisson distribution in a road', required=False)
    args = parser.parse_args()
    
    return args
