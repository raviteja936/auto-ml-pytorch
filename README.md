# Deep Learning Codebase using PyTorch

## Introduction

This is my code base for rapid prototyping of deep learning solutions.
This helps me avoid writing hundreds of lines of boilerplate code for each new problem. 
The idea for creating this is inspired from AutoKeras and Uber Ludwig.

## Features

Currently this repo supports only classification using feed-forward neural nets.
All hyperparameters are expected to be input using a param.json file.

TODO
1. Add Recurrent and Conv Nets
2. Neural architecture search for automated hyperparameter tuning 
3. Utils for saving and loading preprocessed data and models
4. Performance tracking using Tensorboard
5. Exploratory data analysis

## Training

1. Create params.json (see sample in the repo)
2. From command line run: `python experiment.py -p params.json`