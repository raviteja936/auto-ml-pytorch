# Deep Learning Codebase using PyTorch

## Introduction

This is my code base for rapid prototyping and developing quick deep learning solutions.

## Features

Currently this repo supports only classification using feed-forward neural nets.
All hyperparameters are expected to be input using a param.json file.

TODO
1. Add Recurrent and Conv Nets
2. Neural architecture search for automated hyperparameter tuning 
3. Utils for saving and loading preprocessed data and models
4. Performance tracking using Tensorboard 

## Training

1. Create params.json (see sample in the repo)
2. From command line run: `python experiment.py -p params.json`