# SSR1M: A Non-convex Regularized Multi-class Classification Algorithm
## Introduction
This project implements multi-class classification experiments using  the multi-class Logistic loss with a non-convex regularization term. The code is partially derived and adapted from https://github.com/hiroyuki-kasai/SGDLibrary. We would like to sincerely thank the developers of SGDLibrary for their excellent work.

## Project Structure
	+-- Mutl Logistic/                
	|   +-- test_softmax_classifier_usps.m
	|   +-- custom_softmax_regression.m
	|   +-- get_default_options.m
	|   +-- mergeOptions.m
	|   +-- SSR1M.m
	|   +-- stepsize_alg.m
	|   +-- store_infos.m
 
	+-- classification_cifar/                
	|   +-- Model
 			+--Resnet.py
	|   +-- Optimizer
  			+--SSR1M.py
	|   +-- main_cifar10.py

## How to Use
```bash
+-- Mutl Logistic:
|   +-- MATLAB R2016b or above
|   +-- LIBSVM package (for data loading)
```
```bash
+-- classification_cifar:
|   +--  Python 3.9+ and PyTorch:
|   +--  torch, numpy
This will install the following dependencies:
* [torch](https://pytorch.org/) (the library was tested on version  2.5.1+cu121)
* [numpy](https://numpy.org/) (the library was tested on version  1.23.5)
```

## Running Experiments
Basic Training

### Running Experiments
```bash
To run the classifier: test_softmax_classifier_usps.m
```

```bash
python classification_cifar100/main.py --optimizer AdaVAM
```
