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

## How to Use
### Dependencies

MATLAB R2016b or above
LIBSVM package (for data loading)

### Running Experiments
To run the classifier: test_softmax_classifier_usps.m
