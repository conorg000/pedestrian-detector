# Pedestrian detector
This project looks at fine-tuning a pre-trained object detection model so that it performs better at the task of pedestrian detection.

The repo makes heavy use of Tensorflow's [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Set-up instructions are included in the quick start guide, but feel free to refer to the official API documentation if things don't make sense.

Provided below is a quick start guide on how to set things up and start inference asap.

There is also an extended guide discussing the fine-tuning pipeline, in the hope that results can be reproduced.

## Quick start inference
Start with the Python notebook [`model_inference.ipynb`](https://github.com/conorg000/ped-detector/blob/master/model_inference.ipynb). Use it in [Google Colab](https://colab.research.google.com/) to use their [free GPU](https://colab.research.google.com/notebooks/gpu.ipynb).

The notebook contains setup instructions for Tensorflow Object Detection API, loads the fine-tuned model, and performs inference on a test image (or any image you upload to Colab).

## Fine-tuning pipeline
A summary of the method:
- Get a pre-trained object detection model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
- Get training/test data from [MOT](https://motchallenge.net/data/MOT16/).
- Preprocess detection annotations.
- Preprocess training/test data.
- Start training with new data from pre-trained model's checkpoint.
- Export the fine-tuned model.
- Peform inference on test data.
- Evaluate pre-trained model vs fine-tuned model.


### Data
MOT Challenge

### Pre-processing images
Relevant scripts
Relevant TF record commands

### Fine-tuning
Relevant scripts and commands

###
