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
- Pre-process detection annotations.
- Pre-process training/test data.
- Configure pre-trained model.
- Start training with new data from pre-trained model's checkpoint.
- Export the fine-tuned model.
- Perform inference on test data.
- Evaluate pre-trained model vs fine-tuned model.

### Pre-trained model
We use `ssd_mobilenet_v2_coco`, available for download [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

It's MobileNetV2 trained on Microsoft's COCO dataset, topped with a Single Shot MultiBox Detector.

On a GPU, it's supposed to achieve 22 mAP (on COCO test set) and 31 ms inference speed on a single image.

### Data
We use Motion Object Tracking Benchmark challenge data, specifically from 2016 ([MOT16](https://motchallenge.net/data/MOT16/)).

Download the data (1.9GB) and then run the pre-processing scripts below. **Just extract the MOT16 zip file, no need to move the files anywhere yet. The scripts below will deal with that.**

### Pre-process detection annotations
MOT16's data has each video's annotations stored in separate files in different directories. Run [`make_labels.py`](https://github.com/conorg000/ped-detector/blob/master/scripts/make_labels.py) to end up with one file for training annotations (`train_det.txt`) and one file for test annotations (`test_det.txt`). **Change paths in script as needed**. The script also adjusts bounding box coordinates to cater for the upcoming image resizing.

We only use three directories of training data from MOT16 (02, 04, 09) and one directory of test data (01).

### Pre-processing images
We now move and resize the images. Run [`move_images.py`](https://github.com/conorg000/ped-detector/blob/master/scripts/move_images.py) to resize the necessary images (to 600 x 337 pixels) and move them within this repo. The images will end up in `ped-detector/images` directory, in their respective `/train` and `/test` directories.

Then make TFRecords using [`train_tf_records.py`](https://github.com/conorg000/ped-detector/blob/master/scripts/train_tf_records.py) and [`test_tf_records.py`](https://github.com/conorg000/ped-detector/blob/master/scripts/test_tf_records.py). These scripts convert the images and annotations to a data format required by the Object Detection API.

The command for the training data is: `python ped-detector/scripts/train_tf_records.py --output_path=ped-detector/annotations/train.record`

The command for the test data is: `python ped-detector/scripts/test_tf_records.py --output_path=ped-detector/annotations/test.record`

### Configure pre-trained model
We make a new object class mapping file [`ped_label_map.txt`](https://github.com/conorg000/ped-detector/blob/master/annotations/ped_label_map.pbtxt), storing it in `ped-detector/annotations`.

We also adjust the configuration file for `ssd_mobilenet_v2_coco`. This is the [original](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config), but we change the paths and some other metadata to end up with [this](https://github.com/conorg000/ped-detector/blob/master/pre-trained-model/gcolab_ssd_mobilenet_v2_coco.config).

### Fine-tuning
Time to train! Use the API's command (the path to `train.py` depends on where you installed the Object Detection API):

`python /usr/local/lib/python3.6/dist-packages/object_detection/legacy/train.py --logtostderr \
--train_dir=ped-detector/training/ \
--pipeline_config_path=ped-detector/pre-trained-model/gcolab_ssd_mobilenet_v2_coco.config`

We trained for 10 000  steps on Tesla K80 GPU (0.3s per step), and loss was hanging around 4 or 5. More training might help!

### Export the fine-tuned model
Once finished with training, you can export the frozen graph model. The command to do this is:

`python /usr/local/lib/python3.6/dist-packages/object_detection/export_inference_graph.py --input_type image_tensor \
--pipeline_config_path=ped-detector/pre-trained-model/gcolab_ssd_mobilenet_v2_coco.config \
--trained_checkpoint_prefix=ped-detector/training/model.ckpt-24889 \
--output_directory=ped-detector/fine-tuned-model`

where `model.ckpt-24889` is the checkpoint at which we decided to export the model.

### Inference on test data
Using the function in [`model_inference.ipynb`](https://github.com/conorg000/ped-detector/blob/master/model_inference.ipynb), we loop through all test images and perform inference. Results end up in `ped-detector/evaluation` as `/ft_old_detections` (for the fine-tuned model's detections) and `/pt_old_detections` (for the pre-trained model's detections).

### Evaluate model performance
The Tensorflow model spits out results which don't work with the MOT test set or our evaluation script, so we have to clean them up. Basically, Tensorflow's bounding boxes are of the format `[ymin, xmin, ymax, xmax]`, but we want `[xmin, ymin, xmax, ymax]`. Tensorflow's results are also 'relative' coordinates (relative to image size),  whereas we want 'absolute coordinates'.

[`prep_dt_finetuned.py`](https://github.com/conorg000/ped-detector/blob/master/scripts/prep_dt_finetuned.py) cleans up the **fine-tuned model** detections. The results end up in `ped-detector/evaluation/ft_detections`.

[`prep_dt.py`](https://github.com/conorg000/ped-detector/blob/master/scripts/prep_dt.py) cleans up the **pre-trained model** detections. The results end up in `ped-detector/evaluation/pt_detections`.

[`prep_gt.py`](https://github.com/conorg000/ped-detector/blob/master/scripts/prep_gt.py) cleans up the MOT16 groundtruth detections (`test_det.txt`). The results end up `ped-detector/evaluation/groundtruths`.

For evaluation we use Rafael Padilla's easy-to-use [Object Detection Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics#how-to-use-this-project) repo. It's based off PASCAL VOC Challenge metrics and uses mean accuracy precision (mAP) to measure performance. It is very easy to use:
- clone Rafael's repo
- put all of your model detections (i.e. all files in `ped-detector/evaluation/ft_detections`) into `Object-Detection-Metrics/detections`
- put all of your groundtruth files (i.e. all files in `ped-detector/evaluation/groundtruths`) into `Object-Detection-Metrics/groundtruths`
- run `python pascalvoc.py -gtformat xyrb -detformat xyrb`
- mAP will be show on screen along with precision-recall curve
