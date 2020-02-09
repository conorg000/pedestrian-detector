from os import listdir, rename
from os.path import isfile, join
import shutil

dir = 'C:/Users/Conor/Documents/Uni/2019tri3/img/ped-detector/evaluation/'
gt_dir = dir + 'groundtruths/'

def prep_gt(input_file):
    """
    Prepare groundtruth annotations for evaluation by PASCAL metrics
    Takes test_det.txt file and splits detections into a separate file for each image

    Input:
    input_file (text file): list of object annotations for MOT16-01 directory, one line per detection

    Output:
    many files in /evaluation/groundtruths/: each file has the name of the image
    """
    # Get list of all images
    with open(input_file, "r") as file:
        lines = file.readlines()
    start = ''
    for i, line in enumerate(lines):
        # Rename image name
        tokens = line.split(",")
        current = tokens[0]
        xmin = int(float(tokens[1]))
        xmax = int((float(tokens[1]) + float(tokens[3])))
        ymin = int(float(tokens[2]))
        ymax = int((float(tokens[2]) + float(tokens[4])))
        if current != start:
            with open(gt_dir + '{}.txt'.format(current), "w") as imgtxt:
                imgtxt.write('Pedestrian ' + '{} {} {} {}\n'.format(xmin, ymin, xmax, ymax))
        else:
            with open(gt_dir + '{}.txt'.format(current), "a") as imgtxt:
                imgtxt.write('Pedestrian ' + '{} {} {} {}\n'.format(xmin, ymin, xmax, ymax))
        start = current
    return True

prep_gt((dir + 'test_det.txt'))
