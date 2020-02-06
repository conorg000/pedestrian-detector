from os import listdir, rename
from os.path import isfile, join
import shutil
from PIL import Image

#PATH = '/content/'
# Change to absolute path
PATH = 'C:/Users/Conor/Documents/Uni/2019tri3/img/ped-detector/'
train_path = PATH + 'images/train/'
test_path = PATH + 'images/test/'
# Don't use MOT16-05 (odd image size)
train_dirs = ["MOT16-02", "MOT16-04", "MOT16-09", "MOT16-10", "MOT16-11", "MOT16-13"]
# Don't use MOT16-06 (odd image size)
test_dirs = ["MOT16-01","MOT16-03", "MOT16-07", "MOT16-08", "MOT16-12", "MOT16-14"]
out_path = PATH + "images/"

def clean(train_path, test_path, train_dirs, test_dirs, out_path):
    """
    Prepares MOT16 data for converting into TF Records
    Rename MOT16 images and detection labels to include better filenames.

    Parameters:
    train_path (string): path to MOT16 train data
    test_path (string): path to MOT16 test data
    trains_dirs (list of strings): list of directory names in train path
    test_dirs (list of strings): list of directory names in test path

    Output:
    train_det.txt and test_det.txt
    """
    # Examples for testing
    #exam_path = "C:/Users/Conor/Documents/Uni/2019tri3/img/final_proj/MOT16/example/"
    #exam_dirs = ["MOT16-ex"]

    # Start with training directory
    # For each subdirectory in directory, add subdirectory name to line
    for dir in train_dirs:
        with open("{}{}/det/det.txt".format(train_path, dir), "r") as file:
            lines = file.readlines()
        newlines = []
        for i, line in enumerate(lines):
            first = len(line.split(",")[0])
            zeros = (6 - first)
            line = dir + "_" + (zeros * "0") + line
            lines[i] = line

        with open("{}{}/det/det.txt".format(train_path, dir), "w+") as file:
            file.writelines(lines)

        # For each image in img1 subdirectory, rename images
        img_path = train_path + dir + "/img1/"
        images = [f for f in listdir(img_path) if isfile(join(img_path, f))]
        #print(images)
        for img in images:
            rename((img_path + img), (img_path + dir + "_" + img))

    # Same with test directory
    for dir in test_dirs:
        with open("{}{}/det/det.txt".format(test_path, dir), "r") as file:
            lines = file.readlines()
        newlines = []
        for i, line in enumerate(lines):
            first = len(line.split(",")[0])
            zeros = (6 - first)
            line = dir + "_" + (zeros * "0") + line
            lines[i] = line

        with open("{}{}/det/det.txt".format(test_path, dir), "w+") as file:
            file.writelines(lines)

        # For each image in img1 subdirectory, rename images
        img_path = test_path + dir + "/img1/"
        images = [f for f in listdir(img_path) if isfile(join(img_path, f))]
        #print(images)
        for img in images:
            rename((img_path + img), (img_path + dir + "_" + img))

    # Move train files and create train detection labels
    with open(out_path + "train_det.txt", "w") as outfile:
        for dir in train_dirs:
            # Add detection labels to train_det.txt
            with open("{}{}/det/det.txt".format(train_path, dir), "r") as infile:
                for line in infile:
                    outfile.write(line)

    # Move test files and create test detection labels
    with open(out_path + "test_det.txt", "w") as outfile:
        for dir in test_dirs:
            # Add detection labels to test_det.txt
            with open("{}{}/det/det.txt".format(test_path, dir), "r") as infile:
                for line in infile:
                    outfile.write(line)

clean(train_path, test_path, train_dirs, test_dirs, out_path)
