from os import listdir, rename
from os.path import isfile, join
import shutil
from PIL import Image

#PATH = '/content/'
PATH = 'C:/Users/Conor/Documents/Uni/2019tri3/img/ped-detector/'
train_path = PATH + 'images/train/'
test_path = PATH + 'images/test/'
# Don't use MOT16-05 (odd image size)
train_dirs = ["MOT16-02", "MOT16-04", "MOT16-09", "MOT16-10", "MOT16-11", "MOT16-13"]
# Don't use MOT16-06 (odd image size)
test_dirs = ["MOT16-01","MOT16-03", "MOT16-07", "MOT16-08", "MOT16-12", "MOT16-14"]
out_path = PATH + "images/"

def move(train_path, test_path, train_dirs, test_dirs, out_path):
    """
    Collects examples from directories and puts them all together
    Resizes all images to 600 x 337 pixels (reduces file size to 1/5)

    Params:
    train_path (string): path to MOT16 train data
    test_path (string): path to MOT16 test data
    trains_dirs (list of strings): list of directory names in train path
    test_dirs (list of strings): list of directory names in test path
    out_path (string): path to new data directory

    Output:
    Various folders/files in out_path
    """
    # Move all images
    for dir in train_dirs:
        img_path = train_path + dir + "/img1/"
        images = [f for f in listdir(img_path) if isfile(join(img_path, f))]
        for image in images:
            img = Image.open((img_path + image))
            img.thumbnail((600, 600))
            img.save((img_path + image))
            #dest = shutil.copyfile((img_path + image) , (out_path + "train/" + image))

    # Move all images
    for dir in test_dirs:
        img_path = test_path + dir + "/img1/"
        images = [f for f in listdir(img_path) if isfile(join(img_path, f))]
        for image in images:
            img = Image.open((img_path + image))
            img.thumbnail((600, 600))
            img.save((img_path + image))
            #dest = shutil.copyfile((img_path + image) , (out_path + "test/" + image))

move(train_path, test_path, train_dirs, test_dirs, out_path)
