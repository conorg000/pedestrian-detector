from os import listdir, rename
from os.path import isfile, join
import shutil
from PIL import Image

train_det = "C:/Users/Conor/Documents/Uni/2019tri3/img/final_proj/data/train_det.txt"
test_det = "C:/Users/Conor/Documents/Uni/2019tri3/img/final_proj/data/test_det.txt"
# Get all training example names
"""
with open(train_det, "r") as file:
  lines = file.readlines()
train_examples = [(line.split(','))[0] for line in lines]
#print(train_examples)
# Build training examples dictionary
train_dict = {}
for examp in train_examples:
  train_dict[examp] = []
with open(train_det, "r") as file:
  lines = file.readlines()
# Clean each line then add to dict
#print(train_dict)
for line in lines:
    #print(line)
    examp = (line.split(','))[0]
    clean = (line.split(','))[2:6]
    #print(clean)
    (train_dict[examp]).append(clean)
print(train_dict)
"""
img_path = 'C:/Users/Conor/Documents/Uni/2019tri3/img/ped-detector/images/train/MOT16-02/img1/'
image = 'MOT16-02_000001.jpg'
img = Image.open((img_path + image))
img.thumbnail((600, 600))
img.save(('C:/Users/Conor/Documents/Uni/2019tri3/img/ped-detector/' + image))
    #dest = shutil.copyfile((img_path + image) , (out_path + "test/" + image))
