import numpy as np
from object_detection.utils import visualization_utils as vis_util
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
%matplotlib inline

# Change this to the index of training example to viz
EXAMPLE = 0

train_det = "/content/ped-detector/images/train_det.txt"
#test_det = "C:/Users/Conor/Documents/Uni/2019tri3/img/final_proj/data/test_det.txt"

# Get all training example names
with open(train_det, "r") as file:
  lines = file.readlines()
train_examples = [(line.split(','))[0] for line in lines]
#print(train_examples[0])
# Build training examples dictionary
train_dict = {}
for examp in train_examples:
  train_dict[examp] = []
with open(train_det, "r") as file:
  lines = file.readlines()
# Clean each line then add to dict
for line in lines:
    examp = (line.split(','))[0]
    clean = (line.split(','))[2:6]
    (train_dict[examp]).append(clean)

train_path = "/content/ped-detector/images/train/"

example = train_examples[EXAMPLE]
print("Example: ", example)
# Get value for key example
data_list = train_dict[example]
print("Data list: ", data_list)
height = 337 # Image height
width = 600 # Image width
filename = (example + '.jpg').encode('utf8') # Filename of the image. Empty if image is not from file
img_path = (train_path + example + '.jpg').encode('utf8')
image_format = b'jpeg' # b'jpeg' or b'png'
# Coordinates is list of lists, each containing bounding box coordinates
coordinates = data_list
# Plot image
plt.imshow(Image.open(img_path))
# Plot boxes
for box in coordinates:
  cds = [float(i) for i in box]
  plt.gca().add_patch(Rectangle((cds[0],cds[1]),cds[2],cds[3],linewidth=1,edgecolor='r',facecolor='none'))
plt.show()
