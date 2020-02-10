from os import listdir, rename
from os.path import isfile, join
import shutil
import string

dir = 'C:/Users/Conor/Documents/Uni/2019tri3/img/ped-detector/evaluation/'
# path to new finetuned model detections
dt_dir = dir + 'ft_detections/'
# dt_dir = dir + 'pt_detections/'
# path to old finetuned model detections
olddt_dir = dir + 'ft_old_detections/'
#olddt_dir = dir + 'pt_old_detections/'

def prep_gt(input_file):
    """
    Prepare detection annotations for evaluation by PASCAL metrics
    Takes input file, restructures and saves in new directory

    Input:
    input_file (string): file from /old_detections/ to prepare

    Output:
    file in /detections/ with same name as original file
    """
    # Get list of all images
    with open((olddt_dir + input_file), "r") as file:
        lines = file.readlines()
    # Look at first 17 lines (confidence scores)
    # Get confidence scores
    scores = []
    #print(input_file)
    for i, line in enumerate(lines[:17]):
        line = line.translate(str.maketrans('', '', '[]'))
        tokens = line.split()
        tokens = [x.strip() for x in tokens]
        #print(tokens)
        new = [float(x) for x in tokens]
        new = list(filter((lambda x: x > 0.3), new))
        if len(new) != 0:
            #print(new)
            scores += new
        #scores += new
    #print('Scores: ')
    #print(scores)
    # Get detection boxes
    # Un-normalise
    # Starting at line 17, going len(scores) up from there
    boxes = []
    for n, line in enumerate(lines[17:(17+len(scores))]):
        # Rearrange to xmin ymin xmax ymax
        # x * 600
        # y * 337
        line = line.translate(str.maketrans('', '', '[]'))
        #print(line)
        tokens = line.split()
        #print(tokens)
        #new = list(filter((lambda x: x != '0.'), tokens))
        new = [float(x) for x in tokens]
        if len(new) != 0:
            # [ymin, xmin, ymax, xmax] --> [xmin ymin xmax ymax]
            box = [0, 0, 0, 0]
            # xmin
            box[0] = int(float(new[1]) * 600)
            # Get ymin
            box[1] = int(float(new[0]) * 337)
            # xmax
            box[2] = int(float(new[3]) * 600)
            # ymax
            box[3] = int(float(new[2]) * 337)
            boxes.append(box)
    #print('Boxes: ')
    #print(boxes)
    # Zip confidence with bounding box
    res = list(zip(scores, boxes))
    #print(res)
    # Write to file
    with open(dt_dir + '{}'.format(input_file), "w") as imgtxt:
        for thing in res:
            imgtxt.write('Pedestrian ' + '{} {} {} {} {}\n'.format(thing[0], thing[1][0],
             thing[1][1], thing[1][2], thing[1][3]))
    return True

img_path = olddt_dir
files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
for dt_file in files:
    prep_gt(dt_file)
