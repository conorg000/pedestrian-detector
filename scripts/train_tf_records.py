import tensorflow as tf

from object_detection.utils import dataset_util

"""
Usage: in command prompt python train_tf_records.py output_path='path to .record file'
"""

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def mot16_to_tf(coordinates, width, height):
    """
    Converts MOT16 bounding box coordinates into TF_Record-friendly data

    Params:
    coordinates (list of lists): each list containing bounding box data
        (bb_left, bb_top, bb_width, bb_height)
    width (int): width of image
    height (int): height of image

    Returns:
    xmins (list of floats): min x coordinates for each bounding box
    xmaxs (list of floats): max x coords
    ymins (list of floats): min y coords
    ymaxs (list of floats): max y coords
    """
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    for box in coordinates:
        xmin = float(box[0]) / width
        xmax = (float(box[0]) + float(box[2])) / width
        ymin = float(box[1]) / height
        ymax = (float(box[1]) + float(box[3])) / height
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)

    return xmins, xmaxs, ymins, ymaxs


def create_tf_example(example, path, data_list):
  # TODO(user): Populate the following variables from your example.
  height = 1080 # Image height
  width = 1920 # Image width
  filename = (example + '.jpg').encode('utf8') # Filename of the image. Empty if image is not from file
  #encoded_image_data = None
  img_path = path + example + '.jpg'
  with tf.io.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read() # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'
  # Coordinates is list of lists, each containing bounding box coordinates
  coordinates = data_list
  xmins, xmaxs, ymins, ymaxs = mot16_to_tf(coordinates, width, height)
  print("Normalised BB coords: ", xmins, xmaxs, ymins, ymaxs)
  #xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  #xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  #ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  #ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)

  classes_text = [b'Pedestrian'] * (len(coordinates)) # List of string class name of bounding box (1 per box)
  print("Classes: ", classes_text)
  classes = [1] * (len(coordinates)) # List of integer class id of bounding box (1 per box)
  print("Class ids: ", classes)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    # TODO(user): Write code to read in your dataset to examples variable
    # Paths for detection text files
    train_det = "C:/Users/Conor/Documents/Uni/2019tri3/img/final_proj/data/train_det.txt"
    #test_det = "C:/Users/Conor/Documents/Uni/2019tri3/img/final_proj/data/test_det.txt"

    # Get all training example names
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
    for line in lines:
        examp = (line.split(','))[0]
        clean = (line.split(','))[2:6]
        (train_dict[examp]).append(clean)

    train_path = "C:/Users/Conor/Documents/Uni/2019tri3/img/final_proj/data/train/"
    #test_path = "C:/Users/Conor/Documents/Uni/2019tri3/img/final_proj/data/test/"

    for example in train_examples:
        print("Example: ", example)
        # Get value for key example
        data_list = train_dict[example]
        print("Data list: ", data_list)
        tf_example = create_tf_example(example, train_path, data_list)
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
  tf.app.run()
