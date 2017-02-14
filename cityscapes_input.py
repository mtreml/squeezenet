from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

FLAGS = tf.app.flags.FLAGS

# Global constants describing the data set.
IMAGE_WIDTH_ORIG = 2048
IMAGE_HEIGHT_ORIG = 1024
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400
IMAGE_CHANNELS = 3
LABEL_CHANNELS = 1
NUM_CLASSES = 20
MEAN = [72.39239899,  82.90891774,  73.1583588]


# paths to data lists
TRAINLIST = "/some/path/treml/cityscapes/train_CITY_tf.txt"
VALLIST = "/some/path/treml/cityscapes/val_CITY_tf.txt"

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2975

def inputs(phase='train'):    
    
        
    def pr_image(image):
        
        image = tf.image.resize_images(image, IMAGE_HEIGHT, IMAGE_WIDTH)
        image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        
        return image
        
    def pr_label(label):
        
        label = tf.image.resize_images(label, IMAGE_HEIGHT, IMAGE_WIDTH)
        label.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, LABEL_CHANNELS])
        
        return label
            
    return _input_pipeline(phase,
                           processing_image=pr_image,
                           processing_label=pr_label)


def _input_pipeline(phase,
                    processing_image=lambda x: x,
                    processing_label=lambda y: y):
                        
    if phase=='train':
        filelist = TRAINLIST
        num_epochs = FLAGS.max_steps
        batch_size = FLAGS.batch_size
    if phase=='val':
        filelist = VALLIST
        num_epochs = FLAGS.num_examples
        batch_size = 1
    
    # Create filenamelist.
    imagelist, labellist = read_file_list(filelist)
    
    images = ops.convert_to_tensor(imagelist, dtype=dtypes.string)
    labels = ops.convert_to_tensor(labellist, dtype=dtypes.string)
    
    # Makes an input queue.
    print("Num_epochs = ", num_epochs)
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True)
   
    # Reads the actual images from disk.
    image, label = read_images_and_labels_from_disk(input_queue)
    pr_image = processing_image(image)
    pr_label = processing_label(label)    
    
    
    # Set shapes and create train batch.    
    image_batch, label_batch = tf.train.batch([pr_image, pr_label],
                                              batch_size=batch_size)
    print("Image BATCH has dtype", image_batch.dtype, "and shape", image_batch.get_shape())
    print("Label BATCH has dtype", label_batch.dtype, "and shape", label_batch.get_shape())

    # Display the training images in the visualizer.
    tf.image_summary('images',
                     image_batch,
                     max_images=2)
    tf.image_summary('labels uint8',
                     label_batch,
                     max_images=2)
                    

    return image_batch, label_batch
       
    
def read_images_and_labels_from_disk(input_queue):
    """ Reads images and labels from disk and sets their shape

      Args:
        input_queue: created by slice_input_producer
    """
    image_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    image = tf.image.decode_png(image_contents, IMAGE_CHANNELS)
    label = tf.image.decode_png(label_contents, LABEL_CHANNELS)
    image.set_shape([IMAGE_HEIGHT_ORIG, IMAGE_WIDTH_ORIG, IMAGE_CHANNELS])
    label.set_shape([IMAGE_HEIGHT_ORIG, IMAGE_WIDTH_ORIG, LABEL_CHANNELS])
    print("Image has dtype", image.dtype, "and shape", image.get_shape(), "after decoding from disk.")
    print("Label has dtype", label.dtype, "and shape", label.get_shape(), "after decoding from disk.")
    return image, label


def read_file_list(image_list_file):
    
    f = open(image_list_file, 'r')
    imagelist = []
    labellist = []
    for line in f:
        imagename, labelname = line[:-1].split('    ')
        imagelist.append(imagename)
        labellist.append(labelname)
        
    return imagelist, labellist

