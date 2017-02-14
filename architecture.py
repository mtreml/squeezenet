from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import re
import numpy as np
import tensorflow as tf
from math import ceil

import cityscapes_input

FLAGS = tf.app.flags.FLAGS


debug = 1


# Global constants describing the Cityscapes data set.
IMAGE_WIDTH = cityscapes_input.IMAGE_WIDTH
IMAGE_HEIGHT = cityscapes_input.IMAGE_HEIGHT
IMAGE_CHANNELS = cityscapes_input.IMAGE_CHANNELS
NUM_CLASSES = cityscapes_input.NUM_CLASSES
MEAN = cityscapes_input.MEAN

# Constants describing the training process.
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 500.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-4      # Initial learning rate.
wd = 2e-4               # Weight decay


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var
 

def inputs(phase='train'):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  images, labels = cityscapes_input.inputs(phase)

  return images, labels


def inference(images, train=True):
    """Build the model up to where it may be used for inference.
    Parameters
    ----------
    images : Images placeholder, from inputs().
    train : whether the network is used for train of inference
    Returns
    -------
    softmax_linear : Output tensor with the computed logits.
    """
    
    if train:
        batch_size = FLAGS.batch_size
    else:
        batch_size = 1
            
    with tf.name_scope('Processing') :
    
        red, green, blue = tf.split(3, 3, images)

        bgr = tf.concat(3, [
            blue - MEAN[0],
            green - MEAN[1],
            red - MEAN[2],
        ])
    
        bgr.set_shape([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])


#################

###  ENCODER

#################


    conv1 = _convolution_layer(bgr, [3,3,3,64], "conv1")
    pool1 = _max_pool(conv1, 'pool1', debug)
    tf.image_summary("pool1", tf.expand_dims(pool1[:,:,:,0], dim=3))
    
    fire2_squeeze1x1 = _convolution_layer(pool1, [1,1,64,16], "fire2_squeeze1x1")
    fire2_expand1x1 = _convolution_layer(fire2_squeeze1x1, [1,1,16,64], "fire2_expand1x1")
    fire2_expand3x3 = _convolution_layer(fire2_squeeze1x1, [3,3,16,64], "fire2_expand3x3")
    fire2_concat = tf.concat(3, [fire2_expand1x1, fire2_expand3x3])
    
    fire3_squeeze1x1 = _convolution_layer(fire2_concat, [1,1,128,16], "fire3_squeeze1x1")
    tf.image_summary("fire3_squeeze1x1", tf.expand_dims(fire3_squeeze1x1[:,:,:,0], dim=3))
    fire3_expand1x1 = _convolution_layer(fire3_squeeze1x1, [1,1,16,64], "fire3_expand1x1")
    fire3_expand3x3 = _convolution_layer(fire3_squeeze1x1, [3,3,16,64], "fire3_expand3x3")
    fire3_concat = tf.concat(3, [fire3_expand1x1, fire3_expand3x3])
    pool3 = _max_pool(fire3_concat, 'pool3', debug)    
    tf.image_summary("pool3", tf.expand_dims(pool3[:,:,:,0], dim=3))

    fire4_squeeze1x1 = _convolution_layer(pool3, [1,1,128,128], "fire4_squeeze1x1")
    fire4_expand1x1 = _convolution_layer(fire4_squeeze1x1, [1,1,128,128], "fire4_expand1x1")
    fire4_expand3x3 = _convolution_layer(fire4_squeeze1x1, [3,3,128,128], "fire4_expand3x3")
    fire4_concat = tf.concat(3, [fire4_expand1x1, fire4_expand3x3])
    
    fire5_squeeze1x1 = _convolution_layer(fire4_concat, [1,1,256,128], "fire5_squeeze1x1")
    tf.image_summary("fire5_squeeze1x1", tf.expand_dims(fire5_squeeze1x1[:,:,:,0], dim=3))
    fire5_expand1x1 = _convolution_layer(fire5_squeeze1x1, [1,1,128,128], "fire5_expand1x1")
    fire5_expand3x3 = _convolution_layer(fire5_squeeze1x1, [3,3,128,128], "fire5_expand3x3")
    fire5_concat = tf.concat(3, [fire5_expand1x1, fire5_expand3x3])
    pool5 = _max_pool(fire5_concat, 'pool5', debug)
    tf.image_summary("pool5", tf.expand_dims(pool5[:,:,:,0], dim=3))   
    
    fire6_squeeze1x1 = _convolution_layer(pool5, [1,1,256,48], "fire6_squeeze1x1")
    tf.image_summary("fire6_squeeze1x1", tf.expand_dims(fire6_squeeze1x1[:,:,:,0], dim=3))
    fire6_expand1x1 = _convolution_layer(fire6_squeeze1x1, [1,1,48,192], "fire6_expand1x1")
    fire6_expand3x3 = _convolution_layer(fire6_squeeze1x1, [3,3,48,192], "fire6_expand3x3")
    fire6_concat = tf.concat(3, [fire6_expand1x1, fire6_expand3x3])
    
    fire7_squeeze1x1 = _convolution_layer(fire6_concat, [1,1,384,48], "fire7_squeeze1x1")
    tf.image_summary("fire7_squeeze1x1", tf.expand_dims(fire7_squeeze1x1[:,:,:,0], dim=3))
    fire7_expand1x1 = _convolution_layer(fire7_squeeze1x1, [1,1,48,192], "fire7_expand1x1")
    fire7_expand3x3 = _convolution_layer(fire7_squeeze1x1, [3,3,48,192], "fire7_expand3x3")
    fire7_concat = tf.concat(3, [fire7_expand1x1, fire7_expand3x3])
    
    fire8_squeeze1x1 = _convolution_layer(fire7_concat, [1,1,384,64], "fire8_squeeze1x1")
    tf.image_summary("fire8_squeeze1x1", tf.expand_dims(fire8_squeeze1x1[:,:,:,0], dim=3))
    fire8_expand1x1 = _convolution_layer(fire8_squeeze1x1, [1,1,64,256], "fire8_expand1x1")
    fire8_expand3x3 = _convolution_layer(fire8_squeeze1x1, [3,3,64,256], "fire8_expand3x3")
    fire8_concat = tf.concat(3, [fire8_expand1x1, fire8_expand3x3])
    
    fire9_squeeze1x1 = _convolution_layer(fire8_concat, [1,1,512,64], "fire9_squeeze1x1")
    tf.image_summary("fire9_squeeze1x1", tf.expand_dims(fire9_squeeze1x1[:,:,:,0], dim=3))
    fire9_expand1x1 = _convolution_layer(fire9_squeeze1x1, [1,1,64,256], "fire9_expand1x1")
    fire9_expand3x3 = _convolution_layer(fire9_squeeze1x1, [3,3,64,256], "fire9_expand3x3")
    fire9_concat = tf.concat(3, [fire9_expand1x1, fire9_expand3x3])
    
    drop9 = tf.nn.dropout(fire9_concat, keep_prob=0.5, name="drop9")

    score_fr = _convolution_layer(drop9, [1,1,512,NUM_CLASSES], "score_fr")
    tf.image_summary("score_fr", tf.expand_dims(score_fr[:,:,:,0], dim=3))
   
#################

###  DECODER

#################
   
    upscore2 = _upscore_layer(score_fr,
                shape=pool3.get_shape(),
                num_classes=NUM_CLASSES,
                debug=debug, name='upscore2',
                ksize=4, stride=2)  
        
    tf.image_summary("upscore2", tf.expand_dims(upscore2[:,:,:,0], dim=3))
    
    sharpmask3 = SharpMaskBypass(fire5_concat, upscore2, name='sharpmask3')

    upscore4 = _upscore_layer(sharpmask3,
            shape=pool1.get_shape(),
            num_classes=NUM_CLASSES,
            debug=debug, name='upscore4',
            ksize=4, stride=2) 
            
    tf.image_summary("upscore4", tf.expand_dims(upscore4[:,:,:,0], dim=3))
    
    sharpmask2 = SharpMaskBypass(fire3_concat, upscore4, name='sharpmask2')
        
    upscore8 = _upscore_layer(sharpmask2,
            shape=conv1.get_shape(),
            num_classes=NUM_CLASSES,
            debug=debug, name='upscore8',
            ksize=4, stride=2) 
                                         
    tf.image_summary("upscore8", tf.expand_dims(upscore8[:,:,:,0], dim=3))
    
    sharpmask1 = SharpMaskBypass(conv1, upscore8, name='sharpmask1')                                     
                                                                        
    logits = sharpmask1
    
    print("Logits has shape", logits.get_shape())    
        
    # predict for summary
    logits = tf.reshape(logits, (-1, NUM_CLASSES))
    epsilon = tf.constant(value=1e-8)
    logits = logits + epsilon
    predictions = tf.reshape(tf.argmax(logits, dimension=1), (batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1))  
    print("Predictions has dtype", predictions.dtype, "and shape", predictions.get_shape())
    tf.image_summary('labels_prediction',
                     tf.cast(tf.image.grayscale_to_rgb(predictions),
                     dtype=tf.float32),
                     max_images=2)

    return logits
    

def SharpMaskBypass(from_enc, from_dec, name):
    print('SharpMaskBypass %s' % name)
    enc_features = from_enc.get_shape()[3].value   
    from_enc_conv = _convolution_layer(from_enc, [3,3,enc_features,32], name+"_3x3conv_from_enc")     
    concat = tf.concat(3, [from_enc_conv, from_dec])
    
    dec_features = from_dec.get_shape()[3].value     
    to_dec = _convolution_layer(concat, [3,3,dec_features+32,NUM_CLASSES], name+"_3x3conv_to_dec")
    return to_dec


def _max_pool(bottom, name, debug):
    pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)
    print('Layer name: %s' % name)
    print('Layer shape:%s' % str(pool.get_shape()))
    if debug:
        pool = tf.Print(pool, [pool.get_shape()],
                        message='Shape of %s' % name,
                        summarize=4, first_n=1)
    _activation_summary(pool)
    return pool
    
    
def _convolution_layer(bottom, shape, name):
    with tf.variable_scope(name) :
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        
        # get number of input channels
        in_features = bottom.get_shape()[3].value
        out_features = shape[3]
        print('In features: %s' % in_features)
        print('Out features: %s' % out_features)

        # He initialization
        if "sharpmask" in name:
            stddev = 0.0001
        else:
            stddev = (2 / (in_features + out_features))**0.5
        
        filt = _variable_with_weight_decay(shape, stddev, wd) 
        conv = tf.nn.conv2d(bottom, filt, strides=[1, 1, 1, 1], padding='SAME')
        
        conv_biases = _bias_variable([filt.get_shape()[3]], constant=0.0)
        bias = tf.nn.bias_add(conv, conv_biases)
        
        if name == 'score_fr':
            out = bias
        else:
            out = tf.nn.elu(bias)
        
        # Add summary to Tensorboard
        _activation_summary(out)
        return out


def _variable_with_weight_decay(shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal
    distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """

    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable('weights', shape=shape,
                          initializer=initializer)

    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _bias_variable(shape, constant=0.0):
    initializer = tf.constant_initializer(constant)
    return tf.get_variable(name='biases', shape=shape,
                           initializer=initializer)
                           

def _upscore_layer(bottom, shape, num_classes, name, debug, ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name) :
        in_features = bottom.get_shape()[3].value

        if shape is None:
            # Compute shape out of Bottom
            in_shape = bottom.get_shape()

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.pack(new_shape)
        logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
        f_shape = [ksize, ksize, num_classes, in_features]

        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')
        deconv.set_shape(new_shape)
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(deconv.get_shape()))
        if debug:
            deconv = tf.Print(deconv, [deconv.get_shape()],
                              message='Shape of %s' % name,
                              summarize=4, first_n=1)

        _activation_summary(deconv)
        return deconv


def get_deconv_filter(f_shape):
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)



def loss(logits, labels, head=None):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss') :
        
        # adapt logits        
        logits = tf.reshape(logits, (-1, NUM_CLASSES))
        epsilon = tf.constant(value=1e-4)
        logits = logits + epsilon
        
        # create onehot labels
        labels = tf.cast(labels, tf.int64)
        labels = tf.squeeze(labels, squeeze_dims=[3])
        onehot_labels = tf.one_hot(labels, depth=NUM_CLASSES, dtype=tf.int64)
        print("onehot labels", onehot_labels.get_shape())
        tf.image_summary('labels onehot',
                     tf.expand_dims(tf.cast(tf.argmax(onehot_labels, dimension=3), dtype=tf.float32), 3),
                     max_images=2)
        labels = tf.to_float(tf.reshape(onehot_labels, (-1, NUM_CLASSES)))

        softmax = tf.nn.softmax(logits)

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax),
                                           head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
    return loss


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op