from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
import numpy as np
import tensorflow as tf

import cityscapes_input
import architecture

FLAGS = tf.app.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

NUM_CLASSES = cityscapes_input.NUM_CLASSES

tf.app.flags.DEFINE_string('eval_dir', '/some/path/treml/monitoring/tensorflows/2017-01-07-SqueezeNet/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('phase', 'val',
                           """Either 'val', 'test' or 'train'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/some/path/treml/monitoring/tensorflows/2017-01-07-SqueezeNet/2017-01-07_15-45-33',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      print('try')
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter
      step = 0
      while step < num_iter and not coord.should_stop():
        print('step is: ', step)
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      print('except')
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval for a number of steps."""
  with tf.Graph().as_default() as g:
    
    # Get images and labels.
    images, labels = architecture.inputs(phase=FLAGS.phase)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = architecture.inference(images, train=False)
    
    # adapt logits        
    logits = tf.reshape(logits, (-1, NUM_CLASSES))
    epsilon = tf.constant(value=1e-4)
    logits = logits + epsilon
        
    # predict
    predictions = tf.argmax(logits, dimension=1)        
    labels = tf.cast(tf.reshape(labels, shape=predictions.get_shape()), dtype=tf.int64)
    
    # compute accuracy    
    correct_predictions = tf.equal(predictions, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32))

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        architecture.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)
    
    tf.initialize_all_variables()
   
    while True:
      eval_once(saver, summary_writer, accuracy, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()

 
if __name__ == '__main__':
  tf.app.run()
