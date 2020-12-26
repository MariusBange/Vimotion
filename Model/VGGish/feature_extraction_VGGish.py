# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

from __future__ import print_function

import os
import csv
import numpy as np
import argparse
from scipy.io import wavfile
import torch
import six
import time
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import h5py
import random
import sys
from pathlib import Path

current_directory = Path(os.path.abspath(os.path.join(__file__, os.pardir)))
vimotion = Path(current_directory).parent.parent

flags = tf.app.flags

flags.DEFINE_string(
    # 'checkpoint', '/home/minhdanh/Documents/VGGish/vggish_model.ckpt',
    'checkpoint', os.path.join(current_directory, 'vggish_model.ckpt'),
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    # 'pca_params', '/home/minhdanh/Documents/VGGish/vggish_pca_params.npz',
    'pca_params', os.path.join(current_directory, 'vggish_pca_params.npz'),
    'Path to the VGGish PCA parameters file.')

FLAGS = flags.FLAGS

def feature_extraction():
  # In this simple example, we run the examples from a single audio file through
  # the model. If none is provided, we generate a synthetic input.
  start_time = time.time()
  random.seed(123)
  tf.compat.v1.set_random_seed(123) # added by Thao   tf.compat.v1.set_random_seed

  # Read the csv file
  all_VGG_features = [] #
  for path,fold, files in os.walk(audio_folder):
      files.sort()
      count = 0
      for i in range(len(files)):
          row = files[i]  # # Example: row is ACCEDE00091.mp4
          wav_file = os.path.join(audio_folder, row)  # Example: get file name: ACCEDE00091.wav
          print('Processing: ', wav_file)

          examples_batch = vggish_input.wavfile_to_examples(wav_file)

          with tf.Graph().as_default(), tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:  # Use GPU config=tf.ConfigProto(log_device_placement=True)
              # Define the model in inference mode, load the checkpoint, and
              # locate input and output tensors.
              vggish_slim.define_vggish_slim(training = False)
              vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
              features_tensor = sess.graph.get_tensor_by_name(
                  vggish_params.INPUT_TENSOR_NAME)
              embedding_tensor = sess.graph.get_tensor_by_name(
                  vggish_params.OUTPUT_TENSOR_NAME)

              # Run inference and postprocessing.
              [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: examples_batch})   # raw embedding batches: number of seconds x 128
              print('')

          avg_embedding_segments = np.mean(embedding_batch, axis=0)
          all_VGG_features.append(avg_embedding_segments)

          if count%100 == 0:
              print("Processed: ", count)
          count += 1

  all_VGG_features = np.asarray(all_VGG_features)
  # save in h5 file
  # save predicted values
  features_filename = 'VGGish_features.h5'
  h5file = h5py.File(os.path.join(filepath, features_filename), mode='w')
  h5file.create_dataset('data', data= all_VGG_features) #np.asarray(all_VGG_features, dtype=np.float32))
  h5file.close()

  print("Running time: ", time.time() - start_time)

if __name__ == '__main__':
  folder = sys.argv[1]
  del sys.argv[1]
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
  args = parser.parse_args()

  filepath = os.path.join(vimotion, 'static/audioClips' + folder)
  audio_folder = filepath + '/audio'  # the folder containing audio clips

  feature_extraction()
