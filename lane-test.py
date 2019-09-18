# USAGE
# Start the server:
#   python run_keras_server.py
# Submit a request via cURL:
#   curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
# python simple_request.py

# import the necessary packages
import argparse
import os
import time
import datetime
import json
import requests
import base64
from PIL import Image
from io import BytesIO
import cv2
import glog as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import request, redirect, url_for, send_from_directory, render_template
from flask_cors import CORS, cross_origin
from werkzeug import secure_filename
import io
import sys

# custom imports
APP_ROOT_DIR = os.path.join('/aimldl-cod/external/','lanenet-lane-detection')

if APP_ROOT_DIR not in sys.path:
  sys.path.insert(0, APP_ROOT_DIR)

this_dir = os.path.dirname(__file__)
if this_dir not in sys.path:
  sys.path.append(this_dir)

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

CFG = global_config.cfg

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
#   # load the pre-trained Keras model (here we are using a model
#   # pre-trained on ImageNet and provided by Keras, but you can
#   # substitute in your own networks just as easily)
  # global model
  # model = ResNet50(weights="imagenet")
  global net
  net = lanenet.LaneNet(phase='test', net_flag='vgg')

def prepare_image(image):

  # resize the input image and preprocess it
  # image = image.resize(target)
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image = imagenet_utils.preprocess_input(image)

  # return the processed image
  return image

def minmax_scale(input_arr):
  """

  :param input_arr:
  :return:
  """
  min_val = np.min(input_arr)
  max_val = np.max(input_arr)

  output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

  return output_arr


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
  # initialize the data dictionary that will be returned from the
  # view
  data = {"success": False}
  # image = request.files["image"]
  # log.debug("image: {}".format(image))
  # image_name = secure_filename(image.filename)
  # log.debug("image.filename: {}".format(image_name))

  log.info("----------------------------->")
  log.info("request: {}".format(request))
  log.info("request.files: {}".format(request.files))
  image = request.files["image"]
  image_name = secure_filename(image.filename)
  log.info("type_of_image: {}".format(type(image_name)))
  log.info("image.filename: {}".format(image_name))
  print(image_name)

  # ensure an image was properly uploaded to our endpoint
  # if flask.request.method == "POST":
  #   if flask.request.files.get("image"):
  #     # read the image in PIL format
  #     image = flask.request.files["image"].read()
  #     image_name = secure_filename(image.filename)
  #     log.info("image.filename: {}".format(image_name))
  #     image = Image.open(io.BytesIO(image))
      

  #     # convert image to numpy array      
  #     image = np.array(image)
  #     # print(type(image))
  #     # print(image)
      
  #     # preprocess the image and prepare it for classification
  #     # image = prepare_image(image)

  #     # # classify the input image and then initialize the list
  #     # # of predictions to return to the client
  #     # preds = model.predict(image)
  #     # results = imagenet_utils.decode_predictions(preds)
  #     data["predictions"] = []

  #     # model path
  #     weights_path = '/aimldl-cod/external/lanenet-lane-detection/model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt'
  #     log.info("weights_path : {}".format(weights_path))

  #     log.info('Start reading image and preprocessing')
  #     t_start = time.time()
  #     # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
  #     image_vis = image
  #     image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
  #     image = image / 127.5 - 1.0
  #     log.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

  #     input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

  #     binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

  #     postprocessor = lanenet_postprocess.LaneNetPostProcessor()

  #     saver = tf.train.Saver()

  #     # Set sess configuration
  #     sess_config = tf.ConfigProto()
  #     sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
  #     sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
  #     sess_config.gpu_options.allocator_type = 'BFC'

  #     sess = tf.Session(config=sess_config)

  #     with sess.as_default():

  #         saver.restore(sess=sess, save_path=weights_path)

  #         t_start = time.time()
  #         binary_seg_image, instance_seg_image = sess.run(
  #             [binary_seg_ret, instance_seg_ret],
  #             feed_dict={input_tensor: [image]}
  #         )
  #         t_cost = time.time() - t_start
  #         log.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

  #         postprocess_result = postprocessor.postprocess(
  #             binary_seg_result=binary_seg_image[0],
  #             instance_seg_result=instance_seg_image[0],
  #             source_image=image_vis
  #         )
  #         mask_image = postprocess_result['mask_image']

  #         for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
  #             instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
  #         embedding_image = np.array(instance_seg_image[0], np.uint8)

  #         ts = time.time()
  #         st = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y_%H-%M-%S')

  #     sess.close()

  #     pred_json = postprocess_result['pred_json']
  #     data["predictions"].append(pred_json)
  data["image_name"] = image_name

  #     # indicate that the request was a success
      # data["success"] = True
  data["success"] = True

  # # return the data dictionary as a JSON response
  return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
  print(("* Loading Keras model and Flask starting server..."
    "please wait until server has fully started"))
  # load_model()
  # app.run()
  app.run(debug = False, threaded = False, host = '0.0.0.0', port = '5050')
