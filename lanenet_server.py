# USAGE
# Start the server:
#   python run_keras_server.py
# Submit a request via cURL:
#   curl -X POST -F image=@<image> 'http://localhost:<port>/predict'
# Submit a request via Python:
# python simple_request.py

# import the necessary packages
import os
import time
import datetime
import json
import cv2
import glog as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from PIL import Image
from io import BytesIO
import numpy as np
import io
import sys

import flask
from flask import request, redirect, url_for, send_from_directory, render_template
from werkzeug import secure_filename
from flask import Response
from flask import jsonify


from api import apicfg
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

IP = apicfg.FLASK_IP
print("IP: {}".format(IP))

PORT = apicfg.FLASK_PORT
print("PORT: {}".format(PORT))


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
  # if the image mode is not RGB, convert it
  # if image.mode != "RGB":
  #   image = image.convert("RGB")

  # resize the input image and preprocess it
  # image = image.resize(target)
  # image = img_to_array(image)
  # image = np.expand_dims(image, axis=0)
  # image = imagenet_utils.preprocess_input(image)
  image_bytes = image.read()
  # im = np.array(Image.open(io.BytesIO(image_bytes)))
  image = Image.open(io.BytesIO(image_bytes))
  # log.info("image : {}".format(image))
  log.info("image_shape : {}".format(image.size))
  if apicfg.RESIZE == True:
  #   image = image.convert("RGB")  
    image = image.resize((1280,720), Image.ANTIALIAS)
    log.info("Resized_image_shape : {}".format(image.size))

  image = np.array(image)

  # return the processed image
  return image

def allowed_file(filename):
  fn, ext = os.path.splitext(os.path.basename(filename))
  return ext.lower() in apicfg.ALLOWED_IMAGE_TYPE

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
def predict():

  """
  Main function for Lanenet AI API for prediction 
  """
  try:
    # apires = {"success": False}
    t0 = time.time()
    # ensure an image was properly uploaded to our endpoint
    # if flask.request.method == "POST":
      # if flask.request.files.get("image"):
        # read the image in PIL format
        # image = flask.request.files["image"].read()
        # image_name = secure_filename(image.filename)
        # log.info("image.filename: {}".format(image_name))
        # image = Image.open(io.BytesIO(image))
    log.info("----------------------------->")
    # log.info("request: {}".format(request))
    # log.info("request.files: {}".format(request.files))
    image = request.files["image"]
    log.debug("image: {}".format(image))
    image_name = secure_filename(image.filename)
    log.info("image.filename: {}".format(image_name))

    if image and allowed_file(image_name):
      # resizes image from 1920*1080 to 1280*720
      image = prepare_image(image)
      # log.info("image shape is {}".format(image))

      # convert image to numpy array      
      # image = np.array(image)
      # print(type(image))
      # print(image)

      t1 = time.time()
      time_taken_imread = (t1 - t0)
      log.debug('Total time taken in time_taken_imread: %f seconds' %(time_taken_imread))


      # preprocess the image and prepare it for classification
      # image = prepare_image(image)

      # # classify the input image and then initialize the list
      # # of predictions to return to the client
      # preds = model.predict(image)
      # results = imagenet_utils.decode_predictions(preds)

      # # loop over the results and add them to the list of
      # # returned predictions
      # for (imagenetID, label, prob) in results[0]:
      #   r = {"label": label, "probability": float(prob)}
      #   apires["predictions"].append(r)
      # weights_path = '/aimldl-cod/external/lanenet-lane-detection/model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt'
      weights_path = apicfg.WEIGHTS_PATH
      # assert ops.exists(image_path), '{:s} not exist'.format(image_path)

      log.info('Start reading image and preprocessing')
      # t_start = time.time()
      # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
      image_vis = image
      image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
      image = image / 127.5 - 1.0
      # log.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

      input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

      # net = lanenet.LaneNet(phase='test', net_flag='vgg')
      binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

      postprocessor = lanenet_postprocess.LaneNetPostProcessor()

      saver = tf.train.Saver()

      # Set sess configuration
      sess_config = tf.ConfigProto()
      sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
      sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
      sess_config.gpu_options.allocator_type = 'BFC'

      sess = tf.Session(config=sess_config)

      with sess.as_default():

          saver.restore(sess=sess, save_path=weights_path)

          t2 = time.time()
          binary_seg_image, instance_seg_image = sess.run(
              [binary_seg_ret, instance_seg_ret],
              feed_dict={input_tensor: [image]}
          )
          t3 = time.time()
          t_cost = t3 - t2
          log.info('Single image inference cost time: {:.5f}s'.format(t_cost))

          postprocess_result = postprocessor.postprocess(
              binary_seg_result=binary_seg_image[0],
              instance_seg_result=instance_seg_image[0],
              source_image=image_vis
          )
          # visualization
          # mask_image = postprocess_result['mask_image']

          # for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
          #     instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
          # embedding_image = np.array(instance_seg_image[0], np.uint8)

          # plt.figure('mask_image')
          # plt.imshow(mask_image[:, :, (2, 1, 0)])
          # plt.figure('src_image')
          # plt.imshow(image_vis[:, :, (2, 1, 0)])
          # plt.figure('instance_image')
          # plt.imshow(embedding_image[:, :, (2, 1, 0)])
          # plt.figure('binary_image')
          # plt.imshow(binary_seg_image[0] * 255, cmap='gray')
          # # plt.show()

          # ts = time.time()
          # st = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y_%H-%M-%S')

          # cv2.imwrite('instance_mask_image.png', mask_image)
          # cv2.imwrite('binary_mask_image.png', binary_seg_image[0] * 255)
          # cv2.imwrite('source_image-'+st+'.png', postprocess_result['source_image'])

      sess.close()

      t4 = time.time()

      pred_json = postprocess_result['pred_json']
      # apires["result"] = pred_json

      t5 = time.time()
      time_taken_res_preparation = (t5 - t4)
      log.debug('Total time taken in time_taken_res_preparation: %f seconds' %(time_taken_res_preparation))

      t6 = time.time()
      tt_turnaround = (t6 - t0)
      log.debug('Total time taken in tt_turnaround: %f seconds' %(tt_turnaround))

      # with open('pred-'+st+'.json','w') as outfile:
      #         json.dump(pred_json, outfile)

      # return
      res_code = 200
      apires = {
        'type' : 'vidteq-lnd-1',
        'dnnarch' : 'lanenet',
        'image_name' : image_name,
        'result' : pred_json,
        'status_code': res_code,
        'timings': {
          'image_read': time_taken_imread,
          'detect': t_cost,
          # ,'res_preparation': time_taken_res_preparation
          'tt_turnaround': tt_turnaround
        }
      }
    else:
      res_code = 400
      apires = {
        "type": 'vidteq-lnd-1',
        "dnnarch": 'lanenet',
        'image_name' : None,
        "result": None,
        "error": "Invalid Image Type. Allowed Image Types are: {}".format(appcfg.ALLOWED_IMAGE_TYPE),
        'status_code': res_code,
        'timings': {
          'image_read': -1,
          'detect': -1,
          'res_preparation': -1,
          'tt_turnaround': -1
        }
      }
  except Exception as e:
    log.error("Exception in detection", exc_info=True)
    res_code = 500
    apires = {
      "type": None,
      "dnnarch": None,
      'image_name' : None,
      "result": None,
      "error": "Internal Error. Exception in detection.",
      'status_code': res_code,
      'timings': {
        'image_read': -1,
        'detect': -1,
        'res_preparation': -1,
        'tt_turnaround': -1
      }
    }
  log.debug("apires: {}".format(apires)) 


  # indicate that the request was a success
  # apires["success"] = True
  log.debug("apires: {}".format(apires))
  res = Response(json.dumps(apires), status=res_code, mimetype='application/json')
  log.debug("res: {}".format(res))
  # return the apires dictionary as a JSON response
  # return flask.jsonify(apires)
  return res

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
  print(("* Loading Keras model and Flask starting server..."
    "please wait until server has fully started"))
  load_model()
  # app.run()
  app.run(debug = False, threaded = False, host=IP, port=PORT)
