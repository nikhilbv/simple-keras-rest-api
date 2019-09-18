__author__ = 'mangalbhaskar'
__version__ = '1.0'
"""
# API test script
# --------------------------------------------------------
# CREDITS:
# [Pyimagesearch: part-3: Deep learning in production with Keras, Redis, Flask, and Apache](https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/)
# Modded by mangalbhaskar
# --------------------------------------------------------
"""

from threading import Thread
import requests
import time
import datetime

import os
import errno

import apicfg

import base64
import requests

import pandas as pd
import sys

API_URL = apicfg.API_URL
print("API_URL: {}".format(API_URL))

st_rpt = []

this = sys.modules[__name__]
print(this)
def mkdir_p(path):
  """
  mkdir -p` linux command functionality

  References:
  * https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
  """
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def __call_vision_api(n, filepath, im):
    time_stats, data = None, None
    payload = {"image": im}
    res = requests.post(API_URL, files=payload)
    if res.status_code == 200:
      status = True
      print("[INFO] thread {} OK".format(n))

      tt = res.elapsed.total_seconds()

      # data = json.loads(res.text)
      data = res.json()
      K = ['sno', 'filepath', 'status', 'tt_turnaround']+list(data['timings'].keys())
      V = [n, filepath, status, tt] + list(data['timings'].values())
      time_stats = dict(zip(K,V))
      
      ## for testing if predictions are proper or not
      ## comment if only timestats are needed
      time_stats['data'] = data
      st_rpt.append(time_stats)
    else:
      print("[INFO] thread {} FAILED".format(n))
      status = False
      K = ['sno', 'filepath', 'status', 'tt_turnaround']
      V = [n, filepath, status, -1]
      time_stats = dict(zip(K,V))

      ## for testing if predictions are proper or not
      ## comment if only timestats are needed
      time_stats['data'] = None
      st_rpt.append(time_stats)
    print("time_stats: {}".format(time_stats))


def get_as_base64(url):
  ## https://stackoverflow.com/questions/38408253/way-to-convert-image-straight-from-url-to-base64-without-saving-as-a-file-in-pyt
  content = requests.get(url).content
  return base64.b64encode(content)


def call_vision_api_on_image_url(n, filename):
  url = apicfg.IMAGE_API+filename
  print(url)

  im = get_as_base64(url)
  print(type(im))
  # __call_vision_api(n, url, im)


def call_vision_api_on_image_filepath(n, filepath):
  with open(filepath, "rb") as im:
    __call_vision_api(n, filepath, im)


def write_stress_report(data):
  """
  write the report
  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.from_dict.html
  Specify orient='index' to create the DataFrame using dictionary keys as rows:
  """

  timestamp = ("{:%d%m%y_%H%M%S}").format(datetime.datetime.now())
  log_dir = os.getenv('AI_LOGS')
  logpath = os.path.join(log_dir,'api')
  print("logpath: {}".format(logpath))
  
  mkdir_p(logpath)

  filepath = os.path.join(logpath,'stress_test_rpt-'+timestamp)
  print("filepath: {}".format(filepath))

  if data and len(data) > 0:
    # df = pd.DataFrame.from_dict(data, orient="index")
    df = pd.DataFrame(data)
    df.to_csv( filepath+'.csv', index=False )
  else:
    print("No report data!")


def main(cfg, num_requests, target_fn, imglist=None):
  print("num_requests: {}".format(num_requests))

  SLEEP_COUNT = cfg.SLEEP_COUNT
  SLEEP_TIME = cfg.SLEEP_TIME
  image_path = cfg.IMAGE_PATH

  ## loop over the number of threads
  for i in range(0, num_requests):
    ## TODO:
    ## ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()
    # if imglist and len(imglist)==num_requests:
    #   image_path = imglist[i]
    # image_path = imglist[i]
    # start a new thread to call the API
    t = Thread(target=target_fn, args=(i, image_path,))
    t.daemon = True
    t.start()
    time.sleep(SLEEP_COUNT)

  # insert a long sleep so we can wait until the server is finished
  # processing the images
  time.sleep(SLEEP_TIME)

  write_stress_report(st_rpt)
  print("st_rpt: {}".format(st_rpt))


if __name__=='__main__':
  num_requests = apicfg.NUM_REQUESTS
  csvfile = apicfg.CSVFILE
  # fname = 'call_vision_api_on_image_url'
  fname = 'call_vision_api_on_image_filepath'
  query_imglist = None

  ## TODO:
  # df = pd.read_csv(csvfile)
  # imglist = df['file']
  # offset = 291000
  # query_imglist = list(df['file'][offset:])

  ## ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
  # if imglist and len(imglist) >= num_requests:
  #   query_imglist = df['file'][:num_requests]
  # elif imglist and len(imglist) < num_requests:
  #   num_requests = len(df['file'])
  # else:
  #   raise Exception("Not a valid use case", exc_info=True)
  
  fn = getattr(this, fname)
  main(apicfg, num_requests, fn, query_imglist)
  # write_stress_report(st_rpt)
