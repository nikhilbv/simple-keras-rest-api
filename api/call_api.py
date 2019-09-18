__author__ = 'mangalbhaskar'
__version__ = '1.0'
"""
# API test script
# --------------------------------------------------------
# Copyright (c) 2019 Vidteq India Pvt. Ltd.
# Licensed under [see LICENSE for details]
# Written by mangalbhaskar
# --------------------------------------------------------
"""

import requests
import base64
import os

import apicfg

API_URL = apicfg.API_URL
print("API_URL: {}".format(API_URL))

def call_vision_api(args):
  filepath = args.filepath
  time_stats, data = None, None
  with open(filepath,'rb') as im:
    res = requests.post(API_URL
      ,files={"image": im}
      # ,data={"name": name}
    )

    print("res: {}".format(res.status_code))
    if res.status_code == 200:
      tt = res.elapsed.total_seconds()

      data = res.json()

      status = True
      K = ['filepath', 'status', 'tt_turnaround']+list(data['timings'].keys())
      V = [filepath, status, tt] + list(data['timings'].values())
      time_stats = dict(zip(K,V))
    else:
      status = False
      K = ['filepath', 'status', 'tt_turnaround']
      V = [filepath, status, -1]
      time_stats = dict(zip(K,V))
  
  time_stats['data'] = data
  print("time_stats: {}".format(time_stats))

  return time_stats


def parse_args(cfg):
  import argparse

  IMAGE_PATH = cfg.IMAGE_PATH
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--image"
    ,dest="filepath"
    ,default=IMAGE_PATH
    ,required=False)

  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args(apicfg)
  call_vision_api(args)
