import os

FLASK_IP = "0.0.0.0"
FLASK_PORT = "5050"

## Flask python server
API_URL = "http://"+FLASK_IP+":5050/predict"

IMAGE_PATH = "/aimldl-dat/samples/lanenet/7.jpg"
WEIGHTS_PATH = "/aimldl-cod/external/lanenet-lane-detection/model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt"

# Set to true if you want to resize to 720*1280
# RESIZE = 'True'
RESIZE = "Flase"
ALLOWED_IMAGE_TYPE = ['.pdf','.png','.jpg','.jpeg','.tiff','.gif']
# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 500
# NUM_REQUESTS=1
SLEEP_COUNT = 0.05
SLEEP_TIME = 300
SLEEP_TIME = 3

IMAGE_API = 'http://10.4.71.121/stage/maze/vs/trackSticker.php?action=getImage&image='
