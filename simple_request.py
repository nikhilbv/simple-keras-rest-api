# USAGE
# python simple_request.py

# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5050/predict"
# IMAGE_PATH = "dog.jpg"
# IMAGE_PATH = "/aimldl-cod/practice/nikhil/sample-images/7.jpg"
IMAGE_PATH = "/aimldl-cod/practice/nikhil/sample-images/7.jpg"
# image_name = IMAGE_PATH.split('/')[-1].split('.')[0]

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = { "image": image }

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was sucessful
if r["success"]:
	# loop over the predictions and display them
	# for (i, result) in enumerate(r["predictions"]):
	# 	print("{}. {}: {:.4f}".format(i + 1, result["label"],
	# 		result["probability"]))
  # print(r["predictions"])
  print(r)

# otherwise, the request failed
else:
	print("Request failed")
