#!/bin/bash

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
echo $SCRIPTS_DIR

source "${SCRIPTS_DIR}/apicfg.sh"

image=$1
if [ -z ${image} ]; then
  image=${IMAGE_PATH}
fi

echo $image
echo $API_URL

curl -X POST -F image=@${image} "${API_URL}"

## https://stackoverflow.com/questions/19116016/what-is-the-right-way-to-post-multipart-form-data-using-curl
## curl -X POST -F image=@${image} -F q=matterport-coco_things-1 "${API_URL}"

## Ref: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
# The -X flag and POST value indicates we're performing a POST request.
# We supply -F image=@dog.jpg to indicate we're submitting form encoded data. The image key is then set to the contents of the dog.jpg file. Supplying the @ prior to dog.jpg implies we would like cURL to load the contents of the image and pass the data to the request.
# Finally, we have our endpoint: http://localhost:5000/predict
