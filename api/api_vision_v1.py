import requests
import apicfg

## https://stackoverflow.com/questions/11551268/python-post-request-with-image-files
def call_vision_api(q=None, img=None, orgname=None, defaults=False):
  import base64
  import json

  if q is None:
    "tsd-2"

  if img is None:    
    img = '1.jpg'
  
  with open(img,'rb') as im:
    b64im = base64.b64encode(im.read())
    # print(b64im);
    imgname = img.split('/')[-1]
    print("imgname: {}".format(imgname))
    if defaults:
      params = {
        "image": b64im
        ,"name": imgname
        ,"q": q
        ,"orgname": orgname
      }
    else:
      ## check for default values
      params = {
        "image": b64im
        ,"name": imgname
        ,"q": q
        ,"orgname": orgname
      }
    
    # print(params)
    apiurl = apicfg.API_URL_V1
    try:
      # requests.add_header("Content-type", "application/x-www-form-urlencoded; charset=UTF-8")
      res = requests.post(apiurl
        ,data=params
        ,headers={"content-type":"application/x-www-form-urlencoded; charset=UTF-8"}
      )
      # print(res.headers)
      # print(res.text)
      print("-----------------------")

      data = json.loads(res.text)
      print("data: {}".format(data))
      print("-----------------------")

      print("data['api']: {}".format(data['api']))
      
      print("-----------------------")

      detections = data['api']['detections']

      # import matplotlib.ply as plt
    except Exception as e:
      print(e)




if __name__ == '__main__':
  imgpath = apicfg.IMAGE_PATH
  q = 'hmd'
  orgname = 'vidteq'
  call_vision_api(q=q, img=imgpath, orgname=orgname, defaults=False)
