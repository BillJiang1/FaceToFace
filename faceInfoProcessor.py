import faceRecognition
import json
import os
"""
解析json文件，设计的json储存数据格式如下（嵌套字典）：
  facePosition = {
      'image-001.jpg' : {'top': 10, 'bottom': 20, 'left': 30, 'right': 40},
      '2' :{'top': 10, 'bottom': 20, 'left': 30, 'right': 40},
       ...
      'videoSnapshotCode':{...}
  }
"""

jsonPath = 'json/facePositions.json'


# 传入json文件储存路径jsonPath以及视频帧文件名，返回json文件中指定视频截图的人脸位置信息
def jsonResolve(jsonPath, img_name):
    with open(jsonPath, 'r') as file:
        facePosition = json.load(file)
    top = facePosition[img_name]['top']
    bottom = facePosition[img_name]['bottom']
    left = facePosition[img_name]['left']
    right = facePosition[img_name]['right']
    return top, bottom, left, right


# 基于faceRecognition进行人脸识别，并逐帧将人脸位置信息写入json文件
def jsonGenerate(jsonPath):
    json_content = {}
    faceRecognition.get_menu_descriptors()
    for img_name in os.listdir('VideoSnapshot'):
        top,bottom,left,right = faceRecognition.face_recognition('VideoSnapshot/'+img_name,'Phoebe')
        face_pos = {}
        face_pos['top'] = top
        face_pos['bottom'] = bottom
        face_pos['left'] = left
        face_pos['right'] = right
        json_content['{}'.format(img_name)] = face_pos
    with open(jsonPath,'w') as file:
        json.dump(json_content,file)


if __name__ == '__main__':
    # jsonGenerate(jsonPath)

