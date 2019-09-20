import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import glob
import shutil
import pandas as pd
import os, time ,sys
import argparse
import cv2
import numpy as np
import collections

#本程式碼為產出風格分類器之datasets，Ｘ：物件之機率 、 Y :風格標籤

abs_path = os.path.abspath(".")+'/'

formal = os.listdir(abs_path+"Picture/Formal")
smart = os.listdir(abs_path+"Picture/Smart casual")
casual = os.listdir(abs_path+"Picture/Casual")

#test為要拿來產出 datasets之圖片列表
test_path = './test/test0823.txt'
with open(test_path, 'r') as fp:
    test_data = fp.readlines()
test_list =[]
for i in test_data:
    test_list.append(i.split("/")[-1].strip( '\n' ))

#--------------------------------------------------------
modelType = "yolo"  #yolo or yolo-tiny
confThreshold = 0.1  #Confidence threshold
nmsThreshold = 0.6  #Non-maximum suppression threshold
object_Threshold = [0.6,0.5,0.5,0.7,0.6,0.5,0.65,0.6,0.6,0.6,0.5,0.4,0.3,0.3,0.4,0.3,0.3,0.5,0.5,0.3,0.5]

classesFile = "./cfg/deepfashion0827.names"
modelConfiguration = "./cfg/yolov30827.cfg"
modelWeights = "./weights/0827/yolov3_best.weights"


displayScreen = False  #Do you want to show the image on LCD?
outputToFile = True   #output the predicted result to image or video file

#Label & Box
fontSize = 2
fontBold = 1
labelColor = (0,0,255)
boxbold = 3
boxColor = (255,255,255)

if(modelType=="yolo"):
    inpWidth = 608       #Width of network's input image
    inpHeight = 608      #Height of network's input image
else:
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#detect_form.xlsx 欄位為 class(x)＋3種style(y) +10種顏色
form_dict = dict()
for i in classes:
    form_dict[i]=list()
form_dict["Formal"] =list()
form_dict["Smart"] =list()
form_dict["Casual"] =list()
c_names = ['black', 'gray', 'white', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
for i in c_names:
    form_dict[i] =list()



net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#-----------------------------------------------------------------
#偵測顏色比例  input = frame  output=顏色與比例
def getColorList():
    dict = collections.defaultdict(list)

    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list

    #灰色
    lower_gray = np.array([0, 0, 46])
    upper_gray = np.array([180, 43, 220])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    dict['gray']=color_list

    # 白色
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list

    #紅色
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red']=color_list

    # 紅色2
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list

    #橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    #黃色
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    #綠色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    #青色
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    #藍色
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # 紫色
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list

    return dict
def get_color(frame):
    # print('go in get_color')
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    color = dict()
    color_dict = getColorList()
    area_sum=0

    for d in color_dict:
        mask = cv2.inRange(hsv,color_dict[d][0],color_dict[d][1])
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary,None,iterations=2)
        cnts, hiera = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum+=cv2.contourArea(c)
        if d =='red2':
            color['red'] +=sum
        else:
            color[d] = sum
        area_sum+=sum
    for i in color:
        color[i] =color[i]/area_sum
    return color



    # return color
# Get the names of the output layers
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
def postprocess(frame, outs, orgFrame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            confThreshold =object_Threshold[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    confThreshold = min(object_Threshold)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    all_label =[]
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        all_label.append([classes[classIds[i]], '%.2f'%confidences[i] , left, top, left + width, top + height])
    return all_label

###主程式###
count =0
for i in [formal,smart,casual]:
    for f in i:
        #是否要設定在test範圍內，或所有資料
        # if f.split("+")[0].split(".")[0].replace(' ', '') +".jpg" in test_list:
        count = count +1
        # start = time.time()
        if f.split(".")[1] != "DS_Store":# and f in test_list:
            if i ==formal:
                cap = cv2.VideoCapture(abs_path +"Formal/" +f)
            elif i == smart:
                cap = cv2.VideoCapture(abs_path +"Smart casual/"+f)
            else:
                cap = cv2.VideoCapture(abs_path +"Casual/"+f)
            hasFrame, frame = cap.read()
            orgFrame = frame.copy()
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            outs = net.forward(getOutputsNames(net))
            detect_label = postprocess(frame, outs, orgFrame)

            if detect_label ==[]:
                pass
            else:
                ###加顏色比例到form_dict裡
                left_min = 1000000
                right_max = -100
                top_min = 1000000
                bottom_max = -100
                # print(detect_label)

                for dt in detect_label:
                    if dt[3]<top_min:
                        top_min=dt[3]
                    if dt[2]<left_min:
                        left_min =dt[2]
                    if dt[4]>right_max:
                        right_max=dt[4]
                    if dt[5]>bottom_max:
                        bottom_max =dt[5]
                if left_min<0:
                    left_min =0
                if top_min<0:
                    top_min =0
                crop_img = frame[top_min:bottom_max, left_min:right_max]
                color_temp =get_color(crop_img)
                for ct in color_temp:
                    form_dict[ct].append(color_temp[ct])
                # print(color_temp)
                # plt.imshow(crop_img)
                # plt.show()

                ####加label機率到form_dict裡
                label = []
                pro =[]
                for l in detect_label:
                    label.append(l[0])
                # p * A
                    pro.append(float(l[1]))


                for o in form_dict:
                    if o not in ["Formal","Smart","Casual"] and o not in c_names:
                        if o in label:
                            #如果照片中有出現兩個同樣的物件，把那個物件所有機率加總起來
                            o_score =0
                            for oi in range(len(label)):
                                if label[oi] ==o:
                                    o_score += pro[oi]
                            form_dict[o].append(o_score)
                        else:
                            form_dict[o].append("0")
                ####加風格到form_dict裡
                if i==formal:
                    form_dict["Formal"].append("1")
                    form_dict["Smart"].append("0")
                    form_dict["Casual"].append("0")
                elif i==smart:
                    form_dict["Formal"].append("0")
                    form_dict["Smart"].append("1")
                    form_dict["Casual"].append("0")
                else:
                    form_dict["Formal"].append("0")
                    form_dict["Smart"].append("0")
                    form_dict["Casual"].append("1")
        #     print(form_dict)
        # print(f)
        if count % 100 ==0:
            print(round(count/len(test_list),2)) #

data = pd.DataFrame(form_dict)
data.to_excel("./Form/detect_form_[date].xlsx",index=False)