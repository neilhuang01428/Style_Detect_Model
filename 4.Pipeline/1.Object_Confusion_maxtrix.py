import os, time
import argparse
import cv2
import numpy as np
import pickle
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import shutil

##本程式碼目的為：yolov3訓練完照片後，檢視每個標籤的預測成效
#會做三件事情：
# 1.將「FP」、「FN」、「TP」的照片分別顯示在FP_picture_o 、FN_picture_o、TP_picture_o 中檢視
# 2.計算每一個標籤的confusion_matrix 存到Label_performance.xlsx中 ，其中FP的數值會不準，原因是原圖檔沒標到該標籤，所以需要在「FP_picture_o」檢視

#注意：在執行本程式前，需要將FP_picture_o 、FN_picture_o、TP_picture_o  中檔案清空


#讀取所有照片
xml_path = "./Total_xml/"
xml_list = os.listdir(xml_path)


#分好風格之照片，為所有照片加總
abs_path = os.path.abspath(".")+'/'
formal = os.listdir(abs_path+"Formal")
smart = os.listdir(abs_path+"Smart casual")
casual = os.listdir(abs_path+"Casual")

#test 為要測試的範圍
test_path = './test/test0823.txt'
with open(test_path, 'r') as fp:
    test_data = fp.readlines()
test_list =[]
for i in test_data:
    test_list.append(i.split("/")[-1]. strip ( '\n' ))


#物件偵測器
modelType2 = "yolo"  #yolo or yolo-tiny
confThreshold2 = 0.3  #Confidence threshold
nmsThreshold2 = 0.6   #Non-maximum suppression threshold
#每一個label都各有一個閥值 ，也可以統一設定成一個值
object_Threshold = [0.6,0.5,0.5,0.7,0.6,0.5,0.65,0.6,0.6,0.6,0.5,0.4,0.3,0.3,0.4,0.3,0.3,0.5,0.5,0.3,0.5]
classesFile2 = "./cfg/deepfashion0827.names"
modelConfiguration2 = "./cfg/yolov30827.cfg"
modelWeights2 = "./weights/0827/yolov3_best.weights"


displayScreen = False  #Do you want to show the image on LCD?
outputToFile = True   #output the predicted result to image or video file

# Label & Box
fontSize = 2
fontBold = 1
labelColor = (0,0,255)
boxbold = 3
boxColor = (255,255,255)

if(modelType2=="yolo"):
    inpWidth = 608       #Width of network's input image
    inpHeight = 608      #Height of network's input image
else:
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image

#物件
classes2 = None
with open(classesFile2, 'rt') as f:
    classes2 = f.read().rstrip('\n').split('\n')

form_dict = dict()
for i in classes2:
    form_dict[i]=[0,0,0,0]
print(form_dict)

#物件
net2 = cv2.dnn.readNetFromDarknet(modelConfiguration2, modelWeights2)
net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Get the names of the output layers
def calcIOU(one_x, one_y, one_w, one_h, two_x, two_y, two_w, two_h):
    if ((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
        lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
        lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))

        rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
        rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))

        inter_w = abs(rd_x_inter - lu_x_inter)
        inter_h = abs(lu_y_inter - rd_y_inter)

        inter_square = inter_w * inter_h
        union_square = (one_w * one_h) + (two_w * two_h) - inter_square

        calcIOU = inter_square / union_square * 1.0
    else:
        return 0
    return calcIOU
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
def drawPred(frame,labelName, conf center_x, center_y, width, height):

    left = int(center_x - width / 2)
    top = int(center_y - height / 2)
    right = int(center_x + width / 2)
    bottom = int(center_y + height / 2)
    
    labelName = '%s : %.2f'%(labelName,conf)

    #用opencv繪框需要用到「左上角」、「右下角」的座標
    cv2.rectangle(frame, (left, top), (right, bottom), boxColor, boxbold)
    cv2.putText(frame, labelName, (left, top-10), cv2.FONT_HERSHEY_COMPLEX, fontSize, labelColor, fontBold)
    # print(labelName)
def postprocess(frame, outs, orgFrame,classifier):
    #frame 為原始圖片，orgFrame 為處理後的圖片
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    #outs 為所有框框的output 成果 19 x 19 x 3 = 1083 個框框，對應80個類別的機率(前五個數值為，信心值與bbox位置)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classifier=="object":
                T = object_Threshold[classId] #or  confThreshold2(所有標籤都有一樣的Threshold)
            else:
                T =confThreshold
            if confidence > T:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                # left = int(center_x - width / 2)
                # top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                #box
                boxes.append([center_x, center_y, width, height])
    confThreshold2 =min(object_Threshold)
    #選出要print出的框框：選機率最大、不重疊的bbox
    #indices 為NMS 處理後，挑選出的bbox，每個bbox的內容有[index,]
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold2, nmsThreshold2)

    #如果classifier為"person"，或其他， 則return的bbox要先篩選為person類別
    #若classifier為"object" 則 return的 bbox為物件的bbox
    if classifier =="object":
        return [[classIds[i[0]],confidences[i[0]],boxes[i[0]][0],boxes[i[0]][1],boxes[i[0]][2],boxes[i[0]][3],orgFrame] for i in indices]
    else:
        return [[classIds[i[0]],confidences[i[0]],boxes[i[0]][0],boxes[i[0]][1],boxes[i[0]][2],boxes[i[0]][3],orgFrame] for i in indices if classIds[i[0]]==0]


###主程式###
# print(test_list)
count =0
TP_n =0
TN_n =0
FP_n =0
FN_n =0
FP =dict()
FP_pro =dict()
FN =dict()
FN_pro =dict()
TP=dict()
TP_pro =dict()
print(len( test_list))

#所有圖檔街分散在Formal、 Smart casual、Casual 三個資料夾中
#亦可將自行調整來源
for i in [formal,smart,casual]:
    for f in i:
        #有些照片名稱為加上label後之標籤，故要先進行名字清理，並用test_list來限制要辨識之檔案
        if f.split(".")[1] != "DS_Store" and f.split("+")[0].split(".")[0].replace(' ', '') +".jpg" in test_list:
            pic_origin_name =f.split("+")[0].split(".")[0].replace(' ', '')
            count+=1
            if count%100==0:
                print(count)
            truth_label =[]

            x = xml_path + f.split("+")[0].split(".")[0].replace(' ', '')+".xml"
            xml_content = ET.parse(x)
            for elem in xml_content.iter(tag='name'):
                truth_label.append(elem.text)


            if i ==formal:
                cap = cv2.VideoCapture(abs_path +"Picture/Formal/" +f)
            elif i == smart:
                cap = cv2.VideoCapture(abs_path +"Picture/Smart casual/"+f)
            else:
                cap = cv2.VideoCapture(abs_path +"Picture/Casual/"+f)

            hasFrame, frame = cap.read()
            orgFrame = frame.copy()

            # start = time.time()
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

            net2.setInput(blob)
            outs2 = net2.forward(getOutputsNames(net2))
            object_boxs = postprocess(frame, outs2, orgFrame, "object")

            pred_o_label = []
            pred_o_pro = []
#            pred_x =[]
#            pred_y =[]
#            pred_w=[]
#            pred_h=[]

            for o in object_boxs:
                pred_o_label.append(classes2[o[0]])
                pred_o_pro.append(o[1])
#                pred_x.append(o[2])
#                pred_y.append(o[3])
#                pred_w.append(o[4])
#                pred_h.append(o[5])

            for pl in range(len(pred_o_label)):
                if pred_o_label[pl] not in truth_label:
                    #FP : 預測的label 不在ground truth裡
                    if f not in FP:
                        FP[f] =[pred_o_label[pl]]
                        FP_pro[f] =[pred_o_pro[pl]]
                    else:
                        FP[f].append(pred_o_label[pl])
                        FP_pro[f].append(pred_o_pro[pl])
                    form_dict[pred_o_label[pl]][2]+=1
                else:
                    #TP :預測的label 在ground truth裡
                    
                    if f not in TP:
                        TP[f]=[pred_o_label[pl]]
                        TP_pro[f]=[pred_o_pro[pl]]
                    else:
                        TP[f].append(pred_o_label[pl])
                        TP_pro[f].append(pred_o_pro[pl])
                    form_dict[pred_o_label[pl]][0]+=1

            for tl in range(len(truth_label)):
                if truth_label[tl] not in pred_o_label and truth_label[tl] in classes2:
                    #加入FN的dict
                    if f not in FN:
                        FN[f] =[truth_label[tl]]
                    else:
                        FN[f].append(truth_label[tl])
                    form_dict[truth_label[tl]][3]+=1
                elif truth_label[tl] in classes2:
                    #TN
                    form_dict[truth_label[tl]][1]+=1
            # except BaseException:
            #     pass


#FP & FN & TP 調到資料夾中檢查
number_fp = 0
#i 為每張圖片名稱
for i in FP:
    pic_name = str(number_fp)+"_"+ "+".join(str(FP[i][x])+"%.2f"%FP_pro[i][x] for x in range(len(FP[i]))) +".jpg"
    number_fp +=1
    if i in formal:
        shutil.copy(abs_path+"Picture/Formal/" + i,"./Performance/Object/FP_picture_o/"+pic_name)
    elif i in smart:
        shutil.copy(abs_path + "Picture/Smart casual/" + i, "./Performance/Object/FP_picture_o/" + pic_name)
    else:
        shutil.copy(abs_path + "Picture/Casual/" + i, "./Performance/Object/FP_picture_o/" + pic_name)
    # print(i,FP[i])

number_fn =0
#i 為每張圖片名稱
for i in FN:
    pic_name = str(number_fn)+"_"+ "+".join(str(x)for x in FN[i]) +".jpg"
    number_fn +=1
    if i in formal:
        shutil.copy(abs_path+"Picture/Formal/" + i,"./Performance/Object/FN_picture_o/"+pic_name)
    elif i in smart:
        shutil.copy(abs_path + "Picture/Smart casual/" + i, "./Performance/Object/FN_picture_o/" + pic_name)
    else:
        shutil.copy(abs_path + "Picture/Casual/" + i, "./Performance/Object/FN_picture_o/" + pic_name)

number_tp = 0
#i 為每張圖片名稱
for i in TP:
    pic_name = i.split(".")[0].split("+")[0].strip(" ") + "_" + "+".join(str(TP[i][x])+"%.2f"%TP_pro[i][x] for x in range(len(TP[i]))) + ".jpg"
    number_tp += 1
    try:
        if i in formal:
            shutil.copy(abs_path + "Picture/Formal/" + i, "./Performance/Object/TP_picture_o/" + pic_name)
        elif i in smart:
            shutil.copy(abs_path + "Picture/Smart casual/" + i, "./Performance/Object/TP_picture_o/" + pic_name)
        else:
            shutil.copy(abs_path + "Picture/Casual/" + i, "./Performance/Object/TP_picture_o/" + pic_name)
    except BaseException:
        pass


result = pd.DataFrame(form_dict,index=["TP","TN","FP","FN"]).T
result.to_excel("./Performance/Object/Label_performance0830.xlsx")
