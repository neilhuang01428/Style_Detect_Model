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
import operator
import collections
import copy

#說明
#本程式碼為測試風格分類器的成效，方法是偵測完照片中的物件後，將機率餵給風格分類器，判斷風格是否判斷正確，由於一張圖只有一個風格標記
#故將圖片內所有物件都作為該照片風格的Ｘ，因此建議所選之圖片內只有一個人且物件都在人身上

#執行完成式後會有兩個產出
#1. 計算風格的confusion matrix
#2. 將風格認錯的照片 顯示在FP_picture_s資料夾中

#--------------------------------------------------------
clf_file ='./Pickle/final_label/best_xgb_Base model.pickle'

#行人分類器
modelType = "yolo"  #yolo or yolo-tiny
confThreshold = 0.4  #Confidence threshold
nmsThreshold = 0.8  #Non-maximum suppression threshold

classesFile = "./cfg/coco.names"
modelConfiguration = "./cfg/yolov3.cfg"
modelWeights = "./weights/yolov3.weights"

#物件偵測器
modelType2 = "yolo"  #yolo or yolo-tiny
confThreshold2 = 0.4  #Confidence threshold
nmsThreshold2 = 0.6   #Non-maximum suppression threshold

object_Threshold = 0.01 #object 與person 的IOU > objectThreshold 就代表屬於那個person

classesFile2 = "./cfg/deepfashion0827.names"
modelConfiguration2 = "./cfg/yolov30827.cfg"
modelWeights2 = "./weights/0827/yolov3_best.weights"



displayScreen = True  #Do you want to show the image on LCD?
outputToFile = True   #output the predicted result to image or video file

#分好風格之照片
abs_path = os.path.abspath(".")+'/'
formal = os.listdir(abs_path+"Picture/"+"Formal")
smart = os.listdir(abs_path+"Picture/"+"Smart casual")
casual = os.listdir(abs_path+"Picture/"+"Casual")

#test 這邊選擇的照片最好都只有單人
test_path = './test/test0823.txt'
with open(test_path, 'r') as fp:
    test_data = fp.readlines()
test_list =[]
for i in test_data:
    test_list.append(i.split("/")[-1]. strip ( '\n' ))



#Label & Box
fontSize = 2
color_forntSize =1
fontBold = 2
labelColor = (0,0,255)
boxbold = 5
boxColor = (255,255,255)
#--------------------------------------------------------

if(modelType=="yolo"):
    inpWidth = 608       #Width of network's input image
    inpHeight = 608      #Height of network's input image
else:
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image

#行人
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#物件
classes2 = None
with open(classesFile2, 'rt') as f:
    classes2 = f.read().rstrip('\n').split('\n')

#行人
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#物件
net2 = cv2.dnn.readNetFromDarknet(modelConfiguration2, modelWeights2)
net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

parser = argparse.ArgumentParser(description="Do you wish to scan for live hosts or conduct a port scan?")
parser.add_argument("-i", dest='image', action='store', help='Image')
parser.add_argument("-v", dest='video', action='store',help='Video file')


def getColorList():
    dict = collections.defaultdict(list)

    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list

    # 灰色
    lower_gray = np.array([0, 0, 46])
    upper_gray = np.array([180, 43, 220])
    color_list = []
    color_list.append(lower_gray)
    color_list.append(upper_gray)
    dict['gray'] = color_list

    # 白色
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list

    # 紅色
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red'] = color_list

    # 紅色2
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list

    # 橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    # 黃色
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    # 綠色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    # 青色
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    # 藍色
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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color = dict()
    color_dict = getColorList()
    area_sum = 0

    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        if d == 'red2':
            color['red'] += sum
        else:
            color[d] = sum
        area_sum += sum
    for i in color:
        color[i] = color[i] / area_sum
    return color

def posses_conf(one_x, one_y, one_w, one_h, two_x, two_y, two_w, two_h):  #one =人  two =object
    if abs(two_x-one_x)>=one_w or abs(two_y - one_y) >=one_h:
        return False
    else:
        return True

def getROI_Color(roi):
    mean_blue = np.mean(roi[:,:,0])
    mean_green = np.mean(roi[:,:,1])
    mean_red = np.mean(roi[:,:,2])
    actual_name, closest_name = get_colour_name((mean_red, mean_green, mean_blue))
    return actual_name, closest_name, (mean_blue, mean_green, mean_red)

#-----------------------------------------------------------------

# Get the names of the output layers
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(style_type , conf, center_x, center_y, width, height,input="person"):
    left = int(center_x - width / 2)
    top = int(center_y - height / 2)
    right = int(center_x + width / 2)
    bottom = int(center_y + height / 2)
    confident = '%.2f' % conf
    if input !="object":
        if style_type ==0:
            labelName = '%s:%s' % ("Formal", confident)
        elif style_type ==1:
            labelName = '%s:%s' % ("Smart casual", confident)
        elif style_type ==2:
            labelName = '%s:%s' % ("Casual", confident)
        else:
            labelName = '%s' % ("Analyzing...")

        #用opencv繪框需要用到「左上角」、「右下角」的座標
        cv2.rectangle(frame, (left, top), (right, bottom), boxColor, boxbold)
        cv2.putText(frame, labelName, (left, top-10), cv2.FONT_HERSHEY_COMPLEX, fontSize, labelColor, fontBold)
        # print(labelName)
    else:
        labelName = '%s:%s' % (style_type,confident)
        cv2.rectangle(frame, (left, top), (right, bottom), boxColor, boxbold)
        cv2.putText(frame, labelName, (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, fontSize, labelColor, fontBold)

def draw_rectangle(frame, label, center_x, center_y, width, height):
    left = int(center_x - width / 2)
    top = int(center_y - height / 2)
    right = int(center_x + width / 2)
    bottom = int(center_y + height / 2)

    # 用opencv繪框需要用到「左上角」、「右下角」的座標
    cv2.rectangle(frame, (left, top), (right, bottom), boxColor, boxbold)
    cv2.putText(frame, label, (left, top + 10), cv2.FONT_HERSHEY_COMPLEX, color_forntSize, labelColor, fontBold)

# def drawPred(style_type , conf, center_x, center_y, width, height,clf_type ="Best",input="person"):
#     left = int(center_x - width / 2)
#     top = int(center_y - height / 2)
#     right = int(center_x + width / 2)
#     bottom = int(center_y + height / 2)
#     confident = '%.2f' % conf
#     if input !="object":
#         if clf_type =="Best":
#             if style_type ==0:
#                 labelName = '%s:%s' % ("Formal", confident)
#             elif style_type ==1:
#                 labelName = '%s:%s' % ("Smart casual", confident)
#             elif style_type ==2:
#                 labelName = '%s:%s' % ("Casual", confident)
#             else:
#                 labelName = '%s' % ("Analyzing...")
#
#         elif clf_type =="Normal":
#             if style_type =="Formal":
#                 labelName = '%s:%s' % ("Formal", confident)
#             elif style_type =="Smart casual":
#                 labelName = '%s:%s' % ("Smart casual", confident)
#             elif style_type =="Casual":
#                 labelName = '%s:%s' % ("Casual", confident)
#             else:
#                 labelName = '%s' % ("Analyzing...")
#
#         #用opencv繪框需要用到「左上角」、「右下角」的座標
#         cv2.rectangle(frame, (left, top), (right, bottom), boxColor, boxbold)
#         cv2.putText(frame, labelName, (left, top-10), cv2.FONT_HERSHEY_COMPLEX, fontSize, labelColor, fontBold)
#         # print(labelName)
#     else:
#         labelName = '%s:%s' % (style_type,confident)
#         cv2.rectangle(frame, (left, top), (right, bottom), boxColor, boxbold)
#         cv2.putText(frame, labelName, (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, fontSize, labelColor, fontBold)

def postprocess(frame, outs, orgFrame, classifier):
    # frame 為原始圖片，orgFrame 為處理後的圖片
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # outs 為所有框框的output 成果 19 x 19 x 3 = 1083 個框框，對應80個類別的機率(前五個數值為，信心值與bbox位置)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classifier == "object":
                #                CT = object_Threshold[classId]
                CT = object_Threshold
            else:
                CT = confThreshold
            if confidence > CT:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                # left = int(center_x - width / 2)
                # top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                # box
                boxes.append([center_x, center_y, width, height])

    # 選出要print出的框框：選機率最大、不重疊的bbox
    # indices 為NMS 處理後，挑選出的bbox，每個bbox的內容有[index,]
    if classifier == "object":
        NT = nmsThreshold2
    else:
        NT = nmsThreshold
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CT, NT)

    # 如果classifier為"person"，或其他， 則return的bbox要先篩選為person類別
    # 若classifier為"object" 則 return的 bbox為物件的bbox
    if classifier == "object":
        return [[classIds[i[0]], confidences[i[0]], boxes[i[0]][0], boxes[i[0]][1], boxes[i[0]][2], boxes[i[0]][3],
                 orgFrame] for i in indices]
    else:
        return [[classIds[i[0]], confidences[i[0]], boxes[i[0]][0], boxes[i[0]][1], boxes[i[0]][2], boxes[i[0]][3],
                 orgFrame] for i in indices if classIds[i[0]] == 0]

def sty_clf(label, pro, color):
    color_name = ['black', 'gray', 'white', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
    detect_list = [0 for i in range(len(classes2)+len(color_name))]  # 21 label + 10 color
    for i in range(len(label)):
        detect_list[classes2.index(label[i])] += pro[i]

    total_col_name = classes2 + color_name

    for i in range(len(color_name)):
        detect_list[i +  len(classes2)] = color[color_name[i]]
    detect_list = pd.DataFrame(detect_list, index=total_col_name).T

    with open(clf_file, 'rb') as file:
        clf_m = pickle.load(file)
    result = clf_m.predict(detect_list)[0]
    result_proba = clf_m.predict_proba(detect_list)[0][result]
    return result, result_proba


#選擇輸入照片、影片或直接開啟攝像頭
args = parser.parse_args()
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(outputFile, fourcc, 30.0, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
else:
    # Webcam input
    cap = cv2.VideoCapture(0)


#===============================Pipeline=============================




###主程式###
# print(test_list)
count =0
# label_num = len(form_dict)-3

FP =dict()
FP_pro =dict()
label_dict =dict()

# FN =0

form_dict = dict()
for i in ["Formal", "Smart", "Casual"]:
    form_dict[i] = [0, 0, 0]

for i in [formal,smart,casual]:
    for f in i:
        if f.split(".")[1] != "DS_Store"  and f.split("+")[0].strip(" ")+".jpg" in test_list:
            print(f)
            count+=1
            if count%100==0:
                print(count)
            if i ==formal:
                cap = cv2.VideoCapture(abs_path +"Picture/"+"Formal/" +f)
            elif i == smart:
                cap = cv2.VideoCapture(abs_path +"Picture/"+"Smart casual/"+f)
            else:
                cap = cv2.VideoCapture(abs_path +"Picture/"+"Casual/"+f)

            hasFrame, frame = cap.read()
            orgFrame = frame.copy()
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
            net2.setInput(blob)
            outs2 = net2.forward(getOutputsNames(net2))
            object_boxs = postprocess(frame, outs2, orgFrame, "object")

            pred_o_label = []
            pred_o_pro = []
            for o in object_boxs:
                pred_o_label.append(classes2[o[0]])
                pred_o_pro.append(o[1])


            if pred_o_label ==[]:
                pass
            else:
                #進行物件、顏色偵測-->分類

                left_min = 1000000
                right_max = -100
                top_min = 1000000
                bottom_max = -100

                for dt in object_boxs:
                    left = int(dt[2] - dt[4] / 2)
                    top = int(dt[3] - dt[5] / 2)
                    right = int(dt[2] + dt[4] / 2)
                    bottom = int(dt[3] + dt[5] / 2)

                    if top < top_min:
                        top_min = top
                    if left < left_min:
                        left_min = left
                    if right > right_max:
                        right_max = right
                    if bottom > bottom_max:
                        bottom_max = bottom
                if left_min < 0:
                    left_min = 0
                if top_min < 0:
                    top_min = 0
                if bottom_max > frame.shape[0]:
                    bottom_max = frame.shape[0]
                if right_max > frame.shape[1]:
                    right_max = frame.shape[1]
                # print(top_min)
                # print(bottom_max)
                # print(left_min)
                # print(right_max)
                # print(frame.shape)
                crop_img = frame[top_min:bottom_max, left_min:right_max]
                color_temp = get_color(crop_img)
                c = sorted(get_color(crop_img).items(), key=operator.itemgetter(1))[-3:]
                main_color = [i for i in c]
                # main_color_prob =[i[1] for i in c]

                # mainc_color 為物件範圍的主要3個色系
                main_color.reverse()
                result, result_proba = sty_clf(pred_o_label, pred_o_pro, color_temp)

                print(result)
                print(result_proba)

                #confusion matrix
                if i ==formal:
                    form_dict["Formal"][result] +=1
                    if result !=0: # FP 情況
                        if f not in FP:
                            FP[f] =["P%s_A%s"%(result,0)]
                            FP_pro[f] =[result_proba]
                            label_dict[f] =pred_o_label
                        else: #不太有這種情況，除非有重複照片
                            FP[f].append("P%s_A%s"%(result,0))
                            FP_pro[f].append(result_proba)
                            label_dict[f].append((pred_o_label))
                elif i ==smart:
                    form_dict["Smart"][result] += 1
                    if result !=1: # FP 情況
                        if f not in FP:
                            FP[f] =["P%s_A%s"%(result,1)]
                            FP_pro[f] =[result_proba]
                            label_dict[f] =pred_o_label
                        else: #不太有這種情況，除非有重複照片
                            FP[f].append("P%s_A%s"%(result,1))
                            FP_pro[f].append(result_proba)
                            label_dict[f].append((pred_o_label))

                else:
                    form_dict["Casual"][result] += 1
                    if result !=2: # FP 情況
                        if f not in FP:
                            FP[f] =["P%s_A%s"%(result,2)]
                            FP_pro[f] =[result_proba]
                            label_dict[f] =pred_o_label
                        else: #不太有這種情況，除非有重複照片
                            FP[f].append("P%s_A%s"%(result,2))
                            FP_pro[f].append(result_proba)
                            label_dict[f].append((pred_o_label))

result = pd.DataFrame(form_dict, index=["Pred_F", "Pred_S", "Pred_C"])
# print(clf_file)
print(result)
result.to_excel("./Performance/Style/Style_performance.xlsx")

#FP
number_fp = 0
for i in FP:
    pic_name =  i.split(".")[0].split("+")[0].strip(" ")+"_"+str(FP[i][0])+"_"+ str("%.3f"%FP_pro[i][0])+"_"+ "+".join(str(x)for x in label_dict[i]) +".jpg"
    number_fp +=1
    try:
        if i in formal:
            shutil.copy(abs_path+"Picture/Formal/" + i,".Performance/Style/FP_picture_s/"+pic_name)
        elif i in smart:
            shutil.copy(abs_path + "Picture/Smart casual/" + i, ".Performance/Style/FP_picture_s/" + pic_name)
        else:
            shutil.copy(abs_path + "Picture/Casual/" + i, ".Performance/Style/FP_picture_s/" + pic_name)
    except BaseException:
        pass
