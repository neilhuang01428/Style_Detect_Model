import os, time
import argparse
import cv2
import numpy as np
import pickle
import collections
import operator
import pandas as pd

#本程式為影響

#--------------------------------------------------------
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

real_classesFile = "./cfg/deepfashion0827.names" 
modelConfiguration2 = "./cfg/yolov30827.cfg"
modelWeights2 = "./weights/0827/yolov3_best.weights"
clf_file ='./Pickle/final_label/best_xgb_Base model.pickle'

displayScreen = True  #Do you want to show the image on LCD?
outputToFile = True   #output the predicted result to image or video file


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
real_classes = None
with open(real_classesFile, 'rt') as f:
    real_classes  = f.read().rstrip('\n').split('\n')

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
    detect_list = [0 for i in range(len(real_classes)+len(color_name))]  # 21 label + 10 color
    for i in range(len(label)):
        detect_list[real_classes.index(label[i])] += pro[i]

    total_col_name = real_classes + color_name

    for i in range(len(color_name)):
        detect_list[i +  len(real_classes)] = color[color_name[i]]
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

#測試鏡頭fps
cap.set(cv2.CAP_PROP_FPS,60)
fps = cap.get(cv2.CAP_PROP_FPS)
print('fps:',fps)
print(cv2.waitKey(1))

###進入pipeline
#按ESC即跳出
while cv2.waitKey(1) < 0:

    #hasFrame 代表有沒有跳出視窗，frame 代表視窗截圖，在此為還未被處理過、要被輸入到yolo分析的原始圖
    hasFrame, frame = cap.read()
    #resize
    # if (args.image):
    #     frame = cv2.resize(frame,(224,224))
    #沒視窗即將截圖匯出成picture
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv2.waitKey(3000)
        break

    orgFrame = frame.copy() #將要被處理的frame

    #將frame 匯入網路中分析最後輸出 output層的結果：outs
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)#行人偵測的網路:net
    outs = net.forward(getOutputsNames(net)) #yolov3.weights 慢是因為卡在coco 訓練集的分類器有80類別權重（240 mb）
    person_boxs = postprocess(frame, outs, orgFrame,"person")

    if person_boxs==[]: #沒有人就什麼事都沒發生
        pass
    else:
        #有人的話就偵測物件
        net2.setInput(blob)
        outs2 = net2.forward(getOutputsNames(net2))
        object_boxs = postprocess(frame, outs2, orgFrame,"object")
        if object_boxs==[]:
            pass
        else:
            ##有person 又有物件的情況下
            for p in person_boxs:

                label = []
                pro =[]
                object_posi =[]
                posses_boxs = []
                for o in object_boxs:
                    if posses_conf(p[2],p[3],p[4],p[5],o[2],o[3],o[4],o[5])==True:
                        label.append(real_classes[o[0]])
                        pro.append(o[1])
                        posses_boxs.append(o)
                        object_posi.append([o[2],o[3],o[4],o[5]])#

                if label !=[]:
                    #先畫物件的框
                    for n in range(len(label)):
                        drawPred(label[n],pro[n],object_posi[n][0],object_posi[n][1],object_posi[n][2],object_posi[n][3],input="object")

                    #再畫風格分類、色系的框
                    left_min = 1000000
                    right_max = -100
                    top_min = 1000000
                    bottom_max = -100

                    for dt in posses_boxs:
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
                    #main_color_prob =[i[1] for i in c]

                    # mainc_color 為物件範圍的主要3個色系
                    main_color.reverse()
                    style_type ,style_proba = sty_clf(label,pro,color_temp)
                    drawPred(style_type,style_proba,p[2],p[3],p[4],p[5])
                    draw_rectangle(frame, "Color:%s , %s, %s" % (main_color[0][0],main_color[1][0],main_color[2][0]), p[2],p[3],p[4],p[5])

                    print(label)
                    print(pro)

        if (args.image):

            if(outputToFile):
                cv2.imwrite(outputFile, frame.astype(np.uint8))

            if(displayScreen):
                cv2.imshow("Predicted", frame)
        else:

            if(displayScreen):
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("frame",frame)#cv2.resize(frame,(800,600),interpolation = cv2.INTER_CUBIC)
                cv2.waitKey(1)

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
