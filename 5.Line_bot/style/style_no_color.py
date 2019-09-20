import os, time
import argparse
import cv2
import numpy as np
import pickle
import collections


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#model = models.load_model('./model/emo0809_epo50lr6.h5')
#static_alignment_path = os.path.join(os.getcwd(), 'static', 'alignment')

#行人分類器
modelType = "yolo"  #yolo or yolo-tiny
confThreshold = 0.05  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
#objectThreshold = 0.5 #object 與person 的IOU > objectThreshold 就代表屬於那個person
object_Threshold =[0.6,0.5,0.5,0.7,0.6,0.5,0.65,0.6,0.6,0.6,0.5,0.4,0.3,0.3,0.4,0.3,0.3,0.5,0.5,0.3,0.5]
classesFile = "./config/cfg/coco.names"
modelConfiguration = "./config/cfg/yolov3.cfg"
modelWeights = "./config/weights/yolov3.weights"

#物件偵測器
modelType2 = "yolo"  #yolo or yolo-tiny
confThreshold2 = 0.4  #Confidence threshold
nmsThreshold2 = 0.5   #Non-maximum suppression threshold

classesFile2 = "./config/cfg/chinese_label.names" #deepfashion0826.names
modelConfiguration2 = "./config/cfg/yolov30827.cfg"
modelWeights2 = "./config/weights/0827/yolov3_best.weights"




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
#print(classes2)

#行人
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#物件
net2 = cv2.dnn.readNetFromDarknet(modelConfiguration2, modelWeights2)
net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


#----------function---------------
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
# if sum > maxsum :
#     maxsum = sum
#     color = d


# return color

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
                CT = object_Threshold[classId]
            else:
                CT =confThreshold
            if confidence > CT:
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

    #選出要print出的框框：選機率最大、不重疊的bbox
    #indices 為NMS 處理後，挑選出的bbox，每個bbox的內容有[index,]
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    #如果classifier為"person"，或其他， 則return的bbox要先篩選為person類別
    #若classifier為"object" 則 return的 bbox為物件的bbox
    if classifier =="object":
        return [[classIds[i[0]],confidences[i[0]],boxes[i[0]][0],boxes[i[0]][1],boxes[i[0]][2],boxes[i[0]][3],orgFrame] for i in indices]
    else:
        return [[classIds[i[0]],confidences[i[0]],boxes[i[0]][0],boxes[i[0]][1],boxes[i[0]][2],boxes[i[0]][3],orgFrame] for i in indices if classIds[i[0]]==0]

def sty_clf(label,pro,clf="Best"):
    detect_list = [ 0 for i in range(len(classes2))]
    # print(len(detect_list))
    for i in range(len(label)):
        # print(classes2.index(label[i]))
        detect_list[classes2.index(label[i])] +=  pro[i]
    # print(detect_list)
    
    if clf =="Best":
        with open('./config/pickle/final_label/RF.pickle', 'rb') as file:
            clf_m = pickle.load(file)
        result = clf_m.predict([detect_list])[0]
        result_proba = clf_m.predict_proba([detect_list])[0][result]
        return result, result_proba
    elif clf =="Normal":
        style_count = [0,0,0]
        for i in range(len(label)):
            if label[i] in formal_object:
                style_count[0]+=pro[i]
            elif label[i] in smart_casual_object:
                style_count[1]+=pro[i]
            elif label[i] in casual_object:
                style_count[2]+=pro[i]
            else:
                pass
        if style_count[0]/sum(style_count) >= 2/3:
            result = "Formal"
            result_proba = style_count[0]/sum(style_count)
        elif (style_count[0]+style_count[1])/sum(style_count)>= 0.6:
            result = "Smart casual"
            result_proba= (style_count[0] + style_count[1]) / sum(style_count)
        else:
            result = "Casual"
            result_proba = (style_count[2])/sum(style_count)
        return result , result_proba

def predict(img_path):
    result = 'none'
    confidence = 0.00
    cap = cv2.VideoCapture(img_path)
    hasFrame, frame = cap.read()
    orgFrame = frame.copy()
    #將frame 匯入網路中分析最後輸出 output層的結果：outs
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)#行人偵測的網路:net
    outs = net.forward(getOutputsNames(net)) #yolov3.weights 慢是因為卡在coco 訓練集的分類器有80類別權重（240 mb）
    net2.setInput(blob)
    outs2 = net2.forward(getOutputsNames(net2))
    #postprocess = 每個人、每個物件的 classid 、conf 、x y w h
    person_boxs = postprocess(frame, outs, orgFrame,"person")
    object_boxs = postprocess(frame, outs2, orgFrame,"object")
    
    if person_boxs==[] and object_boxs==[]: #沒有人 也沒有物件的狀況
        resp = dict()
        resp["P_isRecognized"] = False
        resp["numOfPeople"] = 0
        resp["style"] = result #None
        resp["style_confidence"] = confidence #0
        resp["O_isRecognized"] =False
        resp["numOfObject"] = 0
        resp["object"]=[]
        resp["object_prob"]=[]
        resp["info"] = "no people and object detected"
        return resp
    elif person_boxs!=[] and object_boxs ==[]:
        resp = dict()
        resp["P_isRecognized"] = True
        resp["numOfPeople"] = len(person_boxs) #人數
        resp["style"] = result #None
        resp["style_confidence"] = confidence #0
        resp["O_isRecognized"] = False
        resp["numOfObject"] = 0
        resp["object"]=[]
        resp["object_prob"]=[]
        resp["info"] = "no people detected"
        return resp
    elif person_boxs==[] and object_boxs !=[]:
        resp = dict()
        resp["P_isRecognized"] = False
        resp["numOfPeople"] = 0 #人數
        label = []
        pro =[]
        object_posi =[]
        for o in object_boxs:
            label.append(classes2[o[0]])
            pro.append(o[1])
            object_posi.append([o[2],o[3],o[4],o[5]])
        clf_type ="Best"
        style_type ,style_proba = sty_clf(label,pro,clf=clf_type)
        resp["style"] = style_type #None
        resp["style_confidence"] = style_proba #0
        resp["O_isRecognized"] = True
        resp["numOfObject"] = len(label)
        resp["object"]=label
        resp["object_prob"]=pro
        resp["info"] = "no people detected but some object detected"
        return resp
    else:
        resp = dict()
        resp["P_isRecognized"] = True
        resp["numOfPeople"] = len(person_boxs) #人數
        resp["style"] = [] #None
        resp["style_confidence"] = [] #0
        resp["O_isRecognized"] = []
        resp["numOfObject"] = []
        resp["object"]=[]
        resp["object_prob"]=[]
        for p in person_boxs:
            label = []
            pro =[]
#            object_posi =[]
            for o in object_boxs:
                # if calcIOU(p[2],p[3],p[4],p[5],o[2],o[3],o[4],o[5]) >= objectThreshold :
                if posses_conf(p[2],p[3],p[4],p[5],o[2],o[3],o[4],o[5])==True:
                    label.append(classes2[o[0]])
                    pro.append(o[1])
#                    object_posi.append([o[2],o[3],o[4],o[5]])
            clf_type ="Best"
            if label!=[]:
                style_type ,style_proba = sty_clf(label,pro,clf=clf_type)
                resp["style"].append(style_type) #None
                resp["style_confidence"].append(style_proba) #0
                resp["O_isRecognized"].append(True)

            else:
                resp["style"].append(None) #None
                resp["style_confidence"].append(None) #0
                resp["O_isRecognized"].append(False)
#                resp["object"].append(None)
#                resp["object_prob"].append(None)
            resp["object"].append(label)
            resp["object_prob"].append(pro)
            resp["numOfObject"].append(len(label))

            
        resp["info"] = "some people and object have been detected"
        return resp


if __name__ == "__main__":

    img = predict("/Users/NeilHuang/Desktop/工作/中國信託/輪調/視覺實驗室/風格辨識/style detect pipline/style分類/test_picture/Dale_F_belt389.jpg")
    print(img)
