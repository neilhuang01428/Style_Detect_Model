import os
import sys
import xml.etree.ElementTree as ET
import glob
import shutil

abs_path = os.path.abspath(".")+'/'
target_anno_path = abs_path + "temp_anno/"
target_picture_path = abs_path + "temp_picture/"

source_pic_list = os.listdir(abs_path + "Picture/")
source_anno_list = os.listdir(abs_path + "Anno/")
wrong_list = os.listdir(abs_path + "wrong/")
wrong_list_name = []

##流程：檢視change label 資料夾後，將錯誤的照片拉到wrong資料夾

#wrong_list_name：將要「處理」的檔案名稱列表
for i in wrong_list:
    #可用條件篩選要處理的標籤
    # if i != "S_man_pant" and i != "C_tank top" and i !="C_man_pant" and i.split(".")[1]!="DS_Store":
    if  i.split(".")[1] != "DS_Store":
        wrong_list_name.append(i.split("+")[0])
print(len(wrong_list_name))


move_count =0
#將wrong_list_name 中要「處理」的檔案，從source data中移到temp資料夾
for i in source_pic_list:
    if i.split(".")[1]!="DS_Store" and i.split(".")[0] in wrong_list_name:
        # 若要用labelImg 調整/檢視標籤的話，可以將picture也搬到temp資料夾
#        shutil.move(abs_path+"Picture/"+i,target_picture_path+i)
        shutil.move(abs_path + "Anno/" + i.split(".")[0]+".xml", target_anno_path + i.split(".")[0]+".xml")
        move_count = move_count +1

print(move_count)
