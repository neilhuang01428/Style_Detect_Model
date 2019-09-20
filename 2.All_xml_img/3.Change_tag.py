import os
import sys
import xml.etree.ElementTree as ET
import glob
import shutil


abs_path = os.path.abspath(".")+'/'

#檢查有無要處理的檔案沒有被抓到temp
# wrong_list = os.listdir(abs_path + "wrong")
# picture_list =os.listdir(abs_path + "temp_picture/")
# print(len(wrong_list))
# print(len(picture_list))
#
# for i in wrong_list:
#     p = i.split("+")[0]+"."+i.split(".")[-1]
#     if p not in picture_list:
#         print(i.split("+")[0])

#批量轉換標籤
chang_list = os.listdir(abs_path+"temp_anno/")
for x in chang_list:
    if x.split(".")[1] !="DS_Store":
        xml_path = abs_path + "temp_anno/" + x.split(".")[0] + ".xml"
        xml_content = ET.parse(xml_path)
        for elem in xml_content.iter(tag='name'):
            if elem.text =="要修改的標籤":
                elem.text = "修改後的標籤"
                xml_content.write(abs_path+"temp_anno/"+x , encoding="utf-8", xml_declaration=True)
                shutil.move(abs_path+"temp_anno/"+x, abs_path + "Anno/" + x)
                print(x)
                # break
