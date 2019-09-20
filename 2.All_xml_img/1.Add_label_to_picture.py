import os
import sys
import xml.etree.ElementTree as ET
import glob
import shutil

abs_path = os.path.abspath(".")+'/'

#將照片檔名加上label之後，複製到check_label資料夾，以供檢視標籤是否標錯
#每次執行重新清空資料夾
target_file =abs_path + 'check_label/'
shutil.rmtree(target_file)
os.mkdir(target_file)

img_list = os.listdir(abs_path+'Picture')
xml_list = os.listdir(abs_path+'Anno')

# count = 0
index =0
for i in img_list:
    if i.split(".")[1] !="DS_Store":
        label = []
        xml_path =abs_path +"Anno/"+ i.split(".")[0]+ ".xml"
        xml_content = ET.parse(xml_path)
        new_name = i.split(".")[0]
        for elem in xml_content.iter(tag='name'):
            new_name = new_name + "+" +elem.text
            #計算某標籤數量
            # if elem.text =="C_T-shirt":
            #     count = count +1
        new_name = new_name + "."+i.split(".")[1]
        #顯示目前跑了多少檔案
        if index %100==0:
            index = index + 1
            print(index)
        try:
            shutil.copyfile(abs_path + 'Picture/' + i, target_file + new_name)
        except BaseException:
            pass
# print(count)