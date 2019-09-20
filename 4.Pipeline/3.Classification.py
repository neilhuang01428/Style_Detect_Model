from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=DeprecationWarning)

import os
import sys ,time
import xml.etree.ElementTree as ET
import glob
import shutil
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter

# Use GridSearchCV to find the best parameters.
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.base import SamplerMixin
from imblearn.ensemble import EasyEnsemble, BalanceCascade
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import EasyEnsemble
from sklearn.model_selection import KFold, StratifiedKFold
def ensemble_method(method,classifier,X_train,y_train,X_test):  # EasyEnsemble和BalanceCascade的方法
    xx, yy = method.fit_sample(X_train, y_train)
    y_prob = np.zeros((len(X_test),3))
    for X_ensemble, y_ensemble in zip(xx, yy):
        model = classifier
        model.fit(X_ensemble, y_ensemble)
        if classifier==lr_cfl or classifier==svm_cfl:
            y_prob +=model.decision_function(X_test)
        elif classifier==xgb_cfl:
            X_test = np.array(X_test)
            y_prob += model.predict_proba(X_test)
        else:
            y_prob += model.predict_proba(X_test)
    y_prob = y_prob / len(xx)
    r = [0,1,2]
    y_pred = [r[int(np.argmax(i))] for i in y_prob]
    return y_pred, y_prob


#test
test_path = './test/test0823.txt'
with open(test_path, 'r') as fp:
    test_data = fp.readlines()
test_list =[]
for i in test_data:
    test_list.append(i.split("/")[-1].strip( '\n' ))

#讀取表格
data = pd.read_excel("./Form/detect_form_final.xlsx")

#選擇測試範圍的
# index = list(range(1,50))+list(range(6001,6050))+list(range(7501,7550))
# data =data.iloc[index,:]
# data.reset_index(inplace=True)
# print(data)

#label_encoder
data["Style_type"] = None
for r in range(len(data["Style_type"])):
    if data["Formal"][r] == 1:
        data.loc[r,"Style_type"]=0
    elif data["Smart"][r] ==1:
        data.loc[r,"Style_type"]=1
    else:
        data.loc[r,"Style_type"]=2

#讓 X 之欄位名稱 ＝label + 顏色 Ｙ欄位名稱為Formal 、Smart、Casual三種風格  或encoder之 0 1 2 分別代表前述之風格
classesFile = './cfg/deepfashion0827.names'
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


###切test、train data
#y為 Style_type = 0 or 1 or 2
x_col =classes
c_names = ['black', 'gray', 'white', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
for i in c_names:
    x_col.append(i)
X_en = data.loc[:,x_col]
y_en = data.loc[:,"Style_type"]

#Scaling
# for i in X_en:
#     X_en[i] = (X_en[i] - X_en[i].mean()) / (X_en[i].std())

#y為3個欄位 formal smart casual
X =data.loc[:,x_col]
y =data.loc[:,["Formal","Smart","Casual"]]
# print(X)
# print(y)

#切train & test的方法  train:test = 8:2  -->後面randomsearch時 會用cross_validation =5 將train 再切成5份
# SK = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
SEED = 2000
# x_train_en, x_test_en, y_train_en, y_test_en = train_test_split(X_en, y_en, test_size=.2, random_state=SEED)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)

# xgb_cfl = xgb.XGBClassifier()
# rf_cfl = RandomForestClassifier()
# lr_cfl= LogisticRegression()
# svm_cfl =svm.SVC()
# classifiers={
#     "XGB":xgb_cfl,
#     # "LR": lr_cfl,
#     "RF":rf_cfl
#     # "SVM": svm_cfl
# }

# sss = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)
# time_total =[]
# for key, classifier in classifiers.items():
#     t0 = time.time()
#     confu_temp=np.array([[0,0,0],[0,0,0],[0,0,0]])
#     for train, test in sss.split(X_en, y_en):
#         prediction , y_score =  ensemble_method(EasyEnsemble(random_state=42),classifier,X_en.iloc[train],y_en.iloc[train],X_en.iloc[test])
#         confu = confusion_matrix(y_en[test],prediction)
#         confu_temp += confu
#
#     time1 = (time.time() - t0)
#     # print(time1)
#     time_total.append(time1)
#     print(confu_temp)




# # 要測試的模型
search_iter = 5
# # Logistic Regression
# log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
# rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, cv = 5, n_iter=search_iter,n_jobs = -1)#n越大越好
#
# # KNears
# knears_params = {"n_neighbors": 3, 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
# rand_knears = RandomizedSearchCV(KNeighborsClassifier(), knears_params,n_iter=search_iter,n_jobs = -1)
#
# # Support Vector Classifier
# svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
# rand_svc = RandomizedSearchCV(svm.SVC(), svc_params,cv = 5,n_iter=search_iter,n_jobs = -1)
#
# # DecisionTree Classifier
# # tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}
# # rand_tree = RandomizedSearchCV(DecisionTreeClassifier(), tree_params,n_iter=search_iter,n_jobs = -1)
#
# RandomForest Classifier
rf_params = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(11, 50, num = 10)],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}
rand_rf =RandomizedSearchCV(RandomForestClassifier(),rf_params,cv = 5,n_iter=search_iter,n_jobs = -1)
#
# Xgboost Classifier
xgb_params={'n_estimators':range(80,200,4),
        'max_depth':range(2,15,1),
        'learning_rate':np.linspace(0.01,2,20),
        'subsample':np.linspace(0.7,0.9,20),
        'colsample_bytree':np.linspace(0.5,0.98,10),
        'min_child_weight':range(1,9,1)}
rand_xgb =RandomizedSearchCV(xgb.XGBClassifier(),xgb_params,cv = 5,n_iter=search_iter,n_jobs=-1)
#
#
classifiers = {
    "XGB" :rand_xgb,
    # "LR": rand_log_reg,
    "RF" :rand_rf
# "SVM": rand_svc,
}



sampling_methods = [
                    'Original'
                    # SMOTE(random_state=42),
                    # SMOTE(random_state=42, kind='borderline1'),
                    # ADASYN(random_state=42),
                    #
                    # SMOTEENN(random_state=42),
                    # SMOTETomek(random_state=42)
                    ]

names = [
        'Base model'
        # 'SMOTE',
        # 'Borderline SMOTE',
        # 'ADASYN',
        # 'SMOTE+ENN',
        # 'SMOTE+Tomek'
         ]

#抽樣方法 x 分類方法
sss = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)
time_total =[]

#以下程式碼為測試Model ＋不同抽樣方法的效果，效果以confusion matrix呈現
#測試結果最後以Random Forest 與 Xgboost 的base model 結果較優，故在此指顯示這兩個最優模型
for key, classifier in classifiers.items():
    for (name, method) in zip(names, sampling_methods):
        t0 = time.time()
        confu_temp=np.array([[0,0,0],[0,0,0],[0,0,0]])
        for train, test in sss.split(X_en, y_en):
            if name == 'Base model':
                model = classifier
                model.fit(X_en.iloc[train], y_en.iloc[train])
                model = model.best_estimator_
            else:
                pipeline = make_pipeline(method,classifier)  # SMOTE happens during Cross Validation not before..
                model = pipeline.fit(X_en.iloc[train], y_en.iloc[train])


            # best_est = model.best_estimator_
            if key =="XGB" and name!='Base model':
                y_pred = model.predict(np.array(X_en.iloc[test]))
            else:
                y_pred = model.predict(X_en.iloc[test])
                # y_score = best_est.decision_function(original_Xtrain[test])
                # y_score = best_est.predict_proba(original_Xtrain[test])
                # if classifier == lr_cfl or classifier==svm_cfl:
                #     y_prob = model.decision_function(X_en.iloc[test])
                # elif classifier == xgb_cfl:
                #     y_prob = model.predict_proba(np.array(X_en.iloc[test]))
                # else:
                #     y_prob = model.predict_proba(X_en.iloc[test])
            # print(key)
            confu = confusion_matrix(y_en[test], y_pred)
            confu_temp += confu

        print(key,name)
        print(confu_temp)
        precision_formal = confu_temp[0][0] / sum(confu_temp[:,0])
        precision_smart = confu_temp[1][1] / sum(confu_temp[:, 1])
        precision_casual = confu_temp[2][2] / sum(confu_temp[:,2])
        recall_formal =confu_temp[0][0] / sum(confu_temp[0])
        recall_smart = confu_temp[1][1] / sum(confu_temp[1])
        recall_casual = confu_temp[2][2] / sum(confu_temp[2])
        print("Formal - Precision:%.2f  Recall%.2f"%(precision_formal,recall_formal))
        print("Smart  - Precision:%.2f  Recall%.2f" % (precision_smart, recall_smart))
        print("Casual - Precision:%.2f  Recall%.2f" % (precision_casual, recall_casual))
        time1 = (time.time() - t0)
        print('花了%.2f秒'%time1)

        # time_total.append(time1)



# download pickle
# 將154~215行的程式碼，只留下最佳解的部分，以此例為xgb。
# 修改以下變數和檔名後，即可執行本程式碼，將model以pickle檔形式儲存
# for (name, method) in zip(names, sampling_methods):
#     if name =='Base model':
#         model = rand_xgb.fit(X_en, y_en)
#         model = model.best_estimator_
#     else:
#         pipeline = make_pipeline(method, rand_xgb)
#         model = pipeline.fit(X_en, y_en)
#     file = open('./pickle/final_label/best_xgb_'+name+'.pickle', 'wb')
#     pickle.dump(model, file)
#     file.close()
