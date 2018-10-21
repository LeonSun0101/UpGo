from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
from label_dict import crop_disease,crop_degree

np.set_printoptions(precision=2)
def get_confusion_matrix(real,pred,normalize=False,):
    cm = confusion_matrix(real, pred)
    np.set_printoptions(precision=2)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


if __name__ == '__main__':
    label = pd.read_csv('label.csv')
    crop = {'苹果': 0, '樱桃': 1, '玉米': 2, '葡萄': 3, '柑桔': 4, '辣椒': 5, '马铃薯': 6, '草莓': 7, '番茄': 8, '桃子': 9, }

    disease = {'健康': 0, '黑星': 1, '灰斑': 2, '锈病': 3, '白粉': 4, '叶斑': 5, '花叶病毒': 6, '黑腐': 7, '轮斑': 8, '褐斑': 9, '黄龙': 10,
               '疮痂': 11, '早疫': 12, '晚疫': 13, '叶枯': 14, '叶霉': 15, '斑点': 16, '斑枯': 17, '蜘蛛': 18, '曲叶病毒': 19}



    path = '/Users/lidandan/Documents/others_pro/crop_disease_detect/result/1020/'
    class_name = 'crop_disease'

    real_path = os.path.join(path,'real.json')
    label_prediction = os.path.join(path,'label_prediction_1.json')
    # crop_class = os.path.join(path,'AgriculturalDisease_validation_annotations_crop_class_1020.json')
    # disease_index = os.path.join(path,'AgriculturalDisease_validation_annotations_disease_index_1020.json')
    # crop_degree = os.path.join(path,'AgriculturalDisease_testA_annotations_crop_degree_1020.json')


    predict_df = pd.read_json(label_prediction)
    real_df = pd.read_json(real_path)

    pre_1 = predict_df[['image_id', class_name]].rename(columns={class_name:class_name+'_pre'})
    real_1 = real_df[['image_id', class_name]]
    # real_pre = real_1.merge(pre_1,how='left')
    confuse_desc = pd.DataFrame(get_confusion_matrix(real_1[class_name],pre_1[class_name+'_pre']),columns=crop_disease,index=crop_disease)




