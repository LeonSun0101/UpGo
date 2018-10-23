from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
from label_dict import crop_disease,crop_degree,crop,disease
from label_dict import crop_disease_label,crop_degree_label

np.set_printoptions(precision=2)
def get_cm(real,pred,normalize=False,):
    cm = confusion_matrix(real, pred)
    np.set_printoptions(precision=2)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


def get_confusion_matrix(real_path,label_prediction,class_name,normalize=False):
    predict_df = pd.read_json(label_prediction)
    real_df = pd.read_json(real_path)

    pre = predict_df[['image_id', class_name]].rename(columns={class_name: class_name + '_pre'})
    real = real_df[['image_id', class_name]]

    columns={'crop':crop,'disease':disease,'crop_disease':crop_disease,'crop_degree':crop_degree}
    confuse_desc = pd.DataFrame(get_cm(real[class_name], pre[class_name + '_pre'],normalize),
                                columns=columns[class_name], index=columns[class_name])
    return confuse_desc

def to_new_label():

    return

if __name__ == '__main__':
    ###################################计算confusing matrix########################################
    path = '/Users/lidandan/Documents/others_pro/crop_disease_detect/result/1020/'
    class_name = 'crop_disease'

    real_path = os.path.join(path,'real.json')
    label_prediction = os.path.join(path,'label_prediction_1.json')

    confuse_desc = get_confusion_matrix(label_prediction,real_path,class_name)
    ####################################生成新标签##################################################
    label_dict = {'crop_disease':crop_disease_label,'crop_degree':crop_degree_label}
    label_name = 'crop_disease'
    old_label = 'disease_class'
    path = 'AgriculturalDisease_train_annotations.json'

    label = pd.read_json(path)
    label[label_name] = label['disease_class'].apply(lambda x:label_dict[label_name][x])
    label.to_json(path+'_'+label_name)




