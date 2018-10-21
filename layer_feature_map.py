import torch
from data_loader.dataset import get_data_loader
from utils.config import process_config
import os
from nets.resnet_module import resnet50
import json


def get_layer_feature(layer,model,data_loader,shape):
    features = []
    labels = []
    feature_data = {'features':features,'labels':labels}
    for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
        shape=(len(batch_x),shape[1],shape[2],shape[3])
        avgpool_layer = model._modules.get(layer)
        embedding = torch.zeros(shape)
        def fun(m, i, o): embedding.copy_(o.data)
        h = avgpool_layer.register_forward_hook(fun)
        h_x = model(batch_x)
        h.remove()

        features.extend(embedding)
        labels.extend(batch_y)
        print('{}/{}'.format(batch_idx+1,len(data_loader)))

        # if batch_idx == 1:
        #     break
    return feature_data
if __name__ == '__main__':
    #定义features保存文件
    feature_save_file = 'train_features.json'
    shape = (16,2048,1,1)
    num_classes = 61

    #加载model
    config = process_config(os.path.join(os.getcwd(), 'configs', 'config.json'))
    train_loader,valid_loader = get_data_loader(config)
    model_path = '/Users/lidandan/Documents/others_pro/model/val_eval_best.pth'
    model_param = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path,map_location='cpu')
    model = resnet50(num_classes=num_classes)
    model.load_state_dict(model_param)

    #获得指定layer的feature
    feature_data = get_layer_feature(layer='avgpool',model=model,data_loader=train_loader,shape=shape)

    X = [feature_data['features'][i].reshape(2048).tolist() for i in range(len(feature_data['features']))]
    Y = [feature_data['labels'][i].tolist() for i in range(len(feature_data['labels']))]

    json.dump({'X':X,'Y':Y},open(os.getcwd()+feature_save_file,'w'))


