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
    feature_save_file = 'valid_features.json'
    shape = (16,2048,1,1)
    num_classes = 61

    #加载model
    config = process_config(os.path.join(os.getcwd(), 'configs', 'config.json'))
    train_loader,valid_loader = get_data_loader(config)
    model_path = config['test_model']
    model_param = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path,map_location='cpu')
    model = resnet50(num_classes=num_classes)
    model.load_state_dict(model_param)

    #获得指定layer的feature
    feature_data = get_layer_feature(layer='avgpool',model=model,data_loader=valid_loader,shape=shape)

    #save
    json.dump(feature_data,open(os.getcwd()+feature_save_file))


