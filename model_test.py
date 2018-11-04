import torch
from data_loader.dataset import get_data_loader
from utils.config import process_config
import os
from nets.resnet_module import resnet50
from nets.resnet_module import resnet101
from nets.senet_module import se_resnet50
from nets.densenet_module import densenet121
import json


def get_result(model,data_loader):
    predictions = []
    labels = []
    result = {'pred':predictions,'labels':labels}
    for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
        batch_x, batch_y = batch_x.cuda(async=True), batch_y.cuda(async=True)
        pred = model(batch_x).data.cpu().numpy()
        predictions.append(pred.tolist())
        labels.append(batch_y.cpu().numpy().tolist())
        print('{}/{}'.format(batch_idx+1,len(data_loader)))

        # if batch_idx == 1:
        #     break
    return result
if __name__ == '__main__':
    #定义result保存文件


    #加载model
    config = process_config(os.path.join(os.getcwd(), 'configs', 'config.json'))
    result_save_file = config['result_save_file']
    num_classes = 61
    train_loader,valid_loader = get_data_loader(config)
    model_path = config['test_model']
    model_param = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path,map_location='cpu')
    model = densenet121(num_classes=num_classes).cuda()
    model.load_state_dict(model_param)

    result = get_result(model=model,data_loader=valid_loader)
    result1 = get_result(model=model,data_loader=train_loader)

    #save
    json.dump(result,open(os.path.join(config['model_result'],'densenet_valid.json'),'w'))
    json.dump(result1,open(os.path.join(config['model_result'],'densenet_train.json'),'w'))

