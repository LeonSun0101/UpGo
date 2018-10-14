import json
with open("./AgriculturalDisease_train_annotations.json", 'r') as f:
    data = json.load(f)
    image=[]
    for element in data:
        #line = line.strip('\n\r').strip('\n').strip('\r')
        #words = line.split(self.config['file_label_separator'])
        # single label here so we use int(words[1])
        image.append((element['image_id'], int(element['disease_class'])))
print(image[0])