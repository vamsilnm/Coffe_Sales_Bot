import json


file_open_train = open('train_sentences.json','r')
json_file = json.load(file_open_train)
print json_file['train_sentences']
file_open_train.close()