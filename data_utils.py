from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

import config

def dev_split(dataset_dir):
    data = pd.read_csv(dataset_dir)[:100]
    x_train, x_val= train_test_split(data, test_size=config.val_split_size, random_state=0)
    return x_train, x_val


def pad(sentences,sentences_types,sentence_labels,max_len,pad_value=0):
    inputs = torch.LongTensor(config.batch_size, max_len).fill_(pad_value)
    input_types = torch.LongTensor(config.batch_size, max_len).fill_(pad_value)
    for i, s in enumerate(zip(sentences,sentences_types)):
        inputs[i, :len(s[0])] = torch.LongTensor(s[0])
        input_types[i, :len(s[1])] = torch.LongTensor(s[1])
    input_labels=torch.LongTensor(sentence_labels)
    return inputs, input_types,input_labels


class MyDataset(Dataset):
    def __init__(self,data,config):
        querys1=data['query1'].values.tolist()
        querys2 = data['query2'].values.tolist()
        self.inputs,self.input_types,self.labels=[],[],[]
        tokenizer=BertTokenizer.from_pretrained(config.pretrain_model_path)
        for q1,q2 in zip(querys1,querys2):
            input_tmp,input_type=[],[]
            q1=tokenizer.tokenize(q1)
            q1=tokenizer.convert_tokens_to_ids(['[CLS]']+q1+['[SEP]'])
            input_tmp+=q1
            input_type+=[0 for _ in range(len(q1))]

            q2=tokenizer.tokenize(q2)
            q2=tokenizer.convert_tokens_to_ids(['[CLS]']+q2+['[SEP]'])
            input_tmp+=q2
            input_type+=[1 for _ in range(len(q2))]

            self.inputs.append(input_tmp)
            self.input_types.append(input_type)

        self.labels=data['label'].values.tolist()
        self.device=config.device

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        input_item = self.inputs[item]
        input_type_item = self.input_types[item]
        label_item = self.labels[item]
        return [input_item,input_type_item,label_item]

    def collate_fn(self,batch):
        sentences=[x[0] for x in batch]
        sentences_type = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        max_len=max(len(s) for s in sentences)
        batch_data,batch_type,batch_labels=pad(sentences,sentences_type,labels,max_len)
        batch_data, batch_type, batch_labels=batch_data.to(config.device),batch_type.to(config.device),batch_labels.to(config.device)
        return [batch_data, batch_type, batch_labels]


























