import torch.nn as nn
from transformers import BertModel
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class BertPair_base(nn.Module):
    def __init__(self,config):
        super(BertPair_base, self).__init__()
        self.bert=BertModel.from_pretrained(config.pretrain_model_path)
        self.linear1=nn.Linear(768,2)
        self.dropout=nn.Dropout(0.2)
        self.label_smooth_loss = LabelSmoothingLoss(classes=2, smoothing=config.smoothing)

    def forward(self,inputs, input_types, labels , task=None):
        '''train: 完成了模型预测输出 + loss计算求和 两个过程
           valid: 完成了模型预测输出
        '''
        if task == 'train' or task == 'val':
            bert_mask = torch.ne(inputs, 0)  # 找到 inputs 中不等于 0 的地方，置为 1（0表示padding）
            bert_enc= self.bert(inputs,attention_mask=bert_mask,token_type_ids=input_types)[0]
            bert_enc = self.dropout(bert_enc)
            ##### 取 mean 之前，应该先把 padding 部分的特征去除！！！
            mask_2 = bert_mask  # 其余等于 1 的部分，即有效的部分
            mask_2_expand = mask_2.unsqueeze_(-1).expand(bert_enc.size()).float()
            sum_mask = mask_2_expand.sum(dim=1)  # 有效的部分“长度”求和
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            bert_enc = torch.sum(bert_enc * mask_2_expand, dim=1) / sum_mask
            #####
            bert_enc = self.linear1(bert_enc)
            loss = self.label_smooth_loss(bert_enc, labels.view(-1))
            return loss, bert_enc
        #测试
        else:
            bert_mask = torch.ne(inputs, 0)  # 找到 inputs 中不等于 0 的地方，置为 1（0表示padding）
            bert_enc= self.bert(inputs,attention_mask=bert_mask,token_type_ids=input_types)[0]
            bert_enc = self.dropout(bert_enc)
            ##### 取 mean 之前，应该先把 padding 部分的特征去除！！！
            mask_2 = bert_mask  # 其余等于 1 的部分，即有效的部分
            mask_2_expand = mask_2.unsqueeze_(-1).expand(bert_enc.size()).float()
            sum_mask = mask_2_expand.sum(dim=1)  # 有效的部分“长度”求和
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            bert_enc = torch.sum(bert_enc * mask_2_expand, dim=1) / sum_mask
            bert_enc = self.linear1(bert_enc)
            out=torch.softmax(bert_enc,dim=-1)
            return out


class BertPair_CNN(nn.Module):
    def __init__(self,config):
        super(BertPair_CNN, self).__init__()
        self.bert=BertModel.from_pretrained(config.pretrain_model_path)
        self.dropout=nn.Dropout(0.2)
        chanel_num=1
        filter_num=128
        dim=768
        self.conv1=nn.Conv2d(chanel_num,filter_num,(2,dim))
        self.conv2 = nn.Conv2d(chanel_num, filter_num, (3, dim))
        self.conv3 = nn.Conv2d(chanel_num, filter_num, (4, dim))
        self.linear1 = nn.Linear(3*filter_num, 2)
        self.label_smooth_loss = LabelSmoothingLoss(classes=2, smoothing=config.smoothing)
        self.pool_way='avg'
        # self.pool_way = 'max'

    def forward(self,inputs, input_types, labels , task=None):
        '''train: 完成了模型预测输出 + loss计算求和 两个过程
           valid: 完成了模型预测输出
        '''
        if task == 'train' or task == 'val':
            bert_mask = torch.ne(inputs, 0)  # 找到 inputs 中不等于 0 的地方，置为 1（0表示padding）
            bert_enc= self.bert(inputs,attention_mask=bert_mask,token_type_ids=input_types)[0]
            #将bert输出中为填充的特征向量去掉
            bert_enc= [layer[starts.nonzero().squeeze(1)]
                                      for layer, starts in zip(bert_enc, bert_mask)]
            #保证输入长度为最大长度
            padded_sequence_output = pad_sequence(bert_enc,batch_first=True)
            bert_enc = self.dropout(padded_sequence_output)

            x = bert_enc.unsqueeze(1)
            x1 = F.relu(self.conv1(x))
            x1 = x1.squeeze(3)
            x2 = F.relu(self.conv2(x))
            x2 = x2.squeeze(3)
            x3 = F.relu(self.conv3(x))
            x3 = x3.squeeze(3)
            if self.pool_way == 'max':
                x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
                x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
                x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
            elif self.pool_way == 'avg':
                x1 = F.avg_pool1d(x1, x1.size(2)).squeeze(2)
                x2 = F.avg_pool1d(x2, x2.size(2)).squeeze(2)
                x3 = F.avg_pool1d(x3, x3.size(2)).squeeze(2)
            x=torch.cat([x1,x2,x3],1)
            #####
            bert_enc = self.linear1(x)
            loss = self.label_smooth_loss(bert_enc, labels.view(-1))
            return loss, bert_enc
        #测试
        else:
            bert_mask = torch.ne(inputs, 0)  # 找到 inputs 中不等于 0 的地方，置为 1（0表示padding）
            bert_enc= self.bert(inputs,attention_mask=bert_mask,token_type_ids=input_types)[0]
            bert_enc= [layer[starts.nonzero().squeeze(1)]
                                      for layer, starts in zip(bert_enc, bert_mask)]
            padded_sequence_output = pad_sequence(bert_enc,batch_first=True)
            bert_enc = self.dropout(padded_sequence_output)

            x = bert_enc.unsqueeze(1)
            x1 = F.relu(self.conv1(x))
            x1 = x1.squeeze(3)
            x2 = F.relu(self.conv2(x))
            x2 = x2.squeeze(3)
            x3 = F.relu(self.conv3(x))
            x3 = x3.squeeze(3)
            if self.pool_way == 'max':
                x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
                x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
                x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
            elif self.pool_way == 'avg':
                x1 = F.avg_pool1d(x1, x1.size(2)).squeeze(2)
                x2 = F.avg_pool1d(x2, x2.size(2)).squeeze(2)
                x3 = F.avg_pool1d(x3, x3.size(2)).squeeze(2)
            x=torch.cat([x1,x2,x3],1)
            #####
            bert_enc = self.linear1(x)
            out=torch.softmax(bert_enc,dim=-1)
            return out



class BertPair_RNN(nn.Module):
    def __init__(self,config):
        super(BertPair_RNN,self).__init__()
        self.bert=BertModel.from_pretrained(config.pretrain_model_path)
        self.dropout=nn.Dropout(0.2)
        self.label_smooth_loss = LabelSmoothingLoss(classes=2, smoothing=config.smoothing)
        self.hidden_size=128
        self.embedding_dim=256
        self.lstm = nn.LSTM(768, self.hidden_size, bidirectional=True, batch_first=True)
        self.W1=nn.Linear(2*self.hidden_size,self.hidden_size)
        self.W2=nn.Linear(self.hidden_size,2)
    def forward(self,inputs, input_types, labels , task=None):
        '''train: 完成了模型预测输出 + loss计算求和 两个过程
           valid: 完成了模型预测输出
        '''
        if task == 'train' or task == 'val':
            bert_mask = torch.ne(inputs, 0)  # 找到 inputs 中不等于 0 的地方，置为 1（0表示padding）
            bert_enc= self.bert(inputs,attention_mask=bert_mask,token_type_ids=input_types)[0]
            bert_enc= [layer[starts.nonzero().squeeze(1)]
                                      for layer, starts in zip(bert_enc, bert_mask)]
            padded_sequence_output = pad_sequence(bert_enc,batch_first=True)
            bert_enc = self.dropout(padded_sequence_output)
            ##### 取 mean 之前，应该先把 padding 部分的特征去除！！！

            output, (final_hidden_state, final_cell_state) = self.lstm(bert_enc)
            y = self.W1(output)  # [batch,seq_len,embedding_dim]
            y = y.permute(0, 2, 1)
            y=F.max_pool1d(y,y.size()[2])
            y=y.squeeze(2)
            bert_enc = self.W2(y)
            loss = self.label_smooth_loss(bert_enc, labels.view(-1))
            return loss, bert_enc
        #测试
        else:
            bert_mask = torch.ne(inputs, 0)  # 找到 inputs 中不等于 0 的地方，置为 1（0表示padding）
            bert_enc= self.bert(inputs,attention_mask=bert_mask,token_type_ids=input_types)[0]
            bert_enc= [layer[starts.nonzero().squeeze(1)]
                                      for layer, starts in zip(bert_enc, bert_mask)]
            padded_sequence_output = pad_sequence(bert_enc,batch_first=True)
            bert_enc = self.dropout(padded_sequence_output)
            output, (final_hidden_state, final_cell_state) = self.lstm(bert_enc)
            y = self.W1(output)  # [batch,seq_len,embedding_dim]
            y = y.permute(0, 2, 1)
            y=F.max_pool1d(y,y.size()[2])
            y=y.squeeze(2)
            bert_enc = self.W2(y)
            out=torch.softmax(bert_enc,dim=-1)
            return out


class LabelSmoothingLoss(nn.Module):
    '''LabelSmoothingLoss
    '''
    def __init__(self, classes, smoothing=0.05, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        tmp=torch.sum(-true_dist * pred, dim=self.dim)
        return torch.mean(tmp)
