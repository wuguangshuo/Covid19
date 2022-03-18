import logging
from torch.utils.data import DataLoader
from transformers.optimization import AdamW,get_cosine_schedule_with_warmup
import pandas as pd

from utils import set_logger,train,evaluate
import config
from data_utils import dev_split,MyDataset
from model import BertPair_base,BertPair_CNN,BertPair_RNN

if __name__=='__main__':
    #设置日志
    set_logger(config.log_dir)
    logging.info('device: {}'.format(config.device))

    #划分训练集
    train_data,val_data=dev_split(config.train_file)

    #读取数据集
    train_dataset=MyDataset(train_data,config)
    val_dataset = MyDataset(val_data,config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=val_dataset.collate_fn,drop_last=True)
    train_size = len(train_dataset)
    val_size=len(val_dataset)

    #采取何种模型
    model_type='rnn'
    #建立模型
    if model_type == 'mlp':
        model=BertPair_base(config)
    if model_type == 'cnn':
        model = BertPair_CNN(config)
    if model_type == 'rnn':
        model = BertPair_RNN(config)
    model.to(config.device)
    #计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    # 模型存储位置
    model_path = './save/' + model_type + '.pkl'
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=config.lr)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,

                                                num_training_steps=config.epoch_num * train_steps_per_epoch)
    # Train the model
    logging.info("--------Start Training!--------")
    train(train_loader, val_loader, model, optimizer, scheduler, train_size,val_size,model_path)

    #test the model
    test_data = pd.read_csv(config.test_file)[:10]
    test_dataset = MyDataset(test_data,config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                              shuffle=False, collate_fn=test_dataset.collate_fn)
    test_size=len(test_dataset)
    testacc=evaluate(test_loader,model,test_size,mode='test')
    logging.info("val acc: {}".format(testacc))
    print('finish')

