import torch
import logging
import os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

from model import BertPair_base,BertPair_CNN,BertPair_RNN
import config

log_dir='./runs/'
writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

def set_logger(log_path):
    logger = logging.getLogger()#用logging.getLogger(name)方法进行初始化
    logger.setLevel(logging.INFO)#设置级别

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)#地址
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)



def train(train_loader,val_loader,model,optimizer,scheduler,train_size,val_size,model_path):
    if os.path.exists(model_path) and config.load_before:
        state = torch.load(model_path)
        if model_path == './save/' + 'mlp' + '.pkl':
            model = BertPair_base(config)
        if model_path == './save/' + 'cnn' + '.pkl':
            model = BertPair_CNN(config)
        if model_path == 'rnn':
            model = BertPair_RNN(config)
        model.to(config.device)
        model.load_state_dict(state['model_state'])
        print('train阶段加载模型完成')
    best_val_acc=0.0
    for epoch in range(1,config.epoch_num+1):
        model.train()
        train_losses,right_num=0,0
        for idx,batch_samples in enumerate(tqdm(train_loader)):
            batch_data,batch_type,batch_labels=batch_samples
            loss,bert_enc=model(batch_data,batch_type,labels=batch_labels,task='train')
            right_num += count_right_num(bert_enc, batch_labels)
            train_losses += loss.item()
            # clear previous gradients, compute gradients of all variables wrt loss
            model.zero_grad()
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
            # performs updates using calculated gradients
            optimizer.step()
            scheduler.step()
        train_loss=float(train_losses)/train_size
        train_acc=float(right_num)/train_size
        logging.info("Epoch: {}, train loss: {}, train acc: {}".format(epoch, train_loss, train_acc))

        val_loss, val_acc = evaluate(val_loader,model,val_size,mode='val')
        logging.info("Epoch: {}, val loss: {}, val acc: {}".format(epoch, val_loss, val_acc))

        #记录
        writer.add_scalar('Training/train loss',train_loss ,epoch)#tensorboard --logdir "./runs"启动
        writer.add_scalar('Validation/val loss', val_loss, epoch)
        writer.add_scalar('Training/train acc',train_acc ,epoch)#tensorboard --logdir "./runs"启动
        writer.add_scalar('Validation/val acc', val_acc, epoch)

        improve_acc = val_acc - best_val_acc
        state = {}
        if improve_acc >1e-5:
            best_val_acc = val_acc
            state['model_state'] = model.state_dict()
            torch.save(state,model_path)
            logging.info("--------Save best model!--------")
            if improve_acc < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best acc
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val acc: {}".format(best_val_acc))
            break
    logging.info("Training Finished!")


def evaluate(data_loader,model,size,mode='val'):
    model.eval()
    val_losses,val_right_num=0,0
    if mode=='val':
        with torch.no_grad():
            val_right_num,right_num=0,0
            for idx, batch_samples in enumerate(tqdm(data_loader)):
                batch_data, batch_type, batch_labels = batch_samples
                loss, bert_enc = model(batch_data, batch_type, labels=batch_labels, task='val')
                right_num = count_right_num(bert_enc, batch_labels)
                val_losses += loss.item()
                val_right_num += right_num
            val_loss=float(val_losses)/size
            val_acc=float(val_right_num)/size
            return val_loss,val_acc
    if mode=='test':
        test_right_num, right_num = 0, 0
        with torch.no_grad():
            for idx, batch_samples in enumerate(tqdm(data_loader)):  # 每一次返回 batch_size 条数据
                batch_data, batch_type, batch_labels = batch_samples
                bert_enc = model(batch_data, batch_type, labels=batch_labels, task='test')
                right_num = count_right_num(bert_enc, batch_labels)
                test_right_num += right_num
            test_acc=float(test_right_num)/size
            return test_acc

def count_right_num(out, label):
    out = out.cpu()
    _, out = torch.max(out, dim=-1)
    out = out.numpy()
    label=label.cpu().numpy()
    return sum(out==label)

