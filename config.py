import torch

log_dir='./log/train.log'
device='cuda' if torch.cuda.is_available() else 'cpu'


train_file = './data/train.csv'
test_file = './data/test.csv'
val_split_size=0.2


model_dir = './save/' + 'model.pth'
load_before=True
patience_num=3
min_epoch_num=5
smoothing=0.05
epoch_num=10
patience=0.01
batch_size=2
clip_grad=5
lr=0.00001
pretrain_model_path='bert-base-chinese'