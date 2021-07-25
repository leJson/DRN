import argparse
import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch_geometric.nn import XConv, fps, global_mean_pool
from show_data import MYData
from datasets import get_dataset
from train_eval_ import run

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
args = parser.parse_args()
import os


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv1 = XConv(0, 48, dim=3, kernel_size=8, hidden_channels=32)
        self.conv2 = XConv(
            48, 96, dim=3, kernel_size=12, hidden_channels=64, dilation=2)
        self.conv3 = XConv(
            96, 192, dim=3, kernel_size=16, hidden_channels=128, dilation=2)
        self.conv4 = XConv(
            192, 384, dim=3, kernel_size=16, hidden_channels=256, dilation=2)
        self.input_size = 1  # feature points size or word size defualt:129

        self.out_size = 1  # the size of prediction for each word
        self.layer_num = 2
        self.rnn_3 = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
                           # bidirectional=True,
                           dropout=0.3,
                           # batch_first=True
                           )
        self.rnn = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
                                 # bidirectional=True,
                                 dropout=0.3,
                                 batch_first=True
                                 )
        self.lin1 = Lin(384, 256)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, num_classes)

    def forward(self, pos, batch):
        # print('pos.shape:', pos.shape)
        # print('batch.shape:', batch.shape)
        batch_size = batch.reshape(-1, 1024).shape[0]
        x = F.relu(self.conv1(None, pos, batch))
        # print('x.shape_conv1:', x.shape)
        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv2(x, pos, batch))
        # print('x.shape_conv2:', x.shape)
        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv3(x, pos, batch))
        t = global_mean_pool(x, batch)

        # print('t.shape:', t.shape)
        # print('x.shape_conv3:', x.shape)
        x = F.relu(self.conv4(x, pos, batch))
        # print('x.shape_conv4:', x.shape)
        # print('x.shape:', x.shape)
        # print('x.reshape:', x.reshape(8, -1, 384).shape)

        # (batch,feature_size,features )
        # print('x.transpose(0,2).shape:', x.reshape(8, -1, 384).transpose(1,2).shape)



        # print('x.shape:', x.shape)
        # input_x: (seq_len, batch, input_size)
        # x, hc = self.rnn(x.reshape(8, -1, 384).transpose(1, 2).shape)
        # x, hc = self.rnn(x.reshape(batch_size, -1, 384).transpose(1, 2))

        x = global_mean_pool(x, batch)
        # x = x.reshape()
        # print('after gloabal', x.shape)
        #x, hc = self.rnn(x.reshape(384, batch_size, -1))
        # x, hc = self.rnn(x.reshape(batch_size, 384, -1))
        # x = x.reshape(batch_size, 384)

        # print('lstm_x.shape:', x.shape)
        # output x:(seq_len, batch, output_size)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)

        return F.log_softmax(x, dim=-1)


# train_dataset, test_dataset = get_dataset(num_points=1024)
# model = Net(train_dataset.num_classes)
#
# data_path = '/home/ljs/datasets/pointcloud_classifier/small_dataset'
# train_data = 'train1.h5'
# test_data = 'test0.h5'
# train_path = os.path.join(data_path, train_data)
# test_path = os.path.join(data_path, test_data)
# train_dataset = MYData(data_path=train_path)
# test_dataset = MYData(data_path=test_path)


# train_dataset, test_dataset = get_dataset(num_points=1024)
data_path = '/home/ljs/datasets/pointcloud_classifier/small_dataset'
# train_data = ['train1.h5', 'train0.h5']
train_data = ['train1.h5']
test_data = ['test0.h5']
# train_path = os.path.join(data_path, train_data)
# test_path = os.path.join(data_path, test_data)


train_dataset = MYData(data_path=data_path, data_name=train_data)
print('train_dataset num:', train_dataset.__len__())
test_dataset = MYData(data_path=data_path, data_name=test_data)

NUM_CLASS = 6


# model = Net(train_dataset.num_classes)
model = Net(NUM_CLASS)
print('ok')
# print(model)
run(train_dataset, test_dataset, model, args.epochs, args.batch_size, args.lr,
    args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay)