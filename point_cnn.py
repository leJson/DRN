import argparse
import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch_geometric.nn import XConv, fps, global_mean_pool
from datasets import get_dataset
from train_eval import run
from show_data import MYData_Lettuce
import os
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=16)  # default: 32
parser.add_argument('--lr', type=float, default=0.001)  # defualt:0.001
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, out_features):
        super(Net, self).__init__()

        self.conv1 = XConv(0, 48, dim=3, kernel_size=8, hidden_channels=32)
        self.conv2 = XConv(
            48, 96, dim=3, kernel_size=12, hidden_channels=64, dilation=2)
        self.conv3 = XConv(
            96, 192, dim=3, kernel_size=16, hidden_channels=128, dilation=2)
        self.conv4 = XConv(
            192, 384, dim=3, kernel_size=16, hidden_channels=256, dilation=2)
        # self.lin1 = Lin(384, 256)
        self.lin1 = Lin(384, out_features)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, out_features)

    def forward(self, pos, batch):

        # print('pos.shape:', pos.shape, 'batch.shape:', batch.shape)
        # print('batch:', batch)
        # pos = pos[:, :3]
        # print('pos.shape:', pos.shape, 'batch.shape:',  batch.shape)
        # print('batch', batch)
        # print('x.shape:', x.shape)
        x = F.relu(self.conv1(None, pos, batch))
        # print('x.shape:', x.shape)
        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]
        x = F.relu(self.conv2(x, pos, batch))
        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]
        x = F.relu(self.conv3(x, pos, batch))
        x = F.relu(self.conv4(x, pos, batch))
        x = global_mean_pool(x, batch)
        # print('x.shape:', x.shape)
        x = self.lin1(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(self.lin1(x))
        #  x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin3(x)
        return x


class Net_LSTM(torch.nn.Module):
    def __init__(self, out_features):
        super(Net_LSTM, self).__init__()
        self.conv1 = XConv(0, 48, dim=3, kernel_size=8, hidden_channels=32)
        self.conv2 = XConv(
            48, 96, dim=3, kernel_size=12, hidden_channels=64, dilation=2)
        self.conv3 = XConv(
            96, 192, dim=3, kernel_size=16, hidden_channels=128, dilation=2)
        self.conv4 = XConv(
            192, 384, dim=3, kernel_size=16, hidden_channels=256, dilation=2)

        self.input_size = 1  # feature points size or word size defualt:129
        self.out_size = 1  # the size of prediction for each word
        self.layer_num = 2 # defualt:2
        # self.rnn_3 = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
        #                    bidirectional=True,
        #                    dropout=0.3,
        #                    # batch_first=True
        #                    )
        self.rnn = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
                                 # bidirectional=True,
                                 dropout=0.3, #defualt:0.3
                                 batch_first=True
                                 )

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.maxpool = torch.nn.AdaptiveMaxPool1d(1)
        # self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        #
        # self.lin1 = Lin(384, 256)
        self.lin1 = Lin(384, out_features)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, out_features)

    def forward(self, pos, batch):
        # batch_size = batch.reshape(-1, 1024).shape[0]
        batch_size = batch.reshape(-1, 1024).shape[0]
        # print('pos.shape:', pos.shape, 'batch.shape:', batch.shape)
        # print('batch:', batch)
        # pos = pos[:, :3]
        # print('pos.shape:', pos.shape, 'batch.shape:',  batch.shape)
        # print('batch', batch)
        # print('x.shape:', x.shape)
        x = F.relu(self.conv1(None, pos, batch))
        # print('x.shape:', x.shape)
        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]
        x = F.relu(self.conv2(x, pos, batch))
        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]
        x = F.relu(self.conv3(x, pos, batch))
        x = F.relu(self.conv4(x, pos, batch))
        x = global_mean_pool(x, batch)
        x, hc = self.rnn(x.reshape(batch_size, 384, -1))
        # x = self.avgpool(x) + self.maxpool(x)
        x = self.avgpool(x)
        # print('x.shape:', x.shape)
        x = x.reshape(batch_size, -1)
        x = self.lin1(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(self.lin1(x))
        #  x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin3(x)
        return x


class Net_LSTM_bak(torch.nn.Module):
    def __init__(self, out_features):
        super(Net_LSTM, self).__init__()
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
        # self.rnn_3 = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
        #                    bidirectional=True,
        #                    dropout=0.3,
        #                    # batch_first=True
        #                    )
        self.rnn = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
                                 bidirectional=True,
                                 dropout=0.3, #defualt:0.3
                                 # batch_first=True
                                 )

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.maxpool = torch.nn.MaxPool1d(2, stride=2)
        # self.lin1 = Lin(384, 256)
        self.lin1 = Lin(384, out_features)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, out_features)

    def forward(self, pos, batch):
        # batch_size = batch.reshape(-1, 1024).shape[0]
        batch_size = batch.reshape(-1, 1024).shape[0]
        # print('pos.shape:', pos.shape, 'batch.shape:', batch.shape)
        # print('batch:', batch)
        # pos = pos[:, :3]
        # print('pos.shape:', pos.shape, 'batch.shape:',  batch.shape)
        # print('batch', batch)
        # print('x.shape:', x.shape)
        x = F.relu(self.conv1(None, pos, batch))
        # print('x.shape:', x.shape)
        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]
        x = F.relu(self.conv2(x, pos, batch))
        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]
        x = F.relu(self.conv3(x, pos, batch))
        x = F.relu(self.conv4(x, pos, batch))
        x = global_mean_pool(x, batch)

        x, hc = self.rnn(x.reshape(batch_size, 384, -1))
        x = self.avgpool(x)
        # print('x.shape:', x.shape)
        x = x.reshape(batch_size, -1)
        x = self.lin1(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(self.lin1(x))
        #  x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin3(x)
        return x


class Net_Lstm__(torch.nn.Module):
    def __init__(self, out_features):
        super(Net_Lstm__, self).__init__()
        self.input_size = 3  # feature points size or word size defualt:129
        self.out_size = 1  # the size of prediction for each word
        self.layer_num = 2
        self.rnn = torch.nn.LSTM(
                                 self.input_size, self.out_size, num_layers=self.layer_num,
                                 bidirectional=True,
                                 dropout=0.3, # default 0.3
                                 batch_first=True
                                 )
        self.lin1 = Lin(4096, 2048)
        self.lin2 = Lin(2048, 512)
        self.lin3 = Lin(512, out_features)

    def forward(self, pos, batch):
        print('pos.shape:', pos.shape, 'batch.shape:', batch.shape)
        batch_size = batch.reshape(-1, 4096).shape[0]
        # print('batch_size:', batch_size)
        # print('pos.shape:', pos.shape)
        pos = pos.reshape(batch_size, 4096, -1)

        print('pos.shape:', pos.shape, 'batch.shape:', batch.shape)

        x, hc = self.rnn(pos)
        # print('x.shape:', x.shape)
        # print('pos.shape:', pos.shape)
        x = x.reshape(batch_size, -1)
        x = self.lin1(x)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        # print('x.shape', x.shape)
        return x
        # return F.log_softmax(x, dim=-1)


class Net_point_lstm(torch.nn.Module):
    def __init__(self, out_features):
        super(Net_point_lstm, self).__init__()
        self.input_size = 3  # feature points size or word size defualt:129
        self.out_size = 128  # the size of prediction for each word
        self.layer_num = 2
        self.rnn = torch.nn.LSTM(self.input_size, self.out_size, num_layers=self.layer_num,
                                 # bidirectional=True,
                                 dropout=0.3,
                                 batch_first=True
                                 )
        self.conv1d_1 = torch.nn.Conv1d(4096, 2048, stride=1, kernel_size=3, padding=True)
        self.conv1d_2 = torch.nn.Conv1d(2048, 1024, stride=1, kernel_size=3, padding=True)
        self.avgpool = torch.nn.AdaptiveAvgPool1d((128))
        self.maxpool = torch.nn.AdaptiveMaxPool1d((128))
        self.conv1d_3 = torch.nn.Conv1d(1024, 1, stride=1, kernel_size=3, padding=True)
        self.conv1d_4 = torch.nn.Conv1d(1024, 1, stride=1, kernel_size=3, padding=True)
        self.lin1 = Lin(4096, 2048)
        self.lin2 = Lin(2048, 512)
        self.lin3 = Lin(512, out_features)
        self.line = Lin(128, out_features)

    def forward(self, pos, batch):
        batch_size = batch.reshape(-1, 4096).shape[0]
        # print('batch_size:', batch_size)
        # print('pos.shape:', pos.shape)
        pos = pos.reshape(batch_size, 4096, -1)
        x, hc = self.rnn(pos)
        # print('lstmx.shape:', x.shape)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.avgpool(x)
        t = self.maxpool(x)
        x = self.conv1d_3(x).reshape(batch_size, -1)
        t = self.conv1d_4(t).reshape(batch_size, -1)
        # print('x.shape:', x.shape)
        # print('t.shape', t.shape)
        res = (x-t)*(x-t)
        # print('res.shape:', res.shape)
        res = self.line(res)
        # # print('pos.shape:', pos.shape)
        # x = x.reshape(batch_size, -1)
        # x = self.lin1(x)
        # x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin3(x)
        # # print('x.shape', x.shape)
        return res
        # return F.log_softmax(x, dim=-1)

# train_dataset, test_dataset = get_dataset(num_points=1024)
# data_path = '/home/ljs/datasets/pointcloud_classifier/small_dataset'
# train_data = ['train1.h5', 'train0.h5']
# train_data = ['train1.h5']
# # tain_data = ['train0.h5']
# test_data = ['test0.h5']
data_path = '/home/ljs/PycharmProjects/data'
# data_name = ['FirstTrainingData']
data_name = ['ori_data']
# train_path = os.path.join(data_path,  train_data)
# test_path = os.path.join(data_path, test_data)
train_dataset = MYData_Lettuce(data_path=data_path, data_name=data_name, data_class='train', points_num=1024)
# print('train_dataset num:', train_dataset.__len__())
test_dataset = MYData_Lettuce(data_path=data_path, data_name=data_name, data_class='test', points_num=1024)
# NUM_CLASS = 6
out_features = 1
# model = Net(train_dataset.num_classes)
# model = Net(out_features)
model = Net_LSTM(out_features)
#model = Net_Lstm(out_features)
# model = Net_point_lstm(out_features)
# print('ok')
run(train_dataset, test_dataset, model, args.epochs, args.batch_size, args.lr,
    args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay)
