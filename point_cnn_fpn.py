import argparse
import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch_geometric.nn import XConv, fps, global_mean_pool
from datasets import get_dataset
from train_eval_ import run
from show_data import MYData
import os
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=16)  # default: 32
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
args = parser.parse_args()


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

        self.lin1 = Lin(384, 256)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, num_classes)
        # fpn conv
        self.deconv = torch.nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=True, bias=False)
        self.conv1d = torch.nn.Conv1d(384, 129, kernel_size=3, stride=1, padding=True, bias=False)
        self.conv1d_downsample = torch.nn.Conv1d(258, 129, kernel_size=3, stride=1, padding=True, bias=False)
        self.conv2d_1 = torch.nn.Conv2d(129, 129, kernel_size=2, stride=(1, 1), padding=0, bias=False)
        self.conv2d_2 = torch.nn.Conv2d(129, 129, kernel_size=3, stride=(1, 2), padding=True, bias=False)
        self.padding = torch.nn.ZeroPad2d(padding=(0, 0, 1, 0))
    def forward(self, pos, batch):
        batch_size = batch.reshape(-1, 1024).shape[0]
        # print('batch_size:', batch_size)
        print('pos.shape:', pos.shape, 'batch.shape:',  batch.shape)
        # print('batch', batch)
        x = F.relu(self.conv1(None, pos, batch))
        # print('xconv1:', x.shape)
        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv2(x, pos, batch))
        # print('xconv2:', x.shape)\


        x_xconv2 = x.reshape(batch_size, 384, 96)
        x_xconv2 = self.conv1d(x_xconv2)
        x_xconv2 = x_xconv2.reshape(batch_size, 1, 129, 96)
        x_xconv2 = self.deconv(x_xconv2)
        x_xconv2 = x_xconv2.reshape(batch_size, 258, 192)
        x_xconv2 = self.conv1d_downsample(x_xconv2)

        # print('x_xconv2.shape:', x_xconv2.shape)

        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv3(x, pos, batch))
        # r融合点1
        # print('xconv2:', x_xconv2.shape)

        x = torch.cat([x.reshape(batch_size, 129, 192, 1), x_xconv2.reshape(batch_size, 129, 192, 1)], dim=3)
        # print('self.padding(x).shape:', self.padding(x).shape)
        x = self.conv2d_1(self.padding(x)).reshape(batch_size * 129, 192)
        # print('after_cat.shape:', x.shape)
        # x = self.conv2d_1(x).reshape(batch_size * 129, 192)
        # print('after_cat.shape:', x.shape)
        # x = (x.reshape(batch_size, 129, 192) + x_xconv2).reshape(129*batch_size, 192)

        x_xconv3 = x.reshape(batch_size, 129, 192)
        x_xconv3 = x_xconv3.reshape(batch_size, 1, 129, 192)
        x_xconv3 = self.deconv(x_xconv3)
        x_xconv3 = x_xconv3.reshape(batch_size, 258, 384)
        x_xconv3 = self.conv1d_downsample(x_xconv3)


        x = F.relu(self.conv4(x, pos, batch))
        # print('before_cat.shape:', x_xconv3.shape)
        # r融合点2
        # print('xconv3:', x.shape)
        # x = (x.reshape(batch_size, 129, 384) + x_xconv3).reshape(129*batch_size, 384)

        x = torch.cat([x.reshape(batch_size, 129, 384, 1), x_xconv3.reshape(batch_size, 129, 384, 1)],dim=3)
        # x = self.conv2d_2(x).reshape(batch_size*129, 384)
        x = self.conv2d_1(self.padding(x)).reshape(batch_size * 129, 384)
        # print('after_cat.shape:', x.shape)
        # print('xconv4:', x.shape)
        x = global_mean_pool(x, batch)
        # print('global_mean_pool:', x.shape)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


# train_dataset, test_dataset = get_dataset(num_points=1024)

data_path = '/home/ljs/datasets/pointcloud_classifier/small_dataset'
# train_data = ['train1.h5', 'train0.h5']
train_data = ['train1.h5']
# tain_data = ['train0.h5']
test_data = ['test0.h5']
# train_path = os.path.join(data_path, train_data)
# test_path = os.path.join(data_path, test_data)
train_dataset = MYData(data_path=data_path, data_name=train_data)
print('train_dataset num:', train_dataset.__len__())
test_dataset = MYData(data_path=data_path, data_name=test_data)
NUM_CLASS = 6


# model = Net(train_dataset.num_classes)
model = Net(NUM_CLASS)
# print('ok')
run(train_dataset, test_dataset, model, args.epochs, args.batch_size, args.lr,
    args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay)
