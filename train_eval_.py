import time
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
# from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.tensorboard import SummaryWriter
from logger import Logger
import datetime
import copy

log_path = './log/result.txt'
log_writer = Logger(log_path, resume=True)


def get_batch(batch_size, points_num=1024):
    arr = torch.ones(points_num, dtype=int)
    arr_sum = torch.zeros(points_num, dtype=int)
    for i in range(1, batch_size):
        arr_sum = torch.cat((arr_sum, arr*i))
    return arr_sum


def save_net(fname, net):
    fpath = os.path.join('weights', fname)
    torch.save(net.cpu().state_dict(), fpath)
    net.train()
    if torch.cuda.is_available():
        net.cuda()


def run(train_dataset, test_dataset, model, epochs, batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay):
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    log_writer.append('start trainning !')
    log_writer.append('start time:' + str(datetime.datetime.now()))
    global_step = 0
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        train(model, optimizer, train_loader, device, writer, global_step)
        test_acc = test(model, test_loader, device, epoch, writer, global_step)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        out_contxt = ' Epoch: {:03d}, Test: {:.4f}, Duration: {:.2f}'.format(
            epoch, test_acc, t_end - t_start)
        log_writer.append(str(datetime.datetime.now()) + out_contxt)
        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']
    log_writer.append('end trainning !')
    log_writer.close()


def train(model, optimizer, train_loader, device, writer, global_step):
    model.train()

    class_0_num, class_1_num, class_2_num = 0, 0, 0
    class_3_num, class_4_num, class_5_num = 0, 0, 0
    class_0_correct_num, class_1_correct_num, class_2_correct_num = 0, 0, 0
    class_3_correct_num, class_4_correct_num, class_5_correct_num = 0, 0, 0

    correct = 0
    for data, label in train_loader:
        optimizer.zero_grad()
        batch_size = data.shape[0]
        data = data.float().reshape(-1, 3).to(device)
        label = label.long().to(device)
        batch = get_batch(batch_size).to(device)
        out = model(data, batch)
        pred = copy.copy(out).max(1)[1]

        class_0 = (0 == label).nonzero(as_tuple=False).view(-1)

        class_0_num += len(class_0)

        class_1 = (1 == label).nonzero(as_tuple=False).view(-1)
        class_1_num += len(class_1)

        class_2 = (2 == label).nonzero(as_tuple=False).view(-1)
        class_2_num += len(class_2)

        class_3 = (3 == label).nonzero(as_tuple=False).view(-1)
        class_3_num += len(class_3)

        class_4 = (4 == label).nonzero(as_tuple=False).view(-1)
        class_4_num += len(class_4)

        class_5 = (5 == label).nonzero(as_tuple=False).view(-1)
        class_5_num += len(class_5)

        class_0_correct_num += pred.eq(label)[class_0].sum().item()
        class_1_correct_num += pred.eq(label)[class_1].sum().item()
        class_2_correct_num += pred.eq(label)[class_2].sum().item()
        class_3_correct_num += pred.eq(label)[class_3].sum().item()
        class_4_correct_num += pred.eq(label)[class_4].sum().item()
        class_5_correct_num += pred.eq(label)[class_5].sum().item()

        correct += pred.eq(label).sum().item()
        loss = F.nll_loss(out, label)
        writer.add_scalar('Loss/train', loss.item(), global_step)
        loss.backward()
        optimizer.step()
        global_step += 1
    class_0_acc = class_0_correct_num / class_0_num
    class_1_acc = class_1_correct_num / class_1_num
    class_2_acc = class_2_correct_num / class_2_num
    class_3_acc = class_3_correct_num / class_3_num
    class_4_acc = class_4_correct_num / class_4_num
    class_5_acc = class_5_correct_num / class_5_num

    train_acc = correct / len(train_loader.dataset)

    writer.add_scalar('train_acc', train_acc, global_step)
    writer.add_scalar(' class_0_acc', class_0_acc, global_step)
    writer.add_scalar(' class_1_acc', class_1_acc, global_step)
    writer.add_scalar(' class_2_acc', class_2_acc, global_step)
    writer.add_scalar(' class_3_acc', class_3_acc, global_step)
    writer.add_scalar(' class_4_acc', class_4_acc, global_step)
    writer.add_scalar(' class_5_acc', class_5_acc, global_step)


def test(model, test_loader, device, epoch, writer, global_step):
    model.eval()
    correct = 0
    class_0_num, class_1_num, class_2_num = 0, 0, 0
    class_3_num, class_4_num, class_5_num = 0, 0, 0
    class_0_correct_num, class_1_correct_num, class_2_correct_num = 0, 0, 0
    class_3_correct_num, class_4_correct_num, class_5_correct_num = 0, 0, 0

    for data, label in test_loader:
        batch_size = data.shape[0]
        data = data.float().reshape(-1, 3).to(device)
        # print('label_len', len(label))
        label = label.long().to(device)
        batch = get_batch(batch_size).to(device)
        out = model(data, batch)
        pred = copy.copy(out).max(1)[1]

        class_0 = (0 == label).nonzero(as_tuple=False).view(-1)
        class_0_num += len(class_0)

        class_1 = (1 == label).nonzero(as_tuple=False).view(-1)
        class_1_num += len(class_1)

        class_2 = (2 == label).nonzero(as_tuple=False).view(-1)
        class_2_num += len(class_2)

        class_3 = (3 == label).nonzero(as_tuple=False).view(-1)
        class_3_num += len(class_3)

        class_4 = (4 == label).nonzero(as_tuple=False).view(-1)
        class_4_num += len(class_4)

        class_5 = (5 == label).nonzero(as_tuple=False).view(-1)
        class_5_num += len(class_5)

        class_0_correct_num += pred.eq(label)[class_0].sum().item()
        class_1_correct_num += pred.eq(label)[class_1].sum().item()
        class_2_correct_num += pred.eq(label)[class_2].sum().item()
        class_3_correct_num += pred.eq(label)[class_3].sum().item()
        class_4_correct_num += pred.eq(label)[class_4].sum().item()
        class_5_correct_num += pred.eq(label)[class_5].sum().item()

        loss = F.nll_loss(out, label)
        writer.add_scalar('Loss/test', loss.item(), global_step)
        correct += pred.eq(label).sum().item()

    class_0_acc = class_0_correct_num / class_0_num
    class_1_acc = class_1_correct_num / class_1_num
    class_2_acc = class_2_correct_num / class_2_num
    class_3_acc = class_3_correct_num / class_3_num
    class_4_acc = class_4_correct_num / class_4_num
    class_5_acc = class_5_correct_num / class_5_num

    test_acc = correct / len(test_loader.dataset)
    writer.add_scalar('test_acc', test_acc, global_step)
    writer.add_scalar(' class_0_acc', class_0_acc, global_step)
    writer.add_scalar(' class_1_acc', class_1_acc, global_step)
    writer.add_scalar(' class_2_acc', class_2_acc, global_step)
    writer.add_scalar(' class_3_acc', class_3_acc, global_step)
    writer.add_scalar(' class_4_acc', class_4_acc, global_step)
    writer.add_scalar(' class_5_acc', class_5_acc, global_step)

    # print('class_0_correct_num,class_0_num, class_0_acc:', class_0_correct_num, class_0_num, class_0_acc)
    # print('class_1_correct_num,class_1_num, class_1_acc:', class_1_correct_num, class_1_num, class_1_acc)
    # print('class_2_correct_num,class_2_num, class_2_acc:', class_2_correct_num, class_2_num, class_2_acc)
    # print('class_3_correct_num,class_3_num, class_3_acc:', class_3_correct_num, class_3_num, class_3_acc)
    # print('class_4_correct_num,class_4_num, class_4_acc:', class_4_correct_num, class_4_num, class_4_acc)
    # print('class_5_correct_num,class_5_num, class_5_acc:', class_5_correct_num, class_5_num, class_5_acc)
    save_net('pointcnn_%s.pth' % epoch, model)
    return test_acc



















