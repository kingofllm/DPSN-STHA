import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import os
import json
import shutil
import torch.utils.data
import torch.optim as optim
from RemSTAM import run_model
from prepareData import load_data
from util import masked_mae, masked_mse, computer_loss
# from utils import adj_weight_matrix, loadGraph, log_string
from early_stopping import EarlyStopping
# from visdom import Visdom

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/PEMS03.json')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = json.loads(f.read())
data = config['data']
dataset = config['dataset']
device_ids = config['device_ids']
d_model = config['d_model']
num_for_predict = config['num_for_predict']
num_of_hour = config['num_of_hour']
nodes = config['nodes']
n_heads = config['n_heads']
num_of_features = config['num_of_features']
epochs = config['epochs']
learning_rate = config['learning_rate']
d_k = config['d_k']
d_v = config['d_v']
batch_size = config['batch_size']
n_layer = config['n_layer']
features_out = config['features_out']
time_features_out = config['time_features_out']
memory_dim = config['memory_dim']

"获取数据集"
filename = data + '/' + dataset + '/' + dataset + '.npz'
all_data = load_data(filename, num_for_predict, num_of_hour, batch_size)

#训练集
train_x = all_data['train']['train_x'][:, :, 0, :]
train_target = all_data['train']['train_target'][:, :, 0, :]
train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).cuda()
train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).cuda()
train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#验证集
val_x = all_data['val']['val_x'][:, :, 0, :]
val_target = all_data['val']['val_target'][:, :, 0, :]
val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).cuda()
val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).cuda()
val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#测试集
test_x = all_data['test']['test_x'][:, :, 0, :]
test_target = all_data['test']['test_target'][:, :, 0, :]
# print(test_target[0:288:12, 110, :])
test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).cuda()
test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).cuda()
test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 图的邻接矩阵
# adj = adj_weight_matrix(data, dataset, nodes, type='connectivity')
# 特征值和特征向量
# spvalue = loadGraph(adj, 12)

# N, N
STAM = np.load(data + '/' + dataset + '/' + dataset + '_STAM' + '.npy')

folder_dir = '%s' % (dataset)
params_path = os.path.join('myexperiments', folder_dir)
if not os.path.exists(params_path):
    os.makedirs(params_path)
elif os.path.exists(params_path):
    shutil.rmtree(params_path)
    os.makedirs(params_path)
else:
    raise SystemExit('Wrong type of model!')

early_stopping = EarlyStopping(params_path)

#creat model
model = run_model(n_layer, d_model, nodes, num_of_hour, num_of_features, d_k, d_v, n_heads,
                  features_out, time_features_out, memory_dim, STAM).cuda()

# model = nn.DataParallel(model, device_ids=device_ids).cuda()

#train model
def main():
    print('------------ params list -------------')
    print('data:', data)
    print('dataset:', dataset)
    print('device_ids:', device_ids)
    print('d_model:', d_model)
    print('num_for_predict:', num_for_predict)
    print('num_of_hour:', num_of_hour)
    print('nodes:', nodes)
    print('n_heads:', n_heads)
    print('num_of_features:', num_of_features)
    print('epochs:', epochs)
    print('learning_rate:', learning_rate)
    print('d_k:', d_k)
    print('d_v:', d_v)
    print('batch_size:', batch_size)
    print('n_layer:', n_layer)
    print('features_out:', features_out)
    print('time_features_out:', time_features_out)
    print('memory_dim:', memory_dim)
    print('-------------- End ----------------')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    hubloss = nn.HuberLoss()
    print("%s Start Training" % dataset)
    print('=====================')
    train_loss = []
    for epoch in range(epochs):
        #train
        model.train()
        t1 = time.time()
        for index, (inputs, target) in enumerate(train_loader):
            optimizer.zero_grad()
            prediction = model(inputs)
            loss = hubloss(prediction, target)
            """
            if index <= 200:
               wind.line(X=[index], Y=[loss.cpu().detach().numpy()], win='train_loss', update='append')
               time.sleep(0.5)
            """
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        t2 = time.time()
        train_length = len(train_loss)
        train_huber_loss = sum(train_loss) / train_length

        print()
        print('Epoch: %.3d' % epoch)
        print('Train_time: {:.4f}s'.format(t2 - t1))
        print('Train_huber_loss: %.4f' % train_huber_loss)

        #val
        val_loss = []
        val_mae = []
        val_mape = []
        val_rmse = []
        model.train(False)
        v1 = time.time()
        with torch.no_grad():
            for index, (inputs, target) in enumerate(val_loader):
                prediction = model(inputs)
                loss = hubloss(prediction, target)
                val_loss.append(loss)
                mae, mape, rmse = computer_loss(prediction, target)
                val_mae.append(mae)
                val_mape.append(mape)
                val_rmse.append(rmse)

        v2 = time.time()

        val_length = len(val_loss)
        val_huber_loss = sum(val_loss) / val_length
        val_mae = sum(val_mae) / val_length
        val_mape = sum(val_mape) / val_length
        val_rmse = sum(val_rmse) / val_length

        print()
        print('Val_time: {:.4f}s'.format(v2 - v1))
        print('Val_mae: %.4f' % val_mae)
        print('Val_mape: %.4f' % val_mape)
        print('Val_rmse: %.4f' % val_rmse)

        early_stopping(val_huber_loss, model)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

    #test
    print()
    print("%s Start Testing" % dataset)
    print('=====================')
    parameter_filename = os.path.join(params_path, 'best_network.pth')
    model.load_state_dict(torch.load(parameter_filename))
    test_mae = []
    test_mape = []
    test_rmse = []
    nodes_30_target = []
    nodes_30_prediction = []
    nodes_130_target = []
    nodes_130_prediction = []
    model.train(False)
    test1 = time.time()
    with torch.no_grad():
        for index, (inputs, target) in enumerate(test_loader):
            prediction = model(inputs)
            """
            nodes_30_target.append(target[:, 30, :])
            nodes_30_prediction.append(prediction[:, 30, :])
            nodes_130_target.append(target[:, 130, :])
            nodes_130_prediction.append(prediction[:, 130, :])
            """
            mae, mape, rmse = computer_loss(prediction, target)
            test_mae.append(mae)
            test_mape.append(mape)
            test_rmse.append(rmse)
    test2 = time.time()

    test_length = len(test_mae)
    test_mae = sum(test_mae) / test_length
    test_mape = sum(test_mape) / test_length
    test_rmse = sum(test_rmse) / test_length

    print()
    print('Test_time: {:.4f}s'.format(test2 - test1))
    print('Test_mae: %.4f' % test_mae)
    print('Test_mape: %.4f' % test_mape)
    print('Test_rmse: %.4f' % test_rmse)

    """
    a = nodes_30_target[0]
    for i in range(len(nodes_30_target) - 1):
        a = torch.cat([a, nodes_30_target[i + 1]], dim=0)

    b = nodes_30_prediction[0]
    for i in range(len(nodes_30_prediction) - 1):
        b = torch.cat([b, nodes_30_prediction[i + 1]], dim=0)

    c = nodes_130_target[0]
    for i in range(len(nodes_130_target) - 1):
        c = torch.cat([c, nodes_130_target[i + 1]], dim=0)

    d = nodes_130_prediction[0]
    for i in range(len(nodes_130_prediction) - 1):
        d = torch.cat([d, nodes_130_prediction[i + 1]], dim=0)

    print(a[0:288:12, :])
    print(b[0:288:12, :])
    print(c[0:288:12, :])
    print(d[0:288:12, :])
    """
if __name__ == "__main__":
    main()














