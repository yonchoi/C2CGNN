#!/usr/bin/env python
# coding: utf-8
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from model.ASTGNN import make_model
from lib.utils import get_adjacency_matrix, get_adjacency_matrix_2direction, compute_val_loss, predict_and_save_results, load_graphdata_normY_channel1
from tensorboardX import SummaryWriter

# read hyper-param settings
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04.conf', type=str, help="configuration file path")
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE, flush=True)

config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config), flush=True)
config.read(args.config)

data_config = config['Data']
training_config = config['Training']
adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
dataset_name = data_config['dataset_name']
model_name = training_config['model_name']
learning_rate = float(training_config['learning_rate'])
start_epoch = int(training_config['start_epoch'])
epochs = int(training_config['epochs'])
fine_tune_epochs = int(training_config['fine_tune_epochs'])
print('total training epoch, fine tune epoch:', epochs, ',' , fine_tune_epochs, flush=True)
batch_size = int(training_config['batch_size'])
print('batch_size:', batch_size, flush=True)
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
direction = int(training_config['direction'])
encoder_input_size = int(training_config['encoder_input_size'])
decoder_input_size = int(training_config['decoder_input_size'])
dropout = float(training_config['dropout'])
kernel_size = int(training_config['kernel_size'])

filename_npz = os.path.join(dataset_name + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '.npz'
num_layers = int(training_config['num_layers']) # 4
d_model = int(training_config['d_model']) # 64
nb_head = int(training_config['nb_head']) # 8
ScaledSAt = bool(int(training_config['ScaledSAt']))  # whether use spatial self attention
SE = bool(int(training_config['SE']))  # whether use spatial embedding
smooth_layer_num = int(training_config['smooth_layer_num'])
aware_temporal_context = bool(int(training_config['aware_temporal_context']))
TE = bool(int(training_config['TE']))
use_LayerNorm = True
residual_connection = True

#### Custom
num_of_vertices = len(adj_in)
encoder_input_size = 1 # F of input
decoder_input_size = 1 # F of predicted
points_per_hour = 10 # Time in
num_for_predict = 10 # Time out
epochs = 100
fine_tune_epochs = 10
batch_size = 4
direction = 1
ScaledSAt = False
SE = False
learning_rate = 1e-3

# direction = 1 means: if i connected to j, adj[i,j]=1;
# direction = 2 means: if i connected to j, then adj[i,j]=adj[j,i]=1
# if direction == 2:
#     adj_mx, distance_mx = get_adjacency_matrix_2direction(adj_filename, num_of_vertices, id_filename)
#
# if direction == 1:
#     adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

## Multiple adjacencies
adj_mxs = np.array([norm_Adj(adj) for adj in adjs])
adj_mxs = torch.from_numpy(adj_mxs).type(torch.FloatTensor).to(DEVICE)
adj_mx = adjs[0] # example

folder_dir = 'MAE_%s_h%dd%dw%d_layer%d_head%d_dm%d_channel%d_dir%d_drop%.2f_%.2e' % (model_name, num_of_hours, num_of_days, num_of_weeks, num_layers, nb_head, d_model, encoder_input_size, direction, dropout, learning_rate)

if aware_temporal_context:
    folder_dir = folder_dir + 'Tcontext'

if ScaledSAt:
    folder_dir = folder_dir + 'ScaledSAt'

if SE:
    folder_dir = folder_dir + 'SE' + str(smooth_layer_num)

if TE:
    folder_dir = folder_dir + 'TE'

print('folder_dir:', folder_dir, flush=True)
params_path = os.path.join('../experiments', dataset_name, folder_dir)

# all the input has been normalized into range [-1,1] by MaxMin normalization
# train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min = load_graphdata_normY_channel1(
#     graph_signal_matrix_filename, num_of_hours,
#     num_of_days, num_of_weeks, DEVICE, batch_size)

net = make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size,
                 d_model, adj_mx, nb_head,
                 num_of_weeks, num_of_days, num_of_hours,
                 points_per_hour, num_for_predict,
                 dropout=dropout, aware_temporal_context=aware_temporal_context,
                 ScaledSAt=ScaledSAt, SE=SE, TE=TE, kernel_size=kernel_size,
                 smooth_layer_num=smooth_layer_num, residual_connection=residual_connection,
                 use_LayerNorm=use_LayerNorm,
                 adj_mxs=adj_mxs)

print(net, flush=True)

# def train_main():

if (start_epoch == 0) and (not os.path.exists(params_path)):  # 从头开始训练，就要重新构建文件夹
    os.makedirs(params_path)
    print('create params directory %s' % (params_path), flush=True)
elif (start_epoch == 0) and (os.path.exists(params_path)):
    shutil.rmtree(params_path)
    os.makedirs(params_path)
    print('delete the old one and create params directory %s' % (params_path), flush=True)
elif (start_epoch > 0) and (os.path.exists(params_path)):  # 从中间开始训练，就要保证原来的目录存在
    print('train from params directory %s' % (params_path), flush=True)
else:
    raise SystemExit('Wrong type of model!')

criterion = nn.L1Loss().to(DEVICE)  # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器，传入所有网络参数
sw = SummaryWriter(logdir=params_path, flush_secs=5)
#
total_param = 0
print('Net\'s state_dict:', flush=True)
for param_tensor in net.state_dict():
    print(param_tensor, '\t', net.state_dict()[param_tensor].size(), flush=True)
    total_param += np.prod(net.state_dict()[param_tensor].size())

print('Net\'s total params:', total_param, flush=True)

print('Optimizer\'s state_dict:')
for var_name in optimizer.state_dict():
    print(var_name, '\t', optimizer.state_dict()[var_name], flush=True)

global_step = 0
best_epoch = 0
best_val_loss = np.inf
#
# train model
if start_epoch > 0:
#
    params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
#
    net.load_state_dict(torch.load(params_filename))
#
    print('start epoch:', start_epoch, flush=True)
#
    print('load weight from: ', params_filename, flush=True)

start_time = time()
#
for epoch in range(start_epoch, epochs):
#
    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
#
    # apply model on the validation data set
    val_loss = compute_val_loss(net, val_loader, criterion, sw, epoch)
#
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(net.state_dict(), params_filename)
        print('save parameters to file: %s' % params_filename, flush=True)
#
    _ = net.train()  # ensure dropout layers are in train mode
#
    train_start_time = time()
#
    for batch_index, batch_data in enumerate(train_loader):
#
        encoder_inputs, decoder_inputs, labels, adjidx = batch_data
        encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
        decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
#
        labels = labels.unsqueeze(-1)
        optimizer.zero_grad()
        adjidx = adjidx.to(torch.int).cpu().numpy()
        outputs = net(encoder_inputs, decoder_inputs, adjidx)
#
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss = loss.item()
#
        global_step += 1
        sw.add_scalar('training_loss', training_loss, global_step)
#
    print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
    print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)

print('best epoch:', best_epoch, flush=True)
#
print('apply the best val model on the test data set ...', flush=True)
#
predict_main(best_epoch, test_loader, test_target_tensor, None, None, 'test')
#
# fine tune the model
optimizer = optim.Adam(net.parameters(), lr=learning_rate*0.1)
print('fine tune the model ... ', flush=True)
for epoch in range(epochs, epochs+fine_tune_epochs):
#
    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
    _ = net.train()  # ensure dropout layers are in train mode
    train_start_time = time()
#
    for batch_index, batch_data in enumerate(train_loader):
#
        encoder_inputs, decoder_inputs, labels, adjidx = batch_data
        encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
        decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
#
        labels = labels.unsqueeze(-1)
        predict_length = labels.shape[2]  # T
        optimizer.zero_grad()
        adjidx = adjidx.to(torch.int).cpu().numpy()
        net.updateAdjMtx(adjidx)
        encoder_output = net.encode(encoder_inputs)
#
        # decode
        decoder_start_inputs = decoder_inputs[:, :, :1, :]
        decoder_input_list = [decoder_start_inputs]
#
        for step in range(predict_length):
            decoder_inputs = torch.cat(decoder_input_list, dim=2)
            predict_output = net.decode(decoder_inputs, encoder_output)
            decoder_input_list = [decoder_start_inputs, predict_output]
#
        loss = criterion(predict_output, labels)
        loss.backward()
        optimizer.step()
        training_loss = loss.item()
#
        global_step += 1
        sw.add_scalar('training_loss', training_loss, global_step)
#
    print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
    print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)
#
    # apply model on the validation data set
    val_loss = compute_val_loss(net, val_loader, criterion, sw, epoch)
#
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(net.state_dict(), params_filename)
        print('save parameters to file: %s' % params_filename, flush=True)
#
print('best epoch:', best_epoch, flush=True)
#
print('apply the best val model on the test data set ...', flush=True)
#
predict_main(best_epoch, test_loader, test_target_tensor, None, None, 'test')


## =============================================================================
def predict_main(epoch, data_loader, data_target_tensor, _max, _min, type):
    '''
    在测试集上，测试指定epoch的效果
    :param epoch: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param type: string
    :return:
    '''
#
    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
#
    print('load weight from:', params_filename, flush=True)
#
    net.load_state_dict(torch.load(params_filename))
#



def predict_main(epoch, data_loader, data_target_tensor, _max, _min, type, outdir='.'):
    '''
    在测试集上，测试指定epoch的效果
    :param epoch: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param type: string
    :return:
    '''
#
    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
#
    print('load weight from:', params_filename, flush=True)
#
    net.load_state_dict(torch.load(params_filename))
#
    # predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type)
#
    _ = net.train(False)  # ensure dropout layers are in evaluation mode
    with torch.no_grad():
        # -------------------------
        test_loader_length = len(test_loader)  # nb of batch
        tmp = []  # batch loss
        for batch_index, batch_data in enumerate(test_loader):
#
            encoder_inputs, decoder_inputs, labels, adjidx = batch_data
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
#
            labels = labels.unsqueeze(-1)
            predict_length = labels.shape[2]  # T
            adjidx = adjidx.to(torch.int).cpu().numpy()
            net.updateAdjMtx(adjidx)
            encoder_output = net.encode(encoder_inputs)
#
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]
            decoder_input_list = [decoder_start_inputs]
#
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]
            #
            loss = criterion(predict_output, labels)
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('test_loss batch %s / %s, loss: %.2f' % (batch_index + 1, test_loader_length, loss.item()))
#
        test_loss = sum(tmp) / len(tmp)
        sw.add_scalar('test_loss', test_loss, epoch)
#
    print(test_loss)
    sample_input  = encoder_inputs[0][:,:,0]  # input
    sample_output = predict_output[0][:,:,0]  # prediction
    sample_labels = labels[0][:,:,0] # truth
    print(sample_output.shape, sample_labels.shape)
#
    from matplotlib import pyplot as plt
#
    plt.figure(figsize=(30,4), dpi=80)
    idx = np.linspace(150,len(sample_output)-1,20).astype('int')
    for j,i in enumerate(idx):
        new_i = j * 20
        _ = plt.plot(range(10+new_i,20+new_i),sample_output[i].detach().cpu().numpy(), color = 'red')
        _ = plt.plot(range(10+new_i,20+new_i),sample_labels[i].cpu().numpy(), color='blue')
        _ = plt.plot(range(0+new_i,10+new_i),sample_input[i].cpu().numpy(), color='green')
#
    plt.savefig(os.path.join(outdir,"pred.svg"))
    plt.close()


if __name__ == "__main__":

    train_main()

    # predict_main(0, test_loader, test_target_tensor, _max, _min, 'test')
