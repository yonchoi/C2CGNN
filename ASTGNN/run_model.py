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
# from model.ASTGNN import make_model
# from lib.utils import get_adjacency_matrix, get_adjacency_matrix_2direction, compute_val_loss, predict_and_save_results, load_graphdata_normY_channel1
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt

from create_dataset import load_npz, create_data_loaders
from modified_classes import make_model, predict_main, compute_val_loss, norm_Adj

def predict_main(data_loader, data_target_tensor, _max, _min, type, outdir='.'):
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
    # predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type)
#
    _ = net.train(False)  # ensure dropout layers are in evaluation mode
    with torch.no_grad():
        # -------------------------
        test_loader_length = len(data_loader)  # nb of batch
        tmp = []  # batch loss
        results = []
        for batch_index, batch_data in enumerate(data_loader):
    #
            encoder_inputs, decoder_inputs, labels, adjidx, metadata = batch_data
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
            results.append([encoder_inputs,
                            predict_output,
                            labels,
                            adjidx,
                            metadata])
    #
        test_loss = sum(tmp) / len(tmp)
        # sw.add_scalar(f'{type}_loss', test_loss, epoch)
    print(test_loss)
    np.savetxt(os.path.join(outdir,f"loss-{type}.txt"), np.array([test_loss]))
#
    from matplotlib import pyplot as plt
    num_predicted_features = predict_output.shape[-1]
    sample_idx = 0
    for i_output in np.arange(num_predicted_features):
        sample_input  = encoder_inputs[sample_idx][:,:,i_output]  # input
        sample_output = predict_output[sample_idx][:,:,i_output]  # prediction
        sample_labels = labels[sample_idx][:,:,i_output] # truth
        print(sample_output.shape, sample_labels.shape)
    #
        plt.figure(figsize=(30,4), dpi=80)
        node_idx = np.linspace(150,len(sample_output)-1,20).astype('int')
        for j,i in enumerate(node_idx):
            output = sample_output[i].detach().cpu().numpy()
            label  = sample_labels[i].detach().cpu().numpy()
            input  = sample_input[i].detach().cpu().numpy()
            time_in  = len(input)
            time_out = len(output)
            length_time = time_in + time_out
            #
            new_i = j * length_time
            _ = plt.plot(range(time_in+new_i,length_time+new_i),output, color = 'red')
            _ = plt.plot(range(time_in+new_i,length_time+new_i),label, color='blue')
            _ = plt.plot(range(0+new_i,time_in+new_i),input, color='green')
    #
        plt.savefig(os.path.join(outdir,f"pred-{type}.svg"))
        plt.close()
#
    adjidxs = np.concatenate([r[-2] for r in results],axis=0)
    metadatas = np.concatenate([r[-1].detach().cpu().numpy() for r in results],axis=0)
    inputs  = np.concatenate([r[0].detach().cpu().numpy() for r in results],axis=0)
    preds   = np.concatenate([r[1].detach().cpu().numpy() for r in results],axis=0)
    labels  = np.concatenate([r[2].detach().cpu().numpy() for r in results],axis=0)
    num_predicted_features = preds.shape[-1]
    n_node = inputs.shape[1]
    for i_adj in np.unique(adjidx):
        adj = adjs[i_adj]
        for i_output in np.arange(num_predicted_features):
            keep_time = adjidxs == i_adj
            adjidx   = adjidxs[keep_time]
            metadata = metadatas[keep_time]
            input    = inputs[keep_time][:,:,:,i_output]
            pred     = preds[keep_time][:,:,:,i_output]
            label    = labels[keep_time][:,:,:,i_output]
            #
            time_in  = input.shape[-1]
            time_out = pred.shape[-1]
            time_total = time_in + time_out
            #
            # Order according to time
            times = np.argsort(metadata)
            idx_time = times[::time_out] # skip
            input,pred,label,metadata = input[idx_time],pred[idx_time],label[idx_time],metadata[idx_time]
            figname = os.path.join(outdir,f"pred-all-{type}.svg")
            plot_continuous(input,label,pred,adj,figname,cont_pred=False)
            # # Combine into continuous frame
            # input_cont = input[:,:,-time_out:].transpose(1,0,2).reshape(n_node,-1) # (N,W,T_inc)
            # input_cont_start = input[0][:,:(time_in-time_out)] # N,
            # input_cont = np.concatenate([input_cont_start,input_cont],axis=1)
            # label_cont = label.transpose(1,0,2).reshape(n_node,-1)
            # pred_cont = pred.transpose(1,0,2).reshape(n_node,-1)
            # node_idx = np.linspace(0,len(pred_cont)-1,5).astype('int')
            # for i_cell in node_idx:
            #     adj_cell = adj[i_cell]
            #     adj_cell[i_cell] = 0
            #     neighbors = list(np.where(adj_cell==1)[0])
            #     neighbors = [i_cell] + neighbors
            #     fig,axes=plt.subplots(len(neighbors),figsize=(20,4*len(neighbors)),
            #                           dpi=80, squeeze=False)
            #     axes = axes.flatten()
            #     for n,ax in zip(neighbors,axes):
            #         pred_  = pred_cont[n]
            #         label_ = label_cont[n]
            #         input_ = input_cont[n]
            #         _ = ax.plot(range(time_in,time_in+len(pred_)),pred_, color = 'red')
            #         _ = ax.plot(range(time_in,time_in+len(pred_)),label_, color='blue')
            #         _ = ax.plot(range(len(input_)),input_, color='green')
            #     plt.savefig(os.path.join(outdir,f"pred-all-{type}-{i_cell}.svg"))
            #     # plt.savefig(os.path.join(outdir,f"pred-all-test-{i_cell}.svg"))
            #     plt.close()


def plot_continuous(input,label,pred,adj,figname,cont_pred=True):
    # Order according to time
    time_in  = input.shape[-1]
    time_out = pred.shape[-1]
    time_total = time_in + time_out
    # Combine into continuous frame
    input_cont = input[:,:,-time_out:].transpose(1,0,2).reshape(n_node,-1) # (N,W,T_inc)
    input_cont_start = input[0][:,:(time_in-time_out)] # N,
    input_cont = np.concatenate([input_cont_start,input_cont],axis=1)
    label_cont = label.transpose(1,0,2).reshape(n_node,-1)
    pred_cont = pred.transpose(1,0,2).reshape(n_node,-1)
    node_idx = np.linspace(0,len(pred_cont)-1,5).astype('int')
    for i_cell in node_idx:
        adj_cell = adj[i_cell]
        adj_cell[i_cell] = 0
        neighbors = list(np.where(adj_cell==1)[0])
        neighbors = [i_cell] + neighbors
        fig,axes=plt.subplots(len(neighbors),figsize=(20,4*len(neighbors)),
                              dpi=80, squeeze=False)
        axes = axes.flatten()
        for n,ax in zip(neighbors,axes):
            # n = n[0]
            input_ = input_cont[n]
            label_ = label_cont[n]
            if cont_pred:
                pred_  = pred_cont[n]
                _ = ax.plot(range(time_in,time_in+len(pred_)),pred_, color = 'red')
            else:
                pred_ = pred[:,n]
                for t,pred_plot in enumerate(pred_):
                    time_start = time_in + t * time_out
                    _ = ax.plot(range(time_start,time_start+time_out),pred_plot, color = 'red')
            _ = ax.plot(range(time_in,time_in+len(label_)),label_, color='blue')
            _ = ax.plot(range(len(input_)),input_, color='green')
        #
        figname_,ext = os.path.splitext(figname)
        figname_ = f"{figname_}-{i_cell}{ext}"
        plt.savefig(figname_)
        plt.close()

samples_dir = '../data/Ram/ERK_ETGs_Replicate1/data_processed/EGF-high/samples.npz'
samples_dir = '../data/Ram/ERK_ETGs_Replicate1/data_processed/Imaging Media/shift-True/samples.npz'

samples = load_npz(samples_dir)
_,n_node,n_time,F_in = samples.input.shape
adjs_all = samples.adj_mxs
print(adjs_all.shape)
adj_name = samples.adj_name
print(adj_name)
adj2idx = dict(zip(adj_name,range(len(adj_name))))

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
# DEVICE = torch.device('cpu')
print("CUDA:", USE_CUDA, DEVICE)

num_of_vertices = n_node
points_per_hour = 10 # Time in
num_for_predict = 10 # Time out
num_for_predict_ = 10     # Time out
dataset_name = 'EGF-Imaging Media/shift-True'
model_name = 'ASTGNN'
learning_rate = 1e-3
start_epoch = 0
epochs = 20
fine_tune_epochs = 10
batch_size = 4

num_of_weeks = 0
num_of_days = 0
num_of_hours = 1
direction = 1 # directed graph
encoder_input_size = F_in # F of input
decoder_input_size = F_in # F of predicted
dropout=0
kernel_size = 3

num_layers = 4
d_model=32
nb_head = 32

ScaledSAt = False # Scale adjacency matrix based on spatial attention
SE = False # Spatial encoding
TE = True  # Temporal encoding
aware_temporal_context = True

smooth_layer_num = 0
use_LayerNorm = True
residual_connection = False

outdir = f'../out/test10/{dataset_name}'
os.makedirs(outdir,exist_ok=True)

for adjtype in adj_name:
# for adjtype in ['dist-0','dist-120','dist-240']:
# for adjtype in ['dist-0','dist-60','dist-120']:
# for adjtype in ['dist-60']:
# for adjtype in ['dist-0','dist-120','neighbor-10','dist-60','neighbor-5']:
    i_adj = adj2idx[adjtype]
    adjs = adjs_all[:,i_adj]
    for num_for_predict_ in [10]:
        for learning_rate in [1e-4]:
            for d_model in [128]:
                for num_layers in [4]:
                    for nb_head in [8]:
                        # for n_hide in [0,1,2,3,5]:
                        # for n_hide in [0,3]:
                        for n_hide in [0]:
                            ## Set folder name where model will be stored
                            folder_dir = (f"Model-{model_name}",
                                          f"adj-{adjtype}",
                                          f"numL-{num_layers}",
                                          f"nbhead-{nb_head}",
                                          f"dmodel-{d_model}",
                                          f"encin-{encoder_input_size}",
                                          f"dir-{direction}",
                                          f"drop-{dropout}",
                                          f"lr-{learning_rate}",
                                          f"pred-{num_for_predict_}",
                                          f"nhide-{n_hide}",
                                          f'rc-{residual_connection}'
                                          )
                            folder_dir = "_".join(folder_dir)

                            if aware_temporal_context:
                                folder_dir = folder_dir + 'Tcontext'

                            if ScaledSAt:
                                folder_dir = folder_dir + 'ScaledSAt'

                            if SE:
                                folder_dir = folder_dir + 'SE' + str(smooth_layer_num)

                            if TE:
                                folder_dir = folder_dir + 'TE'

                            print('folder_dir:', folder_dir, flush=True)
                            params_path = os.path.join(outdir, folder_dir)

                            #### Create dataset
                            dataout = create_data_loaders(samples, batch_size,
                                                          num_for_predict=num_for_predict_,
                                                          train_size=0.8, train_val_size=0.8,
                                                          shuffle=True, DEVICE = DEVICE)

                            train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = dataout

                            ## Multiple adjacencies
                            adj_mxs = np.array([norm_Adj(adj) for adj in adjs])
                            adj_mxs = torch.from_numpy(adj_mxs).type(torch.FloatTensor).to(DEVICE)
                            adj_mx = adjs[0] # example

                            # ==============================================================================

                            model_filename = os.path.join(params_path, 'best_model.pt')

                            if os.path.isfile(model_filename):
                                net = torch.load(model_filename)
                                criterion = nn.L1Loss().to(DEVICE)  # 定义损失函数
                            else:
                                net = make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size,
                                                 d_model, adj_mx, nb_head,
                                                 num_of_weeks, num_of_days, num_of_hours,
                                                 points_per_hour, num_for_predict,
                                                 dropout=dropout, aware_temporal_context=aware_temporal_context,
                                                 ScaledSAt=ScaledSAt, SE=SE, TE=TE, kernel_size=kernel_size,
                                                 smooth_layer_num=smooth_layer_num, residual_connection=residual_connection,
                                                 use_LayerNorm=use_LayerNorm,
                                                 adj_mxs=adj_mxs, n_hide=n_hide)

                                print(net, flush=True)

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
                                        encoder_inputs, decoder_inputs, labels, adjidx, metadata = batch_data
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
                                params_filename = os.path.join(params_path, 'epoch_%s.params' % best_epoch)
                                print('load weight from:', params_filename, flush=True)
                                # net.load_state_dict(torch.load(params_filename))

                                #
                                predict_main(test_loader, test_target_tensor, None, None, 'test', params_path)
                                predict_main(train_loader, train_target_tensor, None, None, 'train', params_path)
                                predict_main(val_loader, val_target_tensor, None, None, 'val', params_path)
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
                                        encoder_inputs, decoder_inputs, labels, adjidx, metadata = batch_data
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
                                params_filename = os.path.join(params_path, 'epoch_%s.params' % best_epoch)
                                print('load weight from:', params_filename, flush=True)
                                # net.load_state_dict(torch.load(params_filename))
                                torch.save(net, model_filename)

                            predict_main(test_loader, test_target_tensor, None, None, 'test-finetune', params_path)
                            predict_main(train_loader, train_target_tensor, None, None, 'train-finetune', params_path)
                            predict_main(val_loader, val_target_tensor, None, None, 'train-finetune', params_path)
                            # Save the best model
