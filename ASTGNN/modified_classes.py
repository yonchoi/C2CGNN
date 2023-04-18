import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from lib.utils import norm_Adj

from model.ASTGNN import subsequent_mask, clones, GCN
from model.ASTGNN import Spatial_Attention_layer, spatialAttentionGCN, spatialAttentionScaledGCN
from model.ASTGNN import SpatialPositionalEncoding, TemporalPositionalEncoding
from model.ASTGNN import SublayerConnection, PositionWiseGCNFeedForward
from model.ASTGNN import attention, MultiHeadAttention
from model.ASTGNN import MultiHeadAttentionAwareTemporalContex_qc_kc
from model.ASTGNN import MultiHeadAttentionAwareTemporalContex_q1d_k1d
from model.ASTGNN import MultiHeadAttentionAwareTemporalContex_qc_k1d
from model.ASTGNN import EncoderDecoder, EncoderLayer, Encoder, DecoderLayer, Decoder
from model.ASTGNN import search_index, make_model

import os
from time import time


def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

def subsequent_mask(size,n_hide=0):
    '''
    mask out subsequent positions.
    :param size: int
    :return: (1, size, size)
    '''
    attn_shape = (size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    n_hide = min(n_hide,size-1)
    # subsequent_mask[kth_diag_indices(subsequent_mask,k=0)] = 1
    for k in range(1,1+n_hide):
        subsequent_mask[kth_diag_indices(subsequent_mask,k=-k)] = 1
    subsequent_mask = np.expand_dims(subsequent_mask,axis=0)

    return torch.from_numpy(subsequent_mask) == 0   # 1 means unreachable; 0 means reachable


# Add
class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask: (batch, T, T)
        :return: x: (batch, N, T, d_model)
        '''
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3) for l, x in
                             zip(self.linears, (query, key, value))]

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


## Modify the GCNs to accomodate for multiple adjacency matrix
class spatialGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(spatialGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def update_adj_matrix(self, sym_norm_Adj_matrix):
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix

    def forward(self, x, sym_norm_Adj_matrix=None):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''

        if sym_norm_Adj_matrix is None:
            sym_norm_Adj_matrix = self.sym_norm_Adj_matrix

        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        sym_norm_Adj_matrix = sym_norm_Adj_matrix.reshape(-1,
                                                          num_of_vertices,
                                                          num_of_vertices) # (B, N, N)

        sym_norm_Adj_matrix = sym_norm_Adj_matrix.broadcast_to(num_of_timesteps,
                                                               batch_size,
                                                               num_of_vertices,
                                                               num_of_vertices).permute(1,0,2,3).reshape(-1, num_of_vertices, num_of_vertices) # (B*T, N, N)
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        assert len(x) == len(sym_norm_Adj_matrix), f"Length of x ({x.shape}) must match length of adj ({sym_norm_Adj_matrix.shape})"

        return F.relu(self.Theta(torch.bmm(sym_norm_Adj_matrix, x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))

class PositionWiseGCNFeedForward(nn.Module):
    def __init__(self, gcn, dropout=.0):
        super(PositionWiseGCNFeedForward, self).__init__()
        self.gcn = gcn
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sym_norm_Adj_matrix=None):
        '''
        :param x:  (B, N_nodes, T, F_in)
        :return: (B, N, T, F_out)
        '''
        return self.dropout(F.relu(self.gcn(x,sym_norm_Adj_matrix=None)))

class PositionWiseGCNFeedForward(nn.Module):
    def __init__(self, gcn, dropout=.0):
        super(PositionWiseGCNFeedForward, self).__init__()
        self.gcn = gcn
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sym_norm_Adj_matrix=None):
        '''
        :param x:  (B, N_nodes, T, F_in)
        :return: (B, N, T, F_out)
        '''
        return self.dropout(F.relu(self.gcn(x,sym_norm_Adj_matrix=None)))


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, gcn, dropout,
                 residual_connection=True, use_LayerNorm=True,
                 n_hide=0):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn # encoder
        self.src_attn = src_attn # decoder
        self.feed_forward_gcn = gcn
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.n_hide = n_hide
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 3)

    def forward(self, x, memory):
        '''
        :param x: (batch_size, N, T', F_in)
        :param memory: (batch_size, N, T, F_in)
        :return: (batch_size, N, T', F_in)
        '''
        m = memory
        tgt_mask = subsequent_mask(x.size(-2), n_hide=self.n_hide).to(m.device)  # (1, T', T')
        # print("tgt_mask")
        # print(x.size(-2))
        # print(tgt_mask)
        # raise Exception('test remove')
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False))  # output: (batch, N, T', d_model)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True))  # output: (batch, N, T', d_model)
            return self.sublayer[2](x, self.feed_forward_gcn)  # output:  (batch, N, T', d_model)
        else:
            x = self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False)  # output: (batch, N, T', d_model)
            x = self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True)  # output: (batch, N, T', d_model)
            return self.feed_forward_gcn(x)  # output:  (batch, N, T', d_model)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_dense, trg_dense, generator, DEVICE, adj_mxs=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_dense
        self.trg_embed = trg_dense
        self.prediction_generator = generator
        self.adj_mxs = adj_mxs # List of normalized adj matrices (A,N,N), A=#ofAdjs
        self.to(DEVICE)
#
    def updateAdjMtx(self,idx):
        if idx is not None:
            adj_mtx = self.adj_mxs[idx]
            for module in [self.encoder,self.decoder]:
                for layer in module.layers:
                    # layer.feed_forward_gcn.gcn.sym_norm_Adj_matrix = adj_mtx
                    layer.feed_forward_gcn.gcn.sym_norm_Adj_matrix.shape
#
    def getAdjmtx(self,idx):
        if idx is None:
            return None
        else:
            return self.adj_mxs[idx]
#
    def encode(self, src, idx=None):
        '''
        src: (batch_size, N, T_in, F_in)
        '''
        self.updateAdjMtx(idx)
        h = self.src_embed(src)
        return self.encoder(h)
        # return self.encoder(self.src_embed(src))
#
    def decode(self, trg, encoder_output, idx=None):
        self.updateAdjMtx(idx)
        return self.prediction_generator(self.decoder(self.trg_embed(trg), encoder_output))
#
    def forward(self, src, trg, idx=None):
        '''
        src:  (batch_size, N, T_in, F_in)
        trg: (batch, N, T_out, F_out)
        '''
        self.updateAdjMtx(idx)
#
        encoder_output = self.encode(src)  # (batch_size, N, T_in, d_model)
#
        return self.decode(trg, encoder_output)


def make_model(DEVICE, num_layers, encoder_input_size, decoder_output_size, d_model, adj_mx, nb_head, num_of_weeks,
               num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=.0, aware_temporal_context=True,
               ScaledSAt=True, SE=True, TE=True, kernel_size=3, smooth_layer_num=0, residual_connection=True, use_LayerNorm=True,
               adj_mxs=None, n_hide=0):
#
    # LR rate means: graph Laplacian Regularization
#
    c = copy.deepcopy
#
    norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor).to(DEVICE)  # 通过邻接矩阵，构造归一化的拉普拉斯矩阵
#
    num_of_vertices = norm_Adj_matrix.shape[0]
#
    src_dense = nn.Linear(encoder_input_size, d_model)
#
    if ScaledSAt:  # employ spatial self attention
        position_wise_gcn = PositionWiseGCNFeedForward(spatialAttentionScaledGCN(norm_Adj_matrix, d_model, d_model), dropout=dropout)
    else:  # 不带attention
        position_wise_gcn = PositionWiseGCNFeedForward(spatialGCN(norm_Adj_matrix, d_model, d_model), dropout=dropout)
#
    trg_dense = nn.Linear(decoder_output_size, d_model)  # target input projection
#
    # encoder temporal position embedding
    max_len = max(num_of_weeks * 7 * 24 * num_for_predict, num_of_days * 24 * num_for_predict, num_of_hours * num_for_predict)
#
    w_index = search_index(max_len, num_of_weeks, num_for_predict, points_per_hour, 7*24)
    d_index = search_index(max_len, num_of_days, num_for_predict, points_per_hour, 24)
    h_index = search_index(max_len, num_of_hours, num_for_predict, points_per_hour, 1)
    en_lookup_index = w_index + d_index + h_index
#
    print('TemporalPositionalEncoding max_len:', max_len)
    print('w_index:', w_index)
    print('d_index:', d_index)
    print('h_index:', h_index)
    print('en_lookup_index:', en_lookup_index)
#
    if aware_temporal_context:  # employ temporal trend-aware attention
        attn_ss = MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout=dropout)  # encoder的trend-aware attention用一维卷积
        attn_st = MultiHeadAttentionAwareTemporalContex_qc_k1d(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout=dropout)
        att_tt = MultiHeadAttentionAwareTemporalContex_qc_kc(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout=dropout)  # decoder的trend-aware attention用因果卷积
    else:  # employ traditional self attention
        attn_ss = MultiHeadAttention(nb_head, d_model, dropout=dropout) # encoder
        attn_st = MultiHeadAttention(nb_head, d_model, dropout=dropout) # decoder
        att_tt = MultiHeadAttention(nb_head, d_model, dropout=dropout) # decoder
#
    if SE and TE:
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)  # decoder temporal position embedding
        decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position), c(spatial_position))
        decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position), c(spatial_position))
    elif SE and (not TE):
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(spatial_position))
        decoder_embedding = nn.Sequential(trg_dense, c(spatial_position))
    elif (not SE) and (TE):
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)  # decoder temporal position embedding
        decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position))
        decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position))
    else:
        encoder_embedding = nn.Sequential(src_dense)
        decoder_embedding = nn.Sequential(trg_dense)
#
    encoderLayer = EncoderLayer(d_model, attn_ss, c(position_wise_gcn), dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)
#
    encoder = Encoder(encoderLayer, num_layers)
#
    decoderLayer = DecoderLayer(d_model, att_tt, attn_st, c(position_wise_gcn),
                                dropout, residual_connection=residual_connection,
                                use_LayerNorm=use_LayerNorm, n_hide=n_hide)
#
    decoder = Decoder(decoderLayer, num_layers)
#
    generator = nn.Linear(d_model, decoder_output_size)
#
    model = EncoderDecoder(encoder,
                           decoder,
                           encoder_embedding,
                           decoder_embedding,
                           generator,
                           DEVICE,
                           adj_mxs=adj_mxs)
    # param init
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
#
    return model

def compute_val_loss(net, val_loader, criterion, sw, epoch):
    '''
    compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param epoch: int, current epoch
    :return: val_loss
    '''
#
    net.train(False)  # ensure dropout layers are in evaluation mode
#
    with torch.no_grad():
#
        val_loader_length = len(val_loader)  # nb of batch
#
        tmp = []  # 记录了所有batch的loss
#
        start_time = time()
#
        for batch_index, batch_data in enumerate(val_loader):
#
            encoder_inputs, decoder_inputs, labels, adj_idx, metadata = batch_data
#
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
#
            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
#
            labels = labels.unsqueeze(-1)  # (B，N，T，1)
#
            predict_length = labels.shape[2]  # T
            adj_idx = adj_idx.to(torch.int).cpu().numpy()
            net.updateAdjMtx(adj_idx)
            # encode
            encoder_output = net.encode(encoder_inputs)
            # print('encoder_output:', encoder_output.shape)
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
            decoder_input_list = [decoder_start_inputs]
            # Autoregressively predict
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]
#
            loss = criterion(predict_output, labels)  # 计算误差
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
#
        print('validation cost time: %.4fs' %(time()-start_time))
#
        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
#
    return validation_loss


def predict_main(epoch, data_loader, data_target_tensor, _max, _min, type, params_path, outdir='.'):
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
