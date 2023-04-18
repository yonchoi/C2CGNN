# import os
# import sys
# sys.path.insert(1, os.path.abspath(".."))

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

from modified_classes import Encoder,EncoderLayer
from modified_classes import MultiHeadAttention, spatialGCN
from modified_classes import PositionWiseGCNFeedForward
from modified_classes import subsequent_mask

import os
from time import time

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True, mask=False):
        super(EncoderLayer, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.self_attn = self_attn
        self.feed_forward_gcn = gcn
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 2)
        self.size = size
        self.mask = mask

    def forward(self, x):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''

        if self.mask:
            mask = subsequent_mask(x.size(-2)).to(x.device)  # (1, T', T')
        else:
            mask = None

        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, query_multi_segment=True, key_multi_segment=True))
            return self.sublayer[1](x, self.feed_forward_gcn)
        else:
            x = self.self_attn(x, x, x, mask, query_multi_segment=True, key_multi_segment=True)
            return self.feed_forward_gcn(x)


class DualEncoder(nn.Module):
    """
    encoder_history: encoder for temporal history of the target
    encoder_signal: encoder for concurrent signal of other sensors
    """
    def __init__(self,
                 encoder1,
                 encoder2,
                 src_dense1,
                 src_dense2,
                 generator1,
                 generator2,
                 generator3,
                 DEVICE,
                 adj_mxs=None,
                 setup=2):
#
        super(DualEncoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.src_embed1 = src_dense1
        self.src_embed2 = src_dense2
        self.generator1 = generator1
        self.generator2 = generator2
        self.generator3 = generator3
        self.adj_mxs = adj_mxs # List of normalized adj matrices (A,N,N), A=#ofAdjs
        self.setup = setup # whether to use both encoders
        self.to(DEVICE)
#
    def updateAdjMtx(self,idx):
        if idx is not None:
            adj_mtx = self.adj_mxs[idx]
            for module in [self.encoder1,
                           self.encoder2]:
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
#     def encode(self, src, idx=None):
#         '''
#         src: (batch_size, N, T_in, F_in)
#         '''
#         self.updateAdjMtx(idx)
#         h = self.src_embed(src)
#         return self.encoder(h)
#         # return self.encoder(self.src_embed(src))
# #
#     def decode(self, trg, encoder_output, idx=None):
#         self.updateAdjMtx(idx)
#         return self.prediction_generator(self.decoder(self.trg_embed(trg), encoder_output))
# #
    def forward(self, src, idx=None):
        '''
        src1: (batch_size, N, T_in, F1)
        src2: (batch_size, N, T_in, F2)
        trg: (batch, N, T_in, F1)
        '''
        src1 = src[:,:,:,[0]]
        src2 = src[:,:,:,1:]

        self.updateAdjMtx(idx)
        h1 = self.src_embed1(src1)
        h2 = self.src_embed2(src2)
#
        h1 = self.encoder1(h1)  # (batch_size, N, T_in, d_model)
        h2 = self.encoder2(h2)  # (batch_size, N, T_in, d_model)
#
        setup = self.setup
        if setup == 2:
            h = self.generator3(torch.cat((h1,h2),dim=-1))
            # h = self.generator1(h1) + self.generator2(h2)
        elif setup == 1:
            h = self.generator2(h2) # Only use sensors
        elif setup == 0:
            h = self.generator1(h1) # Only use history
        else:
            raise Exception("Setup must be either 0,1,2")
#
        return h


def make_model(DEVICE, num_layers, encoder_input_size, decoder_output_size, d_model, adj_mx, nb_head, num_of_weeks,
               num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=.0, aware_temporal_context=True,
               ScaledSAt=True, SE=True, TE=True, kernel_size=3, smooth_layer_num=0, residual_connection=True, use_LayerNorm=True,
               adj_mxs=None, n_hide=0, setup=2):
#
    # LR rate means: graph Laplacian Regularization
#
    c = copy.deepcopy
#
    norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor).to(DEVICE)  # 通过邻接矩阵，构造归一化的拉普拉斯矩阵
#
    num_of_vertices = norm_Adj_matrix.shape[0]
#
#
    if ScaledSAt:  # employ spatial self attention
        position_wise_gcn = PositionWiseGCNFeedForward(spatialAttentionScaledGCN(norm_Adj_matrix, d_model, d_model), dropout=dropout)
    else:  # 不带attention
        position_wise_gcn = PositionWiseGCNFeedForward(spatialGCN(norm_Adj_matrix, d_model, d_model), dropout=dropout)
#
    # trg_dense = nn.Linear(decoder_output_size, d_model)  # target input projection
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
    components = []
    for i in range(2):
        if i == 0:
            src_dense = nn.Linear(1, d_model)
        else:
            src_dense = nn.Linear(encoder_input_size-1, d_model)
    #
        if aware_temporal_context:  # employ temporal trend-aware attention
            attn_ss = MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout=dropout)  # encoder的trend-aware attention用一维卷积
            # attn_st = MultiHeadAttentionAwareTemporalContex_qc_k1d(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout=dropout)
            # att_tt = MultiHeadAttentionAwareTemporalContex_qc_kc(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout=dropout)  # decoder的trend-aware attention用因果卷积
        else:  # employ traditional self attention
            attn_ss = MultiHeadAttention(nb_head, d_model, dropout=dropout) # encoder
            # attn_st = MultiHeadAttention(nb_head, d_model, dropout=dropout) # decoder
            # att_tt = MultiHeadAttention(nb_head, d_model, dropout=dropout) # decoder
    #
        if SE and TE:
            encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)  # decoder temporal position embedding
            # decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
            spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num)
            encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position), c(spatial_position))
            # decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position), c(spatial_position))
        elif SE and (not TE):
            spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num)
            encoder_embedding = nn.Sequential(src_dense, c(spatial_position))
            # decoder_embedding = nn.Sequential(trg_dense, c(spatial_position))
        elif (not SE) and (TE):
            encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)  # decoder temporal position embedding
            # decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
            encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position))
            # decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position))
        else:
            encoder_embedding = nn.Sequential(src_dense)
            # decoder_embedding = nn.Sequential(trg_dense)
    #
        mask = i == 0 # turn on mask to prevent looking ahead for history
        encoderLayer = EncoderLayer(d_model, attn_ss, c(position_wise_gcn), dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm, mask=mask)
    #
        encoder = Encoder(encoderLayer, num_layers)
    #
        generator = nn.Linear(d_model, decoder_output_size)
    #
        components.append(encoder)
        components.append(encoder_embedding)
        components.append(generator)
#
    encoder1,encoder_embedding1,generator1,encoder2,encoder_embedding2,generator2 = components
    # non-linear decoder
    generator3 = nn.Sequential(nn.Linear(d_model*2, d_model),
                               nn.ReLU(),
                               nn.Linear(d_model, d_model),
                               nn.ReLU(),
                               nn.Linear(d_model, decoder_output_size))

    model = DualEncoder(encoder1,
                        encoder2,
                        encoder_embedding1,
                        encoder_embedding2,
                        generator1,
                        generator2,
                        generator3,
                        DEVICE,
                        adj_mxs=adj_mxs,
                        setup=setup)
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
            labels = labels.unsqueeze(-1)  # (B，N，T，1)
#
            predict_length = labels.shape[2]  # T
            adj_idx = adj_idx.to(torch.int).cpu().numpy()
            net.updateAdjMtx(adj_idx)
            # encode
            predict_output = net(encoder_inputs)
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
