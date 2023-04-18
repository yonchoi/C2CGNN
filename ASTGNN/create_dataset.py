import numpy as np
import torch

def sample_time_window(seq, window_in=10, window_out=10):
    """
    seq: Either (n_node, n_time, n_feature) or (n_node, n_time, n_feature)
    """
    if len(seq.shape) == 2:
        seq = np.expand_dims(seq,axis=-1) # (n_node, n_time, n_feature)
#
    assert len(seq.shape) == 3, 'Either put in 3D or 2D array'
#
    time_length = seq.shape[1]
    valid_time_points = np.arange(window_in, time_length - window_out)
#
    seq_in  = []
    seq_out = []
    for t in valid_time_points:
        # For input data
        t_in_start  = t - window_in
        t_in_end    = t
        seq_in.append(seq[:,t_in_start:t_in_end])
        # For output data
        t_out_start = t
        t_out_end   = t + window_out
        seq_out.append(seq[:,t_out_start:t_out_end])
    return np.array(seq_in), np.array(seq_out)

class TemporalDataLoader(object):
#
    def __init__(self, input, target, adj_idx, metadata, adj_mxs=None,adj_name=None):
        self.input    = input
        self.target   = target
        self.adj_idx  = adj_idx
        self.metadata = metadata
        self.adj_mxs  = adj_mxs
        self.adj_name = adj_name
#
    def __getitem__(self, indices):
        return TemporalDataLoader(self.input[indices],
                                  self.target[indices],
                                  self.adj_idx[indices],
                                  self.metadata[indices],
                                  self.adj_mxs,
                                  self.adj_name)
#
    def setAdjMtx(self, adj_mxs, adj_name=None):
        self.adj_mxs = adj_mxs
        if adj_name is not None:
            self.adj_name = adj_name
#
    def __len__(self):
        return len(self.input)
#
    def write(self, dir):
        np.savez(dir,
                 input = self.input,
                 target = self.target,
                 adj_idx = self.adj_idx,
                 metadata = self.metadata,
                 adj_mxs = self.adj_mxs,
                 adj_name = self.adj_name)

def load_npz(dir):
    npz = np.load(dir)
    return TemporalDataLoader(npz['input'],
                              npz['target'],
                              npz['adj_idx'],
                              npz['metadata'],
                              npz['adj_mxs'],
                              npz['adj_name'])

def combine_dataloader(Xs):
    """"""
    input    = np.concatenate([X.input for X in Xs],axis=0)
    target   = np.concatenate([X.target for X in Xs],axis=0)
    adj_idx  = np.concatenate([X.adj_idx for X in Xs],axis=0)
    metadata = np.concatenate([X.metadata for X in Xs],axis=0)
    return TemporalDataLoader(input,target,adj_idx,metadata)


# ==============================================================================
from sklearn.model_selection import train_test_split

def create_data_loaders(custom_dataset, batch_size,
                        train_size=0.5, train_val_size=0.8,
                        shuffle=True, DEVICE = torch.device('cuda:0')):
    '''
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)
    '''
#
    train_idx, test_idx = train_test_split(np.arange(len(custom_dataset)),
                                           stratify=custom_dataset.adj_idx,
                                           train_size=train_size)
    train_idx, val_idx  = train_test_split(train_idx,
                                           stratify=custom_dataset.adj_idx[train_idx],
                                           train_size=train_val_size)
# #
#     train_dataset = custom_dataset[train_idx]
#     train_x = np.moveaxis(train_dataset.input,-2,-1)
#     train_target = train_dataset.target
#     train_adjidx = train_dataset.adj_idx
# #
#     test_dataset = custom_dataset[test_idx]
#     test_x = np.moveaxis(test_dataset.input,-2,-1)
#     test_target = test_dataset.target
#     test_adjidx = test_dataset.adj_idx
# #
#     val_dataset = custom_dataset[val_idx]
#     val_x = np.moveaxis(val_dataset.input,-2,-1)
#     val_target = val_dataset.target
#     val_adjidx = val_dataset.adj_idx
#
    outputs = []
    for datatype, idx in zip(['Train','Val','Test'],[train_idx, val_idx, test_idx]):
        dataset = custom_dataset[idx]
        x = np.moveaxis(dataset.input,-2,-1) # input encoder
        target = dataset.target
        adjidx = dataset.adj_idx
        y_start = x[:, :, 0:1, -1:] # input decoder
        y_start = np.squeeze(y_start, 2)
        y = np.concatenate((y_start, target[:, :, :-1]), axis=2)  # (B, N, T)
        ##
        x_tensor = torch.from_numpy(x).type(torch.FloatTensor).to(DEVICE) # (B, N, F, T)
        y_tensor = torch.from_numpy(y).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
        target_tensor = torch.from_numpy(target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
        adjidx_tensor = torch.from_numpy(adjidx).type(torch.FloatTensor).to(DEVICE)  # (B,)
        ##
        tensordataset = torch.utils.data.TensorDataset(x_tensor,
                                                       y_tensor,
                                                       target_tensor,
                                                       adjidx_tensor)
        loader = torch.utils.data.DataLoader(tensordataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle)
        outputs.append(loader)
        outputs.append(target_tensor)
        print(f'{datatype}:', x_tensor.size(), target_tensor.size())
#
    mean = np.array(0).reshape(1,1,1,1)
    std  = np.array(1).reshape(1,1,1,1)
    outputs.append(mean)
    outputs.append(std)
#
    return outputs

def create_data_loaders_channels(custom_dataset, batch_size, idx_feature=0, t_shift=1,
                                 train_size=0.5, train_val_size=0.8,
                                 split_by_well=True, num_for_predict=-1,
                                 shuffle=True, DEVICE = torch.device('cuda:0')):
    '''
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)
    '''
#
    if split_by_well:
        well_names = custom_dataset.adj_idx
        well_unique = np.unique(well_names)
        assert len(well_unique) > 2, f'To split by well, length of unique wells must be at least 2 instead of {len(well_unique)}'
        well_train = well_unique[2:]
        well_test  = well_unique[[1]]
        well_val   = well_unique[[0]]
        train_idx = np.where(np.isin(well_names, well_train))[0]
        test_idx  = np.where(np.isin(well_names, well_test))[0]
        val_idx   = np.where(np.isin(well_names, well_val))[0]
    else:
        train_idx, test_idx = train_test_split(np.arange(len(custom_dataset)),
                                               stratify=custom_dataset.adj_idx,
                                               train_size=train_size)
        train_idx, val_idx  = train_test_split(train_idx,
                                               stratify=custom_dataset.adj_idx[train_idx],
                                               train_size=train_val_size)
# #
#     train_dataset = custom_dataset[train_idx]
#     train_x = np.moveaxis(train_dataset.input,-2,-1)
#     train_target = train_dataset.target
#     train_adjidx = train_dataset.adj_idx
# #
#     test_dataset = custom_dataset[test_idx]
#     test_x = np.moveaxis(test_dataset.input,-2,-1)
#     test_target = test_dataset.target
#     test_adjidx = test_dataset.adj_idx
# #
#     val_dataset = custom_dataset[val_idx]
#     val_x = np.moveaxis(val_dataset.input,-2,-1)
#     val_target = val_dataset.target
#     val_adjidx = val_dataset.adj_idx
#
    outputs = []
    for datatype, idx in zip(['Train','Val','Test'],[train_idx, val_idx, test_idx]):
        dataset = custom_dataset[idx]
        x = np.moveaxis(dataset.input,-2,-1) # input encoder
        # target = dataset.target
        adjidx = dataset.adj_idx
        metadata = dataset.metadata
        # y = target
        target = x[:,:,idx_feature][:,:,t_shift:]
        idx_non_feature = np.arange(x.shape[2]) != idx_feature
        x = [x[:,:,[idx_feature]][:,:,:,:-t_shift],
             x[:,:,idx_non_feature][:,:,:,t_shift:]]
        x = np.concatenate(x,axis=2)
        y = target
        ##
        x_tensor = torch.from_numpy(x).type(torch.FloatTensor).to(DEVICE) # (B, N, F, T)
        y_tensor = torch.from_numpy(y).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
        target_tensor = torch.from_numpy(target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
        adjidx_tensor = torch.from_numpy(adjidx).type(torch.FloatTensor).to(DEVICE)  # (B,)
        metadata_tensor = torch.from_numpy(metadata).type(torch.int).to(DEVICE)  # (B,)
        ##
        tensordataset = torch.utils.data.TensorDataset(x_tensor,
                                                       y_tensor,
                                                       target_tensor,
                                                       adjidx_tensor,
                                                       metadata_tensor)
        loader = torch.utils.data.DataLoader(tensordataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle)
        outputs.append(loader)
        outputs.append(target_tensor)
        print(f'{datatype}:', x_tensor.size(), target_tensor.size())
#
    mean = np.array(0).reshape(1,1,1,1)
    std  = np.array(1).reshape(1,1,1,1)
    outputs.append(mean)
    outputs.append(std)
#
    return outputs

from scipy.spatial.distance import cdist
from scipy.stats import rankdata

def generate_adj(X,Y, setting=None, min_dist=None, num_neighbor=None):
    """
    Input:
        X: [n_cell x n_timepoint]
        Y: [n_cell x n_timepoint]
    """
    if setting == 'identity':
        adj = np.tile(np.identity(len(X)), (X.shape[1],1,1))
    else:
        if setting is not None:
            metric,threshold = setting.split('-')
            if metric == 'dist':
                min_dist = float(threshold)
            if metric == 'neighbor':
                num_neighbor = int(threshold)
        #
        coord = np.array([X,Y]) # n_coord x n_cell x n_time
        coord = np.swapaxes(coord,0,2)
        dist = np.array([cdist(c,c) for c in coord])
        if min_dist is not None:
            adj = dist <= min_dist
        elif num_neighbor is not None:
            dist_rank = rankdata(dist,axis=-1)
            adj = dist_rank < num_neighbor
        else:
            raise Exception('Specify either min_dist or num_neighbor')
    return adj.astype('float32')
