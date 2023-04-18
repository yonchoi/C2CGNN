import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset

import numpy as np
from copy import deepcopy

class PatienceCounter():

    def __init__(self, epsilon=0,max_count=5):

        self.epsilon=epsilon
        self.max_count=max_count

        # Initialize the counts and min_loss
        self.min_loss = np.Inf
        self.patience_count = 0
        self.stop = False

    def count(self, loss):

        if self.min_loss + self.epsilon > loss:

            self.min_loss = loss
            self.patience_count = 0
            add_count = False
        else:
            self.patience_count += 1
            add_count = True

        if self.patience_count > self.max_count:
            self.stop = True

        return add_count

    def early_stop(self):
        return self.stop


## Set functions for initializing weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def train_epoch(net,
                dataloader,
                criterion,
                optimizer,
                name:str = None,
                train:bool = False,
                scheduler=None):

    total_loss = 0
    num_targets = 0

    for i, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs_dict = {}

        inputs_dict['x']  = data['X']

        if 'M' in data.keys():
            inputs_dict['m'] = data['M']

        targets = data['Y']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(**inputs_dict)
        loss = criterion(outputs, targets)

        if train:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # print statistics
        total_loss += loss.item() * len(targets)
        num_targets += len(targets)

    loss = total_loss / num_targets

    return loss


def train(net, trainloader, testloader,
          writer = None,
          epsilon=0,
          max_count=np.Inf,
          optimizer=None,
          epoch=100,
          criterion=None,
          scheduler=None):

    if criterion is None:
        criterion = F.mse_loss

    # criterion(pred,input) measures cross entropy loss between pred,input
    # pred should be in one-hot encoded format [n_sample, num_base]
    # input should be in categorical(long) format
    if optimizer is None:
        optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-3)
    # weight decay = L2 regularization
    # lr = learning rate

    counter = PatienceCounter(epsilon=epsilon,
                              max_count=max_count)

    #optimize
    best_state_dict = net.state_dict()

    for epoch in range(epoch):  # loop over the dataset multiple times

        for train_type,dataloader in zip(['train','test'],
                               [trainloader,testloader]):

            if not counter.early_stop():

                loss = train_epoch(net        = net,
                                   dataloader = dataloader,
                                   criterion  = criterion,
                                   optimizer  = optimizer,
                                   name  = train_type,
                                   train = train_type == 'train',
                                   scheduler=scheduler)

                print(f'[epoch %d] {train_type} loss: %.3f' % (epoch + 1, loss) )

                if writer is not None:
                    writer.add_scalar(f'Loss/{train_type}', loss, epoch + 1)

                if train_type == 'test':
                    add_count = counter.count(loss)

                    if not add_count:
                        print('state saved')
                        best_state_dict = deepcopy(net.state_dict())

    # Reload the best saved state
    net.load_state_dict(best_state_dict)

class Net(nn.Module):
    # weights initialization with default method similar to kaiming
    def __init__(self,
                 n_input:int,
                 n_output:int,
                 n_input_fc:int=0,
                 n_hiddens = [64,32],
                 mode:str ='flatten',
                 activation=F.relu,
                 kernel_size_pool:int=3,
                 stride:int=1,
                 kernel_size:int=10,
                 conv_channels:list=[16,16],
                 dropout=0.2,
                 batchnorm=False,
                 n_channel:int=1):

        super().__init__()

        self.mode     = mode
        self.n_input  = n_input # n_channel * n_feature
        self.n_ouptut = n_output
        self.n_hiddens = n_hiddens
        self.batchnorm = batchnorm
        self.n_input_fc = n_input if n_input_fc == 0 else n_input_fc
        self.n_channel = n_channel
        self.n_feature = int(self.n_input/self.n_channel)

        if mode == 'conv':

            self.pool = nn.MaxPool1d(kernel_size=kernel_size_pool,
                                     stride=stride)

            self.convs = nn.ModuleList()

            in_channels = n_channel
            for out_channels in conv_channels:
                conv = nn.Conv1d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size)
                self.convs.append(conv)
                in_channels = out_channels

            # calculate required ouptut size
            from torchshape import tensorshape


            outshape = (1, self.n_channel, self.n_feature) # (n_cell,n_channel,n_feature)
            for conv in self.convs:
                outshape = tensorshape(conv,outshape)
                outshape = tensorshape(self.pool,outshape)
            n_output_conv = np.prod(outshape[1:])
            self.fc1 = nn.Linear(n_output_conv, n_input)
            n_in = n_input

        elif mode == 'flatten':
          # self.fcs = []
          # fc = nn.Linear(n_input, self.n_hiddens[0])
          # self.fcs.append(fc)
          # fc = nn.Linear(self.n_hiddens[0], self.n_hiddens[1])
          # self.fcs.append(fc)
          # n_in = self.n_hiddens[1]
          n_in = n_input

        self.fcs = nn.ModuleList()
        if batchnorm:
            self.bns = nn.ModuleList()
        else:
            self.bns = None

        n_in = self.n_input_fc
        for i,n_out in enumerate(self.n_hiddens):
          self.fcs.append(nn.Linear(n_in, n_out))
          if batchnorm:
              self.bns.append(nn.BatchNorm1d(n_out))
          n_in = n_out # Update n_in to be the previous n_out


        self.fc_final = nn.Linear(n_in, n_output)
        self.drop = nn.Dropout(dropout)

        self.activation = activation

    # Forward pass
    def forward(self, x, **kwargs):

        if self.mode == 'conv':
            x = x.view(len(x), self.n_channel, self.n_feature) # [sample,channel*feature] -> [sample,channel,feature]
            # x = torch.unsqueeze(x, 1) # [sample,feature] -> [sample,1,feature]
            for conv in self.convs:
                x = self.pool(self.activation(conv(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = self.activation(self.fc1(x))

        for ii,fc in enumerate(self.fcs):
            x = fc(x)
            if self.batchnorm:
                x = self.bns[ii](x)
            x = self.activation(x)
            x = self.drop(x)

        x = self.fc_final(x)
        return x

    #
    def train(self, **kwargs):
        train(self, **kwargs)


# Create model that takes in metadata

class mixedNet(Net):

    def __init__(self,
                 n_input_meta:int,
                 n_hiddens_meta=[],
                 **kwargs):

        # Adust number of inputs to account for concatenation with meta
        if len(n_hiddens_meta) == 0:
            n_output_meta = n_input_meta
        else:
            n_output_meta = n_hiddens_meta[-1]

        kwargs['n_input_fc'] = kwargs['n_input'] + n_output_meta

        super().__init__(**kwargs)

        self.n_hiddens_meta = n_hiddens_meta
        self.n_input_meta = n_input_meta
        self.n_output_meta = n_output_meta

        # Set FC layers for meta network
        self.fcs_meta = nn.ModuleList()
        n_in = self.n_input_meta
        for i,n_out in enumerate(self.n_hiddens_meta):
          self.fcs_meta.append(nn.Linear(n_in, n_out))
          n_in = n_out # Update n_in to be the previous n_out

    def forward(self,x,m):

        # Run convolution on the input
        if self.mode == 'conv':
            x = x.view(len(x), self.n_channel, self.n_feature) # [sample,channel*feature] -> [sample,channel,feature]
            # x = torch.unsqueeze(x, 1) # [sample,feature] -> [sample,1,feature]
            for conv in self.convs:
                x = self.pool(self.activation(conv(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = self.activation(self.fc1(x))

        # Run FC network on meta
        for fc in self.fcs_meta:
            m = self.activation(fc(m))
            m = self.drop(m)

        # Concatenate x, from input to m, meta
        x = torch.cat((x,m,),dim=1)

        # FC for the final output
        for ii,fc in enumerate(self.fcs):
            x = fc(x)
            if self.batchnorm:
                x = self.bns[ii](x)
            x = self.activation(x)
            x = self.drop(x)

        x = self.fc_final(x)

        return x

class CustomDataset(Dataset):
    """ Custom dataset """

    def __init__(self, X, Y, M=None, P=None, C=None):
        self.X = X
        self.Y = Y
        self.M = M
        self.P = P
        self.C = C

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {"X" : self.X[idx],
                  "Y" : self.Y[idx]}

        if self.M is not None:
            sample['M'] = self.M[idx]

        if self.P is not None:
            sample['P'] = self.P[idx]

        if self.C is not None:
            sample['C'] = self.C[idx]

        return sample
