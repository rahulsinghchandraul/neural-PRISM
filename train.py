import numpy as np
import math
import torch
from torch import nn
from model_downsample import *
import scipy.io as sio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from dataset_phate import *
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu') # if gpu is available use gpu

print(device)

scaler = MinMaxScaler()
# scaler = StandardScaler()
band = 'alpha'

if band == 'delta':   #math.floor(384 * 0.6) #256*6 # theta 427, alpha 854, delta 183
    eeg_seq_len =  183    
elif band == 'theta':
    eeg_seq_len =  427 
elif band == 'alpha':
    eeg_seq_len =  854 

eeg_features = 31

fnirs_seq_len = 41
fnirs_features = 134   #134


# group = 1
# sub = 1
block_test = 6

eeg = sio.loadmat(f'./data/Band_01/eeg_{band}.mat')
eeg = eeg[f'eeg_{band}']

eeg_phate = np.loadtxt(f'./data/eeg_phate3_{band}.txt', delimiter=',')

eeg_train = eeg[np.where( (eeg[:, 32] != block_test) )]
print(eeg_train.shape)

eeg_phate_train = eeg_phate[np.where( (eeg[:, 32] != block_test) )]

eeg_test = eeg[np.where( (eeg[:, 32] == block_test) )]
print(eeg_test.shape)

eeg_phate_test = eeg_phate[np.where( (eeg[:, 32] == block_test) )]

# fnirs = sio.loadmat('./data/fnirs_aligned_ica.mat')
# fnirs = fnirs['fnirs_aligned_ica']

fnirs = sio.loadmat('./data/Band_01/fnirs_decimated.mat')
fnirs = fnirs['fnirs_decimated']

# fnirs = scaler.fit_transform(fnirs)
fnirs[:,0:134] = fnirs[:,0:134] 
print(fnirs[:,137])

fnirs_train = fnirs[np.where(  (fnirs[:, 134] != block_test) )]
fnirs_test = fnirs[np.where( (fnirs[:, 134] == block_test) )]

print(fnirs_train.shape)
print(fnirs_test.shape)

# fnirs = scaler.fit_transform(fnirs)
# fnirs = fnirs * 10.0
# print(fnirs.shape)



batch_size = 16

#### load data


dataset_train = Fnirs134EEGDataset(eeg_train, fnirs_train, eeg_phate_train, eeg_chunk_length = eeg_seq_len, fnirs_chunk_length = fnirs_seq_len, n_samples = 498) # theta 427, alpha 854, delta 183
dataset_val = Fnirs134EEGDataset(eeg_test, fnirs_test, eeg_phate_test, eeg_chunk_length = eeg_seq_len, fnirs_chunk_length = fnirs_seq_len, n_samples = 99) # theta 427, alpha 854, delta 183


train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle = True)
val_loader = DataLoader(dataset_val, batch_size=99, shuffle = True)
############################3

# for j, (eeg_seq, fnirs_seq) in enumerate(val_loader):
#     print(j, eeg_seq.shape, fnirs_seq.shape)



enc_embedding_dim = 128
dec_embedding_dim = 128

dropout = 0.1

#__init__(self, enc_seq_len, dec_seq_len, enc_features, dec_features, enc_embedding_dim, dec_embedding_dim, device):
# model = RecurrentAutoencoder(eeg_seq_len, eeg_seq_len, eeg_features, eeg_features, enc_embedding_dim, dec_embedding_dim, device) # For EEG AE
# model = RecurrentAutoencoder(fnirs_seq_len, eeg_seq_len, fnirs_features, eeg_features, 
#                         enc_embedding_dim, dec_embedding_dim,  dropout, device)  # fNIRS to EEG
# model = RecurrentAutoencoder(eeg_seq_len, fnirs_seq_len, eeg_features, fnirs_features, 
#                         enc_embedding_dim, dec_embedding_dim, dropout, device)  # EEG to fNIRS
# model = model.double() 
# model = model.to(device) 
# ########## load data
# dataset = FnirsEEGDataset(eeg, fnirs, eeg_phate, eeg_chunk_length = eeg_seq_len, fnirs_chunk_length = fnirs_seq_len, n_samples = 597) # theta 427, alpha 854, delta 183
# # train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

# ################################
# validation_split = .05
# shuffle_dataset = True
# random_seed= 566

# # Creating data indices for training and validation splits:
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]

# # print(train_indices)
# print(len(val_indices))

# # Creating PT data samplers and loaders:
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)

# train_loader = DataLoader(dataset, batch_size=batch_size, 
#                                            sampler=train_sampler)
# val_loader = DataLoader(dataset, batch_size=len(val_indices),
#                                                 sampler=valid_sampler)


############################3

# for j, (eeg_seq, fnirs_seq) in enumerate(val_loader):
#     print(j, eeg_seq.shape, fnirs_seq.shape)



# enc_embedding_dim = 128
# dec_embedding_dim = 128

# dropout = 0.1

#__init__(self, enc_seq_len, dec_seq_len, enc_features, dec_features, enc_embedding_dim, dec_embedding_dim, device):
# model = RecurrentAutoencoder(eeg_seq_len, eeg_seq_len, eeg_features, eeg_features, enc_embedding_dim, dec_embedding_dim, device) # For EEG AE
# model = RecurrentAutoencoder(fnirs_seq_len, eeg_seq_len, fnirs_features, eeg_features, 
#                         enc_embedding_dim, dec_embedding_dim,  dropout, device)  # fNIRS to EEG
model = RecurrentAutoencoder(eeg_seq_len, fnirs_seq_len, eeg_features, fnirs_features, 
                        enc_embedding_dim, dec_embedding_dim, dropout, device)  # EEG to fNIRS
model = model.double() 
model = model.to(device) 

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
criterion = nn.MSELoss(reduction='mean').to(device)
epochs = 300

train_loss_all = []
val_loss_all = []

teacher_forcing_ratio = 0.5

for epoch in range(1, epochs+1):
    model = model.train()
    train_losses = []
    for j, (eeg_seq, fnirs_seq, eeg_phate_seq) in enumerate(train_loader):
        optimizer.zero_grad()            # no accumulation
        # print(eeg_seq.shape)
        eeg_seq = eeg_seq.to(device)   # putting sequence to gpu
        fnirs_seq = fnirs_seq.to(device)   # putting sequence to gpu
        eeg_phate_seq = eeg_phate_seq.to(device)
        # print(eeg_phate_seq.shape)
        seq_enc, seq_pred = model(eeg_seq, fnirs_seq, teacher_forcing_ratio)       # EEG to fNIRS prediction
        # seq_enc, seq_pred = model(fnirs_seq, eeg_seq, teacher_forcing_ratio)       # fNIRS to EEG prediction
        # print(eeg_seq[:, 1:,:].shape, seq_pred.shape)
        # print(seq_enc.shape)
        loss1 = criterion(seq_pred, fnirs_seq)  # measuring error
        # dist_ph = eeg_phate_seq[:, :eeg_seq_len -1, :]
        # loss2 = criterion()
        ph_dis = torch.cdist(eeg_phate_seq[:, :eeg_seq_len -1, :], eeg_phate_seq[:, 1:, :], p=2)
        latent_dis = torch.cdist(seq_enc[:, :eeg_seq_len -1, :], seq_enc[:, 1:, :], p=2)
        loss2 = criterion(ph_dis, latent_dis)
        loss = loss1  #+ loss2
        # loss = criterion(seq_pred, eeg_seq)  # measuring error
        loss.backward()                  # backprop
        optimizer.step()
        train_losses.append(loss.item())  # record loss by adding to training losses

    train_loss = np.mean(train_losses)   # computing loss on training and val data for this epoch
    print(f'Epoch {epoch}: train loss = {train_loss}')
    train_loss_all.append(train_loss)

    
    if epoch % 1 == 0:
        model = model.eval()
        val_losses = []
        with torch.no_grad():  # requesting pytorch to record any gradient for this block of code
            for k, (eeg_seq_true_ev, fnirs_seq_true_ev, eeg_phate_seq_true_ev) in enumerate(val_loader): 
                if k==0:
                    # print(eeg_seq_true_ev.shape)
                    eeg_seq_true_ev = eeg_seq_true_ev.to(device)   # putting sequence to gpu
                    fnirs_seq_true_ev = fnirs_seq_true_ev.to(device)   # putting sequence to gpu
                    eeg_phate_seq_true_ev = eeg_phate_seq_true_ev.to(device)
                    seq_enc, seq_pred_ev = model(eeg_seq_true_ev, fnirs_seq_true_ev, 0.1)       # EEG to fNIRS
                    # seq_enc, seq_pred_ev = model(fnirs_seq_true_ev, eeg_seq_true_ev, 0.01)       # fNIRS to EEG
                    loss1 = criterion(seq_pred_ev, fnirs_seq_true_ev)  # measuring error
                    # loss1 = criterion(seq_pred_ev[:, 1:, :], eeg_seq_true_ev[:, 1:, :])  # measuring error
                    val_losses.append(loss1.item())
                    break
                # print(seq_true_ev.shape)
            val_loss = np.mean(val_losses)
            print(f'Epoch {epoch}: Val loss = {val_loss}')
            val_loss_all.append(val_loss)
    
                

torch.save(model, f'./saved_Models/eeg2_fnirs134Decimated_downsample_{band}_Norm01.pth') #g2_3_eeg2fnirs_model sub4_eeg2fnirs_model


# # np.savetxt('train_loss_EEG2fNIRS6.txt', train_loss_all, delimiter=',')
# np.savetxt('train_loss_EEG2fNIRS_g2_3.txt', train_loss_all, delimiter=',')

# # np.savetxt('val_loss_EEG2fNIRS6.txt', val_loss_all, delimiter=',')
# np.savetxt('val_loss_EEG2fNIRS_g2_3.txt', val_loss_all, delimiter=',')

# seq_true_ev = np.asarray(fnirs_seq_true_ev[10, :, : ]) # When predicting fNIRS
seq_true_ev = np.asarray(fnirs_seq_true_ev.cpu()) # when predicting fnirs
# print(seq_true_ev.shape)
sio.savemat('./saved_Results/seq_true_ev_134.mat',  {'seq_true_ev': seq_true_ev})

seq_pred_ev = np.asarray(seq_pred_ev.cpu())
# print(seq_pred_ev.shape)
sio.savemat('./saved_Results/seq_pred_ev_134.mat',  {'seq_pred_ev': seq_pred_ev})



# def train_model(model, train_dataset, n_epochs):
#        # summing error; L1Loss = mean absolute error in torch
#   history = dict(train=[], val=[])                      # recording loss history
#   for epoch in range(1, n_epochs + 1):
#     model = model.train()
#     train_losses = []
#     for seq_true in train_dataset:     # iterate over each seq for train data
#       optimizer.zero_grad()            # no accumulation
#       seq_true = seq_true.to(device)   # putting sequence to gpu
#       seq_pred = model(seq_true)       # prediction
#       loss = criterion(seq_pred, seq_true)  # measuring error
#       loss.backward()                  # backprop
#       optimizer.step()
#       train_losses.append(loss.item())  # record loss by adding to training losses
    

#     # val_losses = []
#     # model = model.eval()
#     # with torch.no_grad():  # requesting pytorch to record any gradient for this block of code
#     #   for seq_true in val_dataset:
#     #       seq_true = seq_true.to(device)   # putting sequence to gpu
#     #       seq_pred = model(seq_true)       # prediction

#     #       loss = criterion(seq_pred, seq_true)  # recording loss

#     #       val_losses.append(loss.item())    # storing loss into the validation losses

#     train_loss = np.mean(train_losses)   # computing loss on training and val data for this epoch
#     history['train'].append(train_loss)

#     print(f'Epoch {epoch}: train loss = {train_loss}')

#   return model.eval(), history      # after training, returning model to evaluation mode

