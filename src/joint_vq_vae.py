import torch 
import torchvision
import torch.nn as nn 
from IPython.display import Image 
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F

import sys
import os
import csv
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from PIL import Image

import pandas as pd
import linecache
import re

from count_vae import CountVAE 
from datasets import STDataset_triple

from vq_vae_flat import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 12345
random.seed(seed)
torch.manual_seed(seed)

########## Data set loading #########################################################

metafile = "../data/meta_train.tsv"
dat1 = STDataset_triple(metafile, iso_cutoff=1.2)
train_loader = DataLoader(dat1, batch_size=32, shuffle=False)

# Generate metadata files for train/test set and instantiate Datasets/DataLoaders
img_size = 256
in_channels = 3

exp_params = {
        'latent_dim': 512,
        'batch_size': 32,
        'img_size': img_size,
        'lr': 1e-4,
        'num_epochs': 20
}

# Instantiate VAE model
D = 32
K = 32
img_size = 16
model = VQVAE(in_channels=3, embedding_dim=D, num_embeddings=K, img_size=16, 
            flat_hidden = 131072, flat_output = 32)

########## Pretrained Vision Model loading, count VAE initialization ################## 

learning_rate = 1e-4
num_classes = exp_params['latent_dim']  #Dimension of the Histology representation we want
n_latent = 20
n_latent_concat = num_classes + n_latent


#Initialize image model


model = model.to(device)

#Initialize count VAE model
nnet_model = CountVAE(
            11162, n_layers = 1, n_latent=20, n_pretrained_features=num_classes,
            n_latent_concat=n_latent_concat, n_hidden=256, gene_likelihood="zinb", dropout_rate=.1,
            n_labels=1
        ).to(device)


######### Parameters to optimize for ##################################################

adam_param_list = []
adam_param_list.append({'params': model.parameters()})
adam_param_list.append({'params': nnet_model.parameters()})


learning_rate = 1e-4
optimiser = torch.optim.Adam(adam_param_list, lr=learning_rate)


######## Training of model ############################################################

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)


def train_joint_vae(nnet, model, train_loader, optimiser, epoch, num_epoch, learning_rate, log_interval):

    count_loss_sum = 0
    trip_loss_sum = 0
    net_loss_sum = 0
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)


    for i, sample_batched in enumerate(train_loader):
        
        optimiser.zero_grad()
        
        images, data, annotation, k_distant, k_near = sample_batched
        images = images.to(device)
        data = data.type(torch.float32).to(device) 
        library = torch.log(torch.sum(data, 1) + 1.0).unsqueeze(1).to(device)
        k_dist_img = k_distant[0][:,0,:,:].to(device)
        k_dist_count = k_distant[1][:,0,:].type(torch.float32).to(device)
        k_near_img = k_near[0][:,0,:,:].to(device)
        k_near_count = k_near[1][:,0,:].type(torch.float32).to(device)

        img_outputs = model(images)
        # Weight KLD portion of loss by fraction of dataset being considered.
        M_N = images.size(0) / len(train_loader.dataset)
        img_loss = model.loss_function(*img_outputs, M_N=M_N)
        
        #Sample Z- latent space from vq vae
        concat_z = model.sample_flat(images).to(device)
        k_dist_concat_z = model.sample_flat(k_dist_img).to(device)
        k_near_concat_z = model.sample_flat(k_near_img).to(device)
        
        #Count vae and count loss
        reconst_loss_train, kl_divergence_train = nnet_model(data, library, concat_z)
        train_loss = torch.mean(reconst_loss_train + kl_divergence_train)
        
        #Sample cuont vae
        data_z = nnet_model.sample_from_posterior_z(data)[2].to(device)
        k_dist_z = nnet_model.sample_from_posterior_z(data)[2].to(device)
        k_near_z = nnet_model.sample_from_posterior_z(data)[2].to(device)
        
        #Concatenaate spaces for triplet loss
        data_z_comb = torch.cat((data_z, concat_z), 1) 
        k_dist_z_comb = torch.cat((k_dist_z, k_dist_concat_z), 1) 
        k_near_z_comb = torch.cat((k_near_z, k_near_concat_z), 1) 

        trip_loss = triplet_loss(data_z_comb, k_near_z_comb, k_dist_z_comb)
        combined_loss = train_loss + img_loss["loss"] + trip_loss

        combined_loss.backward()
        optimiser.step()
        
        if (i + 1) % log_interval == 0:
            print('Epoch [{}/{}], Train loss:{:.6f}'.format(
                epoch + 1, num_epoch, 
                combined_loss.item()
            ))

    save_path_nnet = "/mnt/home/thamamsy/projects/ST_Embed/models/joint_vqvae/jvqvae_triplet_nnet_" + str(epoch) + '.pth' 
    save_path_model = "/mnt/home/thamamsy/projects/ST_Embed/models/joint_vqvae/jvqvae_triplet_model_" + str(epoch) + '.pth'
    
    torch.save(
      nnet_model.state_dict(),
      save_path_nnet
    )

    torch.save(
      model.state_dict(),
      save_path_model
    )
    
#Train loop
num_epoch = 40
log_interval = 10


for epoch in range(num_epoch):
    train_joint_vae(nnet_model, model, train_loader, optimiser, epoch, num_epoch, learning_rate, log_interval)
    
    

