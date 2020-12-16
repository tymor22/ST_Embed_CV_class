
from scvi_functions import *
import os, sys, torch, argparse
import pandas as pd
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import math

from abc import ABC, abstractmethod
from contextlib import ContextDecorator
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kldiv
from torch.distributions import Poisson

# Count VAE model
#https://github.com/YosefLab/scvi-tools/blob/master/scvi/core/modules/vae.py
from typing import Dict, Tuple
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl


class CountVAE(nn.Module):
    """
    Variational auto-encoder model.
    This is an implementation of the scVI model descibed in [Lopez18]_
    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_pretrained_features
        Features to extract from the pretrained model applieid to the Histology
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 20,
        n_layers: int = 1,
        n_pretrained_features: int = 20,
        n_latent_concat: int = 40,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "zinb",
        latent_distribution: str = "normal",
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.n_pretrained_features = n_pretrained_features
        self.n_latent_concat = n_latent_concat

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
        )

        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderSCVI(
            n_latent_concat,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

    def get_latents(self, x, y=None) -> torch.Tensor:
        """
        Returns the result of ``sample_from_posterior_z`` inside a list.
        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)
        Returns
        -------
        type
            one element list of tensor
        """
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(
        self, x, y=None, give_mean=False, n_samples=5000
    ) -> torch.Tensor:
        """
        Samples the tensor of latent values from the posterior.
        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)
        give_mean
            is True when we want the mean of the posterior  distribution rather than sampling (Default value = False)
        n_samples
            how many MC samples to average over for transformed mean (Default value = 5000)
        Returns
        -------
        type
            tensor of shape ``(batch_size, n_latent)``
        """
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC

        if give_mean:
            if self.latent_distribution == "ln":
                samples = Normal(qz_m, qz_v.sqrt()).sample([n_samples])
                z = self.z_encoder.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m
        return [qz_m, qz_v, z]


    def get_sample_scale(
        self, x, batch_index=None, y=None, n_samples=1, transform_batch=None
    ) -> torch.Tensor:
        """
        Returns the tensor of predicted frequencies of expression.
        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size`` (Default value = None)
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)
        n_samples
            number of samples (Default value = 1)
        transform_batch
            int of batch to transform samples into (Default value = None)
        Returns
        -------
        type
            tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        """
        return self.inference(
            x,
            batch_index=batch_index,
            y=y,
            n_samples=n_samples,
            transform_batch=transform_batch,
        )["px_scale"]

    def get_sample_rate(
        self, x, batch_index=None, y=None, n_samples=1, transform_batch=None
    ) -> torch.Tensor:
        """
        Returns the tensor of means of the negative binomial distribution.
        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)`` (Default value = None)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size`` (Default value = None)
        n_samples
            number of samples (Default value = 1)
        transform_batch
            int of batch to transform samples into (Default value = None)
        Returns
        -------
        type
            tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        """
        return self.inference(
            x,
            batch_index=batch_index,
            y=y,
            n_samples=n_samples,
            transform_batch=transform_batch,
        )["px_rate"]

    def get_reconstruction_loss(
        self, x, px_rate, px_r, px_dropout, **kwargs
    ) -> torch.Tensor:
        # Reconstruction Loss
        if self.gene_likelihood == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss

    def inference(
        self, x, library, concat_z, batch_index=None, y=None, n_samples=1, transform_batch=None
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass."""
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, y)
        
        z_concat = torch.cat((z, concat_z), 1) 
        
        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

        if transform_batch is not None:
            dec_batch_index = transform_batch * torch.ones_like(batch_index)
        else:
            dec_batch_index = batch_index

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, z_concat, library, dec_batch_index, y
        )

        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(dec_batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            library=library,
            z_concat=z_concat
        )

    def forward(
        self, x, library, concat_z, batch_index=None, y=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the reconstruction loss and the KL divergences.
        Parameters
        ----------
        x
            tensor of values with shape (batch_size, n_input)
        local_l_mean
            tensor of means of the prior distribution of latent variable l
            with shape (batch_size, 1)
        local_l_var
            tensor of variancess of the prior distribution of latent variable l
            with shape (batch_size, 1)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size`` (Default value = None)
        y
            tensor of cell-types labels with shape (batch_size, n_labels) (Default value = None)
        Returns
        -------
        type
            the reconstruction loss and the Kullback divergences
        """
        # Parameters for z latent distribution
        outputs = self.inference(x, library, concat_z, batch_index, y)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        px_dropout = outputs["px_dropout"]

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        kl_divergence = kl_divergence_z

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        return reconst_loss, kl_divergence
