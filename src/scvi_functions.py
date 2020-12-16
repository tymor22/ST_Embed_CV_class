from abc import ABC, abstractmethod
from contextlib import ContextDecorator
from typing import Iterable, List
import torch
import torch.nn as nn
from torch.nn import functional as F
import collections
from torch.distributions import Normal
from typing import Union, Tuple, Optional
import warnings
from torch.distributions import constraints, Distribution, Gamma, Poisson
from torch.distributions.utils import (
    broadcast_all,
    probs_to_logits,
    lazy_property,
    logits_to_probs,
)

class enable_grad(torch.enable_grad, ContextDecorator):
    pass

class no_grad(torch.no_grad, ContextDecorator):
    pass

class eval_modules:
    def __call__(self, function):
        def wrapper(*args, **kwargs):
            [a.eval() for a in args + tuple(kwargs.items()) if isinstance(a, nn.Module)]
            result = function(*args, **kwargs)
            [a.train() for a in args + tuple(kwargs.items()) if isinstance(a, nn.Module)]
            return result

        return wrapper

def to_cuda(tensors, non_blocking=True):
    return [t.cuda(non_blocking=True) for t in tensors]

''' Defines the abstract types and required signatures for models '''

class BaseModel(ABC):
    @abstractmethod
    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        '''
        :return: reconst_loss, kl_divergence
        '''
        pass

    @abstractmethod
    def get_latents(self, x, y=None):
        '''
        :param x: sample_batch
        :param y: labels
        :return: an array [z1, z2] the sampled variables from bottom to top
        '''
        pass


def one_hot(index, n_cat):
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.unsqueeze(1).type(torch.long), 1)
    #return onehot.type(torch.float32)
    return onehot.type(torch.double)

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


def identity(x):
    return x


class FCLayers(nn.Module):
    """
    A helper class to build fully-connected layers for a neural network.
    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_relu
        Whether to have `ReLU` layers or not
    bias
        Whether to learn bias in linear layers or not
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_relu: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in + sum(self.n_cat_list), n_out, bias=bias),
                            # Below, 0.01 and 0.001 are the default values for `momentum` and `eps` from
                            # the tensorflow implementation of batch norm; we're using those settings
                            # here too so that the results match our old tensorflow code. The default
                            # setting from pytorch would probably be fine too but we haven't tested that.
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.ReLU() if use_relu else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        """
        Forward computation on ``x``.
        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        instance_id
            Use a specific conditional instance normalization (batchnorm)
        x: torch.Tensor
        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        assert len(self.n_cat_list) <= len(
            cat_list
        ), "nb. categorical args provided doesn't match init. params."
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (
                n_cat and cat is None
            ), "cat not provided while n_cat != 0 in init. params."


            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                one_hot_cat = one_hot(cat, n_cat)
                '''
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                '''
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x



# Encoder
class Encoder(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.
    Uses a fully-connected neural network of ``n_hidden`` layers.
    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
        :dropout_rate: Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
    ):
        super().__init__()

        self.distribution = distribution
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)
        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample
        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample
        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = self.z_transformation(reparameterize_gaussian(q_m, q_v))
        return q_m, q_v, latent


# Decoder
class DecoderSCVI(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.
    Uses a fully-connected neural network of ``n_hidden`` layers.
    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        """
        The forward computation for a single sample.
         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``
        Parameters
        ----------
        dispersion
            One of the following
            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library
            library size
        cat_list
            list of category membership(s) for this sample
        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression
        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


def log_zinb_positive(
    x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor, eps=1e-8
):
    """
    Log likelihood (scalar) of a minibatch according to a zinb model.
    Parameters
    ----------
    x
        Data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    pi
        logit of the dropout parameter (real support) (shape: minibatch x vars)
    eps
        numerical stability constant
    Notes
    -----
    We parametrize the bernoulli using the logits, hence the softplus functions appearing.
    """
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return res


def log_nb_positive(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps=1e-8):
    """
    Log likelihood (scalar) of a minibatch according to a nb model.
    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    Notes
    -----
    We parametrize the bernoulli using the logits, hence the softplus functions appearing.
    """
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    return res


def log_mixture_nb(
    x: torch.Tensor,
    mu_1: torch.Tensor,
    mu_2: torch.Tensor,
    theta_1: torch.Tensor,
    theta_2: torch.Tensor,
    pi: torch.Tensor,
    eps=1e-8,
):
    """
    Log likelihood (scalar) of a minibatch according to a mixture nb model.
    pi is the probability to be in the first component.
    For totalVI, the first component should be background.
    Parameters
    ----------
    x
        Observed data
    mu_1
        Mean of the first negative binomial component (has to be positive support) (shape: minibatch x features)
    mu_2
        Mean of the second negative binomial (has to be positive support) (shape: minibatch x features)
    theta_1
        First inverse dispersion parameter (has to be positive support) (shape: minibatch x features)
    theta_2
        Second inverse dispersion parameter (has to be positive support) (shape: minibatch x features)
        If None, assume one shared inverse dispersion parameter.
    pi
        Probability of belonging to mixture component 1
    eps
        Numerical stability constant
    """
    if theta_2 is not None:
        log_nb_1 = log_nb_positive(x, mu_1, theta_1)
        log_nb_2 = log_nb_positive(x, mu_2, theta_2)
    # this is intended to reduce repeated computations
    else:
        theta = theta_1
        if theta.ndimension() == 1:
            theta = theta.view(
                1, theta.size(0)
            )  # In this case, we reshape theta for broadcasting

        log_theta_mu_1_eps = torch.log(theta + mu_1 + eps)
        log_theta_mu_2_eps = torch.log(theta + mu_2 + eps)
        lgamma_x_theta = torch.lgamma(x + theta)
        lgamma_theta = torch.lgamma(theta)
        lgamma_x_plus_1 = torch.lgamma(x + 1)

        log_nb_1 = (
            theta * (torch.log(theta + eps) - log_theta_mu_1_eps)
            + x * (torch.log(mu_1 + eps) - log_theta_mu_1_eps)
            + lgamma_x_theta
            - lgamma_theta
            - lgamma_x_plus_1
        )
        log_nb_2 = (
            theta * (torch.log(theta + eps) - log_theta_mu_2_eps)
            + x * (torch.log(mu_2 + eps) - log_theta_mu_2_eps)
            + lgamma_x_theta
            - lgamma_theta
            - lgamma_x_plus_1
        )

    logsumexp = torch.logsumexp(torch.stack((log_nb_1, log_nb_2 - pi)), dim=0)
    softplus_pi = F.softplus(-pi)

    log_mixture_nb = logsumexp - softplus_pi

    return log_mixture_nb


def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    r"""
    NB parameterizations conversion.
    Parameters
    ----------
    mu
        mean of the NB distribution.
    theta
        inverse overdispersion.
    eps
        constant used for numerical log stability. (Default value = 1e-6)
    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
    if not (mu is None) == (theta is None):
        raise ValueError(
            "If using the mu/theta NB parameterization, both parameters must be specified"
        )
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits


def _convert_counts_logits_to_mean_disp(total_count, logits):
    """
    NB parameterizations conversion.
    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    logits
        success logits.
    Returns
    -------
    type
        the mean and inverse overdispersion of the NB distribution.
    """
    theta = total_count
    mu = logits.exp() * theta
    return mu, theta


def _gamma(theta, mu):
    concentration = theta
    rate = theta / mu
    # Important remark: Gamma is parametrized by the rate = 1/scale!
    gamma_d = Gamma(concentration=concentration, rate=rate)
    return gamma_d


class NegativeBinomial(Distribution):
    """
    Negative binomial distribution.
    One of the following parameterizations must be provided:
    - (`total_count`, `probs`) where `total_count` is the number of failures until
    the experiment is stopped and `probs` the success probability.
    - The (`mu`, `theta`) parameterization is the one used by scvi-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.
    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    probs
        The success probability.
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        validate_args: bool = True,
    ):
        self._eps = 1e-8
        if (mu is None) == (total_count is None):
            raise ValueError(
                "Please use one of the two possible parameterizations. Refer to the documentation for more information."
            )

        using_param_1 = total_count is not None and (
            logits is not None or probs is not None
        )
        if using_param_1:
            logits = logits if logits is not None else probs_to_logits(probs)
            total_count = total_count.type_as(logits)
            total_count, logits = broadcast_all(total_count, logits)
            mu, theta = _convert_counts_logits_to_mean_disp(total_count, logits)
        else:
            mu, theta = broadcast_all(mu, theta)
        self.mu = mu
        self.theta = theta
        super().__init__(validate_args=validate_args)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.mean + (self.mean ** 2) / self.theta

    def sample(
        self, sample_shape: Union[torch.Size, Tuple] = torch.Size()
    ) -> torch.Tensor:
        with torch.no_grad():
            gamma_d = self._gamma()
            p_means = gamma_d.sample(sample_shape)

            # Clamping as distributions objects can have buggy behaviors when
            # their parameters are too high
            l_train = torch.clamp(p_means, max=1e8)
            counts = Poisson(
                l_train
            ).sample()  # Shape : (n_samples, n_cells_batch, n_vars)
            return counts

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                )
        return log_nb_positive(value, mu=self.mu, theta=self.theta, eps=self._eps)

    def _gamma(self):
        return _gamma(self.theta, self.mu)


class ZeroInflatedNegativeBinomial(NegativeBinomial):
    """
    Zero-inflated negative binomial distribution.
    One of the following parameterizations must be provided:
    - (`total_count`, `probs`) where `total_count` is the number of failures until
    the experiment is stopped and `probs` the success probability.
    - The (`mu`, `theta`) parameterization is the one used by scvi-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.
    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    probs
        The success probability.
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    zi_logits
        Logits scale of zero inflation probability.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "zi_probs": constraints.half_open_interval(0.0, 1.0),
        "zi_logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        zi_logits: Optional[torch.Tensor] = None,
        validate_args: bool = True,
    ):

        super().__init__(
            total_count=total_count,
            probs=probs,
            logits=logits,
            mu=mu,
            theta=theta,
            validate_args=validate_args,
        )
        self.zi_logits, self.mu, self.theta = broadcast_all(
            zi_logits, self.mu, self.theta
        )

    @property
    def mean(self):
        pi = self.zi_probs
        return (1 - pi) * self.mu

    @property
    def variance(self):
        raise NotImplementedError

    @lazy_property
    def zi_logits(self) -> torch.Tensor:
        return probs_to_logits(self.zi_probs, is_binary=True)

    @lazy_property
    def zi_probs(self) -> torch.Tensor:
        return logits_to_probs(self.zi_logits, is_binary=True)

    def sample(
        self, sample_shape: Union[torch.Size, Tuple] = torch.Size()
    ) -> torch.Tensor:
        with torch.no_grad():
            samp = super().sample(sample_shape=sample_shape)
            is_zero = torch.rand_like(samp) <= self.zi_probs
            samp[is_zero] = 0.0
            return samp

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        try:
            self._validate_sample(value)
        except ValueError:
            warnings.warn(
                "The value argument must be within the support of the distribution",
                UserWarning,
            )
        return log_zinb_positive(value, self.mu, self.theta, self.zi_logits, eps=1e-08)


class NegativeBinomialMixture(Distribution):
    """
    Negative binomial mixture distribution.
    Parameters
    ----------
    mu1
        Mean of the component 1 distribution.
    mu2
        Mean of the component 2 distribution.
    theta1
        Inverse dispersion for component 1.
    mixture_logits
        Logits scale probability of belonging to component 1.
    theta2
        Inverse dispersion for component 1. If `None`, assumed to be equal to `theta1`.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu1": constraints.greater_than_eq(0),
        "mu2": constraints.greater_than_eq(0),
        "theta1": constraints.greater_than_eq(0),
        "mixture_probs": constraints.half_open_interval(0.0, 1.0),
        "mixture_logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        mu1: torch.Tensor,
        mu2: torch.Tensor,
        theta1: torch.Tensor,
        mixture_logits: torch.Tensor,
        theta2: Optional[torch.Tensor] = None,
        validate_args: bool = True,
    ):

        (
            self.mu1,
            self.theta1,
            self.mu2,
            self.mixture_logits,
        ) = broadcast_all(mu1, theta1, mu2, mixture_logits)

        super().__init__(validate_args=validate_args)

        if theta2 is not None:
            self.theta2 = broadcast_all(mu1, theta2)
        else:
            self.theta2 = None

    @property
    def mean(self):
        pi = self.mixture_probs
        return pi * self.mu1 + (1 - pi) * self.mu2

    @lazy_property
    def mixture_probs(self) -> torch.Tensor:
        return logits_to_probs(self.mixture_logits, is_binary=True)

    def sample(
        self, sample_shape: Union[torch.Size, Tuple] = torch.Size()
    ) -> torch.Tensor:
        with torch.no_grad():
            pi = self.mixture_probs
            mixing_sample = torch.distributions.Bernoulli(pi).sample()
            mu = self.mu1 * mixing_sample + self.mu2 * (1 - mixing_sample)
            if self.theta2 is None:
                theta = self.theta1
            else:
                theta = self.theta1 * mixing_sample + self.theta2 * (1 - mixing_sample)
            gamma_d = _gamma(mu, theta)
            p_means = gamma_d.sample(sample_shape)

            # Clamping as distributions objects can have buggy behaviors when
            # their parameters are too high
            l_train = torch.clamp(p_means, max=1e8)
            counts = Poisson(
                l_train
            ).sample()  # Shape : (n_samples, n_cells_batch, n_features)
            return counts

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        try:
            self._validate_sample(value)
        except ValueError:
            warnings.warn(
                "The value argument must be within the support of the distribution",
                UserWarning,
            )
        return log_mixture_nb(
            value,
            self.mu1,
            self.mu2,
            self.theta1,
            self.theta2,
            self.mixture_logits,
            eps=1e-08,
        )



def compute_vae_elbo(vae, data_loader, **kwargs):
    """
    Computes the ELBO.
    The ELBO is the reconstruction error + the KL divergences
    between the variational distributions and the priors.
    It differs from the marginal log likelihood.
    Specifically, it is a lower bound on the marginal log likelihood
    plus a term that is constant with respect to the variational distribution.
    It still gives good insights on the modeling of the data, and is fast to compute.
    """
    # Iterate once over the data and compute the elbo
    elbo = 0
    for batch_idx, sample_batched in enumerate(data_loader):
        label_id = sample_batched['idx'].to(device)
        label = sample_batched['label'].to(device)
        data_dig = sample_batched['digit']
        mask = sample_batched['mask'].to(device)
        data, library, batch_index = data_dig

        data = data[:, -1, :].to(device)
        data = data.type(torch.float32)

        library = library.to(device)
        train_x = label.type(torch.DoubleTensor).to(device)

        # kl_divergence_global (scalar) should be common across all batches after training
        reconst_loss, kl_divergence = vae(
            data,
            library,
            batch_index=batch_index,
            **kwargs
        )
        elbo += torch.sum(reconst_loss + kl_divergence).item()
    n_samples = len(data_loader.dataset)
    #elbo += kl_divergence_global
    return elbo / n_samples


def compute_vae_reconstruction_error(vae, data_loader, **kwargs):
    """
    Computes log p(x/z), which is the reconstruction error.
    Differs from the marginal log likelihood, but still gives good
    insights on the modeling of the data, and is fast to compute.
    """
    # Iterate once over the data and computes the reconstruction error
    log_lkl = 0
    for batch_idx, sample_batched in enumerate(data_loader):
        label_id = sample_batched['idx'].to(device)
        label = sample_batched['label'].to(device)
        data_dig = sample_batched['digit']
        mask = sample_batched['mask'].to(device)
        data, library, batch_index = data_dig

        data = data[:, -1, :].to(device)
        data = data.type(torch.float32)

        library = library.to(device)
        train_x = label.type(torch.DoubleTensor).to(device)

        # Distribution parameters
        outputs = vae.inference(data, library, batch_index=batch_index, **kwargs)
        px_r = outputs["px_r"]
        px_rate = outputs["px_rate"]
        px_dropout = outputs["px_dropout"]
        bernoulli_params = outputs.get("bernoulli_params", None)

        # Reconstruction loss
        reconst_loss = vae.get_reconstruction_loss(
            data,
            px_rate,
            px_r,
            px_dropout,
            bernoulli_params=bernoulli_params,
            **kwargs
        )

        log_lkl += torch.sum(reconst_loss).item()
    n_samples = len(data_loader.dataset)
    return log_lkl / n_samples
