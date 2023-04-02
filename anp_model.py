from typing import Optional, Any

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter, UninitializedBuffer

from torch.nn import functional as F
from torch.nn import init
# from ._functions import SyncBatchNorm as sync_batch_norm
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn import Module, Linear, Conv2d

class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""

    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            # MODIFIED: 
            # self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            # self.weight = Parameter(torch.zeros(num_features, **factory_kwargs))
            self.weight = Parameter(torch.ones(num_features, **factory_kwargs))
            # self.weight = Parameter(torch.full((num_features,), fill_value=1e-5, **factory_kwargs))
            
            # self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(torch.zeros(num_features, **factory_kwargs))
            # self.bias = Parameter(torch.full((num_features, ), fill_value=1e-5, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        # MODIFIED: 
        self.register_buffer('weight_fixed', torch.ones(num_features, requires_grad=False, **factory_kwargs))
        self.register_buffer('bias_fixed', torch.zeros(num_features, requires_grad=False, **factory_kwargs))
        self.register_buffer('running_mean_fixed', torch.zeros(num_features, requires_grad=False, **factory_kwargs))
        self.register_buffer('running_var_fixed', torch.ones(num_features, requires_grad=False, **factory_kwargs))
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_NormBase, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.is_perturb: bool = True
        
    def enable_perturb(self):
        self.is_perturb = True
    
    def disable_perturb(self):
        self.is_perturb = False
        
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # MODIFIED: 
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            # self.running_mean if not self.training or self.track_running_stats else None,
            self.running_mean_fixed,
            # self.running_var if not self.training or self.track_running_stats else None,
            self.running_var_fixed,
            self.weight if self.is_perturb else self.weight_fixed,
            # self.weight if self.is_perturb else self.weight_fixed,
            # self.weight_fixed,
            self.bias if self.is_perturb else self.bias_fixed,
            # self.bias_fixed,
            # bn_training,
            False,
            # exponential_average_factor,
            1.0,
            # self.eps,
            0.0,
        )
        # return input
        


class _LazyNormBase(LazyModuleMixin, _NormBase):

    weight: UninitializedParameter  # type: ignore[assignment]
    bias: UninitializedParameter  # type: ignore[assignment]

    def __init__(self, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_LazyNormBase, self).__init__(
            # affine and track_running_stats are hardcoded to False to
            # avoid creating tensors that will soon be overwritten.
            0,
            eps,
            momentum,
            False,
            False,
            **factory_kwargs,
        )
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = UninitializedParameter(**factory_kwargs)
            self.bias = UninitializedParameter(**factory_kwargs)
        if self.track_running_stats:
            self.running_mean = UninitializedBuffer(**factory_kwargs)
            self.running_var = UninitializedBuffer(**factory_kwargs)
            self.num_batches_tracked = torch.tensor(
                0, dtype=torch.long, **{k: v for k, v in factory_kwargs.items() if k != 'dtype'})

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.num_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            self.num_features = input.shape[1]
            if self.affine:
                assert isinstance(self.weight, UninitializedParameter)
                assert isinstance(self.bias, UninitializedParameter)
                self.weight.materialize((self.num_features,))
                self.bias.materialize((self.num_features,))
            if self.track_running_stats:
                self.running_mean.materialize((self.num_features,))  # type:ignore[union-attr]
                self.running_var.materialize((self.num_features,))  # type:ignore[union-attr]
            self.reset_parameters()

class PerturbBatchNorm1d(_BatchNorm):
    r"""Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )

class LazyBatchNorm1d(_LazyNormBase, _BatchNorm):
    r"""A :class:`torch.nn.BatchNorm1d` module with lazy initialization of
    the ``num_features`` argument of the :class:`BatchNorm1d` that is inferred
    from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """

    # cls_to_become = BatchNorm1d  # type: ignore[assignment]

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )   
            
class PerturbBatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))            

class LazyBatchNorm2d(_LazyNormBase, _BatchNorm):
    r"""A :class:`torch.nn.BatchNorm2d` module with lazy initialization of
    the ``num_features`` argument of the :class:`BatchNorm2d` that is inferred
    from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """

    # cls_to_become = BatchNorm2d  # type: ignore[assignment]

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim())) 
class PerturbLinear(Linear):
    def __init__(self, layer: Linear):
        self.is_bias = layer.bias != None
        super(PerturbLinear, self).__init__(in_features=layer.in_features, out_features=layer.out_features, bias=self.is_bias)
        
        with torch.no_grad():
            self.weight.copy_(layer.weight)
            if self.is_bias:
                self.bias.copy_(layer.bias)
        
        self.bn = PerturbBatchNorm1d(num_features=layer.out_features, eps=0.0, momentum=0.0, affine=True, track_running_stats=False, device=layer.weight.device, dtype=layer.weight.dtype)
    def forward(self, input: Tensor) -> Tensor:
        return self.bn(F.linear(input, self.weight, self.bias))
            
    #     self.rel_scale_weight = Parameter(torch.zeros((layer.out_features), device=layer.weight.device, dtype=layer.weight.dtype, requires_grad=True))
    #     if self.is_bias:
    #         self.rel_scale_bias = Parameter(torch.zeros((layer.out_features), device=layer.bias.device, dtype=layer.bias.dtype, requires_grad=True))
    # def forward(self, input: Tensor) -> Tensor:
    #     if self.is_bias != None:
    #         return (1 + self.rel_scale_weight) * F.linear(input, self.weight) + (1 + self.rel_scale_bias) * self.bias
    #     return (1 + self.rel_scale_weight) * F.linear(input, self.weight)
        
        # return (1 + self.rel_scale_weight) * F.linear(input, self.weight, self.bias) + self.rel_scale_bias
        # return F.linear(input, self.weight, self.bias)

class PerturbConv2d(Conv2d):
    def __init__(self, layer: Conv2d):
        self.is_bias = layer.bias != None
        super(PerturbConv2d, self).__init__(in_channels=layer.in_channels, 
                                            out_channels=layer.out_channels, 
                                            kernel_size=layer.kernel_size,
                                            stride=layer.stride,
                                            padding=layer.padding,
                                            dilation=layer.dilation,
                                            groups=layer.groups,
                                            padding_mode=layer.padding_mode,
                                            bias=self.is_bias)
        with torch.no_grad():
            self.weight.copy_(layer.weight)
            if self.is_bias:
                self.bias.copy_(layer.bias)
            
        # self.rel_scale_weight = Parameter(torch.empty((layer.out_features), device=layer.weight.device, dtype=layer.weight.dtype, requires_grad=True))
        # if self.is_bias:
        #     self.rel_scale_bias = Parameter(torch.empty((layer.out_features), device=layer.bias.device, dtype=layer.bias.dtype, requires_grad=True))
        
        self.bn = PerturbBatchNorm2d(num_features=layer.out_channels, eps=0.0, momentum=0.0, affine=True, track_running_stats=False, device=layer.weight.device, dtype=layer.weight.dtype)
            
    def forward(self, input: Tensor) -> Tensor:
        return self.bn(self._conv_forward(input, self.weight, self.bias))
        
def enable_perturb(model):
    for name, module in model.named_modules():
        if isinstance(module, PerturbBatchNorm2d) or isinstance(module, PerturbBatchNorm1d):
            module.enable_perturb()


def disable_perturb(model):
    for name, module in model.named_modules():
        if isinstance(module, PerturbBatchNorm2d) or isinstance(module, PerturbBatchNorm1d):
            module.disable_perturb()