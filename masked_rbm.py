import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from qucumber.rbm import BinaryRBM
from qucumber import _warn_on_missing_gpu


class MaskedBinaryRBM(BinaryRBM):
    # Works just like the usual BinaryRBM module, except that the given mask
    #  is applied to the weight-matrix every time the weight matrix is used.
    def __init__(
        self, init_params, masks, gpu=True
    ):
        self.init_params = dict(init_params)
        
        _warn_on_missing_gpu(gpu)
        self.gpu = gpu and torch.cuda.is_available()

        self.device = torch.device("cuda") if self.gpu else torch.device("cpu")
        
        # given masks will use the convention of: 1 = pruned, 0 = kept
        #  in order to use these as multiplicative masks, we need to flip them
        self.masks = {k: (1 - v.to(self.device)) for k, v in masks.items()}
        self.init_params = {k: v.to(self.device)
                            for k, v in self.init_params.items()}

        super(MaskedBinaryRBM, self).__init__(
            num_visible=self.init_params["weights"].shape[1],
            num_hidden=self.init_params["weights"].shape[0], gpu=gpu
        )


    def initialize_parameters(self, init_mode="reinit", **kwargs):
        if init_mode == "reinit":
            self.weights = nn.Parameter(
                self.init_params["weights"] * self.masks["weights"],
                requires_grad=False,
            )
            self.visible_bias = nn.Parameter(
                self.init_params["visible_bias"] * self.masks["visible_bias"],
                requires_grad=False,
            )
            self.hidden_bias = nn.Parameter(
                self.init_params["hidden_bias"] * self.masks["hidden_bias"],
                requires_grad=False,
            )
        elif init_mode == "constant":
            self.weights = nn.Parameter(
                torch.sign(self.init_params["weights"] * self.masks["weights"])
                * torch.std(self.init_params["weights"]),
                requires_grad=False,
            )
            self.visible_bias = nn.Parameter(
                torch.sign(self.init_params["visible_bias"] * self.masks["visible_bias"])
                * torch.std(self.init_params["visible_bias"]),
                requires_grad=False,
            )
            self.hidden_bias = nn.Parameter(
                torch.sign(self.init_params["hidden_bias"] * self.masks["hidden_bias"])
                * torch.std(self.init_params["hidden_bias"]),
                requires_grad=False,
            )
        else:
            raise ValueError("Invalid input '" + init_mode 
                             + "' for 'init_mode'! Must be one of ['reinit', 'constant']")

    @staticmethod
    def create_mask(matrix, p=None, cutoff=None):
        if p is not None:
            vals, _ = matrix.flatten().abs().sort()
            cutoff = vals[int(np.ceil(p * len(vals)))]
            return (matrix.abs() < cutoff).to(dtype=matrix.dtype)
        elif cutoff is not None:
            return (matrix.abs() < cutoff).to(dtype=matrix.dtype)
        else:
            raise ValueError("One of (p, cutoff) must be given!")

    def effective_energy_gradient(self, v):
        v = v.to(self.weights)
        prob = self.prob_h_given_v(v)

        if v.dim() < 2:
            W_grad = -torch.einsum("j,k->jk", (prob, v))
            vb_grad = -v
            hb_grad = -prob
        else:
            W_grad = -torch.matmul(prob.t(), v)
            vb_grad = -torch.sum(v, 0)
            hb_grad = -torch.sum(prob, 0)

        return parameters_to_vector([W_grad * self.masks["weights"].abs(),
                                     vb_grad * self.masks["visible_bias"].abs(),
                                     hb_grad * self.masks["hidden_bias"].abs()])
