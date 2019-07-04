import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from qucumber.rbm import BinaryRBM


class MaskedBinaryRBM(BinaryRBM):
    # Works just like the usual BinaryRBM module, except that the given mask
    #  is applied to the weight-matrix every time the weight matrix is used.

    def __init__(
        self, init_params, masks, gpu=True
    ):
        self.init_params = dict(init_params)
        self.masks = {k: 1 for k in init_params.keys()}

        super(MaskedBinaryRBM, self).__init__(
            self.init_params["weights"].shape[0],
            self.init_params["weights"].shape[1],
            zero_weights=True, gpu=gpu
        )  # no point randomizing weights if we're gonna overwrite them anyway


        # given masks use the convention of: 1 = pruned, 0 = kept
        #  in order to use these as multiplicative masks, we need to flip them
        self.masks = {k: (1 - v.to(self.weights)) for k, v in masks.items()}
        self.init_params = {k: v.to(self.weights) for k, v in self.init_params.items()}

        for name, param in self.named_parameters():
            param = nn.Parameter(self.masks[name] * self.init_params[name])


    def initialize_parameters(self, zero_weights=False):
        """Randomize the parameters of the RBM"""

        gen_tensor = torch.zeros if zero_weights else torch.randn
        self.weights = nn.Parameter(
            (
                gen_tensor(
                    self.num_hidden,
                    self.num_visible,
                    device=self.device,
                    dtype=torch.double,
                )
                * self.masks["weights"]
                / np.sqrt(self.num_visible)
            ),
            requires_grad=False,
        )

        self.visible_bias = nn.Parameter(
            torch.zeros(self.num_visible, device=self.device, dtype=torch.double),
            requires_grad=False,
        )
        self.hidden_bias = nn.Parameter(
            torch.zeros(self.num_hidden, device=self.device, dtype=torch.double),
            requires_grad=False,
        )

    @staticmethod
    def create_mask(matrix, p=0.5):
        vals, _ = matrix.flatten().abs().sort()
        cutoff = vals[int(np.ceil(p * len(vals)))]
        return (matrix.abs() < cutoff).to(dtype=matrix.dtype)

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

        return parameters_to_vector([W_grad * self.masks["weights"],
                                     vb_grad * self.masks["visible_bias"],
                                     hb_grad * self.masks["hidden_bias"]])
