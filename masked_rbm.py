import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
import qucumber


class MaskedBinaryRBM(qucumber.rbm.BinaryRBM):
    # Works just like the usual BinaryRBM module, except that the given mask
    #  is applied to the weight-matrix every time the weight matrix is used.

    def __init__(
        self, init_weights, mask, gpu=True
    ):
        self.mask = mask.to(self.weights) if mask is not None else 1
        self.init_weights = init_weights.to(self.weights)

        super(MaskedBinaryRBM, self).__init__(
            init_weights.shape[0], init_weights.shape[1],
            zero_weights=True, gpu=gpu
        )  # no point randomizing weights if we're gonna overwrite them anyway

        self.weights = nn.Parameter(self.mask * self.init_weights)

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
                * self.mask
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
        return (matrix.abs() >= cutoff).to(dtype=matrix.dtype)

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

        return parameters_to_vector([W_grad * self.mask, vb_grad, hb_grad])
