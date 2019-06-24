import torch
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector
import qucumber


class MaskedBinaryRBM(qucumber.rbm.BinaryRBM):
    # Works just like the usual BinaryRBM module, except that the given mask
    #  is applied to the weight-matrix every time the weight matrix is used.

    def __init__(
        self, num_visible, num_hidden, zero_weights=False, mask=None, gpu=True
    ):
        super(MaskedBinaryRBM, self).__init__(
            num_visible, num_hidden, zero_weights=zero_weights, gpu=gpu
        )

        self.mask = mask.to(self.weights) if mask is not None else 1

    def effective_energy(self, v):
        v = v.to(self.weights)
        if len(v.shape) < 2:
            v = v.unsqueeze(0)

        visible_bias_term = torch.mv(v, self.visible_bias)
        hid_bias_term = F.softplus(
            F.linear(v, self.weights * self.mask, self.hidden_bias)
        ).sum(1)

        return -(visible_bias_term + hid_bias_term)

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

        return parameters_to_vector([W_grad, vb_grad, hb_grad])

    def prob_v_given_h(self, h, out=None):
        if h.dim() < 2:  # create extra axis, if needed
            h = h.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        p = torch.addmm(
            self.visible_bias.data, h, (self.weights * self.mask).data, out=out
        ).sigmoid_()

        if unsqueezed:
            return p.squeeze_(0)  # remove superfluous axis, if it exists
        else:
            return p

    def prob_h_given_v(self, v, out=None):
        if v.dim() < 2:  # create extra axis, if needed
            v = v.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        p = torch.addmm(
            self.hidden_bias.data, v, (self.weights * self.mask).data.t(), out=out
        ).sigmoid_()

        if unsqueezed:
            return p.squeeze_(0)  # remove superfluous axis, if it exists
        else:
            return p
