import torch
import torch.nn as nn
import math

class L2Regularizer(nn.Module):

  def __init__(self, n_learner_params, batch_size):
    super(L2Regularizer, self).__init__()
    self.regularizer = torch.nn.Parameter(torch.ones(1, n_learner_params)*0.1)

    # for Adam
    self.beta1 = 0.9
    self.beta2 = 0.999
    self.step = 0
    self.register_buffer('exp_avg', torch.zeros(batch_size, n_learner_params))
    self.register_buffer('exp_avg_sq', torch.zeros(batch_size, n_learner_params))

  def reset(self):
    self.step = 0
    self.exp_avg.zero_()
    self.exp_avg_sq.zero_()

  def activation(self):
    return 0.01 * nn.functional.softplus(self.regularizer, beta=100)

  def forward(self, loss, grad, z, z_delta, lr):
    # gradient
    grad = grad + self.activation() * z_delta

    # bias
    step = self.step + 1
    bias_correction1 = 1 - self.beta1 ** step
    bias_correction2 = 1 - self.beta2 ** step

    # momentum terms
    exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * grad
    exp_avg_sq = self.beta2 * self.exp_avg_sq + (1 - self.beta2) * torch.mul(grad, grad).detach()

    # update parameters
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
    lr = lr / bias_correction1
    z_delta_new = z_delta - lr * exp_avg / denom

    # update state
    self.step = step
    self.exp_avg.copy_(exp_avg.data)
    self.exp_avg_sq.copy_(exp_avg_sq.data)

    return z_delta_new

  def load_state_dict(self, ckt):
    input_param = ckt['regularizer']
    if input_param.shape != self.regularizer.shape:
      raise Exception('size mismatch for regularizer: copying a param with shape {} from checkpoint, the shape in current model is {}.'.format(input_param.shape, self.regularizer.shape))
    self.regularizer.data.copy_(ckt['regularizer'])
    return
