import math
import torch
from torch.optim.optimizer import Optimizer


class SSR1M(Optimizer):
    r"""Implements Stochastic SR1 with momentum method (SSR1M).

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, optional): coefficient used for computing running averages
            of gradient (momentum parameter) (default: 0.9)
        theta (float, optional): coefficient for exponential moving average of
            curvature information (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-5)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, beta=0.9, theta=0.9, eps=1e-5, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        if not 0.0 <= theta < 1.0:
            raise ValueError("Invalid theta parameter: {}".format(theta))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, beta=beta, theta=theta, eps=eps, weight_decay=weight_decay)
        super(SSR1M, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SSR1M, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SSR1M does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)  # m_t
                    state['curvature'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)  # b_t
                    state['prev_param'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)  # s_{t-1}
                    state['prev_grad'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)  # y_{t-1}

                momentum = state['momentum']
                curvature = state['curvature']
                prev_param = state['prev_param']
                prev_grad = state['prev_grad']

                beta = group['beta']
                theta = group['theta']
                eps = group['eps']

                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                momentum.mul_(beta).add_(1 - beta, grad)
                s_prev = p.data - prev_param
                y_prev = grad - prev_grad

                # diff = y_{t-1} - b_{t-1} * s_{t_1}
                diff = y_prev - curvature * s_prev

                # Compute norm squared of the difference
                diff_norm_sq = torch.norm(diff) ** 2

                # Compute the curvature update term
                curvature_update = (diff * diff) / (diff_norm_sq + eps)

                # b_t = θ * b_{t-1} + (1 - θ) * (diff^2) / (||diff||^2 + ε)
                curvature.mul_(theta).add_(1 - theta, curvature_update)

                # Store current values for next iteration
                prev_param.copy_(p.data)  # x_{t-1}
                prev_grad.copy_(grad)    # g_{t-1}

                # Parameter update: x_{t+1} = x_t - η_t * m_t / sqrt(b_t + ε)
                denom = torch.sqrt(curvature + eps)

                p.data.addcdiv_(-group['lr'], momentum, denom)

        return loss