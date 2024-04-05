from math import pi, exp
import torch
from numpy import arctan
from typing import Optional
from torch import Tensor
from abc import abstractmethod, ABC


class Scheduler(ABC):
    def __int__(self):
        self.alphas = None
        self.betas = None
        self.alphas_hat = None
        self.betas_hat = None


class LinearScheduler(Scheduler):
    def __init__(self, T: int, beta_min: float, beta_max: float):
        """
        :param T:
        :param beta_min:
        :param beta_max:
        """
        self.betas = torch.linspace(beta_min, beta_max, T)
        self.alphas = 1.0 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)
        alpha_hat_t_minus_1 = torch.roll(self.alphas_hat, shifts=1, dims=0)
        alpha_hat_t_minus_1[0] = alpha_hat_t_minus_1[1]
        self.betas_hat = (1 - alpha_hat_t_minus_1) / (1 - self.alphas_hat) * self.betas


class CosineScheduler(Scheduler):
    def __init__(self, T: int, s: float = 0.008):
        """
        Cosine variance scheduler.
        The equation for the variance is:
            alpha_hat = min(cos((t / T + s) / (1 + s) * pi / 2)^2, 0.999)
            beta = 1 - (alpha_hat(t) / alpha_hat(t - 1))
            beta_hat = (1 - alpha_hat(t - 1)) / (1 - alpha_hat(t)) * beta(t)
        """
        self.alphas_hat = torch.pow(torch.cos((torch.arange(T) / T + s) / (1 + s) * pi / 2.0), 2)
        self.alphas_hat_t_minus_1 = torch.roll(self.alphas_hat, shifts=1, dims=0)
        self.alphas_hat_t_minus_1[0] = self.alphas_hat_t_minus_1[1]  # to remove first NaN value
        self.betas = 1.0 - self.alphas_hat / self.alphas_hat_t_minus_1
        self.betas = torch.minimum(self.betas, Tensor([0.999]))
        self.alphas = 1.0 - self.betas
        self.betas_hat = (1 - self.alphas_hat_t_minus_1) / (1 - self.alphas_hat) * self.betas
        self.betas_hat[torch.isnan(self.betas_hat)] = 0.0


def xt_from_x0(alphas_hat, x0, t, eta):
    alpha_hat_t = alphas_hat[t]
    xt = (torch.sqrt(alpha_hat_t).reshape(-1, 1, 1, 1) * x0 +
          torch.sqrt(1 - alpha_hat_t).reshape(-1, 1, 1, 1) * eta)
    return xt


#
#
# class HyperbolicSecant():
#     def __init__(self, T: int, lambda_min: float, lambda_max: float):
#         # pg 3 section 2 for the details about the following eqns
#         self.b = arctan(exp(-lambda_max / 2))
#         self.a = arctan(exp(-lambda_min / 2)) - self.b
#         self._betas = - 2 * torch.log(torch.tan(self.a * torch.linspace(0, 1, T, dtype=torch.float) + self.b))
#         self._alphas = 1.0 - self._betas
#         self._alphas_hat = torch.cumprod(self._alphas, dim=0)
#         self._alphas_hat_t_minus_1 = torch.roll(self._alphas_hat, shifts=1, dims=0)
#         self._alphas_hat_t_minus_1[0] = self._alphas_hat_t_minus_1[1]
#         self._betas_hat = (1 - self._alphas_hat_t_minus_1) / (1 - self._alphas_hat) * self._betas
#
#
#
#
# def sigma_x_t(v: Tensor,
#               t: Tensor,
#               betas_hat: Tensor,
#               betas: Tensor,
#               eps: float = 1e-5) -> Tensor:
#     """
#     Compute the variance at time step t as defined
#     from "Improving Denoising Diffusion probabilistic Models", eqn 15 page 4
#     :param eps:
#     :param v: the neural network "logits" used to compute the variance [BS, C, W, H]
#     :param t: the target time step
#     :param betas_hat: sequence of $\hat{\beta}$ used for variance scheduling
#     :param betas: sequence of $\beta$ used for variance scheduling
#     :return: the estimated variance at time step t
#     """
#     x = torch.exp(v * torch.log(betas[t].reshape(-1, 1, 1, 1) + eps) + (1 - v) * torch.log(
#         betas_hat[t].reshape(-1, 1, 1, 1) + eps))
#     return x
#
#
# def mu_hat_xt_x0(x_t: Tensor,
#                  x_0: Tensor,
#                  t: Tensor,
#                  alphas_hat: Tensor,
#                  alphas: Tensor,
#                  betas: Tensor,
#                  eps: float = 1e-5) -> Tensor:
#     """
#     Compute $\hat{mu}(x_t, x_0)$ of $q(x_{t-1} | x_t, x_0)$
#     from "Improving Denoising Diffusion probabilistic Models", eqn 11 page 2
#     :param eps:
#     :param x_t: The noised image at step t
#     :param x_0: the original image
#     :param t: the time step of $x_t$ [batch_size]
#     :param alphas_hat: sequence of $\hat{\alpha}$ used for variance scheduling [T]
#     :param alphas: sequence of $\alpha$ used for variance scheduling [T]
#     :param betas: sequence of $\beta$ used for variance scheduling [T}
#     :return: the mean of distribution $q(x_{t-1} | x_t, x_0)$
#     """
#     alpha_hat_t = alphas_hat[t].reshape(-1, 1, 1, 1)
#     one_min_alpha_hat = (1 - alpha_hat_t) + eps
#     alpha_t = alphas[t].reshape(-1, 1, 1, 1)
#     alpha_hat_t_1 = alphas_hat[t - 1].reshape(-1, 1, 1, 1)
#     beta_t = betas[t].reshape(-1, 1, 1, 1)
#     x = torch.sqrt(alpha_hat_t_1 + eps) * beta_t / one_min_alpha_hat * x_0 + \
#         torch.sqrt(alpha_t + eps) * (
#                 1 - alpha_hat_t_1) / one_min_alpha_hat * x_t
#     return x
#
#
# def sigma_hat_xt_x0(t: Tensor,
#                     betas_hat: Tensor,
#                     eps: float = 1e-5) -> Tensor:
#     """
#     Compute the variance of of $q(x_{t-1} | x_t, x_0)$
#     from "Improving Denoising Diffusion probabilistic Models", eqn 12 page 2
#     :param eps:
#     :param t: the time step [batch_size]
#     :param betas_hat: the array of beta hats [T]
#     :return: the variance at time step t as scalar [batch_size, 1, 1, 1]
#     """
#     return betas_hat[t].reshape(-1, 1, 1, 1) + eps

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sch1 = CosineScheduler(1000)
    sch2 = LinearScheduler(1000, 0.0001, 0.02)

    plt.plot(sch1.alphas_hat.numpy(), label='Cosine alphas_hat')
    plt.plot(sch2.alphas_hat.numpy(), label='Linear alphas_hat')
    plt.plot(sch1.betas_hat.numpy(), label='Cosine betas_hat')
    plt.plot(sch2.betas_hat.numpy(), label='Linear betas_hat')

    plt.xlabel('t')
    plt.legend()
    plt.show()
