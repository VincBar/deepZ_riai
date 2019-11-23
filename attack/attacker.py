import matplotlib.pyplot as plt

import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch.attacks import LinfPGDAttack, PGDAttack

import sys

sys.path.append("..")
sys.path.append("../code")

import code.verifier
import code.networks
from collections import OrderedDict

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
device="cpu"
from joblib import Parallel, delayed


def load_data(batch_size):
    loader = get_mnist_test_loader(batch_size=batch_size)
    for cln_data, true_label in loader:
        break
    return cln_data.to(device), true_label.to(device)


def load_nets(eps=0, target=0, zonotope=False):
    """
    Load nets

    :param eps: can be number or iterable of size code.verifier.NET_CHOICES
    :param target: int
    :param zonotope: if True return the zonotope nets.
    :return:
    """
    nets = OrderedDict([])
    if type(eps) is int:
        eps = [float(eps)] * len(code.verifier.NET_CHOICES)

    neteps = zip(code.verifier.NET_CHOICES, eps)
    for net_name, eps in neteps:
        net, netZ = code.verifier.load_net(net_name, eps=eps, target=target)
        if zonotope:
            net = netZ
        nets[net_name] = net

    return nets


def create_masks(cln_data, labels, nets):
    nr_nets = len(list(nets.keys()))
    masks = torch.ones((cln_data.shape[0], nr_nets), dtype=torch.bool)
    for i, net in enumerate(nets.values()):
        masks[:, i] = labels == predict_from_logits(net(cln_data))

    return masks


def init_adversary(net, eps):
    return LinfPGDAttack(net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps, nb_iter=40, eps_iter=0.01,
                         rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)


def check_robustness(net, adversary, data, labels):
    """
    Run an adversial attack on the net for all digits in data.

    :param net:
    :param adversary:
    :param data:
    :param labels:
    :return: torch.tensor(dtype=bool) whether adversary attack was successful, i.e. whether an edv. example was found.
    """
    perturbed_data = adversary.perturb(data, labels)
    pred_labels_pert = predict_from_logits(net(perturbed_data))
    return pred_labels_pert == labels


def find_adversarial_examples(data, labels, eps, n_jobs=4, gpu=False):
    """
    Tries to find adversial examples to the digits in data.
    :param eps:
    :param labels:
    :param data:
    :param n_jobs:
    :return: torch.tensor T of shape(nr_nets, nr_digits = data.shape[0]) where entry T_ij is the smallest eps for which
             an adversarial example was found. If none was found T_ij = np.inf.
    """
    nets = load_nets()

    eps_lower_bound = torch.ones((len(nets), data.shape[0])) * np.inf

    job_params = [(net, digit, label) for net in nets.values() for digit, label in zip(data, labels)]
    job_params_inds = [(net_ind, digit_ind) for net_ind in range(len(nets.values())) for digit_ind in
                       range(data.shape[0])]

    for e in eps:
        # get jobs with epsilon == e
        e_params = [(net, init_adversary(net, e), digit, torch.tensor([label])) for net, digit, label in job_params]

        if not gpu:
            jobs = [delayed(check_robustness)(*param) for param in e_params]
            out = Parallel(n_jobs=n_jobs)(jobs)
        else:
            out = []
            for param in e_params:
                out.append(check_robustness(*param))

        # if there is an adversarial example for a (net, digit) combination, take it out and save eps
        adv_job_ind = np.where(np.logical_not(np.array(out)))[0]
        for ind in adv_job_ind[::-1]:
            eps_lower_bound[job_params_inds[ind]] = e

            del job_params[ind]
            del job_params_inds[ind]

    return eps_lower_bound


def verify(data, labels, eps, n_jobs=4, *args, **kwargs):
    """
    Try to verify all digits in data with eps L_inf norm. Any non-finite element of eps will be disregarded and set to
    non_verified.

    :param data:
    :param labels:
    :param eps: torch.tensor with shape (nr_nets, nr_digits)
    :param n_jobs:
    :param args:
    :param kwargs:
    :return: boolean torch.tensor of same shape as eps, whether verifier could verify, for non-finite eps elements, the
             the result will always be False (since no verification is run)
    """
    is_verified = torch.zeros(eps.shape).bool()

    for i, dta in enumerate(zip(data, labels)):
        digit, label = dta
        netZs = load_nets(eps=eps[:, i], target=label, zonotope=True)

        non_robust_net = np.where(eps[:, i] < np.inf)[0]
        netZs = [netZ for i, netZ in enumerate(netZs.values()) if i in non_robust_net]

        jobs = [delayed(code.verifier.analyze)(netZ, digit[None, :], label, *args, **kwargs) for netZ in netZs]
        is_verified[non_robust_net, i] = torch.tensor(Parallel(n_jobs=n_jobs)(jobs), dtype=torch.bool)

    return is_verified


def check_adv_first(data, labels, nr_eps, n_jobs=4, pairwise=True, maxsec=120, *args, **kwargs):
    """
    Find adversarial examples for digits, then try to verify these digits at eps == lower_bound.

    :return:
    """
    eps = np.linspace(0.005, 0.2, nr_eps)
    eps_lower_bound = find_adversarial_examples(data, labels, eps)
    is_verified = verify(data, labels, eps_lower_bound, n_jobs=n_jobs, pairwise=pairwise, maxsec=maxsec,
                         *args, **kwargs)

    # check that none of the instances with an adversarial example is verified
    assert torch.all(torch.logical_not(is_verified))


def check_verify_first(data, labels, eps, nr_eps, n_jobs=4, pairwise=True, maxsec=120, *args, **kwargs):
    """
    Run verification on all digits in data for the
    :param data:
    :param labels:
    :param eps:
    :param n_jobs:
    :param pairwise:
    :param maxsec:
    :param args:
    :param kwargs:
    :return:
    """
    eps_verif = np.ones((len(code.verifier.NET_CHOICES), data.shape[0])) * eps
    is_verified = verify(data, labels, eps_verif, n_jobs=n_jobs, pairwise=pairwise, maxsec=maxsec,
                         *args, **kwargs)

    eps = np.linspace(0.05, eps, nr_eps)
    eps_lower_bound = find_adversarial_examples(data, labels, eps, n_jobs=n_jobs)

    # check that for none of the verified instances an adversarial example can be found
    assert torch.all(eps_lower_bound[is_verified] > torch.Tensor(eps_verif[is_verified]))


if __name__ == '__main__':
    cln_data, true_labels = load_data(1)
    #check_adv_first(cln_data, true_labels, nr_eps=10, n_jobs=1)
    check_verify_first(cln_data, true_labels, 0.15, 10, n_jobs=1)




