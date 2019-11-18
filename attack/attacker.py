import matplotlib.pyplot as plt

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch.attacks import LinfPGDAttack, PGDAttack

import sys

sys.path.append("..")
sys.path.append("../code")

import code.verifier
from collections import OrderedDict

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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
        eps = [eps] * len(code.verifier.NET_CHOICES)

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


def find_adversial_examples(data, labels, eps, n_jobs=4):
    """
    Tries to find adversial examples to the digits in data.
    :param eps:
    :param labels:
    :param data:
    :param n_jobs:
    :return: torch.tensor T of shape(nr_nets, nr_digits = data.shape[0]) where entry T_ij is the lowest eps for which
             an adversial example was found. If none was found T_ij = np.inf.
    """
    nets = load_nets()

    is_robust = torch.ones((len(nets), data.shape[0])) * np.inf

    job_params = [(net, digit, label) for net in nets.values() for digit, label in zip(data, labels)]
    job_params_inds = [(net_ind, digit_ind) for net_ind in range(len(nets.values())) for digit_ind in
                       range(data.shape[0])]

    for e in eps:
        # get jobs with epsilon == e
        e_params = [(net, init_adversary(net, e), digit, torch.tensor([label])) for net, digit, label in job_params]

        jobs = [delayed(check_robustness)(*param) for param in e_params]
        out = Parallel(n_jobs=n_jobs)(jobs)

        # if there is an adversarial example for a (net, digit) combination, take it out and save eps
        adv_job_ind = np.where(np.logical_not(np.array(out)))[0]
        for ind in adv_job_ind[::-1]:
            is_robust[job_params_inds[ind]] = e

            del job_params[ind]
            del job_params_inds[ind]

    return is_robust


def check_soundness(data, labels, is_robust, n_jobs=4, *args, **kwargs):
    """
    Checks the soundness of all zonotope nets. This is done by computing the zonotope relaxation of all (net, digit)
    pairs with the eps in is_robust, i.e. the triple (net, digit, T_ij) should return not_verified. (net, digit) for
    which no adversarial example was found (T_ij = np.inf in this case) are disregarded.

    :param is_robust:
    :return:
    """

    for i, dta in enumerate(zip(data, labels)):
        digit, label = dta
        netZs = load_nets(eps=is_robust[:, i], target=label, zonotope=True)

        should_not_verify_inds = np.where(is_robust[:, i] < np.inf)[0]
        netZs = [netZ for i, netZ in enumerate(netZs.values()) if i in should_not_verify_inds]

        jobs = [delayed(code.verifier.analyze)(netZ, digit[None, :], label, *args, **kwargs) for netZ in netZs]
        out = torch.tensor(Parallel(n_jobs=n_jobs)(jobs), dtype=torch.bool)

        assert torch.all(torch.logical_not(out))


if __name__ == '__main__':
    cln_data, true_labels = load_data(1)
    eps = np.linspace(0.005, 0.2, 10)

    is_robust = find_adversial_examples(cln_data, true_labels, eps)
    check_soundness(cln_data, true_labels, is_robust, n_jobs=1, pairwise=True, tensorboard=False, maxsec=120)
