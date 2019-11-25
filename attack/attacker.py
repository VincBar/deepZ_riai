import sys
sys.path.append("..")
sys.path.append("../code")
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch.attacks import LinfPGDAttack, PGDAttack

import code.verifier
import code.networks
from collections import OrderedDict

from functools import partial
from joblib import Parallel, delayed

import logging

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")


def run_jobs(jobs, joblib=True, n_jobs=4):
    if joblib:
        jobs = [delayed(job)() for job in jobs]
        out = Parallel(n_jobs=n_jobs)(jobs)
    else:
        out = []
        for job in jobs:
            out.append(job())
    return out


def load_data(batch_size):
    loader = get_mnist_test_loader(batch_size=batch_size)
    for cln_data, true_label in loader:
        break
    return cln_data.to(DEVICE), true_label.to(DEVICE)


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
    Run an adversarial attack on the net for all digits in data.

    :param net:
    :param adversary:
    :param data:
    :param labels:
    :return: torch.tensor(dtype=bool) whether adversary attack was successful, i.e. whether an edv. example was found.
    """
    perturbed_data = adversary.perturb(data, labels)
    pred_labels_pert = predict_from_logits(net(perturbed_data))
    return pred_labels_pert == labels


def find_adversarial_examples(data, labels, eps, n_jobs=4, mask=None):
    """
    Tries to find adversarial examples to the digits in data.
    :param eps:
    :param labels:
    :param data:
    :param n_jobs:
    :return: torch.tensor T of shape(nr_nets, nr_digits = data.shape[0]) where entry T_ij is the smallest eps for which
             an adversarial example was found. If none was found T_ij = np.inf.
    """
    nets = load_nets()

    eps_lower_bound = torch.ones((len(nets), data.shape[0])) * np.inf
    eps_lower_bound_inds = torch.ones((len(nets), data.shape[0])) * np.inf

    job_params = [(net, digit, label) for net in nets.values() for digit, label in zip(data, labels)]
    job_params_inds = [(net_ind, digit_ind) for net_ind in range(len(nets.values())) for digit_ind in
                       range(data.shape[0])]

    if not hasattr(eps, '__iter__'):
        eps = [eps]

    for k, e in enumerate(eps):
        # get jobs with epsilon == e
        e_params = [(net, init_adversary(net, e), digit, torch.tensor([label])) for net, digit, label in job_params]

        jobs = [partial(check_robustness, *param) for param in e_params]
        out = run_jobs(jobs, n_jobs=n_jobs, joblib=(n_jobs != 1))

        # if there is an adversarial example for a (net, digit) combination, take it out and save eps
        adv_job_ind = np.where(np.logical_not(np.array(out)))[0]
        for ind in adv_job_ind[::-1]:
            eps_lower_bound[job_params_inds[ind]] = e
            eps_lower_bound_inds[job_params_inds[ind]] = k

            del job_params[ind]
            del job_params_inds[ind]

    return eps_lower_bound, eps_lower_bound_inds


def verify(data, labels, eps, n_jobs=4, maxsec=120, tensorboard=False, pairwise=False):
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
    run_times = torch.zeros(eps.shape)

    for i, dta in enumerate(zip(data, labels)):
        digit, label = dta
        netZs = load_nets(eps=eps[:, i], target=label, zonotope=True)

        non_robust_net = np.where(eps[:, i] < np.inf)[0]
        netZs = [netZ for i, netZ in enumerate(netZs.values()) if i in non_robust_net]

        jobs = [partial(code.verifier.analyze, net=netZ, inputs=digit[None, :], true_label=label,
                        pairwise=pairwise, tensorboard=tensorboard, maxsec=maxsec, time_info=True)
                for netZ in netZs]

        out = run_jobs(jobs, joblib=(n_jobs != 1), n_jobs=n_jobs)

        is_verif, run_time = zip(*out)
        is_verified[non_robust_net, i] = torch.tensor(is_verif, dtype=torch.bool)
        run_times[non_robust_net, i] = torch.tensor(run_time)

    return is_verified, run_times


def check_adv_first(data, labels, nr_eps, n_jobs=4, pairwise=True, maxsec=120, check_smaller=True,
                    *args, **kwargs):
    """
    Search adversarial examples for digits by attacking digits in an increasing L_inf ball until an example for an eps
    is found. Then try to verify these digits at this eps. If the verifier is sound, this must always result in
    non_verified. If check_smaller = True, try to verify the attacked digits at a marginally smaller eps. Failing to
    verify these digits hints to a bad attacker or an imprecise verifier or both.

    Don't use joblib with tensorboard !! Set n_jobs=1 to deactivate joblib.

    :return:
    """
    eps = np.linspace(0.05, 0.2, nr_eps)
    eps_upper_bound, inds = find_adversarial_examples(data, labels, eps)
    is_robust = (eps_upper_bound == np.inf).type(torch.bool)

    # Try to verify non robust examples (with eps != np.inf). This must fail in order for the verifier to be sound.
    is_verified, run_times = verify(data, labels, eps_upper_bound + 0.1, n_jobs=n_jobs, pairwise=pairwise,
                                    maxsec=maxsec, *args, **kwargs)

    # check that none of the instances with an adversarial example is verified
    if not torch.all(torch.logical_not(is_verified)):
        logging.warning('Found adversarial example which was verified by verifier. Verifier is not sound!')

    if check_smaller:
        # create eps tensor with eps smaller than in upper bound. Verification should be likely to succeed in this case.
        # Non verification is due to bad performance of the attacker or an imprecise verification.
        eps_smaller = torch.ones(eps_upper_bound.shape, dtype=torch.float) * np.inf

        # Don't run second verification on instances where the instance is not robust already for the first epsilon.
        # for robust instances, verify with smallest epsilon
        do_check_smaller = (0 != inds).type(torch.bool)
        check_smaller_non_robust = do_check_smaller & torch.logical_not(is_robust)
        eps_smaller[is_robust] = eps[0]
        eps_smaller[check_smaller_non_robust] = torch.tensor(eps[inds[check_smaller_non_robust].int() - 1],
                                                             dtype=torch.float)

        # Try to verify with smaller eps
        is_verified_smaller, run_times_smaller = verify(data, labels, eps_smaller, n_jobs=n_jobs, pairwise=pairwise,
                                                        maxsec=maxsec, *args, **kwargs)

    # write stats
    nets, digits = np.meshgrid(range(eps_upper_bound.shape[0]), range(eps_upper_bound.shape[1]), indexing='ij')
    ret = pd.DataFrame({'verified': is_verified.numpy().flatten(),
                        'robust_upper_bound': eps_upper_bound.numpy().flatten(),
                        'run_time': run_times.numpy().flatten(),
                        'net': nets.flatten(),
                        'digit': digits.flatten(),
                        })

    if check_smaller:
        ret = pd.concat((ret,
                         pd.DataFrame({
                             'verified_smaller': is_verified_smaller.numpy().flatten(),
                             'eps_smaller': eps_smaller.numpy().flatten(),
                             'run_time_smaller': run_times_smaller.numpy().flatten(),
                             'check_smaller': do_check_smaller.numpy().flatten()
                         })), axis=1)

    return ret


def check_verify_first(data, labels, eps, n_jobs=4, pairwise=True, maxsec=120, *args, **kwargs):
    """
    Run verification on all digits in data with Linf = eps. Then

    Don't use joblib with tensorboard !! Set n_jobs=1 to deactivate joblib.

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
    is_verified, run_times = verify(data, labels, eps_verif, n_jobs=n_jobs, pairwise=pairwise, maxsec=maxsec,
                                    *args, **kwargs)

    eps_upper_bound = find_adversarial_examples(data, labels, eps, mask=np.logical_not(is_verified), n_jobs=n_jobs)

    # check that for none of the verified instances an adversarial example can be found
    if not torch.all(eps_upper_bound[is_verified] > torch.Tensor(eps_verif[is_verified])):
        logging.warning('Found adversarial example which was verified by verifier. Verifier is not sound!')

    # write stats
    nets, digits = np.meshgrid(range(eps_upper_bound.shape[0]), range(eps_upper_bound.shape[1]), indexing='ij')
    ret = pd.DataFrame({'verified': is_verified.numpy().flatten(),
                        'robust_upper_bound': eps_upper_bound.numpy().flatten(),
                        'run_time': run_times.numpy().flatten(),
                        'net': nets.flatten(),
                        'digit': digits.flatten(),
                        })

    return ret


if __name__ == '__main__':
    cln_data, true_labels = load_data(1)
    pd.options.display.max_columns = 12

    # don't use joblib with tensorboard !! set n_jobs=1 to deactivate joblib
    print(check_adv_first(cln_data, true_labels, pairwise=False, nr_eps=10, n_jobs=6, maxsec=120, tensorboard=False))

    #check_verify_first(cln_data, true_labels, 0.15, 10, n_jobs=1)




