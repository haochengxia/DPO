# -*- coding: utf-8 -*-
# Copyright (c) Percy.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
dprl.ldp
~~~~~~~~
This module contains ldp algorithms.
"""

from math import ceil, exp, floor, log
import random

import itertools
import numpy as np
from scipy.special import comb
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

from .exceptions import UnImpException


def generate_binary_random(pr, first, second):
    if np.random.random() <= pr:
        return first
    else:
        return second


class Numeric(object):

    def __init__(self):
        self.scope = 'only numeric'

    # ======================================
    # COLLECTING A SINGLE NUMERIC ATTRIBUTES
    # ======================================

    @staticmethod
    def duchi_one_dim_method(tp, epsilon):
        """
        Duchi et al.’s Solution for One-Dimensional Numeric Data

        :param tp: tp is a one-dimensional tuple (t_i \in [-1, 1])
        :param epsilon: privacy param
        """

        prob = (exp(epsilon) - 1) * tp / (2 * exp(epsilon) + 2) + 1 / 2
        t_star = (exp(epsilon) + 1) / (exp(epsilon) - 1)

        if generate_binary_random(prob, 1, 0) == 1:
            return t_star
        else:
            return -t_star

    @staticmethod
    def piecewise_mechanism(tp, epsilon):
        """
        Piecewise Mechanism for One-Dimensional Numeric Data.

        :param tp: tp is a one-dimensional tuple (t_i \in [-1, 1])
        :param epsilon: privacy budget param
        """

        C = (exp(epsilon / 2) + 1) / (exp(epsilon / 2) - 1)
        l_ti = (C + 1) * tp / 2 - (C - 1) / 2
        r_ti = l_ti + C - 1

        x = np.random.uniform(0, 1)  # [0, 1) but in paper [0, 1]
        if x < exp(epsilon / 2) / (exp(epsilon / 2) + 1):
            # [l_ti, r_ti) but in paper [l_ti, r_ti]
            return np.random.uniform(l_ti, r_ti)
        else:
            # [−C, l_ti) \cup [r_ti, C) but in paper [−C, l_ti) \cup (r_ti, C]
            return uniform_two(-C, l_ti, r_ti, C)

    @staticmethod
    def hybrid_mechanism(tp, epsilon):
        """
        Combine PM and Duchi et al’s solution into a new Hybrid Mechanism (HM)

        :param tp: tp is a one-dimensional tuple
        :param epsilon: privacy budget param
        """

        if epsilon > 0.61:
            alpha = 1 - exp(-1 * epsilon / 2)
        else:
            alpha = 0

        if np.random.random() < alpha:
            return Numeric.piecewise_mechanism(tp, epsilon)
        else:
            return Numeric.duchi_one_dim_method(tp, epsilon)

    # ======================================
    # COLLECTING MULTIPLE NUMERIC ATTRIBUTES
    # ======================================

    @staticmethod
    def duchi_multi_dim_method(tp, epsilon):
        """
        Duchi et al.’s Solution for Multidimensional Numeric Data

        :param tp: tp is a d-dimensional tuple (t_i \in [-1, 1]^d)
        :param epsilon: privacy budget param
        """

        d = len(tp)
        if d % 2 != 0:
            # d is odd
            C_d = 2 ** (d - 1) / comb(d - 1, int((d - 1) / 2))
        else:
            # otherwise
            C_d = (2 ** (d - 1) + 1 / 2 * comb(d, int(d / 2))) / comb(d - 1, int(d / 2))

        B = (exp(epsilon) + 1) / (exp(epsilon) - 1) * C_d
        v = [generate_binary_random(1 / 2 + 1 / 2 * tp[j], 1, -1) for j in range(d)]

        t_pos = []
        t_neg = []
        for t_star in itertools.product([-B, B], repeat=d):
            if np.dot(np.array(t_star), np.array(v)) >= 0:
                t_pos.append(t_star)
            else:
                t_neg.append(t_star)

        prob = exp(epsilon) / (exp(epsilon) + 1)
        if generate_binary_random(prob, 1, 0) == 1:
            return np.random.choice(t_pos)
        else:
            return np.random.choice(t_neg)

    @staticmethod
    def wang_multi_dim_method(tp, epsilon, method):
        """
        Wang et al.’s Solution for Multidimensional Numeric Data

        :param tp: tp is a d-dimensional tuple (t_i \in [-1, 1]^d)
        :param epsilon: privacy budget param
        :param method: PM or HM
        """

        d = len(tp)
        t_star = np.zeros(d)
        k = max(1, min(d, floor(epsilon / 2.5)))
        samples = random.sample(list(np.arange(1, d + 1)), k)
        for j in samples:
            j = j -1
            if method == "PM":
                t_star[j] = (d / k) * Numeric.piecewise_mechanism(tp[j], epsilon / k)
            elif method == "HM":
                t_star[j] = (d / k) * Numeric.hybrid_mechanism(tp[j], epsilon / k)
            else:
                raise ValueError("method should either be 'PM' or 'HM'")
        return t_star


class Categorical(object):

    def __init__(self):
        self.scope = 'only categorical'

    @staticmethod
    def wang_one_dim_method(tp, epsilon, A):
        """
        OUE proposed by Tianhao Wang

        :param tp: one dimension categorical date
        :param epsilon: privacy budget param
        :param A: the list of the data domain
        """
        p = 0.5
        q = 1 / (exp(epsilon) + 1)
        oh_vec = np.random.choice([1, 0], size=len(A), p=[q, 1 - q])  # If entry is 0, flip with prob q
        index = A.index(tp)

        if random.random() < p:
            oh_vec[index] = 1

        return oh_vec


class NumericAndCategorical(object):

    def __init__(self):
        self.scope = 'numeric and categorical'

    @staticmethod
    def wang_multi_dim_method(tp, epsilon, method, cates):
        """
        Wang et al.’s Solution for Multidimensional Numeric and Categorical Data

        :param tp: tp is a d-dimensional tuple
        :param epsilon: privacy budget param
        :param method: PM or HM
        :param cates: d-dimension list, if tp[j] is numeric, cates[j] is [], otherwise, cates[j] is data domain
        """

        d = len(tp)
        t_star = np.zeros(d)
        k = max(1, min(d, floor(epsilon / 2.5)))
        samples = random.sample(list(np.arange(1, d + 1)), k)

        for j in samples:
            j = j - 1
            if is_numeric(tp[j], cates[j]):
                if method == "PM":
                    t_star[j] = (d / k) * Numeric.piecewise_mechanism(tp[j], epsilon / k)
                elif method == "HM":
                    t_star[j] = (d / k) * Numeric.hybrid_mechanism(tp[j], epsilon / k)
                else:
                    raise ValueError("method should either be 'PM' or 'HM'")
            else:
                t_star[j] = Categorical.wang_one_dim_method(tp[j], epsilon, cates[j])

        return t_star


class SGD(object):
    # Notice:
    # Each user only participates in at most one iteration

    def __init__(self, x_train, y_train, x_test, y_test, problem, epsilon_list, method):
        """

        :param problem: 'linear-regression' or 'logistic-regression' or 'svm'
        """

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # Basic parameters
        self.epsilon_list = epsilon_list
        self.method = method
        self.problem = problem

        self.d = len(x_train[0])

        # Model parameters
        self.beta = np.zeros(self.d)

    def _submit_gradient(self, i):
        """
        User i submit the gradient added noise
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        d = len(self.x_train[0])

        gradient = None
        if self.problem == 'svm':
            gradient = _svm_train(self.x_train[[i]],
                                  self.y_train[[i]], d, self.beta, device)
        elif self.problem == 'linear-regression':
            gradient = _linear_train(self.x_train[[i]],
                                     self.y_train[[i]], d, self.beta, device)
        elif self.problem == 'logistic-regression':
            gradient = _logistic_train(self.x_train[[i]],
                                       self.y_train[[i]], d, self.beta, device)

        return Numeric.wang_multi_dim_method(gradient, self.epsilon_list[i], self.method)

    def _divide_group(self):
        """
        Divide index into groups. (The group size is G)
        """

        groups = []

        # G = \Omega (d(\log d) \epsilon^2) but epsilon not always the same
        G = ceil(self.d * log(self.d) / np.mean(self.epsilon_list) ** 2)

        n = len(self.y_train)
        index = np.arange(n)
        np.random.shuffle(index)
        groups_size = [G] * (n // G - 1) + [n - G * (n // G - 1)]
        for i, group_size in enumerate(groups_size):
            begin = i * G
            end = i * G + group_size
            groups.append(index[begin: end])

        return groups

    def _eval_beta(self):
        """
        evaluate parameter (beta) utility

        """
        d = len(self.x_test[0])

        model = nn.Linear(in_features=d, out_features=1, bias=False)
        model.weight = torch.nn.parameter.Parameter(
            torch.from_numpy(self.beta).reshape((1, d)), requires_grad=False
        )

        y_pred = model(torch.from_numpy(self.x_test)).numpy()

        # correct for svm
        if self.problem == 'svm':
            y_pred[np.where(y_pred >= 0)] = 1
            y_pred[np.where(y_pred < 0)] = 0

        return accuracy_score(self.y_test, y_pred)

    def eval(self):
        """
        return utility (acc on Test Set) of ldp processed data
        """
        groups = self._divide_group()
        iterations = len(groups)
        learning_rate = 1 / iterations ** (1 / 2)

        for group in groups:
            gradients = []
            for i in group:
                gradient = self._submit_gradient(i)
                gradients.append(gradient)
            # Update parameter
            self.beta -= learning_rate * np.mean(gradients, axis=0)

        # Evaluate the utility (acc on Test Set)
        return self._eval_beta()


def is_numeric(data, cate):
    if data not in cate:
        return True
    else:
        return False


def _svm_train(x, y, d, beta, device, c=0.01):
    x = torch.from_numpy(x).reshape((1, d))
    y = torch.from_numpy(y).reshape((1, 1))

    model = nn.Linear(in_features=d, out_features=1, bias=False)
    model.to(device)
    model.weight = torch.nn.parameter.Parameter(
        torch.from_numpy(beta).reshape((1, d)), requires_grad=True
    )

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    output = model(x).squeeze()  # output is x^T \beta
    weight = model.weight.squeeze()

    loss = torch.mean(torch.clamp(1 - y * output, min=0))
    loss += c / 2.0 * (weight.t() @ weight)  # c is a regularization parameter

    loss.backward()

    grad = model.weight.grad.numpy()[0]
    # Clip
    if grad.all() > 1:
        grad = 1
    elif grad.all() < -1:
        grad = -1

    grad[y[0]] = 0
    return grad


def _linear_train(x, y, d, beta, device, c=0.01):
    # TODO: support linear-regression
    raise UnImpException('_linear_train (for linear regression task)')


def _logistic_train(x, y, d, beta, device, c=0.01):
    # TODO: support logistic-regression
    raise UnImpException('_logistic_train (for logistic regression task)')


def uniform_two(a1, a2, b1, b2):
    # Calc weight for each range
    delta_a = a2 - a1
    delta_b = b2 - b1
    if np.random.random() < delta_a / (delta_a + delta_b):
        return np.random.uniform(a1, a2)
    else:
        return np.random.uniform(b1, b2)

