import torch
import numpy as np

class LogicNN(object):
    def __init__(self, network, rules=[], rule_lambda=[], C=1.):
        """
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
        """
        # self.input = input  # not used
        self.network = network
        self.rules = rules
        self.rule_lambda = np.asarray(rule_lambda, dtype=np.float)
        self.ones = np.ones(len(rules)) * 1.
        # self.pi = pi   # not used
        self.C = C

    def calc_rule_constraints(self,  p_y_pred, new_data=None, new_rule_fea=None):
        # new_data = images in batch
        # new_rule_fea = a list of features calculated for rules
        if new_rule_fea == None:
            new_rule_fea = [None] * len(self.rules)
        distr_all = torch.tensor(0.0)
        for i, rule in enumerate(self.rules):
            distr = rule.log_distribution(self.C * self.rule_lambda[i], p_y_pred, new_rule_fea[i])
            distr_all = distr_all + distr
        distr_all += distr
        #
        # distr_y0 = distr_all[:, 0]
        # distr_y0 = distr_y0.reshape([distr_y0.shape[0], 1])
        # distr_y0_copies = torch.tile(distr_y0, [1, distr_all.shape[1]])
        # distr_all -= distr_y0_copies
        distr_all = torch.maximum(torch.minimum(distr_all, torch.tensor(60.)), torch.tensor(-60.))  # truncate to avoid over-/under-flow
        distr_all *= torch.tensor(-1.)
        return torch.exp(distr_all)

    def predict(self, p_y_pred, new_data, new_rule_fea):
        # p_y_pred - current predictions
        # new_data - images from current batch
        # new_rule_fea - mean of green channel for each predicted box

        q_y_given_x_fea_pred = p_y_pred * self.calc_rule_constraints(p_y_pred, new_data, new_rule_fea)

        # normalize
        n_q_y_given_x_fea_pred = q_y_given_x_fea_pred / torch.sum(q_y_given_x_fea_pred, 1).reshape((-1, 1))

        q_y_pred = torch.argmax(n_q_y_given_x_fea_pred, 1).unsqueeze(1)
        p_y_pred = torch.argmax(p_y_pred, 1).unsqueeze(1)
        return q_y_pred, p_y_pred
