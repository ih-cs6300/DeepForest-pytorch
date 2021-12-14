import torch
import numpy as np

class LogicNN(object):
    def __init__(self, device, network, rules=[], rule_lambda=[], C=1.):
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
        self.device = device

    def calc_rule_constraints(self,  p_y_pred, new_data=None, new_rule_fea=None):
        # new_data = images in batch
        # new_rule_fea = a list of features calculated for rules
        if new_rule_fea == None:
            new_rule_fea = [None] * len(self.rules)
        distr_all = torch.tensor(0.0, requires_grad=True)
        for i, rule in enumerate(self.rules):
            distr = rule.log_distribution(self.C * self.rule_lambda[i], p_y_pred, new_rule_fea[i])
            distr_all = distr_all + distr
        distr_all += distr

        distr_y0 = distr_all[:, 0]
        distr_y0 = distr_y0.reshape([distr_y0.shape[0], 1])
        distr_y0_copies = torch.tile(distr_y0, [1, distr_all.shape[1]])
        distr_all -= distr_y0_copies
        distr_all = torch.maximum(torch.minimum(distr_all, torch.tensor(60.)), torch.tensor(-60.))  # truncate to avoid over-/under-flow
        return torch.exp(distr_all)

    def reg_rule_constraints(self, y_pred, new_data=None, new_rule_fea=None):
        """
        Description: Sums rule outputs for regression based FOL rules
        y_pred - prediction regression values
        new_data- image data
        new_rule_fea - a list of rule features for each rule
        """
        expon_all = torch.tensor(0.0, requires_grad=True)
        for i, rule in enumerate(self.rules):
            expon = rule.expon(self.C * self.rule_lambda[i], y_pred, new_rule_fea[i])
            expon_all = expon_all + expon

        # truncate to avoid over-/under-flow
        expon_all = torch.maximum(torch.minimum(expon_all, torch.tensor(1*0.019802)), torch.tensor(1*-0.02020))
        #expon_all = torch.maximum(torch.minimum(expon_all, torch.tensor(60.)), torch.tensor(-60.))

        return torch.exp(expon_all)

    def predict(self, p_y_pred, new_data, new_rule_fea):
        # p_y_pred - current predictions
        # new_data - images from current batch
        # new_rule_fea - mean of green channel for each predicted box

        #q_y_given_x_fea_pred = p_y_pred * self.calc_rule_constraints(p_y_pred, new_data, new_rule_fea)

        p_y_pred_0 = 1. - p_y_pred
        p_y_pred = torch.cat([p_y_pred_0.reshape(-1, 1), p_y_pred.reshape(-1, 1)], dim=1)
        q_y_given_x_fea_pred = p_y_pred * self.calc_rule_constraints(p_y_pred, new_data, new_rule_fea)

        # normalize
        n_q_y_given_x_fea_pred = q_y_given_x_fea_pred / torch.sum(q_y_given_x_fea_pred, 1).reshape((-1, 1))

        temp = torch.argmax(n_q_y_given_x_fea_pred, 1).unsqueeze(1)
        temp = 1 - temp
        temp = temp.flatten()
        q_y_pred = temp
        #p_y_pred = torch.argmax(p_y_pred[0]['scores2'], 1).unsqueeze(1)
        return q_y_pred

    def scaleBB(self, coords, scaleX, scaleY):
        # takes in bounding box coordinates as [x1, y1, x2, y2] and returns a scaled bounding box with the same centroid
        coords2 = coords.view(-1, 2).to(self.device)

        # transpose coordinates and make them homogenous
        coordsMatrix = torch.vstack([coords2.T, torch.ones([1, coords2.shape[0]], requires_grad=True).to(self.device)]).to(self.device)

        # calculate coordinates of centroid
        # centroid = np.mean(coordsNp[:-1, :], axis=0)
        centroid = torch.mean(coords2, 0)

        # transform to translate to origin, scale, and translate back to centroid
        trans = torch.tensor(
            [[scaleX, 0, centroid[0] * (1 - scaleX)],
             [0, scaleY, centroid[1] * (1 - scaleY)],
             [0, 0, 1]]).to(self.device)

        # multiply matrices
        res = torch.mm(trans, coordsMatrix)[:2, :].T
        res = res.contiguous()

        # return data to original format of a list of tuples
        res = res.view(1, 4)

        return res

    def regress(self, p_y_pred, new_data, new_rule_fea):
        """
        Description: Produces teacher network predictions for regression based FOL
        p_y_pred - student network predictions
        new_data - image data
        new_rule_fea - engineered feature to be used by FOL rule
        """

        scale_factors = self.reg_rule_constraints(p_y_pred, new_data, new_rule_fea)

        q_y_pred = torch.zeros(p_y_pred.shape, requires_grad=True).to(self.device)


        # scale each BBox while keeping the same centroid
        for idx, scale_row in enumerate(scale_factors):
            q_y_pred.data[idx] = self.scaleBB(p_y_pred[idx], scale_row[0], scale_row[1])

        return q_y_pred
