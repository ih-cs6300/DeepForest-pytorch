import torch
import numpy as np

class FOL(object):
    """ First Order Logic (FOL) rules """

    def __init__(self, K, input, fea):
        """ Initialize

    : type K: int
    : param K: the number of classes
    """
        self.input = input
        self.fea = fea
        # Record the data relevance (binary)
        #self.conds = self.conditions(self.input, self.fea)
        # input - original data
        # fea - engineered feature set
        # fol statements divided into condition -> value
        self.K = K

    def conditions(self, X, F):
        results = torch.tensor(list(map(lambda x, f: self.condition_single(x, f), X, F)))
        return results

    def distribution_helper_helper(self, x, f):
        # applies self.value_single to each row of distr passed to it by distribution_helper
        # returns a list of rows with values from evaluation of r(x,y)
        results = list(map(lambda x, k, f: self.value_single(x, k, f), [x] * self.K, range(self.K), [f] * self.K))
        return results

    def distribution_helper(self, w, X, F, conds):
        # w - normalizing factor
        # X - original data
        # F - engineered features
        # conds - conditions for each rule
        # map applies distribution_helper_helper to each row of distribution
        # returns a tensor with values r(x,y)

        nx = X.shape[0]
        distr = torch.ones([nx, self.K], dtype=torch.float)
        distr = torch.tensor(list(map(lambda c, x, f, d: self.distribution_helper_helper(x, f) if c == 1. else d.tolist(), conds, X, F, distr)))
        distr = torch.tensor(list(map(lambda d: (-w * ((torch.min(d) * torch.ones(d.shape)) - d)).tolist(), distr)))  # relative value w.r.t the minimum
        return distr

    """
    Interface function of logic constraints

    The interface is general---only need to overload condition_single(.) and
    value_single(.) below to implement a logic rule---but can be slow

    See the overloaded log_distribution(.) of the BUT-rule for an efficient
    version specific to the BUT-rule
    """

    def log_distribution(self, w, X=None, F=None, config={}):
        """ Return an nxK matrix with the (i,c)-th term
    = - w * (1 - r(X_i, y_i=c))
           if X_i is a grounding of the rule
    = 1    otherwise
    """
        if F == None:
            X, F, conds = self.input, self.fea, self.conds
        else:
            conds = self.conditions(X,F)
        log_distr = self.distribution_helper(w, X, F, conds)
        return log_distr

    """
    Rule-specific functions to be overloaded    

    """

    def condition_single(self, x, f):
        """ True if x satisfies the condition """
        return 0.

    def value_single(self, x, y, f):
        """ value = r(x,y) """
        return 1.


# ----------------------------------------------------
# BUT rule
# -------------------------------------Test accu---------------

class FOL_But(FOL):
    """ x=x1_but_x2 => { y => pred(x2) AND pred(x2) => y } """

    def __init__(self, K, input, fea):
        """ Initialize

    :type K: int
    :param K: the number of classes

    :type fea: theano.tensor.dtensor4
    :param fea: symbolic feature tensor, of shape 3
                fea[0]   : 1 if x=x1_but_x2, 0 otherwise
                fea[1:2] : classifier.predict_p(x_2)
    """
        assert K == 2
        super(FOL_But, self).__init__(K, input, fea)

    """
    Rule-specific functions

    """

    def condition_single(self, x, f):
        # returns a tensor batch_size x 1 of True or False
        return f[0] == 1.

    def distribution_helper_helper(self, x, f):
        # applies self.value_single to each row of distr passed to it by distribution_helper
        # returns a list of rows with values from evaluation of r(x,y)
        results = list(map(lambda x, k, f: self.value_single(x, k, f), [x] * self.K, range(self.K), [f] * self.K))
        return results

    def distribution_helper(self, w, X, F, conds):
        # w - normalizing factor
        # X - original data
        # F - engineered features
        # conds - conditions for each rule
        # map applies distribution_helper_helper to each row of distribution
        # returns a tensor with values r(x,y)

        nx = X.shape[0]
        distr = torch.ones([nx, self.K], dtype=torch.float)
        distr = torch.tensor(list(map(lambda c, x, f, d: self.distribution_helper_helper(x, f) if c == True else d.tolist(), conds, X, F, distr)))
        distr = torch.tensor(list(map(lambda d: (-w * ((torch.min(d) * torch.ones(d.shape)) - d)).tolist(), distr)))  # relative value w.r.t the minimum
        return distr

    def value_single(self, x, y, f):
        ret = torch.mean(torch.tensor([torch.min(torch.tensor([1. - y + f[2], 1.])), torch.min(torch.tensor([1. - f[2] + y, 1.]))]))
        #ret = torch.tensor(ret, dtype=torch.float)
        return (1., ret)[self.condition_single(x, f)]

    """
    Efficient version specific to the BUT-rule

    """
    def log_distribution(self, w, X=None, F=None, config={}):
        """ Return an nxK matrix with the (i,c)-th term
    = - w * (1 - r(X_i, y_i=c))
           if X_i is a grounding of the rule
    = 1    otherwise
    """
        if F == None:
            X, F, conds = self.input, self.fea, self.conds
        else:
            conds = self.conditions(X, F)
        log_distr = self.distribution_helper(w, X, F, conds)
        return log_distr

    def log_distributionOld(self, w, X=None, F=None):
        if F == None:
            X, F = self.input, self.fea
        F_mask = F[:, 0]
        F_fea = F[:, 1:]
        # y = 0
        distr_y0 = w * F_mask * F_fea[:, 0]

        # y = 1
        distr_y1 = w * F_mask * F_fea[:, 1]
        distr_y0 = distr_y0.reshape([distr_y0.shape[0], 1])
        distr_y1 = distr_y1.reshape([distr_y1.shape[0], 1])
        distr = torch.cat([distr_y0, distr_y1], 1)
        return distr


# ----------------------------------------------------
# green rule
# ----------------------------------------------------
class FOL_green(FOL):
    def __init__(self, K, input, fea):
        """ Initialize

    :type K: int
    :param K: the number of classes

    :type fea: theano.tensor.dtensor4
    :param fea: symbolic feature tensor, of shape 3
                fea[0]   : 1 if x=x1_but_x2, 0 otherwise
                fea[1:2] : classifier.predict_p(x_2)
    """
        super().__init__(K, input, fea)



    """
    Rule-specific functions
    """

    def value_single(self, x, y, f):
        ret = torch.mean(torch.tensor([torch.min(torch.tensor([1. - y + f[2], 1.])), torch.min(torch.tensor([1. - f[2] + y, 1.]))]))
        #ret = torch.tensor(ret, dtype=torch.float)
        return (1., ret)[self.condition_single(x, f)]



    def condition_single(self, x, f):
        # returns a tensor batch_size x 1 of True or False
        return f[0] == 1.
