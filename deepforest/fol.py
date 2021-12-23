import torch
import numpy as np

class FOL(object):
    """ First Order Logic (FOL) rules """

    def __init__(self, device, K, input, fea):
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
        self.device = device

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

    w = C * rule_lambda
    X = images from batch; in a tuple
    F = special features calculated for rule; mean green value in box

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
# green rule
# ----------------------------------------------------
class FOL_green(FOL):
    def __init__(self, device, K, input, fea):
        """ Initialize

    :type K: int
    :param K: the number of classes

    :type fea: theano.tensor.dtensor4
    :param fea: symbolic feature tensor, of shape 3
                fea[0]   : 1 if x=x1_but_x2, 0 otherwise
                fea[1:2] : classifier.predict_p(x_2)
    """
        super().__init__(device, K, input, fea)



    """
    Rule-specific functions
    """
    def log_distribution2(self, w, X=None, F=None):
        """ Return an nxK matrix with the (i,c)-th term
    = - w * (1 - r(X_i, y_i=c))
           if X_i is a grounding of the rule
    = 1    otherwise

    w = C * rule_lambda
    X = y_pred a list of dictionaries; 1 dictionary per image; keys are 'boxes', 'scores', 'labels'
    F = special features calculated for rule; mean green value in box

    """
        if F == None:
            X, F, conds = self.input, self.fea, self.conds
        else:
            conds = self.conditions(torch.zeros(X.shape[0], 1), F)
        log_distr = self.distribution_helper(w, X, F, conds)
        return log_distr

    def log_distribution(self, w, X=None, F=None):
        #import pdb; pdb.set_trace()
        f_1 = F.reshape(-1, 1)
        f_0 = 1. - f_1
        f = torch.cat([f_0, f_1], 1)
        f = w * f
        return f


    def value_single(self, x, y, f):
        ret = torch.mean(torch.tensor([torch.min(torch.tensor([1. - y + f, 1.])), torch.min(torch.tensor([1. - f + y, 1.]))]))
        return ret

    def distribution_helper(self, w, X, F, conds):
        # w - normalizing factor
        # X - original data
        # F - engineered features
        # conds - conditions for each rule
        # map applies distribution_helper_helper to each row of distribution
        # returns a tensor with values r(x,y)

        nx = X.shape[0]            # number of bounding boxes in image
        distr = torch.ones([nx, self.K], dtype=torch.float, requires_grad=True)
        distr = torch.tensor(list(map(lambda c, x, f, d: self.distribution_helper_helper(x, f) if c == True else d.tolist(), conds, X, F, distr)))
        distr = torch.tensor(list(map(lambda d: (-w * ((torch.min(d) * torch.ones(d.shape)) - d)).tolist(), distr)))  # relative value w.r.t the minimum
        return distr



    def condition_single(self, x, f):
        # returns a tensor batch_size x 1 of True or False
        # 0 means its labeled as a tree
        return x == 0


# ---------------------------------------------------------------------------
# Competition Rule
# ---------------------------------------------------------------------------
class FOL_competition(FOL):
    def __init__(self, device, K, input, fea):
        """ Initialize

    :type K: int
    :param K: the number of classes

    :type fea: theano.tensor.dtensor4
    :param fea: symbolic feature tensor, of shape 3
                fea[0]   : 1 if x=x1_but_x2, 0 otherwise
                fea[1:2] : classifier.predict_p(x_2)
    """
        super().__init__(device, K, input, fea)
    def expon(self, w, X=None, F=None):
        """
        K - weight for rule
        X - predicted values; BB corners
        F - engineered features
        """

        lbound = np.array([5, 5, 5, 5, 6, 6, 5, 5, 4, 3, 3, 3, 2, 4, 1, 3, 3, 4, 5, 5, 5, 6, 12, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15])
        ubound = np.array([20, 20, 20, 20, 21, 21, 23, 29, 31, 52, 55, 45, 48, 42, 48, 49, 51, 47, 64, 47, 48, 52, 48, 48, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63])

        mapped_lbound = torch.from_numpy(lbound[X['hts'].detach().cpu().numpy().astype(np.int)]).to(self.device).type(torch.float)
        mapped_lbound.requires_grad = True

        mapped_ubound = torch.from_numpy(ubound[X['hts'].detach().cpu().numpy().astype(np.int)]).to(self.device).type(torch.float)
        mapped_ubound.requires_grad = True

        # assume these are optimal values for height and width
        optim_dim = torch.sqrt((0.3073 * (X['hts'] ** 0.9083))/0.01)
        optim_dim = optim_dim.to(self.device)

        

        #if (X['boxes'].shape[0] > 0):
        #   import pdb; pdb.set_trace()


        # calculate the width and height of each bounding box
        width = (X['boxes'][:, 2] - X['boxes'][:, 0]).unsqueeze(1).to(self.device)
        height = (X['boxes'][:, 3] - X['boxes'][:, 1]).unsqueeze(1).to(self.device)

        # calculate the weighted difference between the optimum values and the actual values
        #diff_x = w * (optim_w * torch.ones([X['boxes'].shape[0], 1]).to(self.device) - width)
        #diff_y = w * (optim_h * torch.ones([X['boxes'].shape[0], 1]).to(self.device) - height)

        
        
        diff_x = w * (optim_dim - width)
        diff_y = w * (optim_dim - height)

        #diff_x = torch.where(torch.abs(diff_x) <= (3 * sdev), torch.tensor([0.]).to(self.device), diff_x)
        #diff_y = torch.where(torch.abs(diff_y) <= (3 * sdev), torch.tensor([0.]).to(self.device), diff_y)
        diff_x = torch.where((width >= mapped_lbound) & (width <= mapped_ubound), torch.tensor([0.]).to(self.device), diff_x)
        diff_y = torch.where((height >= mapped_lbound) & (height <= mapped_ubound), torch.tensor([0.]).to(self.device), diff_y)

        # create one matrix with two columns by concatenating diff_x and diff_y
        temp = torch.cat([diff_x, diff_y], dim=1)

        # create a res matrix containing all zeros
        # first column will be the x differences, second column will be the y differences
        res = torch.zeros(X['boxes'].shape[0], 2).to(self.device)

        # set rows where trees where growing in close proximity (competition) to diff values
        # all other rows stay zero
        res[F, :] = temp[F, :]

        return res

# ---------------------------------------------------------------------------
# Bbox_2big Rule
# ---------------------------------------------------------------------------
class FOL_bbox_2big(FOL):
    def __init__(self, device, K, input, fea):
        """ Initialize
        :type K: int
        :param K: the number of classes
        :param fea: symbolic feature tensor, of shape 3
                fea[0]   : 1 if x=x1_but_x2, 0 otherwise
                fea[1:2] : classifier.predict_p(x_2)
        """

        super().__init__(device, K, input, fea)


        # Rule-specific functions

    def log_distribution2(self, w, X=None, F=None):
        """ Return an nxK matrix with the (i,c)-th term
        = - w * (1 - r(X_i, y_i=c))
           if X_i is a grounding of the rule
        = 1    otherwise
        w = C * rule_lambda
        X = y_pred a list of dictionaries; 1 dictionary per image; keys are 'boxes', 'scores', 'labels'
        F = special features calculated for rule; mean green value in box
        """

        if F == None:
            X, F, conds = self.input, self.fea, self.conds
        else:
            conds = self.conditions(torch.zeros(X.shape[0], 1), F)
        log_distr = self.distribution_helper(w, X, F, conds)
        return log_distr

    def log_distribution(self, w, X=None, F=None):
        f_1 = F.reshape(-1, 1)
        f_0 = 1. - f_1
        f = torch.cat([f_0, f_1], 1)
        f = w * f
        return f


    def value_single(self, x, y, f):
        ret = torch.mean(torch.tensor([torch.min(torch.tensor([1. - y + f, 1.])), torch.min(torch.tensor([1. - f + y, 1.]))]))
        return ret

    def distribution_helper(self, w, X, F, conds):
        # w - normalizing factor
        # X - original data
        # F - engineered features
        # conds - conditions for each rule
        # map applies distribution_helper_helper to each row of distribution
        # returns a tensor with values r(x,y)

        nx = X.shape[0]            # number of bounding boxes in image
        distr = torch.ones([nx, self.K], dtype=torch.float, requires_grad=True).to(self.device)
        distr = torch.tensor(list(map(lambda c, x, f, d: self.distribution_helper_helper(x, f) if c == True else d.tolist(), conds, X, F, distr))).to(self.device)
        distr = torch.tensor(list(map(lambda d: (-w * ((torch.min(d) * torch.ones(d.shape)) - d)).tolist(), distr))).to(self.device)  # relative value w.r.t the minimum
        return distr



    def condition_single(self, x, f):
        # returns a tensor batch_size x 1 of True or False
        # 0 means its labeled as a tree
        return x == 0
