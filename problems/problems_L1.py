import torch
import torch.func as ft

class LogisticRegression():
    """
    Class for logistic regression. 
    """
    def __init__(self, device) -> None:
        self.device = device
    
    def predict(self, X, w):
        n, d = X.size()
        Ones = torch.ones((n, 1), device=self.device)
        X = torch.cat([X, Ones], axis=1)
        logits = torch.special.expit(X@w)
        
        return torch.where(logits > 0.5, 1, 0)
    
    def accuracy(self, X, y, w):
        yhat = self.predict(X, w)
        n = torch.numel(y)
        return 100*(torch.sum(torch.where(yhat == y, 1, 0))/n).item()
    
    # Auxillary 
    def logsig(self, w):
        out = torch.zeros_like(w, device=self.device)
        idx0 = w < -33
        out[idx0] = w[idx0]
        idx1 = (w >= -33) & (w < -18)
        out[idx1] = w[idx1] - torch.exp(w[idx1])
        idx2 = (w >= -18) & (w < 37)
        out[idx2] = -torch.log1p(torch.exp(-w[idx2]))
        idx3 = w >= 37
        out[idx3] = -torch.exp(-w[idx3])
        return out
    
    def expit(self, w, y):
        idx = w < 0
        out = torch.zeros_like(w)
        exp_x = torch.exp(w[idx])
        b_idx = y[idx]
        out[idx] = ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
        exp_nx = torch.exp(-w[~idx])
        b_nidx = y[~idx]
        out[~idx] = ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)
        return out
    
    def f_test(self, X, y, l):
        n, d = X.shape
        Ones = torch.ones((n, 1), device=self.device)
        X = torch.cat([X, Ones], axis=1)

        def f(w):
            Xw = X@w 
            return (1/n)*torch.sum((1-y)*Xw - self.logsig(Xw)) + (l/2)*torch.dot(w, w)

        return f

    def mle_logistic_oracle(self, X, y, l=0):
        """
        Maximum likelihood estimation of the logistic regression with regularization.

        # objective is 
        - log L = - sum_i { y_i <xi, w> - log(1 + exp(<x_i, w> )) } 

        We can use the log sum exp trick on the log term as a two term sum to hopefully increase numerical stability. 

        Note: y should be 0/1 encoded labels. l is the L2 regularization parameter.

        Implentation from https://fa.bianp.net/blog/2019/evaluate_logistic/.
        """
        n, d = X.shape
        # include constant as last weight.
        Ones = torch.ones((n, 1), device=self.device)
        X = torch.cat([X, Ones], axis=1)

        # define the negative log likelihood functiom      
        # oracle call

        def oracle(w, vals="fgH"): 
            Xw = X@w 
            fval = (1/n)*torch.sum((1-y)*Xw - self.logsig(Xw)) + (l/2)*torch.dot(w, w)

            if vals=="f":
                return (fval, None, None)
            
            gval = (1/n)*X.T@self.expit(Xw, y) + l*w

            if vals=="fg":
                return (fval, gval, None)

            if vals=="fgH":
                def Hv(v):
                    # H = X^T diag(s(Xw)(1- s(Xw))) X
                    Xv = X@v
                    sXw = torch.special.expit(Xw)
                    return (1/n)*X.T@(sXw*(1 - sXw)*Xv) + l*v
                    
                return (fval, gval, Hv)

        return oracle

class MultinomialRegression():
    """
    Class for the multinomial regression problem. 
    """

    def __init__(self, classes, device=None) -> None:
        self.classes = classes
        self.c = classes-1
        self.device = device

    def get_bias_mask(self, X):
        """
        Computes a mask for the constant parameter for each set of weights.
        """
        n, d = X.shape
        d = d+1
        bias = torch.ones(d*self.c, device=self.device)
        for i in range(1, self.c+1):
            bias[i*d-1] = 0

        return bias 
    
    def predict(self, X, w):
        n, d = X.shape
        Ones = torch.ones((n, 1), device=self.device)
        Zeros = torch.zeros((n, 1), device=self.device)
        X = torch.concatenate([X, Ones], axis=1)
        d = d+1

        W = torch.reshape(w, (self.c, d)).T 
        XW = X@W

        # final column corresponds to dropped class
        XW0 = torch.cat([XW, Zeros], dim=1) 
        Ypred = torch.softmax(XW0, dim=1)
        labels = torch.argmax(Ypred, dim=1)
        
        return labels

    def accuracy(self, X, y, w):
        n, d = X.shape
        yhat = self.predict(X, w)
        return 100*(torch.sum(torch.where(yhat == y, 1, 0))/n).item()

    def get_weights(self, w):
        """
        Get the weights in a formatted form.
        """
        d = int(torch.numel(w)/self.c)
        return torch.reshape(w, (self.c, d)).T 

    def get_oracle(self, X, y, l=0):
        """
        Returns the oracle for the multinomial regression problem.
        """
        n, d = X.shape
        Ones = torch.ones((n, 1), device=self.device)
        Zeros = torch.zeros((n, 1), device=self.device)
        X = torch.concatenate([X, Ones], axis=1)
        d = d+1

        # represent Y as one hot encoded n x c matrix.
        Ydash = torch.nn.functional.one_hot(y) 
        Y = Ydash[:, :self.c] # drop last class as it is implicit
            
        def oracle(w, vals="fgH"):
            # w is a flattened representation of W
            # reshape behaviour means this is better way to unpack weights
            W = torch.reshape(w, (self.c, d)).T 
            XW = X@W # n x c matrix

            ll = -torch.sum(Y*XW, dim=1) # n x 
            XW0 = torch.cat([Zeros, XW], dim=1)
            ll += torch.logsumexp(XW0, dim=1) # exp(0) = 1

            fval = torch.sum(ll)/n + (l/2)*torch.dot(w, w)
            
            if vals=="f":
                return (fval, None, None)
            
            eXW = torch.exp(XW) # n x 
            # now compute normalisation
            S = torch.sum(torch.exp(XW0), dim=1).reshape((n, 1)) # n x 1 sum over classes 
            Stile = torch.tile(S, (1, self.c))
            softmax_probs = eXW/Stile # elementwise division to compute softmax
            # d x c matrix of gradients
            # each col of this matrix is gradient wrt kth weight
            gmat = X.T @(softmax_probs - Y) 
            gval = gmat.T.flatten()/n + l*w

            if vals=="fg":      
                return (fval, gval, None)
            
            if vals=="fgH":
                
                # return hessian vector product function which evaluates hessian product with a tangent 
                def Hv(v):
                    # tangent vector is (d x c)
                    V = v.reshape((self.c, d)).T
                    A = X@V
                    ASprod = A*softmax_probs
                    AS = torch.sum(ASprod, dim=1).reshape((n, 1))
                    rep = AS.repeat((1, self.c))
                    XVd1W = ASprod - softmax_probs*rep
                    Hvv = X.T@XVd1W
                    Hvval = Hvv.T.flatten()
                    
                    return Hvval/n

                return (fval, gval, Hv)

        return oracle
class L1ProxObj():
    """
    Computes the appropriate prox function for L1 regularisation. 
    """

    def __init__(self, l, d, device=None, include_constant=False, bias_mask=None) -> None:
        # l1 regularization parameter of L1 problem.
        self.l = l
        self.device=device
        # mask for the constant terms which the L1 penalty are not applied to.
        # 0s for 

        if include_constant:
            print("L1 penalties are not being applied to bias terms.")
        else:
            print("L1 regularisation applied to all weights.")

        if bias_mask is None and include_constant==True:
            print("Using default bias mask.")
            bias_mask = torch.ones(d, device=self.device)
            bias_mask[-1] = 0
        
        if bias_mask is None and include_constant==False:
            bias_mask = torch.ones(d, device=self.device)

        self.bias_mask = bias_mask

    def get_prox_L1(self):
        """
        Computes the prox operator of L1 norm i.e. the soft thresholding operator. Note that some parameters are not L1 penalised. The prox operator for these paramters is the identity.

        Computes the soft thresholding operator T_t(x).
        """
        def prox(x, t):
            return self.bias_mask*torch.sign(x)*torch.clamp(torch.abs(x) - t*self.l, min=0.0) + (1-self.bias_mask)*x
        
        return prox
    
    def get_L1_reg(self):
        """
        Computes the regularisation term for L1 regularisation.
        """

        def L1(x):
            return self.l*torch.sum(torch.abs(self.bias_mask*x))
        
        return L1

class L1RegularisationDiff():
    """
    Gets the differentiable formulation of L1 regularization as a non-negative constrained problem. 
    """
    def __init__(self, device=None) -> None:
        self.device = device
        
    def gen_x(self, x, d):
        """ Computes the original vector from the 2d non-negative z. """
        xp, xn = x[:d], x[d:]
        return xp - xn
    
    def gen_z(self, x):
        """ Computes the 2xd non-negative vector from x. """
        zero = torch.zeros_like(x)
        xp = torch.max(x, zero)
        xn = -torch.min(x, zero)
        return torch.cat([xp, xn])
         
    def get_objective(self, obj, d, l, include_constant=False, bias_mask=None):
        """
        If include constant is true then the final parameter will remain un penalised by the L1 regularization.

        bias_mask is a mask of the same shape as the parameter vector for the bias terms (1s for weights 0s for bias parameters). The default bias mask will have the final parameter as the constant e.g. for linear regression.
        """
        if include_constant:
            print("L1 penalties are not being applied to bias terms.")
        else:
            print("L1 regularisation applied to all weights.")

        if bias_mask is None and include_constant==True:
            print("Using default bias mask.")
            bias_mask = torch.ones(d, device=self.device)
            bias_mask[-1] = 0

        def oracle(x, vals="fgH"):
            """ New L1 oracle function whcih calls old oracle. """
            xp, xn = x[:d], x[d:]
            if vals=="f":
                fval, _, _ = obj(xp - xn, vals="f")
                return (fval + l*torch.sum(bias_mask*(xp + xn)), None, None)
            
            if vals=="fg":
                fval, gval, _ = obj(xp - xn, vals="fg")

                freg = fval + l*torch.sum(bias_mask*(xp + xn))
                
                gp = gval + bias_mask*l
                gn = -gval + bias_mask*l
                greg = torch.cat([gp, gn])
                return (freg, greg, _)
            
            if vals=="fgH":
                fval, gval, Hv = obj(xp - xn, vals="fgH")
                freg = fval + l*torch.sum(bias_mask*(xp + xn))
                
                gp = gval + bias_mask*l
                gn = -gval + bias_mask*l
                greg = torch.cat([gp, gn])

                def Hvreg(v):
                    # v = (vp, vn) is now a tangent vector.
                    vp, vn = v[:d], v[d:] 
                    Hvxvp = Hv(vp)
                    Hvxvn = Hv(vn)
                    Hvtop = Hvxvp - Hvxvn 
                    return torch.cat([Hvtop, -Hvtop])
                
                return (freg, greg, Hvreg)

        return oracle
    
################################# MLP CODE #################################
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import copy


def make_functional(mod, disable_autograd_tracking=True):
    """
    Essentially the make_functional function from the deprecated functorch package. See:
    - https://pytorch.org/docs/stable/func.migrating.html
    - https://gist.github.com/zou3519/7769506acc899d83ef1464e28f22e6cf
    """
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())
    
    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to('meta')

    def fmodel(new_params_values, *args, **kwargs):
        new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
        return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)
  
    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)

    return fmodel, params_values, params_names


class MLPClassifier(nn.Module):
    """
    Basic MLP classifier.
    """
    def __init__(self, d, l1_shape=100, l2_shape=100, out_classes=2):
        super().__init__()
        self.layer1 = nn.Linear(d, l1_shape, bias=True)
        self.layer2 = nn.Linear(l1_shape, l2_shape, bias=True)
        self.outlayer = nn.Linear(l2_shape, out_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = F.silu(x)
        x = self.layer2(x)
        x = F.silu(x)
        x = self.outlayer(x)
        
        return x

class FunctionalNN():

    def __init__(self, Network, device=None):
        """
        class which generates a functional representation of neural network.
        """
        self.net = Network
        self.device = device
    
    def net_function(self):
        """
        Pytorch usually requires stateful computation of the network through the nn.Module. We want to take a more functional approach. Luckily we can now combine the nn.Module building with the make_functional method
        """ 
        
        self.net = self.net.to(self.device)

        net_functional, self.params, self.param_names = make_functional(self.net, disable_autograd_tracking=True)

        return net_functional

    def net_oracle(self, X, y, mode="rr"):
        """
        Returns an orcale function for self.net as well as a paramter vector. 

        Returns:
            oracle : a function which evaluates the output of this model for a given set of parameters
            params_vec : the initialised parameters of the network 

        Refer to:
            - https://pytorch.org/docs/stable/generated/torch.func.functional_call.html#torch.func.functional_call
            - https://pytorch.org/docs/master/func.migrating.html
        """
        self.net_f = self.net_function()
        params_vec = parameters_to_vector(self.params)
        
        # This function evaluates the loss.
        def loss(w):
            pred = self.net_f(w, X)
            return F.cross_entropy(pred, y, reduction="mean") 

        # autodiff with the tuples version of parameters. 
        mlp_oracle = build_autodiff(loss, mode=mode)

        # translate the parameters 
        def oracle(w, vals="fgH"):
            w_par = self.get_vect_to_params(w)

            if vals=="f":
                fval = loss(w_par)
                return (fval, None, None)
            
            if vals=="fg":
                fval, gval, _ = mlp_oracle(w_par, vals=vals)
                gval = parameters_to_vector(gval)
                return (fval, gval, None)
            
            # get g on its own
            if vals=="fgH":
                fval, gval, _Hv = mlp_oracle(w_par, vals=vals)
                gval = parameters_to_vector(gval)

                def Hv(v):
                    v_par = self.get_vect_to_params(v)
                    return parameters_to_vector(_Hv(v_par))
                
                return (fval, gval, Hv)

        return oracle, params_vec

    def get_bias_mask(self):
        """
        Create a mask for the bias parameters in the network (by name).
        """
        print("Parameter names are")
        print(self.param_names)
        bias_in_name = [0 if "bias" in name else 1 for name in self.param_names]
        
        bias_mask =[]
        for b, v in zip(bias_in_name, self.params):
            bias_mask.append(b*torch.ones_like(v, device=self.device))

        bias_mask = tuple(bias_mask)

        bias_mask = parameters_to_vector(bias_mask)
        return bias_mask
    
    def get_vect_to_params(self, w):
        """
        Returns a tuple of the parameters from a vector format.
        """
        # Here self.w0 serves as a template for the layout.
        w_out = copy.deepcopy(self.params)
        vector_to_parameters(w, w_out)
        return w_out

    def evaluate(self, X, w):
        """
        Must be called after self.mlp().
        """
        w_par = self.get_vect_to_params(w)
        y = self.net_f(w_par, X)
        return y
    
    def accuracy(self, w, X, y):
        n, d = X.shape
        y_p = self.evaluate(X, w)
        y_classes = torch.argmax(y_p, dim=1)
        acc = 100*torch.sum(torch.where(y_classes == y, 1, 0))/n
        return acc.item()
    

def build_autodiff(f, mode="rr"):
    """
    Uses autodifferentiation to return an oracle function.
    """
    def oracle(x, vals="fgH"):   
        if vals=="f":
            fval = f(x)
            return (fval, None, None)
        
        if vals=="fg":
            g_and_f = ft.grad_and_value(f)
            gval, fval = g_and_f(x)
            return (fval, gval, None)

        if vals=="fgH":
            
            g_and_f = ft.grad_and_value(f)

            # forward over reverse differentiation
            if mode == "fr":
                gval, fval = g_and_f(x)
            
                g = lambda x: g_and_f(x)[0]

                def hvp(v):
                    return ft.jvp(g, (x,), (v,))[1]
                
            # reverse over reverse
            elif mode == "rr":
                
                gval, _hvp, fval  = ft.vjp(g_and_f, x, has_aux=True)

                def hvp(v):
                    return _hvp(v)[0] 
                
            else:
                raise ValueError("Invalid differentiation mode" + mode)
            
            return (fval, gval, hvp)
        
    return oracle