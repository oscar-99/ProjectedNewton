import torch
import torch.func as ft


def WHfromX(X, n):
    """ 
    Helper function which extracts W, H from a matrix of the form X = [W.T, H].T
    """
    W = X[:n, :]
    H = X[n:, :].T
    return W, H        
    
def XfromWH(W, H):
    """
    NNMF Helper function which forms the matrix X = [W.T, H].T
    """
    return torch.concatenate([W, H.T])

def vector(X, n, d, r):
    """
    Vectorise NNMF.

    Takes in the structured (n+d, r) matrix X = [W.T, V].T and vectorise into a ((n+d)*r,) vector. 
    """
    return torch.reshape(X, ((n + d)*r,))

def unvector(x, n, d, r):
    """
    Unvectorise NNMF.

    Reshapes the vector x to the matrix X (n + d, r).
    """
    return torch.reshape(x, (n+d, r))        

def TSCAD(l, beta):
    """
    A twice differentiable SCAD type function.
    """
    a = 1/(2*(beta*l**2 - l**2)*(beta**2 - 2*beta + 1))
    b =  (beta + 1)/((l - beta*l)*(beta**2 - 2*beta + 1))
    c = (3*beta)/((beta - 1)*(beta**2 - 2*beta + 1))
    d = (beta**2*l*(beta - 3))/((beta - 1)*(beta**2 - 2*beta + 1))
    e = (l**2*(2*beta - 1))/(2*(beta**3 - 3*beta**2 + 3*beta - 1))

    def poly(x):
        return a*x**4 + b*x**3 + c*x**2 + d*x + e
    
    """
    # Uncomment to verify twice continuous
    x0 = torch.tensor(float(l))
    x1 = torch.tensor(float(l*beta))
    
    print(poly(x0), x0*l)
    print(poly(x1), (beta + 1)*l**2/2)
    dpoly = ft.grad(poly)
    ddpoly = ft.grad(dpoly)
    print(dpoly(x0), l)
    print(dpoly(x1), 0)

    print(ddpoly(x0), 0)
    print(ddpoly(x1), 0)
    """

    def reg_fn(x):
        id0 = x < l
        id1 = torch.logical_and(x >= l, x < beta*l)
        id2 = x >= beta*l

        out = torch.zeros_like(x)
        out[id0] = l*x[id0]
        out[id1] = poly(x[id1])
        out[id2] = (beta+1)*l**2/2
        return torch.sum(out)
    
    return reg_fn

def get_NNMF_obj(Y, loss_type, reg_type, l, shape, **kwargs):
    """
    function which returns the appropriate NNMF objective. 
    """
    n, d, r = shape

    def frob_loss(x):
        X = unvector(x, n, d, r)
        W, H = WHfromX(X, n)

        Yhat = W@H
        R = Y - Yhat

        return 0.5*torch.linalg.norm(R, ord="fro")**2/(n*d)
    
    def cosine_loss(x):
        X = unvector(x, n, d, r)
        W, H = WHfromX(X, n)

        Yhat = W@H

        return (1/n)*torch.sum(1 - torch.cosine_similarity(Y, Yhat, eps=1e-16))

    # Select loss
    if loss_type=="FROB":
        loss = frob_loss
    elif loss_type=="COSINE":
        loss = cosine_loss
    else:
        raise ValueError("Invalid loss function.")

    # Select regularisation. 
    if reg_type=="L2":
        reg_fn = lambda x: 0.5*l*torch.linalg.norm(x)**2

    elif reg_type=="WELSH":
        if "M" not in kwargs.keys():
            raise ValueError("Please include a M parameter.")
        else:
            M = kwargs["M"]

        def welsh(x):
            xdash = 1 - torch.exp(-x**2/(2*M**2))
            return l*torch.sum(xdash)
        
        reg_fn = welsh
    
    elif reg_type=="TSCAD":
        # provide a beta paramter if TSCAD is to be used
        if "a" not in kwargs.keys():
            raise ValueError("Please include a beta kw parameter.")
        else:
            a = kwargs["a"]

        if a <= 1:
            raise ValueError("Choose beta > 1 for valid regularisation.")
        
        reg_fn = TSCAD(l, a)

    elif reg_type is None:
        reg_fn = lambda x : 0
    else:
        raise ValueError("Invalid regularisation.")
    
    return loss, reg_fn


class NNMF():
    """
    Non-negative matrix factorisation oracle based on autodifferentiation.
    """

    def __init__(self, Y, r, loss="FROB", reg=None, l=0.) -> None:        
        # Returns a function which evaluates the loss and regularisation.
        self.loss = loss
        self.reg = reg

        self.r = r
        self.l = l
        self.Y = Y
        self.n, self.d = Y.shape
        self.shape = (self.n, self.d, self.r)

    def get_oracle(self, **kwargs):

        loss, reg = get_NNMF_obj(self.Y, self.loss, self.reg, self.l, self.shape, **kwargs)

        self.loss = loss # for function evaluations directly

        # build the objective
        def obj(x):
            return loss(x) + reg(x)

        g_and_f = ft.grad_and_value(obj)

        def g(x):
            return g_and_f(x)[0]
        
        def oracle(x, vals="fgH"):
            if vals=="f":
                fval = obj(x)
                return (fval, None, None)

            if vals=="fg":
                gval, fval = g_and_f(x)
                return (fval, gval, None)

            if vals=="fgH":
                
                # forward over reverse differentiation
                #def hvp(v):
                #    return ft.jvp(g, (x,), (v,))[1]

                # reverse over reverse differentiation. 
                gval, _hvp, fval  = ft.vjp(g_and_f, x, has_aux=True)

                def hvp(v):
                    return _hvp(v)[0] 

                return (fval, gval, hvp)
            
        return oracle
