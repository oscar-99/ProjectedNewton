import utils.utils as utils
from solvers.Optimizer import OptimizerNonNeg, Optimizer

import torch
import math


class FISTA(Optimizer):
    """
    Optimises a function of the form F = f + h using the FISTA algorithm. Where f is a smooth part of the function and h is a nonsmooth part, e.g., an L1 regularisation term.

    The reference for this section is [Chapter 10.7, First-Order Methods in Optimization, Amir Beck].

    Termination condition is a maximum number of oracle calls reached or a difference in function values which is too small. Line search is reset at each iteration.

    PARAMETERS: 
        obj : The objective oracle for f. Evaluating at x returns a tuple of the form (f, g, Hv), respectively the objective value, gradient and Hv. Hv is not necessary
        prox : the prox operator for the nonsmooth part, h.
        h : the nonsmooth part of the function e.g. the regularisation term.
        alpha0 : an initial step size. 
        eta : line search backtracking parameter.
        tol : termination parameter for subsequent function values.

        ##### SUPER CLASS kwargs #####
        device : the device to run on.
        max_iters : maximum iterations before automatic termination.
        oracle_termination : If True terminate when max_oracles reached.
        verbose : if True print progression.
        folder : the file location where the results will be saved if save_results is true. 
        save_results : if True save the results. 
        max_oracles : Maximum number of oracle calls before termination if oracle termination is True.
        ops_weighting : the relative weighting of function evaluations, gradient evaluations and Hessian vector products.  
    """
    NAME="FISTA"

    def __init__(self, obj, prox, h, alpha0=1, eta=0.5, tol=1e-6, **kwargs) -> None:

        super().__init__(obj, **kwargs)

        self.prox = prox
        self.h = h
        self.max_LS_iters=utils.max_ls_iters(eta, alpha0)
        self.alpha0=alpha0
        self.eta = eta
        self.tol = tol

        self.results.results["settings"]["alpha0"] = alpha0
        self.results.results["settings"]["LS factor"] = eta
        self.results.results["settings"]["termination tol"] = tol

    def step(self):
        # initialise on first iteration
        if self.it == 0:
            self.tk = 1.
            self.yk = self.xk
            self.Fk = torch.inf

        self.last_tk = self.tk
        self.last_xk = self.xk
        self.last_Fk = self.Fk

        # Evaluate oracle calls for step
        self.fyk, self.gyk, _ = self.obj(self.yk, vals="fg") 
        self.f_count += 1 # no free function evaluation.
        self.g_count += 1

        # Evaluate orcale for termination/objective tracking dont count this. (could get it from line search)
        self.fk, _, _ = self.obj(self.xk, vals="f") 
        # Include the regularisation term as the oracle for this method does not include this evaluation.
        self.Fk = self.fk + self.h(self.xk)

        # no other good termination condition
        if torch.abs(self.Fk - self.last_Fk) > self.tol:
            self.alpha = self.alpha0

            i = 0
            while i < self.max_LS_iters:
                # Line search on the smooth part of the function
                # compute the prox
                xdash = self.prox(self.yk - self.alpha*self.gyk, self.alpha)

                fdash, _, _ = self.obj(xdash, vals="f")
                self.f_count += 1

                d = xdash - self.yk
                dnorm = torch.linalg.norm(d)
                
                if fdash <= self.fyk + torch.dot(self.gyk, d) + (1/(2*self.alpha))*dnorm**2:
                    # update xk sequence for computed step size
                    self.xk=xdash
                    break

                self.alpha *= self.eta
                i += 1

            # Compute the update.
            self.tk = (1 + math.sqrt(1 + 4*self.tk**2))/2
            self.yk = self.xk + ((self.last_tk-1)/self.tk)*(self.xk - self.last_xk)

        else:
            return True 
    
    def update_results(self):
        self.results.update(ops=self.ops,
                            t = self.t,
                            alpha=self.alpha,
                            f = self.Fk.item(),
                            Dtype = None,
                            )
        
class ProximalGradientMomentum(Optimizer):
    """
    Optimises a function of the form F = f + h using proximal gradient with momumtum. See [Algorithm 4.1, Accelerated optimization for machine learning].

    PARAMTERS:
        obj : The objective oracle for f. Evaluating at x returns a tuple of the form (f, g, Hv), respectively the objective value, gradient and Hv. Hv is not necessary
        prox : the prox operator for the nonsmooth part, h.
        h : the nonsmooth part of the function e.g. the regularisation term.
        alpha : Fixed step size.
        beta : the momentum term.
        tol : termination parameter for subsequent function values.

        ##### SUPER CLASS kwargs #####
        device : the device to run on.
        max_iters : maximum iterations before automatic termination.
        oracle_termination : If True terminate when max_oracles reached.
        verbose : if True print progression.
        folder : the file location where the results will be saved if save_results is true. 
        save_results : if True save the results. 
        max_oracles : Maximum number of oracle calls before termination if oracle termination is True.
        ops_weighting : the relative weighting of function evaluations, gradient evaluations and Hessian vector products.  
    """
    NAME ="ProxGradMomentum"

    def __init__(self, obj, prox, h, beta=0.9, alpha=1, tol=1e-6, **kwargs) -> None:
        super().__init__(obj, **kwargs)

        self.prox = prox
        self.h = h
        self.alpha = alpha
        self.beta = beta
        self.tol = tol

        self.results.results["settings"]["alpha"] = alpha
        self.results.results["settings"]["beta"] = beta
        self.results.results["settings"]["termination tol"] = tol

    def step(self):
        # We flip the notation from the source x <-> y
        if self.it == 0:
            self.Fk = torch.inf
            self.yk = self.xk

        self.last_yk = self.yk
        self.last_Fk = self.Fk
        # Evaluate oracle for steo requires forward and backwards pass
        # Don't include the function evaluation because we could technically get it from the previous iteration.
        self.fk, self.gk, _ = self.obj(self.xk, vals="fg") 
        self.g_count += 1

        self.Fk = self.fk + self.h(self.xk)

        # Terminate if function changes are too small.
        if torch.abs(self.Fk - self.last_Fk) > self.tol:

            # No line search, fixed step size method.
            self.yk = self.prox(self.xk - self.alpha*self.gk, self.alpha)
            vk = self.yk + self.beta*(self.yk - self.last_yk)

            fyk, _, _ = self.obj(self.yk, vals="f") 
            Fyk = fyk + self.h(self.yk)
            self.f_count += 1

            fvk, _, _ = self.obj(vk, vals="f")
            Fvk = fvk + self.h(vk)
            self.f_count += 1
            
            # update xk sequence for computed step size
            if Fyk <= Fvk:
                self.xk=self.yk
            else:
                self.xk=vk

        else:
            return True 

    def update_results(self):
        self.results.update(ops=self.ops,
                            t = self.t,
                            alpha=self.alpha,
                            f = self.Fk.item(),
                            Dtype = None,
                            )

class ProjectedGradientDescent(OptimizerNonNeg):
    """
    Standard projected gradient descent with line search.

    PARAMETERS: 
        obj : The objective oracle for f. Evaluating at x returns a tuple of the form (f, g, Hv), respectively the objective value, gradient and Hv. Hv is not necessary.
        alpha0 : an initial step size. 
        rho : line search sufficient decrease paramter.
        eta : line search backtracking parameter.

        ##### SUPER CLASS kwargs #####
        device : the device to run on.
        max_iters : maximum iterations before automatic termination.
        oracle_termination : If True terminate when max_oracles reached.
        verbose : if True print progression.
        folder : the file location where the results will be saved if save_results is true. 
        save_results : if True save the results. 
        max_oracles : Maximum number of oracle calls before termination if oracle termination is True.
        ops_weighting : the relative weighting of function evaluations, gradient evaluations and Hessian vector products.
    """
    NAME = "ProjGrad"

    def __init__(self, obj, alpha0=1, rho=1e-4, eta=0.5, **kwargs) -> None:
        super().__init__(obj, **kwargs)

        self.max_LS_iters=utils.max_ls_iters(eta, alpha0)

        self.alpha0=alpha0
        self.rho=rho
        self.eta = eta

        self.results.results["settings"]["alpha0"] = alpha0
        self.results.results["settings"]["rho"] = rho
        self.results.results["settings"]["LS factor"] = eta

        self.Dtype=None # no Dtype is outputed as there is no inner solver

    def step(self):
        # Evaluate oracles.
        self.fk, self.gk, _ = self.obj(self.xk, vals="fg")
        self.g_count += 1 # technically function evaluation for free from line search.

        # Update index sets and approximately active constraints.
        Ak = torch.where(self.xk < self.e_act, 1, 0)

        Ik = 1 - Ak
        self.num_active_constraints = torch.sum(Ak).item()

        gk_active = self.gk*Ak
        xk_active = self.xk*Ak
        gk_inactive = self.gk*Ik

        self.gk_inactive_norm = torch.linalg.norm(gk_inactive).item()
        self.gkxk_active_norm = torch.linalg.norm(gk_active*xk_active).item()
        self.min_gk_active = torch.min(gk_active).item()
        self.active_min_grad= -(min(self.min_gk_active, 0)) # absolute value of most negative active gradient
        
        if self.min_gk_active < -math.sqrt(self.eg) or self.gkxk_active_norm > self.eg or self.gk_inactive_norm > self.eg:
            self.alpha = self.alpha0
            # LINESEARCH
            i=0
            while i < self.max_LS_iters:
                Pxk = utils.Proj(self.xk - self.alpha*self.gk)
                fPxk, _, _ = self.obj(Pxk, vals="f")
                self.f_count += 1
                
                reduction = self.rho*torch.dot(self.gk, Pxk - self.xk)

                if fPxk <= self.fk + reduction:
                    break

                self.alpha *= self.eta
                i+=1
            
            self.xk = utils.Proj(self.xk - self.alpha*self.gk)  
            return False

        else:
            return True


