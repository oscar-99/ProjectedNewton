from solvers.inner_solvers import cg
from solvers.Optimizer import OptimizerNonNeg
import utils.utils as utils

import torch
import math

class NewtonCGAlternatingSteps(OptimizerNonNeg):
    """
    The "alternating CG" variant from https://arxiv.org/pdf/2103.15989.pdf. Takes projected gradient steps until active set optimality. Takes Newton-CG steps once active set optimal.

    PARAMETERS: 
        obj : The objective oracle. Evaluated at x returns a tuple of the form (f, g, Hv), respectively the objective value, gradient and Hessian 
        acc : the relative error tolerance for the residual.
        alpha0 : an initial step size. 
        rho: a line search reduction tolerance.
        eta : line search decrease parameter
        eH : the Hessian dampening paramter. By default is sqrt(eg).

        ##### SUPER CLASS kwargs #####
        eg : The first order termination tolerance. 
        e_active : the active set tolerance (defaults to sqrt(eg) if not provided).
        device : the device to run on.
        max_iters : maximum iterations before automatic termination.
        oracle_termination : If True terminate when max_oracles reached.
        verbose : if True print progression.
        folder : the file location where the results will be saved if save_results is true. 
        save_results : if True save the results. 
        max_oracles : Maximum number of oracle calls before termination if oracle termination is True.
        ops_weighting : the relative weighting of function evaluations, gradient evaluations and Hessian vector products.  
    """

    NAME="NewtonCGAlt"

    def __init__(self, obj, acc=None, alpha0=1, rho=1e-4, eta=0.5, eH=None, **kwargs) -> None:

        super().__init__(obj, **kwargs)

        if acc == None:
            raise ValueError("Provide a acc paramter (cg residual tolerance for termination).")

        if eH is None:
            self.eH = math.sqrt(self.eg)
        else:
            self.eH = eH

        self.max_LS_iters=utils.max_ls_iters(eta, alpha0)

        self.acc = acc
        self.alpha0=alpha0
        self.rho=rho
        self.eta=eta

        # record some important paramter settings
        self.results.results["settings"]["acc"] = acc
        self.results.results["settings"]["alpha0"] = alpha0
        self.results.results["settings"]["rho"] = rho
        self.results.results["settings"]["LS factor"] = eta

    def step(self) -> bool:
        # Evaluate oracles.
        self.fk, self.gk, self.Hvk = self.obj(self.xk)
        self.g_count += 1 # get the function evaluation for free technically due to line search

        Ak = torch.where(self.xk < self.e_act, 1, 0)
        Ik = 1 - Ak
        self.num_active_constraints = torch.sum(Ak).item()
        self.num_inactive_constraints = torch.sum(Ik).item()

        gk_active = self.gk*Ak
        xk_active = self.xk*Ak
        gk_inactive = self.gk*Ik
        Hvk_inactive = utils.Hv_masked(Ik, self.Hvk)

        self.gk_inactive_norm = torch.linalg.norm(gk_inactive).item()
        self.gkxk_active_norm = torch.linalg.norm(gk_active*xk_active).item()
        self.min_gk_active = torch.min(gk_active).item()
        self.active_min_grad= -(min(self.min_gk_active, 0))

        # Alternating steps between gradient and Newton-CG
        if self.num_active_constraints > 0 and (self.min_gk_active < -math.sqrt(self.eg) or self.gkxk_active_norm > self.eg):
            self.alpha = self.alpha0
            self.Dtype = "GRAD"
            # LINESEARCH
            i=0
            while i < self.max_LS_iters:
                Pxk = utils.Proj(self.xk - self.alpha*self.gk)
                fPxk, _, _ = self.obj(Pxk, vals="f")
                self.f_count += 1
                
                reduction = (1/2)*torch.dot(self.gk, Pxk - self.xk)

                if fPxk < self.fk + reduction:
                    break

                self.alpha *= self.eta
                i+=1
            
            self.xk = Pxk 
            return False
        
        elif self.num_inactive_constraints > 0 and (self.gk_inactive_norm > self.eg):
            # Now use CG solve in the inactive subspace 
            d, Hd, self.Dtype, Hvs = cg(Hvk_inactive, gk_inactive, self.eH, self.acc, device=self.device)
            self.Hv_count += Hvs

            if self.Dtype == "NPC": 
                d = d/torch.linalg.norm(d)
                dHd = torch.dot(d, Hd)
                pk_inactive = -sign(torch.dot(d, gk_inactive))*torch.abs(dHd)*d

            elif self.Dtype == "SOL":
                pk_inactive = d

            # LINESEARCH
            self.alpha = self.alpha0
            pk_inactive_norm = torch.linalg.norm(pk_inactive)
            i=0
            while i < self.max_LS_iters:
                Pxk = utils.Proj(self.xk + self.alpha*pk_inactive)
                fPxk, _, _ = self.obj(Pxk, vals="f")
                self.f_count += 1
                reduction = -self.rho*math.sqrt(self.eg)*self.alpha**2*pk_inactive_norm**2

                if fPxk < self.fk + reduction:
                    break

                self.alpha *= self.eta
                i+=1
            
            self.xk = utils.Proj(self.xk + self.alpha*pk_inactive)  
            return False
        
        else:
            return True
        

def sign(x):
    """
    Tie broken sign function.
    """
    if x >= 0:
        return 1
    else:
        return -1