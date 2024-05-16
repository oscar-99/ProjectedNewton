from solvers.inner_solvers import minres
from solvers.Optimizer import OptimizerNonNeg
import utils.utils as utils

import torch
import math

class NewtonMRTwoMetricOptimal(OptimizerNonNeg):
    """
    A class for the two metric projection newton-MR algorithm. This is the optimal version of the Newton-MR two metric projection.

    The main difference from the standard version is eliminating the step in the active set once active set optimality is achieved.

    PARAMETERS: 
        obj : The objective oracle. Evaluated at x returns a tuple of the form (f, g, Hv), respectively the objective value, gradient and Hessian vector product function.
        theta : tolerance for the MR system solve.
        alpha0 : an initial step size. 
        rho: a line search reduction tolerance.
        eta : line search decrease parameter
        forward_tracking : If true line search will forawd track from alpha=1. Can be useful due to the NPC step case. 
        bertsekas_active_set : uses the Bertsekas active set.
        active_grad_scaling : applies MR scaling to the active gradient portion.

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

    NAME = "NewtonMRTMP"

    def __init__(self, obj, theta=None, alpha0=1, rho=1e-4, eta=0.5, bertsekas_active_set=False, active_grad_scaling=False, forward_tracking=True, **kwargs) -> None:
        # handles the more generic parameters and initialises results
        super().__init__(obj, **kwargs)

        if theta == None:
            raise ValueError("Provide a theta paramter (minres residual tolerance).")

        self.max_LS_iters=utils.max_ls_iters(eta, alpha0)

        self.theta=theta
        self.alpha0=alpha0
        self.rho=rho
        self.eta=eta
        self.forward_tracking = forward_tracking
        self.bertsekas_active_set=bertsekas_active_set
        self.active_grad_scaling=active_grad_scaling

        # record some important paramter settings
        self.results.results["settings"]["theta"] = theta
        self.results.results["settings"]["alpha0"] = alpha0
        self.results.results["settings"]["rho"] = rho
        self.results.results["settings"]["LS factor"] = eta
        self.results.results["settings"]["bertsekas_active_set"] = bertsekas_active_set
        self.results.results["settings"]["active_grad_scaling"] = active_grad_scaling
        self.results.results["settings"]["forward_tracking_enabled"] = forward_tracking

    def step(self):
        # Evaluate oracles.
        self.fk, self.gk, self.Hvk = self.obj(self.xk)
        self.g_count += 1 # get function value for free due to line search

        # Update index sets and approximately active constraints.
        if self.bertsekas_active_set:
            Ak = torch.where((self.xk <= self.e_act), 1, 0)*torch.where((self.gk > 0), 1, 0)
        else:
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
        self.active_min_grad= -(min(self.min_gk_active, 0)) # absolute value of most negative active gradient
        
        if self.min_gk_active < -math.sqrt(self.eg) or self.gkxk_active_norm > self.eg or self.gk_inactive_norm > self.eg:
                
            # COMPUTE ACTIVE STEP
            # for this version of the algorithm the type of active set depends on which termination condition is unsatisfied.
            if self.num_active_constraints > 0 and (self.min_gk_active < -math.sqrt(self.eg) or self.gkxk_active_norm > self.eg):
                pk_active = -gk_active
            else:
                pk_active=torch.zeros_like(gk_active)
            
            if self.num_inactive_constraints > 0:
                pk_inactive, self.Dtype, Hvs = minres(Hvk_inactive, gk_inactive, math.sqrt(self.eg)*self.theta, device=self.device)
            else:
                pk_inactive = torch.zeros_like(gk_inactive, device=self.device)
                self.Dtype = "SOL"
                Hvs = 0

            if self.Dtype != "SOL" and self.Dtype != "NPC":
                raise ValueError("Unknown inner solver flag: " + self.Dtype)

            pk = pk_inactive + pk_active
            self.Hv_count += Hvs

            self.alpha = self.alpha0
            # LINESEARCH
            i=0
            Pxk = utils.Proj(self.xk + self.alpha*pk)
            # Get just objective
            fPxk, _, _ = self.obj(Pxk, vals="f")
            self.f_count += 1
            reduction = self.rho*(torch.dot(gk_active, Pxk - self.xk) + self.alpha*torch.dot(gk_inactive, pk_inactive))

            if fPxk >= self.fk + reduction:
                self.alpha *= self.eta
                i+=1
                while i < self.max_LS_iters:
                    Pxk = utils.Proj(self.xk + self.alpha*pk)
                    # Get just objective
                    fPxk, _, _ = self.obj(Pxk, vals="f")
                    self.f_count += 1
                    
                    reduction = self.rho*(torch.dot(gk_active, Pxk - self.xk) + self.alpha*torch.dot(gk_inactive, pk_inactive))

                    if fPxk < self.fk + reduction:
                        break

                    self.alpha *= self.eta
                    i+=1
            else:
                if self.forward_tracking and self.Dtype=="NPC":
                    self.alpha *= 1/self.eta # this will increase step size

                    while i < self.max_LS_iters:
                        Pxk = utils.Proj(self.xk + self.alpha*pk)
                        # Get just objective
                        fPxk, _, _ = self.obj(Pxk, vals="f")
                        self.f_count += 1

                        reduction = self.rho*(torch.dot(gk_active, Pxk - self.xk) + self.alpha*torch.dot(gk_inactive, pk_inactive))

                        # If LS is unsatisfied terminate the iteration with the last alpha.
                        if fPxk >= self.fk + reduction:
                            self.alpha = self.alpha*self.eta
                            break

                        self.alpha *= 1/self.eta 
                        i += 1
            
            self.xk = utils.Proj(self.xk + self.alpha*pk)  
            return False

        else:
            return True    


