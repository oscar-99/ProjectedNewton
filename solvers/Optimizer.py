from utils.utils import Results
import time
import math

class Optimizer:
    """
    ABC for an optimiser.
    """
    NAME = None

    def __init__(self, obj, device=None, max_iters=1000, verbose=False, folder=None, save_results=True, oracle_termination=False, max_oracles=1e6, ops_weighting=(1, 1, 2)) -> None:
        self.obj = obj
        self.f_count = 0
        self.g_count = 0
        self.Hv_count = 0
        self.max_iters=max_iters
        self.max_oracles = max_oracles
        self.oracle_termination = oracle_termination
        self.ops_weighting = ops_weighting

        self.device = device
        self.verbose = verbose
        self.folder = folder

        self.is_small = []

        
        self.save_results = save_results
        if save_results:
            if folder is None:
                raise ValueError("To save results provide a folder.")
            
            self.results = Results(folder)

        self.results.results["name"] = self.NAME
        # save parameter settings.
        self.results.results["settings"]["max_iters"] = max_iters
        self.results.results["settings"]["max_oracles"] = max_oracles
        self.results.results["settings"]["ops_weighting"] = ops_weighting
        self.results.results["settings"]["oracle termination"] = oracle_termination

    def step(self) -> bool:
        """
        Compute a single step of the algorithm. Returns true if the algorithm has converged to a tolerance
        """
        pass

    def run(self, x0):
        """
        Runs the optimisation algorithm
        """
        self.xk = x0
        
        t0 = time.time()
        self.t = 0.0

        self.it = 0 
        self.flag = "RUNNING"
        while self.it < self.max_iters and self.flag=="RUNNING":
            # store ops and time at beginning of iteration.
            self.t = time.time() - t0 
            self.ops = (self.f_count, self.g_count, self.Hv_count) 
            if self.verbose and self.it % 100 == 0:
                print("Iteration k =",self.it)

            converged = self.step()

            if converged:
                # These won't be computed by step if we terminate
                self.alpha = None
                self.Dtype = None
                self.flag = "TOL"

            else:
                if self.small_step_test():
                    self.alpha = None
                    self.Dtype = None
                    self.flag="SmallStepSize"

                # If ops at start over limit terimate iteration after populating
                if self.oracle_termination:
                    if self.oracle_termination_test(self.ops):
                        self.alpha = None
                        self.Dtype = None
                        self.flag = "MaxOracleCalls"    

            self.update_results()
            self.it += 1

        else:
            # Hit maximum iteration tolerance
            if self.flag=="RUNNING":
                self.flag = "MAXITERS"

        self.results.results["termination flag"] = self.flag
        print("{} terminated in {} iterations ({:.2f} sec) with flag {}.".format(self.NAME, self.it, self.t, self.flag))

        if self.save_results:
            self.save()

        return self.xk, self.flag
    
    def save(self):
        self.results.save(self.NAME)

    def save_val(self, label, val):
        self.results.results[label] = val
        self.save()

    def small_step_test(self):
        """
        Terminate the iteration if step size is too small for many iterations in a row.
        """
        SMALL = 1e-12
        
        if self.alpha < SMALL:
            self.is_small.append(True)
        else:
            self.is_small.append(False) 
        
        if len(self.is_small) > 5:
            self.is_small.pop(0)
        
        # If last 5 steps have been small 
        if len(self.is_small)==5 and all(self.is_small):
            return True
        else:
            return False
        
    def oracle_termination_test(self, ops):
        """
        Returns true if max_oracles is exceeded.
        """
        f_weight, g_weight, Hv_weight = self.ops_weighting
        f_c, g_c, Hv_c = ops
        oracle_count = f_c*f_weight + g_c*g_weight + Hv_c*Hv_weight

        if oracle_count > self.max_oracles:
            return True
        
        else:
            return False
        
    def update_results(self):
        """
        Updates the results dictionary for the current iteration.
        """
        pass
        

class OptimizerNonNeg(Optimizer):
    """
    ABC for a nonnegative optimiser.
    """

    def __init__(self, obj, eg=None, e_act=None, **kwargs) -> None:
        super().__init__(obj, **kwargs)

        if eg is None:
            raise ValueError("No optimality tolerance supplied.")
        self.eg=eg

        if e_act == None:
            self.e_act = math.sqrt(self.eg)
        else:
            self.e_act=e_act

        # record some important paramter settings
        self.results.results["settings"]["eg"] = eg
        self.results.results["settings"]["e_act"] = e_act
        
    def update_results(self):
        """
        Updates the results dictionary for the current iteration.
        """
        self.results.update(ops=self.ops,
                            t = self.t,
                            alpha=self.alpha,
                            Dtype=self.Dtype, 
                            f=self.fk.item(), 
                            inactive_grad_norm=self.gk_inactive_norm, active_min_grad=self.active_min_grad, active_gx_norm=self.gkxk_active_norm,
                            num_active_constraints=self.num_active_constraints)
        

        