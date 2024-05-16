import torch.func as ft
import torch
import math

def minres(Hv, g, eta, debug=False, device=None):
    """
    Implementation of a MINRES solver with NPC detection.
    
    Solves the least squares problem min ||Hx + g||^2 where A is symmetric but not necessarily positive definite. Includes non positive curvature direction detection.

    PARAMETERS:
        Hv : a Hessian vector product function which computes $Hv$ for a symmetric hessian H.  
        g : a solution vector.
        eta : Inexactness tolerance for the ||Hr_k|| <= eta||Hs_k|| condition

    RETURNS:
        (d, Dtype, Hv_count) : the search direction, solution type 'SOL' or 'NPC' and the number of Hv products.
        
    REFERENCES:
        - http://web.stanford.edu/group/SOL/software/minres/
        - https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/minres/minres.py
        - https://arxiv.org/pdf/2206.05732.pdf
    """
    # ZERO tolerance for calculation
    EPS = 1e-16

    # Initialize parameters 
    betak = torch.linalg.norm(g) # \beta_{k+1}
    rk = -g

    if betak < EPS:
        return (-g, "SOL", 0)

    # Initial basis vectors and search directions.
    # vk_old is v_{k-1}
    vk_old = torch.zeros_like(rk)
    vk = rk/betak
    dk_old = torch.zeros_like(rk)
    dk = torch.zeros_like(rk)
    xk = torch.zeros_like(rk)

    # Rotation parameters 
    ck = -1 
    sk = 0
    delta_1_k = 0
    epsilon_k = 0

    # RHS parameters
    phik = betak
    phi0 = phik
    tauk = 0

    # Modify this later
    k = 1
    max_iters = g.size()[0] + 1 # Shouldnt take more than g.size iterations.
    Hv_count = 0

    if debug:
        debug_state = {"Vk":[]}

    # Main solver loop 
    while k <= max_iters:
        # Generate new basis vector. 
        qk = Hv(vk)
        Hv_count+=1
        alphak = torch.dot(vk, qk).item()

        qk = qk - betak*vk_old - alphak*vk      
        betak = torch.linalg.norm(qk) # betak+1

        # Apply previous rotation.
        delta_2_k = ck*delta_1_k + sk*alphak 
        gamma_1_k = sk*delta_1_k - ck*alphak 

        epsilon_k_old = epsilon_k
        epsilon_k = sk*betak
        delta_1_k = -ck*betak

        # Test for non positive curvature.
        curvature = ck*gamma_1_k # Store the curvature.
        
        # Inexact test 
        # || H rk || <= eta ||H xk||
        if phik*torch.sqrt(gamma_1_k**2 + delta_1_k**2) < eta*torch.sqrt(phi0**2 - phik**2):
            if debug:
                return ((xk, "SOL", Hv_count), debug_state)
            #print("MINRES iters", k)
            #print("MINRES curvature", curvature)
            return (xk, "SOL", Hv_count)
        
        if curvature >= -EPS:
            #print("MINRES iters", k)
            #print("MINRES curvature", curvature)
            if debug:
                return ((rk, "NPC", Hv_count), debug_state)
                
            return (rk, "NPC", Hv_count)

        gamma_2_k = torch.sqrt(gamma_1_k**2 + betak**2)

        if gamma_2_k > EPS:
            # Update rotations 
            ck = gamma_1_k/gamma_2_k
            sk = betak/gamma_2_k

            # RHS
            tauk = ck*phik
            phik = sk*phik 

            # Search directions and iterates
            old_d_term = epsilon_k_old*dk_old
            dk_old = dk 
            dk = (vk - delta_2_k*dk - old_d_term)/gamma_2_k

            xk = xk + tauk*dk

            # Residual and basis vectors 
            if betak > EPS:
                vk_old = vk
                vk = qk/betak
                
                # update the lancosz vector list
                if debug:
                    debug_state["Vk"].append(vk)
                    debug_state["num iters"] = k

                rk = (sk**2)*rk - phik*ck*vk
                
        else:
            ck = 0
            sk = 1
            tauk = 0

        k += 1

    else:
        if debug:
                return ((xk, "MaxIter", Hv_count), debug_state)
        
        return (xk, "MaxIter", Hv_count)
    

def cg(Hv, g, eps, acc, M = 0, device=None):
    """
    Implementation of the capped conjugate gradient methods https://arxiv.org/pdf/1803.02924.pdf. Solves the problem:

        Hx = -g

    Note that for the problem to be solved H must be positive definite. If H is not positive definite a negative curvature direction of H will be returned.

    PARAMETERS:
        Hv : a Hessian vector product function which computes $Hv$ for a symmetric hessian H.    
        g : a RHS vector.
        eps : a damping parameter. 
        acc : desired relative accuracy.
        M : estimate of the norm of H, defaults to 0.
        device : the device the algorithm is being run on.

    RETURNS:
        d : The direction returned by the method. 
        dtype : "SOL" or "NPC" depending on whether a full solution or non positive curvature direction is detected.
    """
    n  = g.size()[0]
    max_iters = n + 1
    Hv_count=0

    # Hessian vector product 
    Hvb =  lambda v: Hv(v) + 2*eps*v # Hb = Hbar 
    normg = torch.linalg.norm(g)

    # Initialization of parameters 
    k = 0
    y = torch.zeros((n,), device=device)
    # Y = [y] # Storage of y vectors. necessary for last case which never triggers.
    r = g
    p = -g 
    rr = torch.dot(r, r)
    rrold = rr

    # Initialize matrix vector products and norms
    Hbp = Hvb(p) 

    Hv_count+=1
    Hbpold = Hbp

    Hp = Hbp - 2*eps*p 
    Hby = torch.zeros_like(y) # H0 = 0

    normr = torch.sqrt(rr)
    normp = torch.linalg.norm(p)
    normHp = torch.linalg.norm(Hp)
    pTHbp = torch.dot(p, Hbp)

    # Storage of alphas and residual norms
    residual_norms = [normr]
    alphas = []

    # Update M estimate 
    if normHp > M*normp:
        M = normHp/normp

    kappa, zetahat, tau, Tcap =  update_para(M, eps, acc)
    
    # Initial direction is non positive curvature.
    if pTHbp < eps*(normp**2):
        return (p, Hp, "NPC", Hv_count)

    while k < max_iters:
        # CG operations
        alpha = rr/pTHbp
        # alphas.append(alpha)
        
        y = y + alpha*p
        # Y.append(y) 
        Hby = Hby + alpha*Hbp # Hb y_{j+1} = Hb y_j + \alpha Hb p_j
        r = r + alpha*Hbp
        rrold = rr # ||rj||^2
        rr = torch.dot(r, r)
        beta = rr/rrold  
        p = -r + beta*p # update p
        
        Hbpold = Hbp 
        Hbp = Hvb(p)
        Hv_count+=1

        Hbr = beta*Hbpold - Hbp  # Hb r_{j+1} = beta Hb p_j - Hp_{j+1}
        pTHbp = torch.dot(p, Hbp)
        # CG iterations complete.
        k += 1
        
        # Handle matrix norm estimates.
        # Undo the dampening for each of the estimates.
        Hp  = Hbp - 2*eps*p
        Hy = Hby - 2*eps*y
        Hr = Hbr - 2*eps*r
        normHp  = torch.linalg.norm(Hp)
        normHy = torch.linalg.norm(Hy)
        normHr = torch.linalg.norm(Hr)

        normr = torch.sqrt(rr)
        # residual_norms.append(normr)
        normy = torch.linalg.norm(y)
        normp = torch.linalg.norm(p)

        # Update M if needed 
        if normHp  > M*normp:
            M = (normHp/normp).item()
            kappa, zetahat, tau, Tcap = update_para(M, eps, acc)

        if normHy > M*normy:
            M = (normHy/normy).item()
            kappa, zetahat, tau, Tcap = update_para(M, eps, acc)
        
        if normHr > M*normr:
            M = (normHr/normr).item()
            kappa, zetahat, tau, Tcap = update_para(M, eps, acc)

        # Now check for negative curvature conditions/termination conditions. 
        if torch.dot(y, Hby) < eps*normy**2:
            return (y, Hy, "NPC", Hv_count)

        # Check if relative residual criteria satisfied. 
        elif normr <= zetahat*normg:
            # in the constrained case need to verify ||r||_inf < c_mu
            return (y, Hy, "SOL", Hv_count)

        # Negative curvature detected.  
        elif pTHbp < eps*normp**2:
            return (p, Hp, "NPC", Hv_count)
        
        # Handle the case where residual is not decreasing sufficiently
        elif normr > math.sqrt(Tcap)*tau**(k/2)*normg:
            # never seen this condition trigger so raise an error so I can investigate
            # this negative curvature detection mechanism requires access to the iterates.
            raise ValueError("CG detected negative curvature due to insufficient decrease in residual.")
            print("CG terminated due to insufficient decrease in residual") # 
            """
            alpha = rr/pTHbp
            y = y + alpha*p

            for yi in Y:
                d = yi - y
                if torch.dot(d, Hvb(d))/torch.dot(d, d) < eps:
                    return (d, "NPC", Hv_count)
            # This is a faster option (no matrix vector products) but is hard to test/

        
            i = self.search_for_NPC(residual_norms, alphas, eps)
            yi = Y[i]
            d = y - yi
            self.results_dict["termination_flag"] = "NPC: residual decrease insufficient"
            return (d, "NPC")
            """
            
    else:
        return (y, "MaxIter", Hv_count)

def update_para(M, eps, acc):
    """
    Helper method for the CG capped CG method that updates the parameters as the norm estimate varies.
    """
    kappa = (M + 2*eps)/eps
    zetahat = acc/(3*kappa)
    tau = math.sqrt(kappa)/(math.sqrt(kappa) + 1)
    Tcap = 4*kappa**4/(1 - math.sqrt(tau))**2

    return kappa, zetahat, tau, Tcap