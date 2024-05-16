
EXAMPLE_NAME="L1_logistic_MNIST"

def run(save_folder=None):
    from problems.problems_L1 import L1RegularisationDiff, LogisticRegression
   
    import torch
    import utils.utils as utils
    import math
    from problems.problems_L1 import L1RegularisationDiff, LogisticRegression, L1ProxObj

    from torchvision.datasets import MNIST

    # Double precision.
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Folder for storage. 
    if save_folder is None:
        folder = utils.create_folder(EXAMPLE_NAME)
    else:
        folder = utils.create_folder(save_folder, ts=False)
    print("Results will be saved in: ")
    print(folder)

    # Data importation and processing.
    dataset = MNIST("./datasets", download=True) 
    X, y = utils.process_pytorch_image_dataset(dataset, device=device)
    y  = y % 2
    X = X/torch.max(X)

    n, d = X.size()
    print("Running mnist logistic regression")
    print("(n, d) data matrix", (n, d))
    d += 1 # for constant term
    print("Problem dimension d=", d)

    # Build objective function.
    logistic = LogisticRegression(device=device)
    obj = logistic.mle_logistic_oracle(X, y, 0)
    l = 1e-3 # L1 regularisation parameter

    # Differentiable formulation of the L1 regularisation
    L1Reg = L1RegularisationDiff(device=device)
    diff_L1 = L1Reg.get_objective(obj, d, l, include_constant=True)

    # Prox formulation of objective
    L1prox = L1ProxObj(l, d, device=device, include_constant=True)
    prox_L1 = L1prox.get_prox_L1()
    penalty_L1 = L1prox.get_L1_reg()

    # Generate initial point
    x0 = torch.zeros(d, device=device)
    z0 = L1Reg.gen_z(x0) # diffable objection needs positive and negative part

    # Parameter values
    # Termination conditions
    FISTA_tol = 1e-8 
    eg = 1e-8
    e_act = math.sqrt(eg)
    max_iters = 1e7
    max_oracles = 1e5
    ops_weighting = (1, 1, 4)
    oracle_termination = True
    verbose = False

    # Line search paramters 
    eta = 0.5 
    rho = 1e-4
    rho_cg = 0.2
    alpha0 = 1
    
    # Inner solver tolerances
    theta = 1e-2/math.sqrt(eg)
    acc = 0.5

    # Import and run each of the methods we are testing.
    from solvers.ProjectedNewtonMR import NewtonMRTwoMetricOptimal
    from solvers.GradientMethods import ProjectedGradientDescent, FISTA
    from solvers.ProjectedNewtonCG import NewtonCGAlternatingSteps

    # A dry run helps with "warmup" for timing results. Other methods are already "prewarmed" by the MR run.
    # With no dry run first iteration of MR takes excessive amount of time due to pytorch prewarming. This doesn't seem to affect the other runs as much for some reason.
    newton_mr_dry = NewtonMRTwoMetricOptimal(diff_L1, theta=theta, alpha0=alpha0, e_act=e_act, eg=eg, rho=rho, eta=eta, verbose=verbose, folder=folder, device=device, max_iters=5, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    _, _ = newton_mr_dry.run(z0)

    newton_mr = NewtonMRTwoMetricOptimal(diff_L1, theta=theta, alpha0=alpha0, e_act=e_act, eg=eg, rho=rho, eta=eta, verbose=verbose, folder=folder, device=device, max_iters=max_iters, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    z_mr, FLAG_MR = newton_mr.run(z0)

    fista = FISTA(obj, prox_L1, penalty_L1, alpha0=alpha0, eta=eta, tol=FISTA_tol, max_iters=max_iters, verbose=verbose, folder=folder, device=device, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    x_f, FLAG_f = fista.run(x0)

    proj_grad = ProjectedGradientDescent(diff_L1, alpha0=alpha0, rho=rho, eta=eta, eg=eg, e_act=e_act, max_iters=max_iters, verbose=verbose, folder=folder, device=device, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    z_pg, FLAG_PG = proj_grad.run(z0)

    newton_cg_alt = NewtonCGAlternatingSteps(diff_L1, acc=acc, alpha0=alpha0, eg=eg, e_act=e_act, rho=rho_cg, eta=eta, max_iters=max_iters, verbose=verbose, folder=folder, device=device, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    z_cg_alt, flag_cg_alt = newton_cg_alt.run(z0)

    # Recover the paramters and record the accuracy/sparsity of the solutions 
    methods = [newton_mr, fista, proj_grad, newton_cg_alt]
    
    # diffable objectives converted back.
    x_mr = L1Reg.gen_x(z_mr, d)
    x_cg = L1Reg.gen_x(z_cg_alt, d)
    x_pg = L1Reg.gen_x(z_pg, d)
    sols = [x_mr, x_f, x_pg, x_cg]

    for x, m in zip(sols, methods):
        m.save_val("Train accuracy", logistic.accuracy(X, y, x))
        m.save_val("Sparsity", (100*torch.sum(x==0.0)/(d-1)).item())