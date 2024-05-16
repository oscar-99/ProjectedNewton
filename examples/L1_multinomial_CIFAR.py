EXAMPLE_NAME="L1_multinomial_CIFAR10"

def run(save_folder=None):
    from problems.problems_L1 import L1RegularisationDiff, MultinomialRegression, L1ProxObj
    import utils.utils as utils

    import math
    import torch
    from torchvision.datasets import CIFAR10

    # Double precision.
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Folder for storage. 
    if save_folder is None:
        folder = utils.create_folder(EXAMPLE_NAME)
    else:
        folder = utils.create_folder(save_folder, ts=False)

    # Load in and preprocess data
    dataset = CIFAR10("./datasets", download=True) 
    X, y = utils.process_pytorch_image_dataset(dataset, device=device)
    X = torch.Tensor(X).to(device)
    y = torch.Tensor(y).to(device)
    # normalise images to [0, 1] scale
    X = X/torch.max(X)
    n, d = X.size()

    # Setup objective function
    print("(n, d) data matrix", (n, d))
    d += 1 # for constant term
    classes = 10
    problems_dim = d*(classes-1)
    print("Problem dimension: ", problems_dim)

    # Set up multinomial regression oracle
    multi_regression = MultinomialRegression(classes, device=device)
    multi_oracle = multi_regression.get_oracle(X, y, l=0)
    bias_mask = multi_regression.get_bias_mask(X)
    num_non_bias_params = torch.sum(bias_mask)
    l = 1e-5 # L1 regularisation parameter

    # Differentiable reformulation
    L1Reg = L1RegularisationDiff(device=device)
    diff_L1 = L1Reg.get_objective(multi_oracle, problems_dim, l, include_constant=True, bias_mask=bias_mask)
    print("L1 regularisation paramter l=", l)

    # Prox formulation
    L1prox = L1ProxObj(l, d, device=device, include_constant=True, bias_mask=bias_mask)
    prox_L1 = L1prox.get_prox_L1()
    penalty_L1 = L1prox.get_L1_reg()

    # Generate initial point
    x0 = torch.zeros(problems_dim, device=device)
    z0 = L1Reg.gen_z(x0)

    # Parameter values
    # Termination conditions
    FISTA_tol = 1e-8 
    eg = 1e-8
    e_act = math.sqrt(eg)
    max_iters = 1e7
    max_oracles = 1e6
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

    newton_mr = NewtonMRTwoMetricOptimal(diff_L1, theta=theta, alpha0=alpha0, e_act=e_act, eg=eg, rho=rho, eta=eta, verbose=verbose, folder=folder, device=device, max_iters=max_iters, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    z_mr, FLAG_MR = newton_mr.run(z0)

    fista = FISTA(multi_oracle, prox_L1, penalty_L1, alpha0=alpha0, eta=eta, tol=FISTA_tol, max_iters=max_iters, verbose=verbose, folder=folder, device=device, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    x_f, FLAG_f = fista.run(x0)

    proj_grad = ProjectedGradientDescent(diff_L1, alpha0=alpha0, rho=rho, eg=eg, e_act=e_act, max_iters=max_iters, verbose=verbose, folder=folder, device=device, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    z_pg, FLAG_PG = proj_grad.run(z0)

    newton_cg_alt = NewtonCGAlternatingSteps(diff_L1, acc=acc, alpha0=alpha0, eg=eg, e_act=e_act, rho=rho_cg, max_iters=max_iters, verbose=verbose, folder=folder, device=device, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    z_cg_alt, flag_cg_alt = newton_cg_alt.run(z0)

    # Recover the paramters and record the accuracy/sparsity of the solutions 
    methods = [newton_mr, fista, proj_grad, newton_cg_alt]

    # diffable objectives converted back.
    x_mr = L1Reg.gen_x(z_mr, problems_dim)
    x_cg = L1Reg.gen_x(z_cg_alt, problems_dim)
    x_pg = L1Reg.gen_x(z_pg, problems_dim)
    sols = [x_mr, x_f, x_pg, x_cg]

    for x, m in zip(sols, methods):
        m.save_val("Train accuracy", multi_regression.accuracy(X, y, x))
        m.save_val("Sparsity", (100*torch.sum(x==0.0)/(num_non_bias_params)).item())

