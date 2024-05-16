EXAMPLE_NAME="L1_MLP_fashion"

def run(save_folder=None):
    from problems.problems_L1 import FunctionalNN, MLPClassifier, L1RegularisationDiff, L1ProxObj
    import utils.utils as utils
    import math

    import torch
    import torch.nn
    from torchvision.datasets import FashionMNIST

    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Running on", device)

    # Folder for saving results
    if save_folder is None:
        folder = utils.create_folder(EXAMPLE_NAME)
    else:
        folder = utils.create_folder(save_folder, ts=False)
    print("Results will be saved in: ")
    print(folder)
    
    # Setup data
    dataset = FashionMNIST("./datasets", download=True) 
    X, y = utils.process_pytorch_image_dataset(dataset, device=device)
    X = X/torch.max(X)
    n, data_dim = X.shape
    print("data dim", (n, data_dim))
    print("Class labels", torch.unique(y))

    # Get MLP oracle
    MLP = MLPClassifier(data_dim, l1_shape=100, l2_shape=100, out_classes=10)
    mlp = FunctionalNN(MLP, device=device)
    mlp_obj, x0 = mlp.net_oracle(X, y)
    d = x0.shape[0]
    bias_mask = mlp.get_bias_mask()
    num_non_bias_params = torch.sum(bias_mask)
    print("problem dimension", d)

    # Create L1 regularised objective
    l = 1e-3
    l1 = L1RegularisationDiff(device=device)
    diff_L1 = l1.get_objective(mlp_obj, d, l, include_constant=True, bias_mask=bias_mask) 
    z0 = l1.gen_z(x0)

    # Create L1 prox objective
    L1prox = L1ProxObj(l, d, device=device, include_constant=True, bias_mask=bias_mask)
    prox_L1 = L1prox.get_prox_L1()
    penalty_L1 = L1prox.get_L1_reg()

    # Parameter values
    # Termination conditions
    PGM_tol = 1e-8 
    eg = 1e-8
    e_act = math.sqrt(eg)
    max_iters = 1e7
    max_oracles = 1e6
    ops_weighting = (1, 1, 4)
    oracle_termination = True
    verbose = False

    # PGM settings
    beta = 0.9 # momentum term
    alpha = 1e-3 # constant step size

    # Line search paramters 
    eta = 0.5 
    rho = 1e-4
    rho_cg = 0.2
    alpha0 = 1

    # Inner solver tolerances
    theta = 1/math.sqrt(eg)
    acc = 0.5

    # Import and run each of the methods we are testing.
    from solvers.ProjectedNewtonMR import NewtonMRTwoMetricOptimal
    from solvers.GradientMethods import ProjectedGradientDescent, ProximalGradientMomentum
    from solvers.ProjectedNewtonCG import NewtonCGAlternatingSteps

    def save_acc_sparsity(x, m):
        acc = mlp.accuracy(x, X, y)
        print("method accuracy", acc)
        m.save_val("Train accuracy", acc)
        m.save_val("Sparsity", (100*torch.sum(x==0.0)/num_non_bias_params).item())

    newton_mr = NewtonMRTwoMetricOptimal(diff_L1, theta=theta, alpha0=alpha0, e_act=e_act, eg=eg, rho=rho, eta=eta, verbose=verbose, folder=folder, device=device, max_iters=max_iters, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    z_mr, FLAG_MR = newton_mr.run(z0)

    pg_momentum = ProximalGradientMomentum(mlp_obj, prox_L1, penalty_L1, beta=beta, alpha=alpha, tol=PGM_tol, max_iters=max_iters, verbose=verbose, folder=folder, device=device, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    x_pgm, FLAG_PGM = pg_momentum.run(x0)

    proj_grad = ProjectedGradientDescent(diff_L1, alpha0=alpha0, rho=rho, eg=eg, e_act=e_act, max_iters=max_iters, verbose=verbose, folder=folder, device=device, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    z_pg, FLAG_PG = proj_grad.run(z0)

    newton_cg = NewtonCGAlternatingSteps(diff_L1, acc=acc, alpha0=alpha0, eg=eg, e_act=e_act, rho=rho_cg, max_iters=max_iters, verbose=verbose, folder=folder, device=device, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    z_cg_alt, flag_cg_alt = newton_cg.run(z0)

    methods = [newton_mr, pg_momentum, proj_grad, newton_cg]

    # diffable objectives converted back.
    x_mr = l1.gen_x(z_mr, d)
    x_pg = l1.gen_x(z_pg, d)
    x_cg = l1.gen_x(z_cg_alt, d)
    sols = [x_mr, x_pgm, x_pg, x_cg]

    for x, m in zip(sols, methods):
        save_acc_sparsity(x, m)
