EXAMPLE_NAME="NNMF_image_noncvx"

def run(save_folder=None):
    import torch
    import utils.utils as utils
    import math
    import os
    from sklearn.datasets import fetch_olivetti_faces

    from problems.problems_NNMF import NNMF, XfromWH, WHfromX, unvector, vector
    
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Folder for saving results
    if save_folder is None:
        folder = utils.create_folder(EXAMPLE_NAME)
    else:
        folder = utils.create_folder(save_folder, ts=False)
    print("Results will be saved in: ")
    print(folder)
    
    # DATA LOADING
    data_path = os.path.join("datasets", "faces")

    # Load in dataset.
    Y, y = fetch_olivetti_faces(data_home=data_path, download_if_missing=True, return_X_y=True) 

    Y = torch.tensor(Y, device=device, dtype=torch.float64) # data already 0-1
    n, d = Y.shape

    print("Y data shape", (n, d))

    # rank 10 representation
    r = 10
    l = 1e-4 # regularisation paramter.

    nnmf = NNMF(Y, r, loss="FROB", reg="TSCAD", l=l)
    oracle = nnmf.get_oracle(a=3)

    W0 = torch.abs(torch.randn((n, r), device=device))
    H0 = torch.abs(torch.randn((r, d), device=device))
    Y0 = W0@H0
    max_Y0 = torch.sqrt(torch.max(Y0))

    # Ensure resulting text is on 0 to 1 scale.
    W0 = W0/max_Y0
    H0 = H0/max_Y0

    X0 = XfromWH(W0, H0)
    x0 = vector(X0, n, d, r)
    print("Opt problem dimension", x0.shape[0])

    # Parameter values
    # Termination conditions
    PGM_tol = 1e-8 
    eg = 1e-8
    e_act = math.sqrt(eg)
    max_iters = 1e7
    max_oracles = 1e5
    ops_weighting = (1, 1, 2)
    oracle_termination = True
    verbose = False

    # PGM settings
    beta = 0.9 # momentum term
    alpha = 1 # constant step size

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

    newton_mr = NewtonMRTwoMetricOptimal(oracle, theta=theta, alpha0=alpha0, e_act=e_act, eg=eg, rho=rho, eta=eta, verbose=verbose, folder=folder, device=device, max_iters=max_iters, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    x_mr, FLAG_MR = newton_mr.run(x0)

    pg_momentum = ProximalGradientMomentum(oracle, utils.prox_NN, utils.h_NN, beta=beta, alpha=alpha, tol=PGM_tol, max_iters=max_iters, verbose=verbose, folder=folder, device=device, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    x_pgm, FLAG_PGM = pg_momentum.run(x0)

    proj_grad = ProjectedGradientDescent(oracle, alpha0=alpha0, rho=rho, eg=eg, e_act=e_act, max_iters=max_iters, verbose=verbose, folder=folder, device=device, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    x_pg, FLAG_PG = proj_grad.run(x0)

    newton_cg = NewtonCGAlternatingSteps(oracle, acc=acc, alpha0=alpha0, eg=eg, e_act=e_act, rho=rho_cg, max_iters=max_iters, verbose=verbose, folder=folder, device=device, oracle_termination=oracle_termination, ops_weighting=ops_weighting, max_oracles=max_oracles)
    x_cg, flag_cg_alt = newton_cg.run(x0)

    methods = [newton_mr, pg_momentum, proj_grad, newton_cg]
    sols = [x_mr, x_pgm, x_pg, x_cg]

    def save_additional_results(x, method):
        # save the r x d representation for this example to see the faces!
        X = unvector(x, n, d, r)
        W, H = WHfromX(X, n)

        # cpu + float32 + list format
        H = H.cpu()
        H = H.to(dtype=torch.float32)
        H = H.tolist()

        sparsity = 100*(torch.sum(x ==0.0)/torch.numel(x))
        loss = nnmf.loss(x)
        print("sparsity", sparsity.item())
        print("loss", loss.item())

        method.save_val("Representation", H)
        method.save_val("sparsity", sparsity.item())
        method.save_val("loss", loss.item()) 

    for x, m in zip(sols, methods):
        save_additional_results(x, m)
    
    