EXAMPLE_NAME="NNMF_text_cosine"

def run(save_folder=None):
    import torch
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from problems.problems_NNMF import NNMF, XfromWH, WHfromX, unvector, vector
    import utils.utils as utils
    import math

    import os

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
    
    # Load data
    data_path = os.path.join("datasets", "news20")
    news20 = fetch_20newsgroups(subset="all", data_home=data_path, remove=('headers', 'footers', 'quotes'))

    X, y = news20.data, news20.target
    y = torch.tensor(y, device=device)

    n_features = 1000
    vectorizer = TfidfVectorizer(max_features=n_features)
    X = vectorizer.fit_transform(X)

    X = X.toarray()
    Y = torch.tensor(X, device=device) # Y already between 0-1
    
    # still some all zero rows, mostly empty messages/single word messages.
    not_equal_zero_indices = []
    for yi in Y:
        not_equal_zero_indices.append(~torch.all(yi == torch.zeros_like(yi)))

    Y = Y[not_equal_zero_indices]
    y = y[not_equal_zero_indices]

    n, d = Y.shape
    print("data dimension", (n, d))
    r = 20

    nnmf = NNMF(Y, r, loss="COSINE", reg=None, l=0)
    oracle = nnmf.get_oracle()

    # Initialisation
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
    max_oracles = 1e6
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
        # save the r x d to see top words.
        X = unvector(x, n, d, r)
        W, H = WHfromX(X, n)

        # cpu + float32 + list format
        H = H.cpu()
        H = H.to(dtype=torch.float32)
        H = H.tolist()
        sparsity = 100*(torch.sum(x==0.0)/torch.numel(x))
        loss = nnmf.loss(x)
        print("sparsity", sparsity.item())
        print("loss", loss.item())

        method.save_val("Representation", H)
        method.save_val("sparsity", sparsity.item())
        method.save_val("loss", loss.item()) 

    for x, m in zip(sols, methods):
        save_additional_results(x, m)