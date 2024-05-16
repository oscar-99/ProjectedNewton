import unittest
import torch

from solvers.inner_solvers import minres, cg
import utils.utils as utils

# Run some rudimentary tests on the inner solvers.
torch.set_default_dtype(torch.float64)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SEED = 0

class test_minres(unittest.TestCase):
    test_eps = 1e-8
    
    def test_small_diag_matrix(self):
        """
        Tests the system solver on a small diagonal PD system.
        """
        eta = 1e-4
        A = torch.diag(torch.tensor([1., 2., 3., 4.]))
        xstar = torch.ones((4,))
        b = A@xstar 
        Av = lambda v : A@v

        x, Dtype, _ = minres(Av, -b, eta)

        # At the moment just using defaults tolerance
        bhat = Av(x)
        r = bhat - b
        self.assertEqual(Dtype, "SOL")
        self.assertTrue(inexactness_test(A, b, x, eta))

        # in this case can test residual directly using lower bound on Hr and upper bound on Hx
        self.assertTrue(torch.linalg.norm(r) < eta*4*torch.linalg.norm(x))       

    def test_small_diag_npc(self):
        """
        Tests whether MINRES detects the NPC direction in a small diagonal matrix a manual test verifies that x^TAx < 0. 
        """
        eta = 0 # 0 tolerance should mean we dont terminate early
        A = torch.diag(torch.tensor([1., 2., -1., 3.]))
        Av = lambda v: A@v
        xstar = torch.ones((4,))

        b = A@xstar

        x, Dtype, _ = minres(Av, b, eta)

        self.assertEqual(Dtype, "NPC")
        self.assertTrue(torch.dot(x, A@x).item() < 0)

    def test_large_PD_matrix(self):
        """
        Tests some random large PD matrices with known spectrum. Test passes if the inexact decrease condition ||Ar || <= eta ||A x|| is met by manually verifying. 
        """
        N = 100
        eta = 1e-4

        g = torch.Generator()
        g.manual_seed(0)
        A, b, xstar = utils.build_symmetric_matrix_problem(N, g, mineval=1, maxeval=100)
        Av = lambda v: A@v

        x, Dtype, _ = minres(Av, -b, eta)

        self.assertEqual(Dtype, "SOL")
        self.assertTrue(inexactness_test(A, b, x, eta))

    def test_large_NPC_matrix(self):
        """
        Tests some random large matrix with one negative eigenvalue. Test passes if the Dtrpe is NPC and a manual test verifies that x^TAx < 0.
        """
        N = 100
        eta = 1e-4

        g = torch.Generator()
        g.manual_seed(0)
        A, b, xstar = utils.build_symmetric_matrix_problem(N, g, mineval=-1, maxeval=100)
        Av = lambda v: A@v

        xstar = torch.randn((N,), generator=g)
        b = A@xstar 

        x, Dtype, _ = minres(Av, b, eta)

        self.assertEqual(Dtype, "NPC")
        self.assertTrue((torch.dot(x, A@x)) < 0)

    def test_large_PSD_matrix(self): 
        """
        Tests large matrix which is almost rank deficient. Solution vector b is random normal and so b is not in Range(A).

        MINRES passes the test with the 0 eigenvalues if EPS=1e-6.
        """
        N = 100
        etas = [1e-2, 1e-4] # allows for mix of SOL and NPC

        g = torch.Generator()
        g.manual_seed(0)

        d = torch.linspace(1, 10, N-2)
        d = torch.concatenate([d, torch.tensor([-1e-6, -1e-6])])
        A = utils.build_symmetric_matrix_from_diag(d, g)
        Av = lambda v: A@v

        for j in range(100):
            eta = etas[j%2]
            b = torch.randn((N,), generator=g)
            b = b/torch.linalg.norm(b)

            x, Dtype, _ = minres(Av, -b, eta)        
            self.assertTrue(Dtype == "NPC" or "SOL")

            if Dtype =="NPC":
                self.assertTrue(torch.dot(x, A@x).item() < 1e-6)

            else:
                r = A@x-b
                self.assertTrue(inexactness_test(A, b, x, eta))


class test_cg(unittest.TestCase):
    test_eps = 1e-6

    def test_small_diag_matrix(self):
        """
        Tests the system solver on a small diagonal PD system.
        """
        acc = 1e-4
        A = torch.diag(torch.tensor([1., 2., 3., 4.]))
        xstar = torch.ones((4,))
        b = A@xstar 
        Av = lambda v: A@v

        x, Ax, Dtype, _ = cg(Av, -b, self.test_eps, acc)

        # At the moment just using defaults tolerance
        bhat = Av(x)
        r = bhat - b
        self.assertEqual(Dtype, "SOL")
        self.assertTrue(residual_test(A, b, x, acc=0.1))

        self.assertTrue(torch.norm(Ax - bhat).item() < acc)
                
        # in this case can test residual directly using lower bound on Hr and upper bound on Hx

    def test_small_diag_npc(self):
        """
        Tests whether CG detects the NPC direction in a small diagonal matrix a manual test verifies that x^TAx < 0. 
        """
        eps = 1e-6
        acc = 1e-4 # small tolerance should mean we dont terminate early
        A = torch.diag(torch.tensor([1., 2., -1., 3.]))
        Av = lambda v: A@v
        xstar = torch.ones((4, ))

        b = A@xstar

        x, Ax, Dtype, _ = cg(Av, -b, eps, acc)

        self.assertEqual(Dtype, "NPC")
        self.assertTrue(torch.dot(x, A@x).item() < 0)

        self.assertTrue(torch.norm(Ax - A@x).item() < acc)

    def test_large_PD_matrix(self):
        """
        Tests some random large PD matrices with known spectrum. Test passes if the inexact decrease condition ||Ar || <= eta ||A x|| is met by manually verifying. 
        """
        N = 100
        acc = 1e-4

        g = torch.Generator()
        g.manual_seed(0)
        A, b, xstar = utils.build_symmetric_matrix_problem(N, g, mineval=1, maxeval=100)
        Av = lambda v: A@v

        x, Ax, Dtype, _ = cg(Av, -b, self.test_eps, acc)

        self.assertEqual(Dtype, "SOL")
        self.assertTrue(residual_test(A, b, x, acc=0.1))
        self.assertTrue(torch.norm(Ax - A@x).item() < acc)

    def test_large_NPC_matrix(self):
        """
        Tests some random large matrix with one negative eigenvalue. Test passes if the Dtrpe is NPC and a manual test verifies that x^TAx < 0.
        """
        N = 100
        acc = 1e-4

        g = torch.Generator()
        g.manual_seed(0)
        A, b, xstar = utils.build_symmetric_matrix_problem(N, g, mineval=-1, maxeval=100)
        Av = lambda v: A@v

        xstar = torch.randn((N,), generator=g)
        b = A@xstar 

        x, Ax, Dtype, _ = cg(Av, -b, self.test_eps, acc)

        self.assertEqual(Dtype, "NPC")
        self.assertTrue(torch.dot(x, A@x).item() < 0)
        self.assertTrue(torch.norm(Ax - A@x).item() < acc)

    def test_large_PSD_matrix(self): 
        """
        Tests a 100 random large matrices which is almost rank deficient. Solution vector b is random normal and so b is not in Range(A).

        MINRES passes the test with the 0 eigenvalues if EPS=1e-6.
        """
        N = 100
        accs = [1e-2, 1e-4] # allows for mix of SOL and NPC?

        g = torch.Generator()
        g.manual_seed(0)

        # CG 
        d = torch.linspace(1, 10, N-2)
        d = torch.concatenate([d, torch.tensor([-1e-4, -1e-4])])
        A = utils.build_symmetric_matrix_from_diag(d, generator=g)
        Av = lambda v: A@v

        for j in range(100):
            acc = accs[j%2]
            b = torch.randn((N,), generator=g)

            b = b/torch.linalg.norm(b)

            x, Ax, Dtype, _ = cg(Av, -b, self.test_eps, acc)   
                 
            self.assertTrue(Dtype == "NPC" or "SOL")
            self.assertTrue(torch.norm(Ax - A@x).item() < acc)

            if Dtype =="NPC":
                self.assertTrue(torch.dot(x, A@x) < 1e-6)

            else:
                self.assertTrue(residual_test(A, b, x, acc=0.1))

class test_autodiff_mlp(unittest.TestCase):

    def test_hvp_mlp(self):
        from problems.problems_L1 import FunctionalNN, MLPClassifier
        import utils.utils as utils
        import math

        import torch
        import torch.nn
        from torch.nn.utils import parameters_to_vector, vector_to_parameters
        from torchvision.datasets import FashionMNIST

        torch.manual_seed(0)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Setup data
        dataset = FashionMNIST("./datasets", download=True) 
        X, y = utils.process_pytorch_image_dataset(dataset, device=device)
        X = X/torch.max(X)
        n, data_dim = X.shape

        # Get MLP oracle
        MLP = MLPClassifier(data_dim, l1_shape=100, l2_shape=100, out_classes=10)
        mlp = FunctionalNN(MLP, device=device)
        mlp_obj_rr, x0 = mlp.net_oracle(X, y, mode="rr")
        mlp_obj_fr, _ = mlp.net_oracle(X, y, mode="fr")
        d = x0.shape[0]

        test_eps = 1e-8

        for i in range(10):
            x = torch.rand_like(x0)

            fval_rr, gval_rr, hvp_rr = mlp_obj_rr(x)
            fval_fr, gval_fr, hvp_fr = mlp_obj_fr(x)

            self.assertTrue(abs(fval_fr - fval_rr) < test_eps)
            self.assertTrue(torch.linalg.norm(gval_rr - gval_fr) < test_eps)

            for j in range(10):
                v = torch.randn_like(x0)
                self.assertTrue(torch.linalg.norm(hvp_rr(v) - hvp_fr(v)) < test_eps)


class test_L1_oracle(unittest.TestCase):
    """
    Manually verify the L1 oracle is working as intended. 
    """

    def test_logistic(self):
        from problems.problems_L1 import LogisticRegression, L1ProxObj, L1RegularisationDiff, build_autodiff
        from torchvision.datasets import MNIST

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        eps=1e-12

        dataset = MNIST("./datasets", download=True) 
        X, y = utils.process_pytorch_image_dataset(dataset, device=device)
        y  = y % 2
        X = X/torch.max(X)

        n, d = X.size()

        logistic = LogisticRegression(device=device)
        obj = logistic.mle_logistic_oracle(X, y, 0)
        l = 1e-3 # L1 regularisation parameter

        # Differentiable formulation of the L1 regularisation
        L1Reg = L1RegularisationDiff(device=device)
        diff_L1 = L1Reg.get_objective(obj, d+1, l, include_constant=True)

        def f_test(x):
            return obj(x)[0]

        # Verify we get the same results when evaluating the L1 oracle and manual L1.
        for i in range(10):
            x0 = torch.randn(d+1, device=device)
            z0 = L1Reg.gen_z(x0)

            # Verify z is the negative part
            self.assertTrue(torch.linalg.norm((z0[:d+1] - z0[d+1:]) - x0) < eps)

            fval_L1, _, _ = diff_L1(z0)

            fval, _, _ = obj(x0)

            manual_reg = l*torch.sum(torch.abs(x0[:d])) # Last parameter should be bias masked as it corresponds to the constant

            # Check functon values match
            self.assertTrue(torch.abs(fval + manual_reg - fval_L1) < eps)

        # Check derivatives are correct against autodiff
        for i in range(5):
            x = torch.randn(d+1, device=device)
            _, gval, hvp = obj(x)

            oracle_auto = build_autodiff(f_test)
            _, gval_auto, hvp_auto = oracle_auto(x)

            self.assertTrue(torch.linalg.norm(gval - gval_auto) < eps)

            for i in range(5):
                v = torch.randn(d+1, device=device)

                self.assertTrue(torch.linalg.norm(hvp(v)-hvp_auto(v))< eps)
            

        # Sanity check if constant is masked properly evaluating [0,...,0, 1] results in same value i.e. no L1 penalty.
        x1 = torch.zeros(d+1, device=device)
        x1[-1] = 1
        z1 = L1Reg.gen_z(x1)
        
        self.assertTrue(z1[d].item() == 1.) # (d+1)th entry should be 1

        fval_L1, _, _ = diff_L1(z1)
        fval, _, _ = obj(x1)

        self.assertTrue(torch.abs(fval_L1 - fval) < eps)

    def test_softmax(self):
        from problems.problems_L1 import L1RegularisationDiff, build_autodiff,MultinomialRegression, L1ProxObj
        import utils.utils as utils

        import math
        import torch
        from torchvision.datasets import CIFAR10

        # Double precision.
        torch.set_default_dtype(torch.float64)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Load in and preprocess data
        dataset = CIFAR10("./datasets", download=True) 
        X, y = utils.process_pytorch_image_dataset(dataset, device=device)
        X = torch.Tensor(X).to(device)
        y = torch.Tensor(y).to(device)
        # normalise images to [0, 1] scale
        X = X/torch.max(X)
        n, d = X.size()

        # Setup objective function
        d += 1 # for constant term
        classes = 10
        problems_dim = d*(classes-1)

        # Set up multinomial regression oracle
        multi_regression = MultinomialRegression(classes, device=device)
        multi_oracle = multi_regression.get_oracle(X, y, l=0)
        bias_mask = multi_regression.get_bias_mask(X)

        eps = 1e-12

        def f_test(x):
            return multi_oracle(x)[0]

        # Check derivatives are correct against autodiff
        for i in range(5):
            x = torch.randn(problems_dim, device=device)
            _, gval, hvp = multi_oracle(x)

            oracle_auto = build_autodiff(f_test)
            _, gval_auto, hvp_auto = oracle_auto(x)

            self.assertTrue(torch.linalg.norm(gval - gval_auto) < eps)

            for i in range(5):
                v = torch.randn(problems_dim, device=device)

                self.assertTrue(torch.linalg.norm(hvp(v)-hvp_auto(v))< eps)

        # test whether L1 formulations match

        l = 1e-5 # L1 regularisation parameter
        # Differentiable reformulation
        L1Reg = L1RegularisationDiff(device=device)
        diff_L1 = L1Reg.get_objective(multi_oracle, problems_dim, l, include_constant=True, bias_mask=bias_mask)

        # test bias mask
        for i in range(10):
            x0 = torch.randn(problems_dim, device=device)
            bias = 1 - bias_mask

            x_non_bias = x0*bias_mask
            z_non_bias = L1Reg.gen_z(x_non_bias)
            fval_L1_non_bias, _, _ = diff_L1(z_non_bias)
            fval_non_bias, _, _  = multi_oracle(x_non_bias)

            # manually compute check the penalties match in case of no bias terms.
            self.assertTrue(torch.abs(fval_non_bias + l*torch.sum(torch.abs(x_non_bias)) - fval_L1_non_bias) < eps)

            # on the other hand if we only have the bias function evaluations will directly match.
            x_bias = x0*bias
            z_bias = L1Reg.gen_z(x_bias)
            fval_L1_bias, _, _ = diff_L1(z_bias)
            fval_bias, _, _  = multi_oracle(x_bias)        
            self.assertTrue(torch.abs(fval_L1_bias - fval_bias)<eps)

    
def inexactness_test(A, b, x, eta):
    r = A@x-b
    return torch.linalg.norm(A@r) <= eta*torch.linalg.norm(A@x)

def residual_test(A, b, x, acc=0.1):
    r = A@x-b
    return torch.linalg.norm(r) <= acc*torch.linalg.norm(b)



if __name__ == '__main__':
    unittest.main()