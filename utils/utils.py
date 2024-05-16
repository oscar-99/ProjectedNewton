import torch
import numpy as np
import math

import json
import os
import datetime

class Results:
    """
    A results class wrapping a dict which stores and updates the results of iterations.
    """
    FOLDER = os.path.join(os.getcwd(), "results")

    def __init__(self, folder) -> None:
        # Results dictionary has a reserved key for the paramter settings
        self.results = {"settings" : {}}
        self.save_location = os.path.join(self.FOLDER, folder)

    def update(self, **kwargs):
        """
        This should be called only once per iteration.
        """
        for k, v in kwargs.items():
            if k not in self.results.keys():
                self.results[k] = []
                
            self.results[k].append(v)

    def save(self, id):
        name = id + ".json"
        with open(os.path.join(self.save_location, name), "w") as f:
            json.dump(self.results, f, indent=4)

    def load(folder, file):
        res = Results(folder)
        with open(os.path.join(folder, file), "r") as f:
            res.results = json.load(f)

        return res

def Proj(x):
    """
    Project onto the non negative orthant
    """
    EPS = 1e-16 # positivity tolerance 
    # indices of active set 
    I = torch.where(x<EPS, 0.0, 1.)
    return x*I


def prox_NN(x, t):
    """
    Nonnegative constraint prox functions. 
    """
    # t is dummy paramter so that the number of arguments is correct
    return torch.clamp(x, min=0.0)

def h_NN(x):
    """
    Indicator for the nonnegativity constraint.
    """
    # Return a dummy value as projection handles this. 
    return 0.0
    
def build_small_PD_diag(device=None):
    """
    Small positive definite diagonal matrix.
    """
    A = torch.diag(torch.array([1, 2, 3, 4]), device=device)
    xstar = torch.ones((4,), device=device)

    b = A@xstar 

    return A, b, xstar

def build_symmetric_matrix_problem(N, generator, mineval=1, maxeval=100, device=None):
    """
    Generate a symmetric matrix. 
    """
    d = torch.linspace(mineval, maxeval, N)
    D = torch.diag(d)

    # Generate a random orthogonal matrix
    U = random_orthogonal(N, generator=generator, device=device)
    A = U@D@U.T # A is symmetric.

    xstar = torch.randn((N,), generator=generator, device=device)
    b = A@xstar 

    return A, b, xstar

def build_symmetric_matrix_from_diag(d, generator=None, device=None):
    """
    Builds a symmetric matrix from a diagonal vector. 
    """
    N = len(d)
    D = torch.diag(d)
    U = random_orthogonal(N, device=device)
    A = U@D@U.T

    return A

def random_orthogonal(n, generator=None, device=None):
    """
    Generate a random orthogonal matrix.
    """
    A = torch.randn((n, n), generator=generator, device=device)
    U, _, _ = torch.linalg.svd(A)
    return U

def process_pytorch_image_dataset(data, device=None):
    """
    Process a dataset into a torch tensor. 
    """
    X = []
    y = []
    for image, label in data:
        # intermediate numpy representation
        image_int = np.array(image).flatten()
        X.append(image_int)
        y.append(label)
    
    X = np.array(X) # supposedly numpy conversion first is faster.
    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device)

    return X, y

def load_adults_data(download_again=False, device=None):
    """
    Downloads and preprocess the adults dataset. Once complete dataset is stored as a csv. Adults has imbalanced classes. 
    """
    from ucimlrepo import fetch_ucirepo
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    path = os.path.join("datasets", "adults")
    file = os.path.join(path, "adult.csv")

    if len(os.listdir(path)) == 0 or download_again:
        print("Downloading and storing dataset")
        adult = fetch_ucirepo(id=2)
        data = adult.data.original
        data.to_csv(file, index=False)

    data = pd.read_csv(file)
    data = data.replace("?", np.nan)
    data = data.dropna()

    # get labels and data.
    y = data["income"]
    X = data.drop("income", axis=1)
    pd.set_option("display.max_columns", None)

    # one hot encode categories.
    X = pd.get_dummies(X)
    X = X.replace([True, False], [1, 0])

    # process labels.
    y = y.replace([">50K", "<=50K"], [">50K.", "<=50K."])
    y = pd.get_dummies(y, drop_first=True)
    y = y.replace([True, False], [1, 0])
    y = y.values
    y = torch.tensor(y, device=device).flatten()

    # scale data including categoricals (this is done in HTF)
    scaler = StandardScaler()
    X = X.values
    X = scaler.fit_transform(X)
    X = torch.tensor(X, device=device)
    return X, y

def load_covertype_data(download_again=False, device=None):
    """
    Loads and processes the covertype data.
    """
    from ucimlrepo import fetch_ucirepo
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    path = os.path.join("datasets", "covtype")
    file = os.path.join(path, "covtype.csv")

    if len(os.listdir(path)) == 0 or download_again:
        print("Downloading and storing dataset.")
        covtype = fetch_ucirepo(id=31)
        data = covtype.data.original
        data.to_csv(file, index=False)

    # apparently no missing values which makes this fairly simple 
    data = pd.read_csv(file)
    X = data.drop("Cover_Type", axis=1).values
    y = data["Cover_Type"].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device).flatten() - 1

    return X, y

def max_ls_iters(theta, alpha0, min_step=1e-20):
    """
    Computes an appropriate number of line search iterations for a given reduction amount.
    """
    
    # roughly want 
    # alpha0*(theta)**n = min_step
    return math.ceil(math.log(min_step/alpha0)/math.log(theta)) + 1

def clip_zeros(l, eps=1e-16):
    """
    Clip the zeros in a list for log plotting.
    """
    return [max(eps, i) for i in l]

def Hv_masked(mask, Hv):
    """
    Produces a Hessian vector product function restricted to the index set specified by mask. Mask is a vector with 1s in the "active" indices and 0s elsewhere. 
    """
    return lambda v : (mask*Hv(v*mask))

def timestamp():
    """
    Creates a time stamp as a string.
    """
    date = datetime.datetime.now()
    time = date.strftime("%H%M")
    ts = "_{}_{}_{}_{}".format(time, date.day, date.month, date.year)
    return ts

def create_folder(example_name, ts=True):
    """
    Handles the creation of a time stamped folder with the given name. 
    """
    if ts:
        folder_name= example_name+timestamp()
    else:
        folder_name = example_name

    folder = os.path.join(Results.FOLDER, folder_name)
    
    if folder_name not in os.listdir(Results.FOLDER):
        os.mkdir(folder)

    return folder
