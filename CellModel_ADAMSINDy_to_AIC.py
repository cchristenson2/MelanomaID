import MelanomaModel as mm
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time
import scipy as scp

torch.manual_seed(1)

method = 'Radau'

model_idx = [0] #Sensitive cell state
dt = 0.05
T_SOLVE = 100
plot = True

protein_indices = [8,10,12,17]

def addToDataset(X,DX,P,X_new,DX_new,P_new):
    """Append a new experiment's cell states, derivatives, and protein signals to the dataset."""
    X   = torch.cat((X,X_new.unsqueeze(0)),dim=0)
    DX  = torch.cat((DX,DX_new.unsqueeze(0)),dim=0)
    P = torch.cat((P,P_new.unsqueeze(0)),dim=0)
    
    return X,DX,P

def POOL_DATA(X,P):
    """Build the candidate matrix by prepending a ones column and concatenating
    cell states and protein signals. Shape: (sets, T, 1+3+18)."""
    sets, n, _ = X.shape
    return torch.cat((torch.ones(sets,n,1),X,P),dim=2)

def ZERO_COEFF(l,threshold):
    """Hard-threshold: set any coefficient with |value| < threshold to exactly zero (in-place)."""
    with torch.no_grad():
        for elem in l:
            elem [torch.abs(elem) < threshold] = 0.0
    return l

def boundVar(varlist,bounds):
    """Clip all tensors in varlist to [bounds[0], bounds[1]] in-place. Used to keep
    nonlinear Hill constants in a physically meaningful range during training."""
    for elem in varlist:
        elem[elem < bounds[0]] = bounds[0]
        elem[elem > bounds[1]] = bounds[1]
    return varlist

def flattenTorchList(x):
    """Concatenate a list of 1-D tensors and flatten to a single 1-D tensor."""
    return torch.flatten(torch.cat(x))

def getLibrary(state):
    """Build the candidate function library for a given cell state (S, D, or R).

    Args:
        state: 'S', 'D', or 'R'

    Returns:
        dict with keys: 'terms' (names), 'prolif_hill_idx', 'lin_hill_idx',
        'term_types', 'numbers', 'self_propagate'

    Term types:
        0 = constant
        1 = proliferation-Hill: state * (1 - T/theta) * Hill(protein)  [self-propagating]
        2 = linear-Hill: S * Hill(|Δprotein|) * sigmoid(±Δprotein)     [transition term]
    """
    term_types     = []
    numbers        = []
    self_propagate = []
    
    states = ['S','D','R','T']
    proteins = ['iRAS','aRAS','iRAF_wt','aRAF_wt','RAF_m','iMEK','aMEK','iERK',
                'aERK','iPI3K','aPI3K','iAKT','aAKT','ATP','cAMP','iPKA',
                'aPKA','MITF']
    
    prolif_hill_idx = [] #type = 1
    lin_hill_idx    = [] #type = 2
    
    curr_prolif_hill = 0
    curr_lin_hill    = 0

    term_names = ['constant'] #type = 0
    term_types.append(0)
    numbers.append(0)
    self_propagate.append(True)
    
    if state == 'S' or state == 'R':
        for elem in proteins:
            term_names.append('Log('+state+')*Hill('+elem+')')
            term_types.append(1)
            numbers.append(curr_prolif_hill)
            curr_prolif_hill+=1
            self_propagate.append(True)
            prolif_hill_idx.append([states.index(state),states.index('T'),proteins.index(elem)])
            
        for elem in proteins:
            term_names.append('Lin('+'S'+')*Hill('+elem+')')
            term_types.append(2)
            numbers.append(curr_lin_hill)
            curr_lin_hill+=1
            if state == 'S':
                self_propagate.append(False)
            else:
                self_propagate.append(True)
            lin_hill_idx.append([states.index('S'),proteins.index(elem)])
            
    elif state == 'D':
        for elem in proteins:
            term_names.append('Lin('+state+')*Hill('+elem+')')
            term_types.append(2)
            numbers.append(curr_lin_hill)
            curr_lin_hill+=1
            self_propagate.append(True)
            lin_hill_idx.append([states.index('S'),proteins.index(elem)])
    
    #if no terms set to None
    if len(lin_hill_idx)==0:
        lin_hill_idx = None
    else:
        lin_hill_idx = np.array(lin_hill_idx)
    if len(prolif_hill_idx)==0:
        prolif_hill_idx = None
    else:
        prolif_hill_idx = np.array(prolif_hill_idx)
        
    return {'terms':term_names,'lin_hill_idx':lin_hill_idx,
            'prolif_hill_idx':prolif_hill_idx,'state':state,
            'term_types':term_types,'numbers':numbers,
            'self_propagate':self_propagate}

class Hill_term(torch.nn.Module):
    """Hill-style saturation term: x / (K + x). K is a learnable constant."""
    def __init__(self,K):
        super().__init__()
        self.K = K

    def forward(self, x):
        return x / (self.K + x)

class Log_term(torch.nn.Module):
    """Logistic growth term: x * (1 - y/theta). Implements density-dependent
    proliferation where y is total cell count and theta is the carrying capacity."""
    def __init__(self,theta):
        super().__init__()
        self.theta = theta

    def forward(self, x, y):
        return x*(1 - (y/self.theta))

class ADAM_SINDy_MODEL(torch.nn.Module):
    """SINDy model used during ADAM optimization. Evaluates the full candidate library
    with learnable coefficients (a) and nonlinear constants (K1, theta, K2).
    Masks terms with biologically incorrect sign to enforce known constraints.

    Args:
        a:               coefficient vector (learnable)
        K1:              Hill saturation constants for proliferation terms (learnable)
        theta:           carrying capacity for logistic growth terms (learnable)
        K2:              Hill saturation constants for transition terms (learnable)
        library_package: output of getLibrary for this state
        steady_state:    protein steady-state values; deviations drive transitions
    """
    def __init__(self,a,K1,theta,K2,library_package,steady_state):
        super().__init__()
        self.a  = a
        self.K1 = K1
        self.theta = theta
        self.K2 = K2
        
        self.prolif_hill = Hill_term(self.K1)
        self.prolif      = Log_term(self.theta)
        self.lin_hill    = Hill_term(self.K2)
        
        self.lib = library_package
        self.steady_state = steady_state
        
        self.sigmoid_sign = torch.tensor([-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,
                                          -1.0,1.0,1.0,-1.0,1.0,1.0,1.0,-1.0,
                                          1.0,1.0])
    
    def forward(self, candidates):
        con  = candidates[:,:,0].unsqueeze(2)
        x    = candidates[:,:,1:4] #order: S, D, R - add T
        proteins = candidates[:,:,4:]
        d_proteins = proteins - self.steady_state

        x = torch.cat((x, torch.sum(x,dim=2).unsqueeze(2)),dim=2)

        terms = con

        if isinstance(self.lib['prolif_hill_idx'],np.ndarray):
            temp_prolif  = self.prolif(x[:,:,self.lib['prolif_hill_idx'][:,0]],
                                      x[:,:,self.lib['prolif_hill_idx'][:,1]])
            temp_hill    = self.prolif_hill(proteins[:,:,self.lib['prolif_hill_idx'][:,2]])
            prolif_hill = temp_prolif * temp_hill
            terms = torch.cat((terms, prolif_hill),dim=2)
        if isinstance(self.lib['lin_hill_idx'],np.ndarray):
            temp_lin  = x[:,:,self.lib['lin_hill_idx'][:,0]]
            temp_hill = self.lin_hill(torch.abs(d_proteins[:,:,self.lib['lin_hill_idx'][:,1]]))
            temp_heaviside = self.sigmoid_sign*d_proteins[:,:,self.lib['lin_hill_idx'][:,1]]
            temp_heaviside = (temp_heaviside > 0).float()
            lin_hill  = temp_lin * temp_hill * temp_heaviside
            terms = torch.cat((terms, lin_hill),dim=2)   
        #Zero terms that correlate to biologically incorrect coefficients
        for i in range(terms.shape[2]):
            if self.lib['self_propagate'][i] == True:
                if self.a[i] < 0.0:
                    terms[:,:,i] = 0.0
            else:
                if self.a[i] > 0.0:
                    terms[:,:,i] = 0.0
        return terms @ self.a

class ADAM_SINDy_MODEL_permute(torch.nn.Module):
    """SINDy model used during AIC permutation search. Evaluates a specific subset
    (permutation) of library terms with fixed, refitted coefficients and nonlinear
    constants. Returns numpy output for use with scipy.optimize.least_squares."""
    def __init__(self,library_package,steady_state):
        super().__init__()
        
        self.lib = library_package
        self.steady_state = steady_state
        
        self.sigmoid_sign = torch.tensor([-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,
                                          -1.0,1.0,1.0,-1.0,1.0,1.0,1.0,-1.0,
                                          1.0,1.0],dtype=torch.float64)
    
    def forward(self, candidates, permutation, coeff, prolif_hill_k,
                prolif_hill_cc, lin_hill_k):
        con  = candidates[:,:,0].unsqueeze(2)
        x    = candidates[:,:,1:4] #order: S, D, R - add T
        proteins = candidates[:,:,4:]
        d_proteins = proteins - self.steady_state

        x = torch.cat((x, torch.sum(x,dim=2).unsqueeze(2)),dim=2)

        types   = self.lib['term_types']
        numbers = self.lib['numbers']

        if permutation[0] == 0:
            terms = con
            start = 1
        else:
            start = 0
            terms = torch.empty((con.shape[0],con.shape[1],0),device=device)

        prolif_hill_ind = 0
        lin_hill_ind    = 0

        for elem in permutation[start:]:
            curr_type = types[elem]
            ind       = numbers[elem]
            if curr_type == 1:
                term_prolif = Log_term(prolif_hill_cc)
                term_hill   = Hill_term(prolif_hill_k[prolif_hill_ind])

                temp_prolif  = term_prolif(x[:,:,self.lib['prolif_hill_idx'][ind,0]],
                                          x[:,:,self.lib['prolif_hill_idx'][ind,1]])
                temp_hill    = term_hill(proteins[:,:,self.lib['prolif_hill_idx'][ind,2]])
                prolif_hill = (temp_prolif * temp_hill).unsqueeze(2)
                terms = torch.cat((terms, prolif_hill),dim=2)
                prolif_hill_ind += 1

            elif curr_type == 2:
                term_hill = Hill_term(lin_hill_k[lin_hill_ind])

                temp_lin  = x[:,:,self.lib['lin_hill_idx'][ind,0]]
                temp_hill = term_hill(torch.abs(d_proteins[:,:,self.lib['lin_hill_idx'][ind,1]]))
                temp_heaviside = self.sigmoid_sign[self.lib['lin_hill_idx'][ind,1]]*\
                                     d_proteins[:,:,self.lib['lin_hill_idx'][ind,1]]
                temp_heaviside = (temp_heaviside > 0).double()
                lin_hill  = (temp_lin * temp_hill * temp_heaviside).unsqueeze(2)
                terms = torch.cat((terms, lin_hill),dim=2)
                lin_hill_ind += 1
        
        dx = terms @ coeff
        return dx.cpu().detach().numpy()
    
def getParamNum(permutation, term_types):
    """Count the total number of free parameters for a given term permutation.
    Constant terms cost 1. Proliferation-Hill (type 1) and linear-Hill (type 2)
    terms each cost 2 (coefficient + saturation constant K)."""
    k = 0
    for elem in permutation:
        if term_types[elem] == 0:
            k += 1
        elif term_types[elem] == 1:
            k += 2
        elif term_types[elem] == 2:
            k += 2
    return k

class TO_SOLVER():
    """Callable wrapper for ADAM_SINDy_MODEL_permute, used as the residual function
    for scipy.optimize.least_squares. Unpacks the flat parameter vector x into
    linear coefficients, proliferation-Hill constants, and linear-Hill constants,
    then returns residuals. The .final() method returns the model output (not residuals)."""
    def __init__(self,model,data,candidates,permutation,term_types):
        self.model = model
        self.data  = data
        self.candidates  = candidates
        self.permutation = permutation
        self.term_types  = term_types
    def __call__(self,x):
        n = len(x)
        try:
            n_ph = np.array(self.term_types[self.term_types==1]).size * 1
        except:
            n_ph = 0
        try:
            n_lh = np.array(self.term_types[self.term_types==2]).size * 1
        except:
            n_lh = 0
            
        l = n-(n_ph+n_lh)
        
        coeff = x[:l]
        if n_ph != 0:
            temp = x[l:l+n_ph].copy()
            prolif_hill_k  = temp
            
        else:
            prolif_hill_k = None
            
        if n_lh != 0:
            temp = x[l+n_ph:].copy()
            lin_hill_k = temp
        else:
            lin_hill_k = None

        output = self.model(self.candidates,self.permutation,coeff,
                            prolif_hill_k,1.0,lin_hill_k)
        
        return (self.data.cpu().detach().numpy() - output).reshape(-1)
    
    def final(self,x):
        n = len(x)
        try:
            n_ph = np.array(self.term_types[self.term_types==1]).size * 1
        except:
            n_ph = 0
        try:
            n_lh = np.array(self.term_types[self.term_types==2]).size * 1
        except:
            n_lh = 0
            
        l = n-(n_ph+n_lh)
        
        coeff = x[:l]
        if n_ph != 0:
            temp = x[l:l+n_ph].copy()
            prolif_hill_k  = temp
            
        else:
            prolif_hill_k = None
            
        if n_lh != 0:
            temp = x[l+n_ph:].copy()
            lin_hill_k = temp
        else:
            lin_hill_k = None

        output = self.model(self.candidates,self.permutation,coeff,
                            prolif_hill_k,1.0,lin_hill_k)
        return output

def isin_tolerance(A, B, tol):
    A = np.asarray(A)
    B = np.asarray(B)

    Bs = np.sort(B) # skip if already sorted
    idx = np.searchsorted(Bs, A)

    linvalid_mask = idx==len(B)
    idx[linvalid_mask] = len(B)-1
    lval = Bs[idx] - A
    lval[linvalid_mask] *=-1

    rinvalid_mask = idx==0
    idx1 = idx-1
    idx1[rinvalid_mask] = 0
    rval = A - Bs[idx1]
    rval[rinvalid_mask] *=-1
    return np.minimum(lval, rval) <= tol
    
generator   = np.random.default_rng(seed=1)
net_params  = mm.RandomizeMAPKParams(generator)
cell_params = mm.RandomizeCellParams(generator)

device = torch.device('cpu')
print("AVAILABLE PROCESSOR:", device, '\n')
print('Note: GPU is supported but may run slower due to complex candidate libraries')
###############################################################################
# Solve for MAPK steady state #################################################
# Run MAPKModel for T=200 with no drug to reach steady state from all-ones IC.
# Steady-state protein values are used as initial conditions for perturbation sims
# and as the reference point for computing protein deviations in CellModel.
T_temp = 100
t_temp  = np.arange(0,T_temp+dt,dt)
    
initial = np.ones(18,)
RAF_inhib = np.zeros((len(t_temp),4))
inputs = np.ones((len(t_temp),3))
Network = mm.MAPKModel(RAF_inhib, inputs, net_params, t_temp)
X_temp = scp.integrate.solve_ivp(Network, (0,T_temp), initial, t_eval = t_temp,
                                 method=method, rtol=10**(-12),
                                 atol=10**(-12)*np.ones_like(initial)).y.T
steady_state = X_temp[-1,:]

###############################################################################
# Build dataset ###############################################################
# For each perturbation: simulate MAPKModel at high resolution (dt=0.001) to
# get protein trajectories P, then simulate CellModel driven by P. Downsample
# both to the analysis dt, compute dX/dt via torch.gradient, and collect into
# tensors X_data (cells), P_data (proteins), dX_data (derivatives).
T = T_SOLVE
t  = torch.arange(0,T+dt,dt)
t_eval  = np.arange(0,T+0.001,0.001) #High resolution time data to downsample for model fitting

initial_cells = np.array([0.5,0.0,0.0])

on    = torch.ones((len(t),1))
empty = torch.zeros((len(t),1))
sigmoid = (torch.sigmoid(5.0*(t-10)) - torch.sigmoid(5.0*(t-40))).unsqueeze(1)

sigmoid_raf = torch.cat((sigmoid,empty,empty,empty),dim=1).detach().numpy() 

constant_inputs    = torch.cat((on,on,on),dim=1).detach().numpy()

perturbations = [(sigmoid_raf, constant_inputs),]

X_data = []; dX_data = []; P_data = [];
t  = np.arange(0,T+dt,dt)
for i, curr in enumerate(perturbations):
    inhib, inputs = curr
    
    Network = mm.MAPKModel(inhib, inputs, net_params, t)
    
    t_eval = np.arange(0,T+0.001,0.001)
    meas_indices = isin_tolerance(np.round(t_eval,decimals=4),np.round(t,decimals=4), 1e-5)
    
    inhib, inputs = curr
    
    Network = mm.MAPKModel(inhib, inputs, net_params, t)
    P = scp.integrate.solve_ivp(Network, (0,T), steady_state, t_eval = t_eval,
                                method='Radau', rtol=10**(-12),
                                atol=10**(-12)*np.ones_like(steady_state)).y.T
    P = P[meas_indices,:]
    
    t_eval = np.arange(0,T+0.001,0.001)
    meas_indices = isin_tolerance(np.round(t_eval,decimals=4),np.round(t,decimals=4), 1e-5)
    
    CellModel = mm.CellModel(steady_state[protein_indices], P[:,protein_indices], cell_params, t)
    out = scp.integrate.solve_ivp(CellModel, (0,T), initial_cells, t_eval = t_eval,
                                method='Radau', rtol=10**(-12),
                                atol=10**(-12)*np.ones_like(initial_cells))
    X = out.y.T
    X = X[meas_indices,:]
    
    X = torch.tensor(X,dtype=torch.float32)
    P = torch.tensor(P,dtype=torch.float32)
    
    if plot == True:
        mm.plotFullNetwork(mm.detachTorch(torch.tensor(t)), mm.detachTorch(P).T,
                           title='Protein Dataset '+str(i+1))
        
        mm.plotCells(mm.detachTorch(torch.tensor(t)), mm.detachTorch(X).T,
                           title='Cell Dataset '+str(i+1))
        
    dX_dt = torch.gradient(X, spacing = dt, dim = 0)[0]
    
    try:
        X_data,dX_data,P_data = addToDataset(X_data,dX_data,P_data,X,dX_dt,P)
    except:
        X_data  = X.unsqueeze(0)
        dX_data = dX_dt.unsqueeze(0)
        P_data  = P.unsqueeze(0)
    
###############################################################################
# Setup ADAM_SINDy Model ######################################################   
A_CANDIDATES = POOL_DATA(X_data, P_data).to(device)

###############################################################################
# Initialize ADAM_SINDy training ##############################################
# Two Adam optimizers run in parallel:
#   optim_COEFF_ADT: updates all coefficients and nonlinear constants (K values)
#   optim_weights:   updates the L1 sparsity weights (promotes coefficient dropout)
# Both use step-decay learning rate schedules. Nonlinear constants are clipped
# to [0.05, 5.0] after each step via boundVar.
Epochs              = 25000
lr                  = 1e-2
lr_sparsity         = 1e-3
step_epoch          = 5000
step_epoch_sparsity = 5000
decay_rate          = 0.50
tolerance           = 1e-4

all_model_names  = ['S','D','R']
model_names  = [all_model_names[i]  for i in model_idx]

optimized_params = []
coeff_list       = []
nonlin_list      = []

for i, elem in enumerate(model_names):
    temp=eval('getLibrary(model_names['+str(i)+'])')
    exec('COEFF_'+elem+'=torch.ones((len(temp[\'terms\']),),requires_grad=True,device=device)')
    exec('optimized_params.append(COEFF_'+elem+')')
    exec('coeff_list.append(COEFF_'+elem+')')
    
    if isinstance(temp['prolif_hill_idx'],np.ndarray):
        exec('PROLIF_HILL_CC_'+elem+'=torch.ones((len(temp[\'prolif_hill_idx\']),))')
        #Off to limit to only a single non-linear parameter optimized
        # exec('optimized_params.append(PROLIF_HILL_CC_'+elem+')')
        # exec('nonlin_list.append(PROLIF_HILL_CC_'+elem+')')
        
        exec('PROLIF_HILL_K_'+elem+'=torch.ones((len(temp[\'prolif_hill_idx\']),),requires_grad=True,device=device)')
        exec('optimized_params.append(PROLIF_HILL_K_'+elem+')')
        exec('nonlin_list.append(PROLIF_HILL_K_'+elem+')')
    else:
        exec('PROLIF_HILL_CC_'+elem+'=None') 
        exec('PROLIF_HILL_K_'+elem+'=None')
     
    if isinstance(temp['lin_hill_idx'],np.ndarray):
        exec('LIN_HILL_K_'+elem+'=torch.ones((len(temp[\'lin_hill_idx\']),),requires_grad=True,device=device)')
        exec('optimized_params.append(LIN_HILL_K_'+elem+')')
        exec('nonlin_list.append(LIN_HILL_K_'+elem+')')
    else:
        exec('LIN_HILL_K_'+elem+'=None')
    
    exec(elem+'_MODEL=ADAM_SINDy_MODEL(COEFF_'+elem+',PROLIF_HILL_K_'+elem+',\
             PROLIF_HILL_CC_'+elem+',\
             LIN_HILL_K_'+elem+',temp,torch.tensor(steady_state,dtype=torch.float32))')
        
WEIGHTS  = torch.nn.Parameter(1e-3*torch.ones((flattenTorchList(coeff_list).numel(),1)), requires_grad= True)
optim_COEFF_ADT = torch.optim.Adam(optimized_params, lr=lr, betas = (0.9,0.99),
                                   eps = 10**-15)
optim_weights   = torch.optim.Adam([WEIGHTS], lr=lr_sparsity, betas = (0.9,0.99),
                                   eps = 10**-15)
scheduler_ADT     = torch.optim.lr_scheduler.StepLR(optim_COEFF_ADT,
                                                    step_size=step_epoch, 
                                                    gamma=decay_rate)
scheduler_weights = torch.optim.lr_scheduler.StepLR(optim_weights,   
                                                    step_size=step_epoch_sparsity, 
                                                    gamma=decay_rate)
Loss_data     = torch.empty(size=(Epochs, 1))
loss_function = torch.nn.MSELoss()
dX_data = dX_data[:,:,np.array(model_idx)].to(device)
if len(dX_data.shape) == 2:
    dX_data.unsqueeze(2)
    
mag = torch.max(torch.abs(dX_data))

###############################################################################
# Train #######################################################################
t_start = time.time()
print('Starting optimization')
for epoch in range(Epochs):
    # WEIGHTS = enforce_sparsity(WEIGHTS,model_idx)
    
    output_data = torch.zeros((dX_data.shape)).to(device)
    for i, elem in enumerate(model_names):
        output_data[:,:,i] = eval(elem+'_MODEL(A_CANDIDATES)')
    
    loss_l2 = loss_function (dX_data, output_data)     
    loss_l1 = torch.linalg.matrix_norm(torch.abs(WEIGHTS)*\
                                       flattenTorchList(coeff_list).unsqueeze(1),ord = 1)
    loss_elas = 1e-6*torch.linalg.matrix_norm(flattenTorchList(coeff_list).unsqueeze(1),ord=2)
    loss_epoch = loss_l2 + mag*loss_l1 + loss_elas
                      
    optim_COEFF_ADT.zero_grad()
    optim_weights.zero_grad()
    
    loss_epoch.backward()

    with torch.no_grad():
        optim_COEFF_ADT.step()
        Loss_data [epoch] = loss_l2.detach()
        
        nonlin_list = boundVar(nonlin_list, [0.05,5.0])

        optim_weights.step()
        coeff_list = ZERO_COEFF(coeff_list,tolerance)
        
    if epoch%1000 == 0:
        print('LOSS DATA, [EPOCH =', epoch,  ']:',  Loss_data [epoch].item())
        print('LEARNING RATE:', optim_COEFF_ADT.param_groups[0]['lr'])
        print ("*"*85)

    scheduler_ADT.step()
    scheduler_weights.step()
    
SINDY_time = time.time() - t_start
coeff_list = ZERO_COEFF(coeff_list,5e-3)

if plot == True:
    #Plot ADAM-SINDy loss figure to ensure convergence and movement
    plt.figure(figsize=(10, 6))
    plt.plot(Loss_data.detach().numpy())
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(False)
    plt.show()

full_model = torch.zeros((dX_data.shape)).to(device)
for i, elem in enumerate(model_names):
    full_model[:,:,i] = eval(elem+'_MODEL(A_CANDIDATES)')
    
TRUE_PARAMS = []
TRUE_COEFF = []
for i, elem in enumerate(model_names):
    temp=eval('getLibrary(model_names['+str(i)+'])')
    exec('COEFF_'+elem+'_TRUE=torch.zeros((len(temp[\'terms\']),))')
    exec('TRUE_PARAMS.append(COEFF_'+elem+'_TRUE)')
    exec('TRUE_COEFF.append(COEFF_'+elem+'_TRUE)')
    
    if isinstance(temp['prolif_hill_idx'],np.ndarray):
        exec('PROLIF_HILL_CC_'+elem+'_TRUE=torch.ones((len(temp[\'prolif_hill_idx\']),))')
        exec('TRUE_PARAMS.append(PROLIF_HILL_CC_'+elem+'_TRUE)')
        
        exec('PROLIF_HILL_K_'+elem+'_TRUE=torch.ones((len(temp[\'prolif_hill_idx\']),))')
        exec('TRUE_PARAMS.append(PROLIF_HILL_K_'+elem+'_TRUE)')
    else:
        exec('PROLIF_HILL_CC_'+elem+'_TRUE=None') 
        exec('PROLIF_HILL_K_'+elem+'_TRUE=None')
     
    if isinstance(temp['lin_hill_idx'],np.ndarray):
        exec('LIN_HILL_K_'+elem+'_TRUE=torch.ones((len(temp[\'lin_hill_idx\']),))')
        exec('TRUE_PARAMS.append(LIN_HILL_K_'+elem+'_TRUE)')
    else:
        exec('LIN_HILL_K_'+elem+'_TRUE=None')
    
    exec(elem+'_MODEL_TRUE=ADAM_SINDy_MODEL(COEFF_'+elem+'_TRUE,PROLIF_HILL_K_'+elem+'_TRUE,\
             PROLIF_HILL_CC_'+elem+'_TRUE,\
             LIN_HILL_K_'+elem+'_TRUE,temp,steady_state)')

#S model
if 0 in model_idx:
    exec('COEFF_S_TRUE[9]  = cell_params[\'r_S\']')
    exec('COEFF_S_TRUE[27] = -cell_params[\'k_S\']')
    exec('COEFF_S_TRUE[29] = -cell_params[\'k_SR\']')
    exec('COEFF_S_TRUE[36] = -cell_params[\'k_SD\']')
    
    exec('PROLIF_HILL_K_S_TRUE[8] = cell_params[\'K_rS\']')
    exec('PROLIF_HILL_CC_S_TRUE[8] = cell_params[\'theta\']')

    exec('LIN_HILL_K_S_TRUE[8] = cell_params[\'K_kS\']')
    
    exec('LIN_HILL_K_S_TRUE[10] = cell_params[\'K_kSR\']')
    
    exec('LIN_HILL_K_S_TRUE[17] = cell_params[\'K_kSD\']')
    
#D model
if 1 in model_idx:
    exec('COEFF_D_TRUE[18] = cell_params[\'k_SD\']')
    
    exec('LIN_HILL_K_D_TRUE[17] = cell_params[\'K_kSD\']')
#R model
if 2 in model_idx:
    exec('COEFF_R_TRUE[13] = cell_params[\'r_R\']')
    exec('COEFF_R_TRUE[29] = cell_params[\'k_SR\']')
    
    exec('PROLIF_HILL_K_R_TRUE[12] = cell_params[\'K_rR\']')
    exec('PROLIF_HILL_CC_R_TRUE[12] = cell_params[\'theta\']')
    
    exec('LIN_HILL_K_R_TRUE[10] = cell_params[\'K_kSR\']')
    

if plot == True:
    data = None
    true = None
    for elem in model_names:
        data = eval('COEFF_'+elem+'.cpu().detach().numpy()')
        true = eval('COEFF_'+elem+'_TRUE.cpu().detach().numpy()')
    
    true_idx = np.argwhere(true!=0)
    
    colors = []
    for i in range(len(data)):
        if i in true_idx:
            colors.append('blue')
        else:
            colors.append('red')
    colors = np.array(colors)
            
    idx = np.argsort(np.abs(data))
    
    custom_legend = [plt.Line2D([0], [0], color='blue', lw=4),
                     plt.Line2D([0], [0], color='red', lw=4)]
    
    fig = plt.figure(figsize=(10,6),layout='constrained')
    fig.suptitle(model_names[0] + ' - Post ADAM-SINDy terms and fit')
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2,2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2:])
    
    ax1.bar(np.arange(len(data)),np.abs(data)[idx],color=colors[idx])
    ax1.set_title('Sorted magnitudes')
    ax1.set_xlabel('Terms')
    ax1.set_ylabel('Coeff Magnitude')
    ax1.legend(custom_legend,['Correct','Incorrect'])
    
    ax2.bar(np.arange(len(data)),data,color=colors)
    ax2.set_title('Coefficients')
    ax2.set_xlabel('Terms')
    ax2.set_ylabel('Coeff Value')
    ax2.legend(custom_legend,['Correct','Incorrect'])
    
    ax3.plot(t, mm.detachTorch(dX_data[0,:,0]).T, color='blue',marker='o')
    ax3.plot(t, mm.detachTorch(full_model[0,:,0]).T, color='red',linestyle='--')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Derivative')
    plt.show()

###############################################################################
# AIC permutation search ######################################################
# Take the nonzero terms from ADAM-SINDy and enumerate all subsets (up to size 4).
# Each subset is refitted with scipy.optimize.least_squares and scored by AIC.
# The permutation with the lowest AIC is selected as the final sparse model.
#Get permutations of terms based on ADAM-SINDy outputs
nonzeros = eval('torch.nonzero(COEFF_'+model_names[0]+')')
nonzeros = nonzeros.detach().numpy()
nonzeros = [int(nonzeros[i,0]) for i in range(len(nonzeros))]    

import itertools
permutations = []
for r in range(1,min((len(nonzeros)+1),5)):
    temp = list(itertools.combinations(nonzeros,r))
    temp = [list(temp[elem]) for elem in range(len(temp))]
    permutations.extend(temp)

AIC = np.zeros((len(permutations),)) 
BIC = np.zeros((len(permutations),)) 
params = []
fits = []

lib = eval(model_names[0]+'_MODEL.lib')
term_types = lib['term_types']

A_CANDIDATES = A_CANDIDATES.to(torch.float64)

# data_to_fit = X_data[:,:,model_idx[0]].to(device)
data_to_fit = dX_data.clone().to(device)
n = (len(t)*data_to_fit.shape[0])

t_start = time.time()
print('Total permutations:', len(permutations))
for i, perm in enumerate(permutations):
    curr_uses_self = np.array([lib['self_propagate'][i] for i in perm])
    model = ADAM_SINDy_MODEL_permute(lib, steady_state)
    curr_types = np.array([term_types[i] for i in perm])
    
    MINIMIZER = TO_SOLVER(model, torch.squeeze(data_to_fit), A_CANDIDATES,
                          perm, curr_types)
    
    pnum = getParamNum(perm, term_types)
    params_guess = np.ones(pnum)
    
    low  = np.full(pnum, -10.0)
    high = np.full(pnum, 10.0)
    
    l = len(perm)

    for j in range(l):
        if curr_uses_self[j] == False:
            params_guess[j] = -1.0
            high[j] = 0.0
        else:
            params_guess[j] = 1.0
            low[j] = 0.0
    
    if pnum > l:
        for j in range(l,pnum):
            low[j] = 0.01
            high[j] = 5.0
            
    out = least_squares(MINIMIZER,params_guess,method='dogbox',
                        bounds=(low,high),loss='soft_l1',max_nfev=100)
    params_fit = out.x.copy()
    
    x_fit = MINIMIZER.final(params_fit)
    
    sse = np.sum((x_fit - torch.squeeze(data_to_fit).detach().numpy())**2)
    
    AIC[i] = n*np.log(sse/n) + 2*pnum
    BIC[i] = n*np.log(sse/n) + pnum*np.log(n)
    
    params.append(params_fit)
    fits.append(x_fit)
                
    if i % 100 == 0:
        print('Permutations done: ',i)
print('Permutation search done')
    
AIC_time = time.time() - t_start

def getTruePermutation(name):
    """Return the ground-truth term indices for a given cell state.
    Used to compare ADAM-SINDy + AIC recovery against the known model structure."""
    if 'S' in name:
        return [9,27,29,36]
    elif 'D' in name:
        return [18]
    elif 'R' in name:
        return [13,29]


true_perm = getTruePermutation(model_names[0])  

#Extract identified model through Tau thresholding
ind = np.argmin(AIC)
sort_ind = np.argsort(AIC)
temp_mag = np.max(np.abs(mm.detachTorch(dX_data)))
tau = 30*temp_mag

sorted_perms = []
sorted_aic = []
for j in range(len(sort_ind)):
    sorted_aic.append(AIC[sort_ind[j]])
    sorted_perms.append(permutations[sort_ind[j]])

cut = 0
best_AIC = AIC[ind]
for j in range(1,len(sort_ind)):
    if np.abs(100*(sorted_aic[j] - best_AIC)/best_AIC) < tau:
        cut = j
    else:
        break
    
perms_out = []
for j in range(cut+1):
    perms_out.append(sorted_perms[j])
    
#count number of times a term appears
temp_counts = np.zeros(len(lib['terms']))
for j, elem1 in enumerate(perms_out):
    for k, elem2 in enumerate(elem1):
        temp_counts[elem2]+=1
        
max_counts = np.max(temp_counts)
        
fit_permute = np.argwhere(temp_counts==max_counts)
if fit_permute.size > 1:
    fit_permute = np.squeeze(fit_permute)
else:
    fit_permute = fit_permute[:,0]
fit_permute = [fit_permute[j] for j in range(fit_permute.size)]  

if set(fit_permute) == set(true_perm):
    print('Model correctly identified')
else:
    print('Model incorrectly identified')

###############################################################################
# Print string representation of true and fit models ##########################
term_names = lib['terms']

true_terms = [term_names[i] for i in true_perm]
fit_terms  = [term_names[i] for i in fit_permute]

print('\n' + '='*60)
print('State: ' + model_names[0])
print('='*60)
print('True model:')
print('  d' + model_names[0] + '/dt = ' + ' + '.join(true_terms))
print('\nFit model:')
print('  d' + model_names[0] + '/dt = ' + ' + '.join(fit_terms))
print('='*60 + '\n')