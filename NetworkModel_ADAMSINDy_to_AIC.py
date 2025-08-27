import MelanomaModel as mm
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time
import scipy as scp

torch.manual_seed(1)

method = 'Radau'

model_idx = [0] #iRAS protein used in testing
dt = 0.5
T_SOLVE = 60
plot = True
sparsity_enforce = True
t  = torch.arange(0,T_SOLVE+dt,dt)

empty = torch.zeros((len(t),1))
on    = torch.ones((len(t),1))

sigmoid = (torch.sigmoid(5.0*(t-10)) - torch.sigmoid(5.0*(t-40))).unsqueeze(1)

#Inputs
constant_inputs    = torch.cat((on,on,on),dim=1).detach().numpy() #stimulated: RTK, random: GPCR, PPTase

RTK_stim = torch.cat((sigmoid+on,on,on),dim=1).detach().numpy()
GPCR_stim = torch.cat((on,sigmoid+on,on),dim=1).detach().numpy()
PPTASE_stim = torch.cat((on,on,sigmoid+on),dim=1).detach().numpy()

test_raf = torch.cat((sigmoid,empty,empty,empty),dim=1).detach().numpy()    #RAF on, no MEK, no ERK
no_drug = torch.cat((empty,empty,empty,empty),dim=1).detach().numpy()

#Build all perturbations for the simulated data
perturbations = [(test_raf, constant_inputs),
                 (no_drug, RTK_stim),
                 (no_drug, GPCR_stim),
                 (no_drug, PPTASE_stim),]

generator = np.random.default_rng(seed=1)
net_params = mm.RandomizeMAPKParams(generator)

torch.manual_seed(1)

device = torch.device('cpu')
print("AVAILABLE PROCESSOR:", device, '\n')
print('Not optimized for running on GPU due to complex libraries')

def addToDataset(X,DX,RAF,GF,X_new,DX_new,RAF_new,GF_new):
    X   = torch.cat((X,X_new.unsqueeze(0)),dim=0)
    DX  = torch.cat((DX,DX_new.unsqueeze(0)),dim=0)
    RAF = torch.cat((RAF,RAF_new.unsqueeze(0)),dim=0)
    GF  = torch.cat((GF,GF_new.unsqueeze(0)),dim=0)
    
    return X,DX,RAF,GF

def POOL_DATA(X,RAF,inputs):
    sets, n, _ = X.shape
    return torch.cat((torch.ones(sets,n,1),X,inputs,RAF),dim=2)

def enforce_sparsity(W,model_idx):
    #Enforces soft constraint on source and decay terms depending on the state
    both = [0,2,4,5,7,9,11,15]
    with torch.no_grad():
        cnt = 0
        for elem in model_idx:
            if elem in both:
                W[0,cnt] = 0.0
                W[1,cnt] = 0.0
            else:
                W[1,cnt] = 0.0
                
            cnt+=1
    return W

def ZERO_COEFF(l,threshold):
    with torch.no_grad():
        for elem in l:
            elem [torch.abs(elem) < threshold] = 0.0
    return l

def boundVar(varlist,bounds):
    for elem in varlist:
        elem[elem < bounds[0]] = bounds[0]
        elem[elem > bounds[1]] = bounds[1]
    return varlist

def getLibrary(state, analog, flip):
    term_types = []
    numbers    = []
    uses_self  = []
    
    term_names = ['constant'] #type = 0
    term_types.append(0)
    numbers.append(0)
    uses_self.append(False)
    
    lin_idx   = []   #type = 1
    drug_idx  = []  #type = 2
    bilin_idx = [] #type = 3
    mm2_idx   = []   #type = 4
    hill_idx  = []  #type = 5
    
    states = ['iRAS','aRAS','iRAF_wt','aRAF_wt','RAF_m','iMEK',
              'aMEK','iERK','aERK','iPI3K','aPI3K','iAKT','aAKT','ATP','cAMP',
              'iPKA','aPKA','MITF','RTK','GPCR','PPTase']
    
    term_names.append(state)
    term_types.append(1)
    numbers.append(0)
    lin_idx.append(states.index(state))
    uses_self.append(True)
    
    #Known inputs (drug targets)
    if state == 'RAF_m':
        term_names.append(state+'*'+'RAF_inhib')
        term_types.append(2)
        numbers.append(0)
        drug_idx.append([states.index(state), 0])
        uses_self.append(True)
        
    if state == 'iMEK':
        term_names.append(state+'*'+'MEK_inhib')
        term_types.append(2)
        numbers.append(0)
        drug_idx.append([states.index(state), 1])
        uses_self.append(True)
        
    if state == 'iERK':
        term_names.append(state+'*'+'ERK_inhib')
        term_types.append(2)
        numbers.append(0)
        drug_idx.append([states.index(state), 2])
        uses_self.append(True)
        
    if state == 'iRAS':
        term_names.append(state+'*'+'RAS_inhib')
        term_types.append(2)
        numbers.append(0)
        drug_idx.append([states.index(state), 3])
        uses_self.append(True)
        
    curr_mm    = 0
    curr_bilin = 0
    curr_hill  = 0    
    #States with activations   
    if state not in ['MITF','RAF_m','ATP','cAMP']:
        if flip == False:
            #Self and analog with all possible enzymes
            for elem in states:
                if elem != state:
                    term_names.append('MM(substrate: '+state+', enzyme: '+elem+')')
                    term_types.append(4)
                    numbers.append(curr_mm)
                    curr_mm += 1
                    mm2_idx.append([states.index(state), states.index(elem)])
                    uses_self.append(True)
            for elem in states:
                if elem != analog:
                    term_names.append('MM(substrate: '+analog+', enzyme: '+elem+')')
                    term_types.append(4)
                    numbers.append(curr_mm)
                    curr_mm += 1
                    mm2_idx.append([states.index(analog), states.index(elem)])
                    uses_self.append(False)
        else:
            #Self and analog with all possible enzymes
            for elem in states:
                if elem != analog:
                    term_names.append('MM(substrate: '+analog+', enzyme: '+elem+')')
                    mm2_idx.append([states.index(analog), states.index(elem)])
                    term_types.append(4)
                    numbers.append(curr_mm)
                    curr_mm += 1
                    uses_self.append(False)
            for elem in states:
                if elem != state:
                    term_names.append('MM(substrate: '+state+', enzyme: '+elem+')')
                    mm2_idx.append([states.index(state), states.index(elem)])
                    term_types.append(4) 
                    numbers.append(curr_mm)
                    curr_mm += 1
                    uses_self.append(True)
    #States with no activation (RAF_m and MITF)
    elif state not in ['ATP','cAMP']:
        #Self - bilinear combinations
        for elem in states:
            if elem != state:
                term_names.append(state + '*' + elem)
                term_types.append(3)
                numbers.append(curr_bilin)
                curr_bilin += 1
                bilin_idx.append([states.index(state), states.index(elem)])
                uses_self.append(True)
        #Hill combinations of all proteins for MITF
        if state == 'MITF':
            for elem in states:
                if elem != state:
                    term_names.append('Hill('+elem)
                    term_types.append(5)
                    numbers.append(curr_hill)
                    curr_hill += 1
                    hill_idx.append(states.index(elem))
                    uses_self.append(False)
    else:
        for elem in states:
            if elem != 'ATP':
                term_names.append('MM(substrate: ATP, enzyme: ' + elem)
                term_types.append(4)
                numbers.append(curr_mm)
                curr_mm += 1
                mm2_idx.append([states.index('ATP'), states.index(elem)])
                if state == 'ATP':
                    uses_self.append(True)
                else:
                    uses_self.append(False)
                    
        if state == 'ATP':
            for elem in states:
                if elem != state:
                    term_names.append('Hill('+elem)
                    term_types.append(5)
                    numbers.append(curr_hill)
                    curr_hill += 1
                    hill_idx.append(states.index(elem))  
                    uses_self.append(False)
            
    #if no terms set to None
    if len(lin_idx)==0:
        lin_idx = None
    else:
        lin_idx = np.array(lin_idx)
    if len(drug_idx)==0:
        drug_idx = None
    else:
        drug_idx = np.array(drug_idx)
    if len(bilin_idx)==0:
        bilin_idx = None
    else:
        bilin_idx = np.array(bilin_idx)
    if len(mm2_idx)==0:
        mm2_idx = None
    else:
        mm2_idx = np.array(mm2_idx)
    if len(hill_idx)==0:
        hill_idx = None
    else:
        hill_idx = np.array(hill_idx)
        
    return {'terms':term_names,'lin_idx':lin_idx,'drug_idx':drug_idx,
            'bilin_idx':bilin_idx,'mm2_idx':mm2_idx,'hill_idx':hill_idx,
            'state':state,'analog':analog,'term_types':term_types,
            'numbers':numbers,'uses_self':uses_self}

def flattenTorchList(x):
    return torch.flatten(torch.cat(x))
    
class MM2_term(torch.nn.Module):
    def __init__(self,K):
        super().__init__()
        self.K = K
        
    def forward(self, x, y):
        return (x * y) / (self.K + x)
    
class Hill_term(torch.nn.Module):
    def __init__(self,K):
        super().__init__()
        self.K = K
        
    def forward(self, x):
        return x / (self.K + x)

#Used for the candidate identfication from the library
class ADAM_SINDy_MODEL(torch.nn.Module):
    def __init__(self,a,K2,K3,library_package):
        super().__init__()
        self.a = a
        self.K2 = K2
        self.K3 = K3
        
        self.mm2  = MM2_term(self.K2)
        self.hill = Hill_term(self.K3)
        
        self.lib = library_package
    
    def forward(self, candidates):
        con  = candidates[:,:,0].unsqueeze(2)
        x    = candidates[:,:,1:22] #order: EGFR, iRAS, aRAS, iRAF_wt, aRAF_wt
                                          # RAF_m, iMEK, aMEK, iERK, aERK, MITF           
                                          # iPI3K, aPI3K, iAKT, aAKT
        drug = candidates[:,:,22:] #RAF_inhib, MEK_inhib, ERK_inhib, RAS_inhib
        
        terms = con
        
        if isinstance(self.lib['lin_idx'],np.ndarray):
            terms = torch.cat((terms, x[:,:,self.lib['lin_idx']]),dim=2)
            
        if isinstance(self.lib['drug_idx'],np.ndarray):
            terms = torch.cat((terms, x[:,:,self.lib['drug_idx'][:,0]]*\
                               drug[:,:,self.lib['drug_idx'][:,1]]),dim=2)
                
        if isinstance(self.lib['bilin_idx'],np.ndarray):
            terms = torch.cat((terms, x[:,:,self.lib['bilin_idx'][:,0]]*\
                               x[:,:,self.lib['bilin_idx'][:,1]]),dim=2)
            
        if isinstance(self.lib['mm2_idx'],np.ndarray):
            terms = torch.cat((terms, self.mm2(x[:,:,self.lib['mm2_idx'][:,0]],
                                               x[:,:,self.lib['mm2_idx'][:,1]])),dim=2)
            
        if isinstance(self.lib['hill_idx'],np.ndarray):
            terms = torch.cat((terms, self.hill(x[:,:,self.lib['hill_idx']])),dim=2)
        #Zero terms that correlate to biologically incorrect coefficients
        for i in range(2,terms.shape[2]):
            if self.lib['uses_self'][i] == True:
                if self.a[i] > 0.0:
                    terms[:,:,i] = 0.0
            else:
                if self.a[i] < 0.0:
                    terms[:,:,i] = 0.0
        return terms @ self.a

#Used for the permutation testing
class ADAM_SINDy_MODEL_permute(torch.nn.Module):
    def __init__(self,library_package,starts,dt,initial):
        super().__init__()
        self.lib = library_package
        self.starts = starts
        self.dt = dt
        self.initial = initial
    def forward(self, candidates, permutation, coeff, mm2con_fit, hill_fit):
        con  = candidates[:,:,0].unsqueeze(2)
        x    = candidates[:,:,1:22]
        drug = candidates[:,:,22:]
        
        types   = self.lib['term_types']
        numbers = self.lib['numbers']
        
        if permutation[0] == 0:
            terms = con
            start = 1
        else:
            terms = (x[:,:,self.lib['lin_idx'][0]]).unsqueeze(2)
            start = 1
        
        mm2_curr_ind = 0
        hill_curr_ind = 0
        
        for elem in permutation[start:]:
            curr_type = types[elem]
            ind       = numbers[elem]
            if curr_type == 1:
                terms = torch.cat((terms, (x[:,:,self.lib['lin_idx'][ind]]).unsqueeze(2)),dim=2)
            elif curr_type == 2:
                terms = torch.cat((terms, (x[:,:,self.lib['drug_idx'][ind,0]]*\
                                   drug[:,:,self.lib['drug_idx'][ind,1]]).unsqueeze(2)),dim=2)
            elif curr_type == 3:
                terms = torch.cat((terms, (x[:,:,self.lib['bilin_idx'][ind,0]]*\
                                   x[:,:,self.lib['bilin_idx'][ind,1]]).unsqueeze(2)),dim=2)
            elif curr_type == 4:
                temp_mm2 = MM2_term(mm2con_fit[mm2_curr_ind])
                terms = torch.cat((terms, temp_mm2(x[:,:,self.lib['mm2_idx'][ind,0]],
                                                  x[:,:,self.lib['mm2_idx'][ind,1]]).unsqueeze(2)),dim=2)
                mm2_curr_ind += 1
            elif curr_type == 5:
                temp_hill = Hill_term(hill_fit[hill_curr_ind])
                terms = torch.cat((terms, temp_hill(x[:,:,self.lib['hill_idx'][ind]]).unsqueeze(2)),dim=2)
                hill_curr_ind += 1

        dx = terms @ coeff
        temp = dx.detach().numpy()
        return temp
        
def getParamNum(permutation, term_types):
    k = 0
    for elem in permutation:
        if (term_types[elem] == 0 or term_types[elem] == 1 or 
            term_types[elem] == 2 or term_types[elem] == 3):
            k += 1
        elif term_types[elem] == 4 or term_types[elem] == 5:
            k += 2
    return k
                  
class TO_SOLVER():
    def __init__(self,model,data,candidates,permutation,term_types):
        self.model = model
        self.data  = data
        self.candidates  = candidates
        self.permutation = permutation
        self.term_types  = term_types
    def __call__(self,x):
        #convert x to coeff, mm_con and mm2_con
        n = len(x)
        try:
            nl2 = np.array(self.term_types[self.term_types==4]).size
        except:
            nl2 = 0
        try:
            hill = np.array(self.term_types[self.term_types==5]).size
        except:
            hill = 0
            
        l = n-(nl2+hill)
        
        coeff = x[:l]
        if nl2 != 0:
            mm2_con = x[l:l+nl2]
        else:
            mm2_con = None
        if hill != 0:
            hill_con = x[l+nl2:]
        else:
            hill_con = None

        output = self.model(self.candidates,self.permutation,
                                coeff,mm2_con,hill_con)
        
        return (self.data.cpu().detach().numpy() - output).reshape(-1)
    
    def final(self,x):
        #convert x to coeff, mm_con and mm2_con
        n = len(x)
        try:
            nl2 = np.array(self.term_types[self.term_types==4]).size
        except:
            nl2 = 0
        try:
            hill = np.array(self.term_types[self.term_types==5]).size
        except:
            hill = 0
            
        l = n-(nl2+hill)
        
        coeff = x[:l]
        if nl2 != 0:
            mm2_con = x[l:l+nl2]
        else:
            mm2_con = None
        if hill != 0:
            hill_con = x[l+nl2:]
        else:
            hill_con = None
        
        output = self.model(self.candidates,self.permutation,
                                coeff,mm2_con,hill_con)
        return output
    
###############################################################################
# Initialize model ############################################################    
T_temp = 200
t_temp  = np.arange(0,T_temp+dt,dt)

###############################################################################
# Solve for steady state - no drug ############################################
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
T = T_SOLVE
t  = np.arange(0,T+dt,dt)
t_eval  = np.arange(0,T+0.001,0.001) #High resolution time data to downsample for model fitting

X_data = torch.empty((0,len(t),len(steady_state)))
dX_data = torch.empty((0,len(t),len(steady_state)))

RAF_inhib_data = torch.empty((0,len(t),4))
input_data = torch.empty((0,len(t),3))

for i, curr in enumerate(perturbations):
    RAF_inhib, inputs = curr
    
    Network = mm.MAPKModel(RAF_inhib, inputs, net_params, t)
    
    X = scp.integrate.solve_ivp(Network, (0,T), steady_state, t_eval = t_eval,
                                method=method, rtol=10**(-12),
                                atol=10**(-12)*np.ones_like(steady_state)).y.T
    
    meas_indices = np.isin(np.round(t_eval,decimals=4),np.round(t,decimals=4))
    X = X[meas_indices,:]
    
    X = torch.tensor(X,dtype=torch.float32)
    if plot == True:
        mm.plotFullNetwork(mm.detachTorch(torch.tensor(t)), mm.detachTorch(X).T,
                           title='Protein Datasets')
    
    dX_dt = torch.gradient(X, spacing = dt, dim = 0)[0]
    
    RAF_inhib = torch.tensor(RAF_inhib,dtype=torch.float32)
    inputs = torch.tensor(inputs,dtype=torch.float32)
    
    X_data   = torch.cat((X_data,X.unsqueeze(0)),dim=0)
    dX_data  = torch.cat((dX_data,dX_dt.unsqueeze(0)),dim=0)
    RAF_inhib_data = torch.cat((RAF_inhib_data,RAF_inhib.unsqueeze(0)),dim=0)
    input_data  = torch.cat((input_data,inputs.unsqueeze(0)),dim=0)

# ###############################################################################
# # Setup ADAM_SINDy Model ######################################################   
t = torch.tensor(t)
A_CANDIDATES = POOL_DATA(X_data, RAF_inhib_data, input_data).to(device)
A_CANDIDATES = A_CANDIDATES.to(torch.float32)

# ###############################################################################
# # Initialize ADAM_SINDy training ##############################################
Epochs              = 25000
lr                  = 1e-2
lr_sparsity         = 1e-3
step_epoch          = 5000
step_epoch_sparsity = 5000
decay_rate          = 0.50
tolerance           = 5e-5

#Specific model setup for each state to prevent rewriting for each
all_model_names  = ['iRAS','aRAS','iRAF_wt','aRAF_wt','RAF_m','iMEK','aMEK',
                    'iERK','aERK','iPI3K','aPI3K','iAKT','aAKT','ATP','cAMP',
                    'iPKA','aPKA','MITF']
#Is the protein part of an active/inactive pair
all_analog_names = ['aRAS','iRAS','aRAF_wt','iRAF_wt',None,'aMEK','iMEK','aERK',
                    'iERK','aPI3K','iPI3K','aAKT','iAKT',None,None,'aPKA','iPKA',None]
#Flip used for active states so that inactive terms come first in library
all_flip         = [False,True,False,True,False,False,True,False,True,False,
                    True,False,True,False,False,False,True,False]
#Active used for determining whether term coefficients should be positive or negative
active           = [False,True,False,True,True,False,True,
                    False,True,False,True,False,True,False,True,
                    False,True,True]

model_names  = [all_model_names[i]  for i in model_idx]
analog_names = [all_analog_names[i] for i in model_idx]
flip         = [all_flip[i]         for i in model_idx]

optimized_params = []
coeff_list = []
nonlin_list = []
for i, elem in enumerate(model_names):
    temp=eval('getLibrary(model_names['+str(i)+'],analog_names['+str(i)+'],flip['+str(i)+'])')
    exec('COEFF_'+elem+'=torch.ones((len(temp[\'terms\']),),requires_grad=True,device=device)')
    exec('optimized_params.append(COEFF_'+elem+')')
    exec('coeff_list.append(COEFF_'+elem+')')
        
    if isinstance(temp['mm2_idx'],np.ndarray):
        exec('MM2_CONSTANTS_'+elem+'=torch.ones((len(temp[\'mm2_idx\']),),requires_grad=True,device=device)')
        exec('optimized_params.append(MM2_CONSTANTS_'+elem+')')
        exec('nonlin_list.append(MM2_CONSTANTS_'+elem+')')
    else:
        exec('MM2_CONSTANTS_'+elem+'=None')   
    
    if isinstance(temp['hill_idx'],np.ndarray):
        exec('HILL_CONSTANTS_'+elem+'=torch.ones((len(temp[\'hill_idx\']),),requires_grad=True,device=device)')
        exec('optimized_params.append(HILL_CONSTANTS_'+elem+')')
        exec('nonlin_list.append(HILL_CONSTANTS_'+elem+')')
    else:
        exec('HILL_CONSTANTS_'+elem+'=None')  
            
    exec(elem +'_MODEL=ADAM_SINDy_MODEL(COEFF_'+elem+',MM2_CONSTANTS_'+elem+'\
                                        ,HILL_CONSTANTS_'+elem+',temp)')

WEIGHTS  = torch.nn.Parameter(1e-1*torch.ones((flattenTorchList(coeff_list).numel(),1)),
                              requires_grad= True)
torch.nn.init.normal_(WEIGHTS, mean=0, std=1e-1)
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

###############################################################################
# Train #######################################################################
t_start = time.time()
print('Starting optimization')
for epoch in range(Epochs):
    if sparsity_enforce == True:
        WEIGHTS = enforce_sparsity(WEIGHTS,model_idx)
    
    output_data = torch.zeros((dX_data.shape)).to(device)
    for i, elem in enumerate(model_names):
        output_data[:,:,i] = eval(elem+'_MODEL(A_CANDIDATES)')
    
    loss_l2 = loss_function (dX_data, output_data)
    loss_l1 = torch.linalg.matrix_norm(torch.abs(WEIGHTS)*\
                                       flattenTorchList(coeff_list).unsqueeze(1),ord = 1)
    loss_epoch = loss_l2 + loss_l1
                      
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
coeff_list = ZERO_COEFF(coeff_list,1e-3)

if plot == True:
    #Plot ADAM-SINDy loss figure to ensure convergence and movement
    plt.figure(figsize=(10, 6))
    plt.plot(Loss_data.detach().numpy())
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(False)
    plt.show()

#Full model analysis
full_model = torch.zeros((dX_data.shape)).to(device)
for i, elem in enumerate(model_names):
    full_model[:,:,i] = eval(elem+'_MODEL(A_CANDIDATES)')
    
full_mse = torch.mean((dX_data - full_model)**2)

TRUE_PARAMS = []
TRUE_COEFF = []
for i, elem in enumerate(model_names):
    temp=eval(elem+'_MODEL.lib')
    exec('COEFF_'+elem+'_TRUE=torch.zeros((len(temp[\'terms\']),),device=device)')
    exec('TRUE_PARAMS.append(COEFF_'+elem+'_TRUE)')
    exec('TRUE_COEFF.append(COEFF_'+elem+'_TRUE)')
        
    if isinstance(temp['mm2_idx'],np.ndarray):
        exec('MM2_CONSTANTS_'+elem+'_TRUE=torch.ones((len(temp[\'mm2_idx\']),),device=device)')
        exec('TRUE_PARAMS.append(MM2_CONSTANTS_'+elem+'_TRUE)')
    else:
        exec('MM2_CONSTANTS_'+elem+'_TRUE=None') 
    
    if isinstance(temp['hill_idx'],np.ndarray):
        exec('HILL_CONSTANTS_'+elem+'_TRUE=torch.ones((len(temp[\'hill_idx\']),),device=device)')
        exec('TRUE_PARAMS.append(HILL_CONSTANTS_'+elem+'_TRUE)')
    else:
        exec('HILL_CONSTANTS_'+elem+'_TRUE=None') 
        
    exec(elem +'_MODEL_TRUE=ADAM_SINDy_MODEL(COEFF_'+elem+'_TRUE,MM2_CONSTANTS_'+elem+'_TRUE,\
             HILL_CONSTANTS_'+elem+'_TRUE,temp)')
            
#RAS models
if 0 in model_idx:
    exec('COEFF_iRAS_TRUE[0] = net_params[\'s_iRAS\']')
    exec('COEFF_iRAS_TRUE[1] = -net_params[\'mu_iRAS\']')
    exec('COEFF_iRAS_TRUE[2] = -net_params[\'k_RAS_inhib\']')
    exec('COEFF_iRAS_TRUE[20] = -net_params[\'V_aRAS\']')
    exec('COEFF_iRAS_TRUE[30] = net_params[\'V_iRAS\']')

    exec('MM2_CONSTANTS_iRAS_TRUE[17] = net_params[\'K_aRAS\']')
    exec('MM2_CONSTANTS_iRAS_TRUE[27] = net_params[\'K_iRAS\']')
if 1 in model_idx:
    exec('COEFF_aRAS_TRUE[1] = -net_params[\'mu_aRAS\']')
    exec('COEFF_aRAS_TRUE[19] = net_params[\'V_aRAS\']')
    exec('COEFF_aRAS_TRUE[29] = -net_params[\'V_iRAS\']')

    exec('MM2_CONSTANTS_aRAS_TRUE[17] = net_params[\'K_aRAS\']')
    exec('MM2_CONSTANTS_aRAS_TRUE[27] = net_params[\'K_iRAS\']')
#RAF_wt models
if 2 in model_idx:
    exec('COEFF_iRAF_wt_TRUE[0] = net_params[\'s_iRAF_wt\']')
    exec('COEFF_iRAF_wt_TRUE[1] = -net_params[\'mu_iRAF_wt\']')
    exec('COEFF_iRAF_wt_TRUE[3] = -net_params[\'V_aRAF\']')
    exec('COEFF_iRAF_wt_TRUE[41] = net_params[\'V_iRAF\']')
    
    exec('MM2_CONSTANTS_iRAF_wt_TRUE[1] = net_params[\'K_aRAF\']')
    exec('MM2_CONSTANTS_iRAF_wt_TRUE[39] = net_params[\'K_iRAF\']')
if 3 in model_idx:
    exec('COEFF_aRAF_wt_TRUE[1] = -net_params[\'mu_aRAF_wt\']')
    exec('COEFF_aRAF_wt_TRUE[3] = net_params[\'V_aRAF\']')
    exec('COEFF_aRAF_wt_TRUE[41] = -net_params[\'V_iRAF\']')
    
    exec('MM2_CONSTANTS_aRAF_wt_TRUE[1] = net_params[\'K_aRAF\']')
    exec('MM2_CONSTANTS_aRAF_wt_TRUE[39] = net_params[\'K_iRAF\']')
#RAF_m model
if 4 in model_idx:
    exec('COEFF_RAF_m_TRUE[0] = net_params[\'s_RAF_m\']')
    exec('COEFF_RAF_m_TRUE[1] = -net_params[\'mu_RAF_m\']')
    exec('COEFF_RAF_m_TRUE[2] = -net_params[\'k_RAF_inhib\']')
#MEK models
if 5 in model_idx:
    exec('COEFF_iMEK_TRUE[0] = net_params[\'s_iMEK\']')
    exec('COEFF_iMEK_TRUE[1] = -net_params[\'mu_iMEK\']')
    exec('COEFF_iMEK_TRUE[6] = -net_params[\'V_aMEK\']')
    exec('COEFF_iMEK_TRUE[7] = -net_params[\'V_aMEK_m\']')
    exec('COEFF_iMEK_TRUE[42] = net_params[\'V_iMEK\']')
    exec('COEFF_iMEK_TRUE[2]  = -net_params[\'k_MEK_inhib\']')
    
    exec('MM2_CONSTANTS_iMEK_TRUE[3] =  net_params[\'K_aMEK\']')
    exec('MM2_CONSTANTS_iMEK_TRUE[4] =  net_params[\'K_aMEK_m\']')
    exec('MM2_CONSTANTS_iMEK_TRUE[39] =  net_params[\'K_iMEK\']')
if 6 in model_idx:
    exec('COEFF_aMEK_TRUE[1] = -net_params[\'mu_aMEK\']')
    exec('COEFF_aMEK_TRUE[5] = net_params[\'V_aMEK\']')
    exec('COEFF_aMEK_TRUE[6] = net_params[\'V_aMEK_m\']')
    exec('COEFF_aMEK_TRUE[41] = -net_params[\'V_iMEK\']')

    exec('MM2_CONSTANTS_aMEK_TRUE[3] =  net_params[\'K_aMEK\']')
    exec('MM2_CONSTANTS_aMEK_TRUE[4] =  net_params[\'K_aMEK_m\']')
    exec('MM2_CONSTANTS_aMEK_TRUE[39] =  net_params[\'K_iMEK\']')
#ERK models
if 7 in model_idx:
    exec('COEFF_iERK_TRUE[0] = net_params[\'s_iERK\']')
    exec('COEFF_iERK_TRUE[1] = -net_params[\'mu_iERK\']')
    exec('COEFF_iERK_TRUE[9] = -net_params[\'V_aERK\']')
    exec('COEFF_iERK_TRUE[42] = net_params[\'V_iERK\']')
    exec('COEFF_iERK_TRUE[2]  = -net_params[\'k_ERK_inhib\']')
    
    exec('MM2_CONSTANTS_iERK_TRUE[6] =  net_params[\'K_aERK\']')
    exec('MM2_CONSTANTS_iERK_TRUE[39] =  net_params[\'K_iERK\']')
if 8 in model_idx:
    exec('COEFF_aERK_TRUE[1] = -net_params[\'mu_aERK\']')
    exec('COEFF_aERK_TRUE[8] = net_params[\'V_aERK\']')
    exec('COEFF_aERK_TRUE[41] = -net_params[\'V_iERK\']')
    
    exec('MM2_CONSTANTS_aERK_TRUE[6] =  net_params[\'K_aERK\']')
    exec('MM2_CONSTANTS_aERK_TRUE[39] =  net_params[\'K_iERK\']')
#PI3K models
if 9 in model_idx:
    exec('COEFF_iPI3K_TRUE[0] = net_params[\'s_iPI3K\']')
    exec('COEFF_iPI3K_TRUE[1] = -net_params[\'mu_iPI3K\']')
    exec('COEFF_iPI3K_TRUE[19] = -net_params[\'V_aPI3K_RTK\']')
    exec('COEFF_iPI3K_TRUE[20] = -net_params[\'V_aPI3K_GPCR\']')
    exec('COEFF_iPI3K_TRUE[30] = net_params[\'V_iPI3K\']')
    
    exec('MM2_CONSTANTS_iPI3K_TRUE[17] = net_params[\'K_aPI3K_RTK\']')
    exec('MM2_CONSTANTS_iPI3K_TRUE[18] = net_params[\'K_aPI3K_GPCR\']')
    exec('MM2_CONSTANTS_iPI3K_TRUE[28] = net_params[\'K_iPI3K\']')
if 10 in model_idx:
    exec('COEFF_aPI3K_TRUE[1] = -net_params[\'mu_aPI3K\']')
    exec('COEFF_aPI3K_TRUE[19] = net_params[\'V_aPI3K_RTK\']')
    exec('COEFF_aPI3K_TRUE[20] = net_params[\'V_aPI3K_GPCR\']')
    exec('COEFF_aPI3K_TRUE[30] = -net_params[\'V_iPI3K\']')
    
    exec('MM2_CONSTANTS_aPI3K_TRUE[17] = net_params[\'K_aPI3K_RTK\']')
    exec('MM2_CONSTANTS_aPI3K_TRUE[18] = net_params[\'K_aPI3K_GPCR\']')
    exec('MM2_CONSTANTS_aPI3K_TRUE[28] = net_params[\'K_iPI3K\']')
#AKT models
if 11 in model_idx:
    exec('COEFF_iAKT_TRUE[0] = net_params[\'s_iAKT\']')
    exec('COEFF_iAKT_TRUE[1] = -net_params[\'mu_iAKT\']')
    exec('COEFF_iAKT_TRUE[12] = -net_params[\'V_aAKT\']')
    exec('COEFF_iAKT_TRUE[41] = net_params[\'V_iAKT\']')
    
    exec('MM2_CONSTANTS_iAKT_TRUE[10] =  net_params[\'K_aAKT\']')
    exec('MM2_CONSTANTS_iAKT_TRUE[39] =  net_params[\'K_iAKT\']')
if 12 in model_idx:
    exec('COEFF_aAKT_TRUE[1] = -net_params[\'mu_aAKT\']')
    exec('COEFF_aAKT_TRUE[12] = net_params[\'V_aAKT\']')
    exec('COEFF_aAKT_TRUE[41] = -net_params[\'V_iAKT\']')

    exec('MM2_CONSTANTS_aAKT_TRUE[10] =  net_params[\'K_aAKT\']')
    exec('MM2_CONSTANTS_aAKT_TRUE[39] =  net_params[\'K_iAKT\']')
#ATP model
if 13 in model_idx:
    exec('COEFF_ATP_TRUE[1] = -net_params[\'mu_ATP\']')
    exec('COEFF_ATP_TRUE[38] = net_params[\'s_ATP\']')
    exec('COEFF_ATP_TRUE[20] = -net_params[\'V_acAMP\']')
    
    exec('MM2_CONSTANTS_ATP_TRUE[18] = net_params[\'K_acAMP\']')
    exec('HILL_CONSTANTS_ATP_TRUE[16] = net_params[\'K_sATP\']')
#cAMP model
if 14 in model_idx:
    exec('COEFF_cAMP_TRUE[1] = -net_params[\'mu_cAMP\']')
    exec('COEFF_cAMP_TRUE[20] = net_params[\'V_acAMP\']')
    
    exec('MM2_CONSTANTS_cAMP_TRUE[18] = net_params[\'K_acAMP\']')
#PKA models
if 15 in model_idx:
    exec('COEFF_iPKA_TRUE[0] = net_params[\'s_iPKA\']')
    exec('COEFF_iPKA_TRUE[1] = -net_params[\'mu_iPKA\']')
    exec('COEFF_iPKA_TRUE[16] = -net_params[\'V_aPKA\']')
    exec('COEFF_iPKA_TRUE[41] = net_params[\'V_iPKA\']')
    
    exec('MM2_CONSTANTS_iPKA_TRUE[14] = net_params[\'K_aPKA\']')
    exec('MM2_CONSTANTS_iPKA_TRUE[39] = net_params[\'K_iPKA\']')
if 16 in model_idx:
    exec('COEFF_aPKA_TRUE[1] = -net_params[\'mu_aPKA\']')
    exec('COEFF_aPKA_TRUE[16] = net_params[\'V_aPKA\']')
    exec('COEFF_aPKA_TRUE[41] = -net_params[\'V_iPKA\']')
    
    exec('MM2_CONSTANTS_aPKA_TRUE[14] = net_params[\'K_aPKA\']')
    exec('MM2_CONSTANTS_aPKA_TRUE[39] = net_params[\'K_iPKA\']')
#MITF model
if 17 in model_idx:
    exec('COEFF_MITF_TRUE[38] = net_params[\'s_MITF\']')
    exec('COEFF_MITF_TRUE[1] = -net_params[\'mu_MITF\']')
    exec('COEFF_MITF_TRUE[10] = -net_params[\'k_MITF\']')
    
    exec('HILL_CONSTANTS_MITF_TRUE[16] = net_params[\'K_sMITF\']')

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
    fig.suptitle(model_names[0])
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
    
    ax3.plot(mm.detachTorch(t), mm.detachTorch(dX_data[0,:,0]).T, color='blue',marker='o')
    ax3.plot(mm.detachTorch(t), mm.detachTorch(full_model[0,:,0]).T, color='red',linestyle='--')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Derivative')
    plt.show()

#Get permutations of terms based on ADAM-SINDy outputs
nonzeros = eval('torch.nonzero(COEFF_'+model_names[0]+')')
nonzeros = nonzeros.detach().numpy()
nonzeros = [int(nonzeros[i,0]) for i in range(len(nonzeros))]

try:
    nonzeros.remove(0)
except:
    pass
try:
    nonzeros.remove(1)
except:
    pass

import itertools
permutations = []
for r in range(1,min((len(nonzeros)+1),4)):
    permutations.extend(list(itertools.combinations(nonzeros,r)))
    
both = [0,2,4,5,7,9,11,15]
for i, elem in enumerate(permutations):
    if model_idx[0] in both:
        permutations[i] = [0,1]+list(elem)
    else:
        permutations[i] = [1]+list(elem)

AIC = np.zeros((len(permutations),)) 
params = []
fits = []

lib = eval(model_names[0]+'_MODEL.lib')
term_types = lib['term_types']

temp_types = np.array(term_types)
n_c_terms = np.array(temp_types[temp_types==0]).size
n_l_terms = np.array(temp_types[temp_types==1]).size
n_d_terms = np.array(temp_types[temp_types==2]).size
n_bl_terms = np.array(temp_types[temp_types==3]).size
n_mm2_terms = np.array(temp_types[temp_types==4]).size
n_hill_terms = np.array(temp_types[temp_types==5]).size

start_c = 0
start_l = n_c_terms
start_d = start_l + n_l_terms
start_bl = start_d + n_d_terms
start_mm2 = start_bl + n_bl_terms
start_hill = start_mm2 + n_mm2_terms

starts = [start_c,start_l,start_d,start_bl,start_mm2,start_hill]

coeff = eval('COEFF_'+model_names[0])

A_CANDIDATES = A_CANDIDATES.to(torch.float64)

data_to_fit = dX_data.clone()

solver_outputs = []

t_start = time.time()
model = ADAM_SINDy_MODEL_permute(lib, starts, dt, steady_state[model_idx[0]])
print('Total permutations:', len(permutations))
for i, perm in enumerate(permutations):
    curr_uses_self = np.array([lib['uses_self'][i] for i in perm])
    model = ADAM_SINDy_MODEL_permute(lib, starts, dt,
                                     steady_state[model_idx[0]])
    curr_types = np.array([term_types[i] for i in perm])
    
    MINIMIZER = TO_SOLVER(model, torch.squeeze(data_to_fit), A_CANDIDATES,
                          perm, curr_types)
    
    pnum = getParamNum(perm, term_types)
    params_guess = np.ones(pnum)
    
    low  = np.full(pnum, -10.0)
    high = np.full(pnum, 10.0)
    
    nl2 = np.array(curr_types[curr_types==4]).size 
    hill = np.array(curr_types[curr_types==5]).size 
    l = pnum-nl2-hill

    for j in range(l):
        if curr_uses_self[j] == True:
            params_guess[j] = -1.0
            high[j] = 0.0
        else:
            params_guess[j] = 1.0
            low[j] = 0.0
    
    if nl2 != 0 or hill!= 0:
        for j in range(l,pnum):
            low[j] = 0.01
            high[j] = 5.0
            
    out = least_squares(MINIMIZER,params_guess,method='dogbox',bounds=(low,high),
                        gtol = 1e-12, ftol = 1e-12, xtol = 1e-12, max_nfev = 100,
                        loss = 'cauchy')
    # out = least_squares(MINIMIZER,params_guess,method='lm',x_scale=1).x
    solver_outputs.append(out)
    params_fit = out.x.copy()
    
    x_fit = MINIMIZER.final(params_fit)
    
    sse = np.sum((x_fit - torch.squeeze(data_to_fit).detach().numpy())**2)
    
    AIC[i] = len(t)*np.log(sse/len(t)) + 2*pnum
    params.append(params_fit)
    fits.append(x_fit)
                
    if i % 100 == 0:
        print('Permutations done: ',i)
print('Permutation search done')
    
AIC_time = time.time() - t_start

####Save results in dictionary
post_sindy_params = optimized_params
post_sindy_out    = full_model.cpu().detach().numpy()

def getTruePermutation(name):
    if 'iRAS' in name:
        return [[0,1,2,20,30],[0,1,20,30]]
    elif 'aRAS' in name:
        return [1,19,29]
    elif 'iRAF_wt' in name:
        return [0,1,3,41]
    elif 'aRAF_wt' in name:
        return [1,3,41]
    elif 'RAF_m' in name:
        return [0,1,2]
    elif 'iMEK' in name:
        return [[0,1,2,6,7,42],[0,1,6,7,42]]
    elif 'aMEK' in name:
        return [1,5,6,41]
    elif 'iERK' in name:
        return [[0,1,2,9,42],[0,1,9,42]]
    elif 'aERK' in name:
        return [1,8,41]
    elif 'iPI3K' in name:
        return [0,1,19,20,30]
    elif 'aPI3K' in name:
        return [1,19,20,30]
    elif 'iAKT' in name:
        return [0,1,12,41]
    elif 'aAKT' in name:
        return [1,12,41]
    elif 'ATP' in name:
        return [1,18,40]
    elif 'cAMP' in name:
        return [1,20]
    elif 'iPKA' in name:
        return [0,1,16,41]
    elif 'aPKA' in name:
        return [1,16,41]
    elif 'MITF' in name:
        return [1,10,38]       

true_perm = getTruePermutation(model_names[0])  
ind = np.argmin(AIC)
fit_permute = permutations[ind]
best_params = params[ind]

true_fit = torch.zeros((dX_data.shape)).to(device)
for i, elem in enumerate(model_names):
    true_fit[:,:,i] = eval(elem+'_MODEL_TRUE(A_CANDIDATES.to(torch.float32))')

fit_tc = fits[ind]    
for i in range(4):
    plt.plot(mm.detachTorch(t),dX_data[i,:,0].detach().numpy(),color='blue',marker='o')
    plt.plot(mm.detachTorch(t),fit_tc[i,:],color='red')
    plt.show()

sort_ind = np.argsort(AIC)
sorted_perm = []
sorted_params = []
sorted_aic = []
for i in sort_ind:
    sorted_perm.append(permutations[i])
    sorted_params.append(params[i])
    sorted_aic.append(AIC[i])
sorted_aic = np.array(sorted_aic)
    