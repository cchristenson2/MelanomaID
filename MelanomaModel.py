import numpy as np
import torch
import matplotlib.pyplot as plt

###############################################################################
# Sub-cellular melanoma model - MAPK ##########################################
class MAPKModel(torch.nn.Module):
    ################################################################
    ##### MEK, ERK, and RAS inhibitor in model were not tested #####
    ################################################################
    def __init__(self,drug,inputs,p,t):
        super().__init__()
        self.drug = drug.T
        self.inputs = inputs.T
        self.p = p
        self.t = t
    def vecToDict(self,vec):
        out = {}
        for i, elem in enumerate(self.param_names):
            out[elem] = vec[i]
        return out
    def forward(self, t, states):
        p = self.p
        
        iRAS, aRAS, iRAF_wt, aRAF_wt, RAF_m, iMEK, aMEK, iERK, aERK, \
        iPI3K, aPI3K, iAKT, aAKT, ATP, cAMP, iPKA, aPKA, MITF = states

        RTK, GPCR, PPTASE  = self.inputs.copy()
        
        RTK = np.interp(t,self.t,RTK)
        GPCR = np.interp(t,self.t,GPCR)
        PPTASE = np.interp(t,self.t,PPTASE)
        
        RAF_inhib,MEK_inhib,ERK_inhib,RAS_inhib = self.drug.copy()
        
        RAF_inhib = np.interp(t,self.t,RAF_inhib)
        MEK_inhib = np.interp(t,self.t,MEK_inhib)
        ERK_inhib = np.interp(t,self.t,ERK_inhib)
        RAS_inhib = np.interp(t,self.t,RAS_inhib)
        
        if not isinstance(p,dict):
            p = self.vecToDict(p)

        #Translation of iRAS - decay - activation + deactivation
        d_iRAS = p['s_iRAS'] - p['mu_iRAS']*iRAS - \
                    p['V_aRAS']*RTK*iRAS/(p['K_aRAS']+iRAS) +\
                    p['V_iRAS']*aERK*aRAS/(p['K_iRAS']+aRAS) -\
                    p['k_RAS_inhib']*iRAS*RAS_inhib

        # - decay + activation - deactivation
        d_aRAS = -p['mu_aRAS']*aRAS + \
                    p['V_aRAS']*RTK*iRAS/(p['K_aRAS']+iRAS) -\
                    p['V_iRAS']*aERK*aRAS/(p['K_iRAS']+aRAS)   
        
        #Trasnlation of iRAF_wt - decay + activation - deactivation
        d_iRAF_wt = p['s_iRAF_wt'] - p['mu_iRAF_wt']*iRAF_wt - \
                        p['V_aRAF']*aRAS*iRAF_wt/(p['K_aRAF'] + iRAF_wt) +\
                        p['V_iRAF']*PPTASE*aRAF_wt/(p['K_iRAF']+aRAF_wt)
        # - decay + activation - deactivation            
        d_aRAF_wt = - p['mu_aRAF_wt']*aRAF_wt + \
                        p['V_aRAF']*aRAS*iRAF_wt/(p['K_aRAF'] + iRAF_wt) -\
                        p['V_iRAF']*PPTASE*aRAF_wt/(p['K_iRAF']+aRAF_wt)
        
        #Translation of RAF_m - decay - inhibition
        d_RAF_m = p['s_RAF_m'] - p['mu_RAF_m']*RAF_m -\
                    (p['k_RAF_inhib']*RAF_inhib*RAF_m)
        
        #Translation of MEK - decay - activation through WildType - activation 
        #   through mutant + deactivation
        d_iMEK = p['s_iMEK'] - p['mu_iMEK']*iMEK -\
                    p['V_aMEK']*aRAF_wt*iMEK/(p['K_aMEK'] + iMEK) -\
                    p['V_aMEK_m']*RAF_m*iMEK/(p['K_aMEK_m'] + iMEK) +\
                    p['V_iMEK']*aMEK*PPTASE/(p['K_iMEK'] + aMEK) -\
                    (p['k_MEK_inhib']*MEK_inhib*iMEK)
       
        # - decay + activation through WildType + activation through mutant - deactivation
        d_aMEK = -p['mu_aMEK']*aMEK +\
                    p['V_aMEK']*aRAF_wt*iMEK/(p['K_aMEK'] + iMEK) +\
                    p['V_aMEK_m']*RAF_m*iMEK/(p['K_aMEK_m'] + iMEK) -\
                    p['V_iMEK']*PPTASE*aMEK/(p['K_iMEK'] + aMEK)
        
        #Translation of ERK - decay - activation - deactivation
        d_iERK = p['s_iERK'] - p['mu_iERK']*iERK -\
                    p['V_aERK']*aMEK*iERK/(p['K_aERK'] + iERK) +\
                    p['V_iERK']*PPTASE*aERK/(p['K_iERK']+aERK) -\
                    (p['k_ERK_inhib']*ERK_inhib*iERK)

        # - decay + activation - deactivation
        d_aERK = - p['mu_aERK']*aERK + \
                    p['V_aERK']*aMEK*iERK/(p['K_aERK'] + iERK) -\
                    p['V_iERK']*PPTASE*aERK/(p['K_iERK']+aERK)            
       
        #Translation of PI3K - decay - activation through EGFR - activation
        #   through RAS + inactivation
        d_iPI3K = p['s_iPI3K'] - p['mu_iPI3K']*iPI3K -\
                    p['V_aPI3K_GPCR']*GPCR*iPI3K/(p['K_aPI3K_GPCR']+iPI3K) -\
                    p['V_aPI3K_RTK']*RTK*iPI3K/(p['K_aPI3K_RTK']+iPI3K) +\
                    p['V_iPI3K']*aERK*aPI3K/(p['K_iPI3K'] + aPI3K)
        # - decay + activation through EGFR + activation through RAS -
        #   inactivation
        d_aPI3K = -p['mu_aPI3K']*aPI3K +\
                    p['V_aPI3K_GPCR']*GPCR*iPI3K/(p['K_aPI3K_GPCR']+iPI3K) +\
                    p['V_aPI3K_RTK']*RTK*iPI3K/(p['K_aPI3K_RTK']+iPI3K) -\
                    p['V_iPI3K']*aERK*aPI3K/(p['K_iPI3K'] + aPI3K)
        
        #Translation of AKT - decay - activation + inactivation
        d_iAKT = p['s_iAKT'] - p['mu_iAKT']*iAKT -\
                    p['V_aAKT']*aPI3K*iAKT/(p['K_aAKT'] + iAKT) +\
                    p['V_iAKT']*PPTASE*aAKT/(p['K_iAKT'] + aAKT)
        # - decay + activation - inactivation
        d_aAKT = -p['mu_aAKT']*aAKT +\
                    p['V_aAKT']*aPI3K*iAKT/(p['K_aAKT'] + iAKT) -\
                    p['V_iAKT']*PPTASE*aAKT/(p['K_iAKT'] + aAKT)
                    
        d_ATP = p['s_ATP']*MITF/(p['K_sATP']+MITF) - p['mu_ATP']*ATP -\
                    p['V_acAMP']*GPCR*ATP/(p['K_acAMP'] + ATP)
                    
        d_cAMP = -p['mu_cAMP']*cAMP + p['V_acAMP']*GPCR*ATP/(p['K_acAMP'] + ATP)
        
        d_iPKA = p['s_iPKA'] - p['mu_iPKA']*iPKA -\
                    p['V_aPKA']*cAMP*iPKA/(p['K_aPKA'] + iPKA) +\
                    p['V_iPKA']*PPTASE*aPKA/(p['K_iPKA'] + aPKA)
                    
        d_aPKA = -p['mu_aPKA']*aPKA +\
                    p['V_aPKA']*cAMP*iPKA/(p['K_aPKA'] + iPKA) -\
                    p['V_iPKA']*PPTASE*aPKA/(p['K_iPKA'] + aPKA)
                    
        #Translation of MITF - ERK mediated decay - decay
        d_MITF = p['s_MITF']*aPKA/(p['K_sMITF'] + aPKA) - p['k_MITF']*aERK*MITF - p['mu_MITF']*MITF
                    
        return np.stack([d_iRAS, d_aRAS, d_iRAF_wt, d_aRAF_wt,d_RAF_m, 
                            d_iMEK, d_aMEK, d_iERK, d_aERK, d_iPI3K, d_aPI3K, 
                            d_iAKT, d_aAKT, d_ATP, d_cAMP, d_iPKA, d_aPKA, 
                            d_MITF])
            
def RandomizeMAPKParams(generator):
    params_velocities = ['V_aRAS','V_iRAS','V_aRAF','V_iRAF','V_aMEK','V_iMEK',
                         'V_aERK','V_iERK','V_aPI3K_RTK','V_iPI3K','V_aAKT',
                         'V_iAKT','V_aPI3K_GPCR','V_acAMP','V_aPKA','V_iPKA']
    params_constants = ['K_aRAS','K_iRAS','K_aRAF','K_iRAF','K_aMEK','K_aMEK_m',
                        'K_iMEK','K_aERK','K_iERK','K_aPI3K_RTK','K_iPI3K',
                        'K_aAKT','K_iAKT','K_aPI3K_GPCR','K_acAMP','K_aPKA',
                        'K_iPKA','K_sMITF','K_sATP']
    params_sources = ['s_iRAS','s_iRAF_wt','s_RAF_m','s_iMEK','s_iERK',
                      's_iPI3K','s_iAKT','s_iPKA']
    params_deaths = ['mu_RTK','mu_iRAS','mu_aRAS','mu_iRAF_wt','mu_aRAF_wt',
                     'mu_RAF_m','mu_iMEK','mu_aMEK','mu_iERK','mu_aERK','mu_MITF',
                     'mu_iPI3K','mu_aPI3K','mu_iAKT','mu_aAKT','mu_ATP','mu_cAMP',
                     'mu_iPKA','mu_aPKA']
    params_rates = ['k_MITF']
    params_hill_sources = ['s_ATP','s_MITF']
    
    p = {}
    for elem in params_velocities:
        p[elem] = generator.uniform(low=2.0,high=5.0,size=(1,))[0]
    for elem in params_constants:
        p[elem] = generator.uniform(low=0.5,high=3.0,size=(1,))[0]
    for elem in params_sources:
        p[elem] = generator.uniform(low=0.5,high=2.0,size=(1,))[0]
    for elem in params_deaths:
        p[elem] = generator.uniform(low=0.1,high=1.0,size=(1,))[0]
    for elem in params_rates:
        p[elem] = generator.uniform(low=0.1,high=1.0,size=(1,))[0]
    for elem in params_hill_sources:
        p[elem] = generator.uniform(low=2.0,high=5.0,size=(1,))[0]
        
    #Fast decrease for cAMP
    p['mu_cAMP'] = generator.uniform(low=0.5,high=1.0,size=(1,))[0]
    
    #Ensure mutant raf has greater affinity for mek than non-mutant
    p['V_aMEK_m'] = generator.uniform(low=1.0,high=2.0,size=(1,))[0] * p['V_aMEK'].copy()
    
    #Ensure RAF_m has high affinity for inhibitor
    p['k_RAF_inhib'] = generator.uniform(low=5.0,high=10.0,size=(1,))[0]
    p['k_MEK_inhib'] = generator.uniform(low=5.0,high=10.0,size=(1,))[0]
    p['k_ERK_inhib'] = generator.uniform(low=5.0,high=10.0,size=(1,))[0]
    p['k_RAS_inhib'] = generator.uniform(low=5.0,high=10.0,size=(1,))[0]
    return p

###############################################################################
# Cellular melanoma model - 3 phenotypes ######################################

class CellModel(torch.nn.Module):
    def __init__(self, initial, proteins, p, t):
        super().__init__()
        self.initial = initial
        self.proteins = proteins
        self.p = p
        self.t = t
    def sigmoid(self,x):
        return 1/(1+torch.exp(-500*(x-0.05)))
    def vecToDict(self,vec):
        out = {}
        for i, elem in enumerate(self.param_names):
            out[elem] = vec[i]
        return out
    def forward(self, t, states):
        p = self.p
        
        S,D,R = states
        T = S+D+R
        
        ERK  = np.interp(t,self.t,self.proteins[:,0])
        AKT  = np.interp(t,self.t,self.proteins[:,2])
        
        d_ERK  = ERK - self.initial[0]
        d_PI3K = np.interp(t,self.t,self.proteins[:,1]) - self.initial[1]
        d_MITF = np.interp(t,self.t,self.proteins[:,3]) - self.initial[3]
        
        if not isinstance(p,dict):
            p = self.vecToDict(p)

        dS = (p['r_S']*ERK**p['n_rS'])/(p['K_rS']**p['n_rS'] + ERK**p['n_rS'])*\
                S*(1-((T)/(p['theta']))) - \
                S*p['k_S']*(torch.abs(d_ERK)**p['n_kS'])/\
                    (p['K_kS']**p['n_kS'] + torch.abs(d_ERK)**p['n_kS'])*\
                    self.sigmoid(-d_ERK) - \
                S*p['k_SD']*(d_MITF**p['n_kSD'])/\
                    (p['K_kSD']**p['n_kSD'] + d_MITF**p['n_kSD'])*\
                    self.sigmoid(d_MITF) - \
                S*p['k_SR']*(d_PI3K**p['n_kSR'])/(p['K_kSR']**p['n_kSR'] + d_PI3K**p['n_kSR'])*\
                self.sigmoid(d_PI3K)
        
        dD = S*p['k_SD']*(d_MITF**p['n_kSD'])/\
                (p['K_kSD']**p['n_kSD'] + d_MITF**p['n_kSD'])*\
                self.sigmoid(d_MITF)
                
        dR = (p['r_R']*AKT**p['n_rR'])/(p['K_rR']**p['n_rR'] + AKT**p['n_rR'])*\
                R*(1-((T)/(p['theta']))) + \
                S*p['k_SR']*(d_PI3K**p['n_kSR'])/\
                    (p['K_kSR']**p['n_kSR'] + d_PI3K**p['n_kSR'])*\
                    self.sigmoid(d_PI3K)
        
        return torch.stack([dS,dD,dR])

def RandomizeCellParams(generator):
    params_prolif = ['r_S','r_R']
    params_constants = ['K_rS','K_rR','K_kS','K_kSD','K_kSR']
    params_coop = ['n_rS','n_rR','n_kS','n_kSD','n_kSR']
    
    p = {}
    for elem in params_prolif:
        p[elem] = torch.tensor(generator.uniform(low=0.10,high=0.50,size=(1,))[0],dtype=torch.float32)
    
    p['k_S']  = torch.tensor(generator.uniform(low=0.1,high=0.3,size=(1,))[0],dtype=torch.float32)
    p['k_SD'] = torch.tensor(generator.uniform(low=0.01,high=0.05,size=(1,))[0],dtype=torch.float32)
    p['k_SR'] = torch.tensor(generator.uniform(low=0.01,high=0.05,size=(1,))[0],dtype=torch.float32)
    
    for elem in params_constants:
        p[elem] = torch.tensor(generator.uniform(low=0.1,high=1.0,size=(1,))[0],dtype=torch.float32)
    for elem in params_coop:
        p[elem] = torch.tensor(generator.uniform(low=0.5,high=2.5,size=(1,))[0],dtype=torch.float32)
        
    p['theta'] = torch.tensor(1.0,dtype=torch.float32)
    
    return p

###############################################################################
# Plotting functions ##########################################################

#Plot all time courses from the network model    
def plotFullNetwork(t,X,compare = [],title=None):
    fig, axes = plt.subplots(3,6,layout='constrained',figsize=(18,9))
    Labels = ['iRAS', 'aRAS', 'iRAF_wt', 'aRAF_wt', 'RAF_m', 'iMEK',
              'aMEK', 'iERK', 'aERK','iPI3K', 'aPI3K', 'iAKT',
              'aAKT','ATP','cAMP','iPKA','aPKA','MITF']
    cnt = 0
    for row, ax_row in enumerate(axes):
        for col, ax in enumerate(ax_row):
            try:
                ax.plot(t,X[cnt,:])
                if len(compare) != 0:
                    ax.plot(t,compare[cnt,:])
                ax.set_xlabel('Time')
                ax.set_ylabel(Labels[cnt])
            except:
                continue
            cnt+=1
    if title:
        fig.suptitle(title)
    plt.show() 

#Plot all time courses from the cell model
def plotCells(t,X,title=None):
    fig, ax = plt.subplots(1,1,layout='constrained',figsize=(15,6))
    Labels = ['S','D','R']
    for i in range(3):
        ax.plot(t,X[i,:],label=Labels[i])
    ax.plot(t,np.sum(X,axis=0),label='Total')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Cell Number')
    ax.legend(loc='upper left')
    # ax.set_ylim([0.0,1.5])
    
    if title:
        fig.suptitle(title)
    plt.show()

###############################################################################
# Random ######################################################################   
def detachTorch(x):
    return x.cpu().detach().numpy()        
        
        