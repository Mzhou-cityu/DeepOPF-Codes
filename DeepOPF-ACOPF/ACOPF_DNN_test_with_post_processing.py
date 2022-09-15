# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:26:56 2019

@author: xpan
"""
#AC-OPF-Case30
import time
from pypower.runpf import runpf
from pypower.runpf1 import runpf1
from pypower.ppoption import ppoption
from pypower.runopf import runopf
from scipy.sparse import csr_matrix as sparse
from pypower.loadcase import loadcase
import numpy as np
total = 0
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Function
from pypower.ext2int import ext2int1
torch.manual_seed(1)    # reproducible

# parameter
training_amout = 10000
test_amout = 2500
NN_input_number = 0 
NN_output_number = 0   
neural_number = 512
batch_size_training = 32  
batch_size_valtest = 1  
#learning parameters
learning_rate = 1e-3
num_epoches = 100

Bus_number = 0
All_Gen_number = 0
Slack_bus = 0
standval = 0
SYNC_GEN_index_list = []
Active_GEN_index_list = []
Active_GEN_index_list2PG_train = []
Gen_index_list = []
Slack_GEN_index_list = []   

P_Bus = np.zeros([1,1], dtype ='float32')
Q_Bus = np.zeros([1,1], dtype ='float32')
P_Load = np.zeros([1,1], dtype ='float32')
Q_Load = np.zeros([1,1], dtype ='float32')
V_Bus = np.zeros([1,1], dtype ='float32')
NumOfGen = 0
Node_index = np.zeros([1,1],dtype='float32')
Lowbound_Volt=np.zeros([1,1],dtype='float32')
Upbound_Volt=np.zeros([1,1],dtype='float32')
Lowbound_Slack_V=np.zeros([1,1],dtype='float32')
Upbound_Slack_V=np.zeros([1,1],dtype='float32')
PG_Lowbound=np.zeros([1,1],dtype='float32')
PG_Upbound=np.zeros([1,1],dtype='float32')
PG_Lowbound_Active=np.zeros([1,1],dtype='float32')
PG_Upbound_Active=np.zeros([1,1],dtype='float32')
Load_index = np.zeros([1,1],dtype='float32')
Gen_index = np.zeros([1,1], dtype ='float32')
Branch_list = []

class Penalty_ACPF(Function):
    @staticmethod
    def forward(ctx, NN_input, load):
        ctx.save_for_backward(NN_input, load)
        # detach so we can cast to NumPy
        NN_input, load = NN_input.detach(), load.detach()
        NN_input_np = NN_input.numpy()
        #PG      
        Pred_PG_train = PG_Lowbound_Active+(PG_Upbound_Active-PG_Lowbound_Active)*NN_input_np[:,0:PG_Lowbound_Active.shape[1]]
        #Slack_bus_Voltage
        Pred_Slack_V_train = Lowbound_Slack_V+(Upbound_Slack_V-Lowbound_Slack_V)*NN_input_np[:,PG_Lowbound_Active.shape[1]]
        Pred_Slack_V_train = Pred_Slack_V_train.reshape(NN_input_np.shape[0],-1)
        #Voltage
        Pred_Volt_train = Lowbound_Volt+(Upbound_Volt-Lowbound_Volt)*NN_input_np[:,PG_Lowbound_Active.shape[1]+1:NN_output_number]
        Pred_Volt_train = np.dot(Pred_Volt_train,Bus2Gen)
        #input
        train_x_np = load.numpy()
        #P_D_temp_train =  train_x_np[:,0:NN_input_number]
        #Q_D_temp_train =  train_x_np[:,NN_input_number:2*NN_input_number] 
        P_D_temp_train =  (P_D_train_std+1e-8)*train_x_np[:,0:NN_input_number]+P_D_train_mean
        Q_D_temp_train =  (Q_D_train_std+1e-8)*train_x_np[:,NN_input_number:2*NN_input_number]+(Q_D_train_mean)                
        
        result = 0
        for i in range(NN_input_np.shape[0]):
            #solving PF
            mpc_pf_train = loadcase('pglib_opf_case2000_goc_rev.py')
            Qg_index_zero_limit = np.where((mpc_pf_train['gen'][:,3]==0) & (mpc_pf_train['gen'][:,4] == 0))
            Qg_index_no_zero_limit = np.where((mpc_pf_train['gen'][:,3]!=0) & (mpc_pf_train['gen'][:,4] != 0))
            #modify QG
            mpc_pf_train['gen'][Qg_index_zero_limit,3] = 10
            mpc_pf_train['gen'][Qg_index_zero_limit,4] = -10
            mpc_pf_train['gen'][Qg_index_no_zero_limit,3] = mpc_pf_train['gen'][Qg_index_no_zero_limit,3]*5
            mpc_pf_train['gen'][Qg_index_no_zero_limit,4] = mpc_pf_train['gen'][Qg_index_no_zero_limit,4]*5            
            #pd,qd
            mpc_pf_train["bus"][:,2] =  P_D_temp_train[i,:]
            mpc_pf_train["bus"][:,3] =  Q_D_temp_train[i,:]


            global Q_G_all_train_mean 
            global P_G_all_train_mean 
            global V_all_train_mean 
            global Theta_all_train_mean
            global V_G_all_train_mean        
            #initialization of Pg and Qg
            # mpc_pf_train["bus"][:,7] = V_all_train_mean
            # mpc_pf_train["bus"][:,8] = Theta_all_train_mean             
            # mpc_pf_train["gen"][:,1] = P_G_all_train_mean
            # mpc_pf_train["gen"][:,2] = Q_G_all_train_mean
            
            #set NN predicted variables
            #pg,vg
            mpc_pf_train["gen"][SYNC_GEN_index_list,1] = 0
            mpc_pf_train["gen"][FIX_GEN_index_list,1] = mpc_pf_train["gen"][FIX_GEN_index_list,8]
            mpc_pf_train["gen"][Active_GEN_index_list,1] = Pred_PG_train[i,:]*standval                                        
            mpc_pf_train["gen"][:,5] = Pred_Volt_train[i,:]
            #slack vg
            mpc_pf_train["gen"][Slack_GEN_index_list,5] = Pred_Slack_V_train[i,:]

            ppopt = ppoption()
            ppopt = ppoption(ppopt, OUT_ALL=0, VERBOSE=False, ENFORCE_Q_LIMS=False) 

            #run power flow
            r1_pf = runpf1(mpc_pf_train,ppopt)
            
            #r1_pf = decouple_flow(mpc_pf_train)                                       
            penalty = zero_order_penalty_abs(r1_pf)
            result = result + penalty
            
        return torch.as_tensor(result)

    @staticmethod
    def backward(ctx, grad_output):
#        grad_output = grad_output.detach()
        NN_input, load = ctx.saved_tensors
        NN_input_temp, load_temp = NN_input.detach(), load.detach()
        NN_input_modified_temp = NN_input_temp.clone().detach().numpy()

        h=1e-4
        Est_grad = np.zeros([NN_input.shape[0],NN_input.shape[1]],dtype ='float32')
        for i in range(NN_input.shape[0]):
            Original_value = NN_input_modified_temp[i,:]
            #sample random unit vector
            vector1 = sample_spherical(1,NN_output_number)
            vector1 = np.squeeze(vector1)
            #estimate gradient
            NN_input_modified_temp[i,:] = Original_value + vector1*h
            #PG      
            Pred_PG_train_plus_h = PG_Lowbound_Active+(PG_Upbound_Active-PG_Lowbound_Active)*NN_input_modified_temp[i,0:PG_Lowbound_Active.shape[1]]
            #Slack_bus_Voltage
            Pred_Slack_V_train_plus_h = Lowbound_Slack_V+(Upbound_Slack_V-Lowbound_Slack_V)*NN_input_modified_temp[i,PG_Lowbound_Active.shape[1]]
            #Voltage
            Pred_Volt_train_plus_h = Lowbound_Volt+(Upbound_Volt-Lowbound_Volt)*NN_input_modified_temp[i,PG_Lowbound_Active.shape[1]+1:NN_output_number]                 
            train_x_np = load_temp.numpy()
            
            P_D_temp_train =  train_x_np[i,0:NN_input_number]
            Q_D_temp_train =  train_x_np[i,NN_input_number:2*NN_input_number]                   
            #if do pre-processing            
#            P_D_temp_train =  (P_D_train_std+1e-8)*train_x_np[i,0:NN_input_number]+P_D_train_mean
#            Q_D_temp_train =  (Q_D_train_std+1e-8)*train_x_np[i,NN_input_number:2*NN_input_number]+(Q_D_train_mean)                    
            
            #solving PF
            mpc_pf_train_puls_h = loadcase('pglib_opf_case2000_goc_rev.py')
            [m,n] = mpc_pf_train_puls_h["bus"].shape
            count = 0
            for index in range(0,m):
#                if mpc_pf_train_puls_h["bus"][index,2] != 0 or mpc_pf_train_puls_h["bus"][index,3] != 0:
                mpc_pf_train_puls_h["bus"][index,2] =  P_D_temp_train[count]
                mpc_pf_train_puls_h["bus"][index,3] =  Q_D_temp_train[count]
                count = count + 1
            
            [m_gen,n_gen]=mpc_pf_train_puls_h["gen"].shape
            count_PG = 0
            count_Volt = 0 
            for index in range(0,m_gen):
#                gen_index = np.where(mpc_pf_train_puls_h["bus"][:,0] == mpc_pf_train_puls_h["gen"][index,0])[0][0]
#                if mpc_pf_train_puls_h["bus"][gen_index,1] != 3:
                if index in SYNC_GEN_index_list:
                    mpc_pf_train_puls_h["gen"][index,1] =  0
                    mpc_pf_train_puls_h["gen"][index,5] =  Pred_Volt_train_plus_h[count_Volt]
                    count_Volt = count_Volt + 1 
                elif index in Active_GEN_index_list:
                    mpc_pf_train_puls_h["gen"][index,1] =  Pred_PG_train_plus_h[0,count_PG]*standval
                    mpc_pf_train_puls_h["gen"][index,5] =  Pred_Volt_train_plus_h[count_Volt]                                          
                    count_PG = count_PG+1
                    count_Volt = count_Volt + 1 
                else:
                    mpc_pf_train_puls_h["gen"][index,5] =  Pred_Slack_V_train_plus_h[0,0]
            #run power flow
            ppopt = ppoption()
            ppopt = ppoption(ppopt, OUT_ALL=0, VERBOSE=False, ENFORCE_Q_LIMS=False)             
            r1_plus_h_pf = runpf1(mpc_pf_train_puls_h,ppopt)
            penalty_plus_h = zero_order_penalty_abs(r1_plus_h_pf)                    
            #################################################################################################################
            #estimate gradient
            NN_input_modified_temp[i,:] = Original_value - vector1*h         
            #PG      
            Pred_PG_train_minus_h = PG_Lowbound_Active+(PG_Upbound_Active-PG_Lowbound_Active)*NN_input_modified_temp[i,0:PG_Lowbound_Active.shape[1]]             
            #Slack_bus_Voltage
            Pred_Slack_V_train_minus_h = Lowbound_Slack_V+(Upbound_Slack_V-Lowbound_Slack_V)*NN_input_modified_temp[i,PG_Lowbound_Active.shape[1]]
            #Voltage
            Pred_Volt_train_minus_h = Lowbound_Volt+(Upbound_Volt-Lowbound_Volt)*NN_input_modified_temp[i,PG_Lowbound_Active.shape[1]+1:NN_output_number]                    
            #solving PF
            mpc_pf_train_minus_h = loadcase('pglib_opf_case2000_goc_rev.py')
                
            count = 0
            [m_bus,n_bus]=mpc_pf_train_minus_h["bus"].shape
            for index in range(0,m_bus):
#                if mpc_pf_train_minus_h["bus"][index,2] != 0 or mpc_pf_train_minus_h["bus"][index,3] != 0:
                mpc_pf_train_minus_h["bus"][index,2] =  P_D_temp_train[count]
                mpc_pf_train_minus_h["bus"][index,3] =  Q_D_temp_train[count]
                count = count + 1
 
            [m_gen,n_gen]=mpc_pf_train_minus_h["gen"].shape
            count_PG = 0
            count_Volt = 0 
            for index in range(0,m_gen):
#                gen_index = np.where(mpc_pf_train_puls_h["bus"][:,0] == mpc_pf_train_minus_h["gen"][index,0])[0][0]
#                if mpc_pf_train_minus_h["bus"][gen_index,1] != 3:
                if index in SYNC_GEN_index_list:
                    mpc_pf_train_minus_h["gen"][index,1] =  0
                    mpc_pf_train_minus_h["gen"][index,5] =  Pred_Volt_train_minus_h[count_Volt]
                    count_Volt = count_Volt + 1 
                elif index in Active_GEN_index_list:
                    mpc_pf_train_minus_h["gen"][index,1] =  Pred_PG_train_minus_h[0,count_PG]*standval
                    mpc_pf_train_minus_h["gen"][index,5] =  Pred_Volt_train_minus_h[count_Volt]                                          
                    count_PG = count_PG+1
                    count_Volt = count_Volt + 1 
                else:
                    mpc_pf_train_minus_h["gen"][index,5] =  Pred_Slack_V_train_minus_h[0,0]
            #run power flow
            ppopt = ppoption()
            ppopt = ppoption(ppopt, OUT_ALL=0, VERBOSE=False, ENFORCE_Q_LIMS=False)            
            r1_minus_h_pf = runpf1(mpc_pf_train_minus_h,ppopt)
            penalty_minus_h = zero_order_penalty_abs(r1_minus_h_pf)                                                             
            #gradient descent update
            gradient_estimate = (penalty_plus_h - penalty_minus_h)*vector1/(2*h)
            Est_grad[i,:] = gradient_estimate
            #recover
            NN_input_modified_temp[i,:] =  Original_value               
#        print(Est_grad)
#        print("\n")
#        print(grad_output)
        return torch.from_numpy(Est_grad)*grad_output, None

# neural network definition
class Neuralnetwork(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, 4*n_hidden)
        self.layer2 = nn.Linear(4*n_hidden, 2*n_hidden)
        #self.layer3 = nn.Linear(2*n_hidden, n_hidden)
        self.layer4 = nn.Linear(2*n_hidden, out_dim)
        #Layer for AC-PF
        self.layer6 = Penalty_ACPF.apply

    def forward(self, x):
        x0 = F.relu(self.layer1(x))
        x1 = F.relu(self.layer2(x0))
        #x2 = F.relu(self.layer3(x1))
        x_sol = torch.sigmoid(self.layer4(x1))
        #Layer for AC-PF
        x_penalty = self.layer6(x_sol,x)
        return x_sol, x_penalty


#penalty term (absolute value)
def zero_order_penalty_abs(pf_results=None):
    standval = pf_results[0]["baseMVA"]
    ctol = 1e-03
    #PV-PG_slack_bus
    On_index = np.where(pf_results[0]["gen"][:,7]==1)[0]    
    Pg_temp1 = (pf_results[0]["gen"][On_index,9]-pf_results[0]["gen"][On_index,1])/standval
    Pg_temp1[Pg_temp1 < ctol]=0
    Pg_temp2 = (pf_results[0]["gen"][On_index,1]-pf_results[0]["gen"][On_index,8])/standval
    Pg_temp2[Pg_temp2 < ctol]=0   
    PG_penalty = np.abs(Pg_temp1)+np.abs(Pg_temp2)

	
    #PV-QG
    Qg_temp1 = (pf_results[0]["gen"][On_index,4]-pf_results[0]["gen"][On_index,2])/standval
    Qg_temp1[Qg_temp1 < ctol] = 0
    Qg_temp2 = (pf_results[0]["gen"][On_index,2]-pf_results[0]["gen"][On_index,3])/standval
    Qg_temp2[Qg_temp2 < ctol] = 0
    QG_penalty = np.abs(Qg_temp1)+np.abs(Qg_temp2)
	
	
#    #PQ-V
    PQ_index = np.where(pf_results[0]["bus"][:,1]==1)[0]
    V_temp1 = pf_results[0]["bus"][PQ_index,12]-pf_results[0]["bus"][PQ_index,7]
    V_temp1[V_temp1 < ctol] = 0
    V_temp2 = pf_results[0]["bus"][PQ_index,7]-pf_results[0]["bus"][PQ_index,11]
    V_temp2[V_temp2 < ctol] = 0
    V_penalty = np.abs(V_temp1)+np.abs(V_temp2)


    #Branch
    Ff = abs(pf_results[0]["branch"][:, 13] + 1j * pf_results[0]["branch"][:, 14])
    Ft = abs(pf_results[0]["branch"][:, 15] + 1j * pf_results[0]["branch"][:, 16])

    [m_branch,n_branch]=pf_results[0]["branch"].shape
    Branch_index = np.where(pf_results[0]["branch"][:,5] != 0)[0]
    Branch_bound = pf_results[0]["branch"][Branch_index, 5]
    Ff_temp = (Ff[Branch_index]-Branch_bound)/standval
    ctol = 1e-2 
    Ff_temp[Ff_temp< ctol] = 0
    Ft_temp = (Ft[Branch_index]-Branch_bound)/standval
    Ft_temp[Ft_temp< ctol] = 0    
    Ff_penalty = np.abs(Ff_temp)
    Ft_penalty = np.abs(Ft_temp)

#    
    return np.sum(PG_penalty)+np.sum(QG_penalty)+np.sum(V_penalty)+np.sum(Ff_penalty)+np.sum(Ft_penalty)

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    #vec /= np.linalg.norm(vec, axis=0)
    return vec


def model_generation(mpc=None):
    #read admittance matrix from pypower
    global standval
    
    standval, bus, gen, branch = \
    mpc['baseMVA'], mpc['bus'], mpc['gen'], mpc['branch']

    ## switch to internal bus numbering and build admittance matrices
    _, bus, gen, branch = ext2int1(bus, gen, branch)
    
    #genrator
    mpc_gen=gen
    ## check that all buses have a valid BUS_TYPE
    bt = bus[:, 1]    
    ## determine which buses, branches, gens are connected and
    ## in-service
    n2i = sparse((range(bus.shape[0]), (bus[:, 0], np.zeros(bus.shape[0]))),
        shape=(max(bus[:, 0].astype(int)) + 1, 1))
    n2i = np.array( n2i.todense().flatten() )[0, :] # as 1D array
    bs = (bt != None)                               ## bus status
    gs = ( (gen[:, 7] > 0) &          ## gen status
    bs[ n2i[gen[:, 0].astype(int)] ] )
    mG = len(gs.ravel().nonzero()[0])    ## on and connected
#    nG = gen.shape[1]
    # gen_on = np.where(mpc_gen[:,7]!=0)[0]

    global PG_Lowbound
    global PG_Upbound

    PG_Lowbound = np.zeros([1, mG-1],dtype='float32')
    PG_Upbound = np.zeros([1, mG-1],dtype='float32')

    Load_index_list=[]
    global V_non_load
    global SYNC_GEN_index_list
    global FIX_GEN_index_list
    global Active_GEN_index_list
    global Slack_GEN_index_list
    global Volt_PV_index_list
    global Volt_PV_GEN_index_list
    global All_gen_list
    global PV_to_Gen_index_list
    #index transfer
    global Slack_bus
    global slack_bus_index
    PV_to_Gen_index_temp_list=[]
    for j in range(0,bus.shape[0],1):
#        if (bus[j,2] != 0 or bus[j,3] != 0):
        Load_index_list.append(bus[j,0])   
        if (bus[j,1] == 3):
            Slack_bus = int(bus[j,0])
            slack_bus_index = j
        if (bus[j,1] == 2):
            Volt_PV_index_list.append(int(j))
            PV_to_Gen_index_temp_list.append(int(bus[j,0]))
        if bus[j,1] != 1:
            non_load_index_list.append(int(j))
        
    
    gen_index = 0
    slack_bus_index = 0
    for j in range(0,mpc['gen'].shape[0],1):
        if Slack_bus!= mpc['gen'][j,0] and mpc['gen'][j,7] !=0:
            Volt_PV_GEN_index_list.append(j)
            gen_idx=PV_to_Gen_index_temp_list.index(mpc['gen'][j,0])
            PV_to_Gen_index_list.append(gen_idx)
            PG_Upbound[0,gen_index] = mpc['gen'][j,8]/standval
            PG_Lowbound[0,gen_index] = mpc['gen'][j,9]/standval
            gen_index = gen_index +1
            if (mpc['gen'][j,8] == 0 and mpc['gen'][j,9] == 0):
                SYNC_GEN_index_list.append(j)
            elif (mpc['gen'][j,8] > 0 and mpc['gen'][j,8] == mpc['gen'][j,9]):
                FIX_GEN_index_list.append(j)
            else:     
                Active_GEN_index_list.append(j)
#                active_gen_index += 1
        elif mpc['gen'][j,7] !=0:
            Slack_GEN_index_list.append(j)


    #Vm of bus 2 Vm of gen
    global Bus2Gen
    Bus2Gen = np.zeros([len(Volt_PV_index_list), mpc['gen'].shape[0]],dtype='float32')
    for i in range(0,len(Volt_PV_index_list)):
        gen_idx = np.where((Volt_PV_index_list[i]==mpc['gen'][:,0]) & (mpc['gen'][:,7] == 1))[0]
        Bus2Gen[i,gen_idx] = 1

    global PG_Lowbound_Active
    PG_Lowbound_Active = np.zeros([1,len(Active_GEN_index_list)],dtype='float32')
    global PG_Upbound_Active
    PG_Upbound_Active = np.zeros([1,len(Active_GEN_index_list)],dtype='float32')
    
    for j in range(0,PG_Upbound_Active.shape[1]):
        PG_Upbound_Active[0,j] = mpc['gen'][Active_GEN_index_list[j],8]/standval
        PG_Lowbound_Active[0,j] = mpc['gen'][Active_GEN_index_list[j],9]/standval
        if Active_GEN_index_list[j] >slack_bus_index:
            Active_GEN_index_list2PG_train.append(Active_GEN_index_list[j]-1)
        else:
            Active_GEN_index_list2PG_train.append(Active_GEN_index_list[j])

            
    global Gen_index
    global Load_index
    Gen_index = np.array(non_load_index_list)
    Load_index = np.array(Load_index_list)      

    global NN_input_number
    global NN_output_number
    #input/output of NN    
    NN_output_number = len(Active_GEN_index_list)+len(Volt_PV_index_list)+1
    NN_input_number = Load_index.shape[0]
    global All_Gen_number
    All_Gen_number = mG
    
    global Lowbound_Volt
    global Upbound_Volt
    global Lowbound_Slack_V
    global Upbound_Slack_V
    for i in range(mpc['gen'].shape[0]):
        gen_bus_index = np.where(mpc['gen'][i,0]== mpc['bus'][:,0])[0][0]
        if mpc['bus'][gen_bus_index,1]== 3 and mpc['gen'][i,7] !=0:
            Lowbound_Slack_V[0,0] = np.power(mpc['bus'][gen_bus_index,12],1)
            Upbound_Slack_V[0,0] = np.power(mpc['bus'][gen_bus_index,11],1)

    Lowbound_Volt = np.zeros((len(Volt_PV_index_list),),dtype='float32')
    Upbound_Volt = np.zeros((len(Volt_PV_index_list),),dtype='float32')
    Lowbound_Volt = Lowbound_Volt.T
    Upbound_Volt = Upbound_Volt.T
    index = 0
    for j in range(0,bus.shape[0],1):
        if (bus[j,1] == 2):
            Lowbound_Volt[index] = np.power(mpc['bus'][j,12],1)
            Upbound_Volt[index] = np.power(mpc['bus'][j,11],1)            
            index = index + 1
        elif bus[i,1]== 3:
            Lowbound_Slack_V[0,0] = np.power(bus[i,12],1)
            Upbound_Slack_V[0,0] = np.power(bus[i,11],1)
            
if __name__ == '__main__':
    # cores = mp.cpu_count()
    Volt_PV_index_list = []
    Volt_PV_GEN_index_list = []
    Bus_to_Gen_list = []
    SYNC_GEN_index_list = []
    FIX_GEN_index_list = []    
    Active_GEN_index_list = []
    Slack_GEN_index_list = []
    Active_GEN_index_list2PG_train = []
    non_load_index_list = []
    PV_to_Gen_index_list = []
    #load case
    mpc = loadcase('pglib_opf_case30_ieee.py')    
    dir = ''
    model_generation(mpc)    
    #read data
    P_D = np.load(dir+'P_D.npy').astype(np.float32) 
    Q_D = np.load(dir+'Q_D.npy').astype(np.float32)
    P_G_all = np.load(dir+'P_G_all.npy').astype(np.float32)
    P_G = P_G_all[:,Active_GEN_index_list]
    Q_G = np.load(dir+'Q_G_all.npy').astype(np.float32)    
    # Va = np.load('Va_data.npy').astype(np.float32)     
    Vm = np.load(dir+'Vm.npy').astype(np.float32)
    V =  Vm[:,Volt_PV_index_list]
    Slack_V = Vm[:,Slack_bus]    
    #divide into 1000 per group
    segment1 = P_D.shape[0]/1000
    #generate data array
    num_of_group = int(np.floor(P_D.shape[0] / segment1))    
    ############################################## 
    # Preprocessing training data
    segment = int(training_amout/1000)
    P_D_train = P_D[0:num_of_group * (segment)]
    #P_D_train_normolization = (P_D_train)
    #if do pre-processing
    P_D_train_mean = np.mean(P_D_train, axis=0)
    P_D_train_std = np.std(P_D_train, axis=0)
    P_D_train_std_tensor = torch.from_numpy(P_D_train_std) 
    P_D_train_mean_tensor = torch.from_numpy(P_D_train_mean) 
    P_D_train_normolization = (P_D_train - P_D_train_mean) / (P_D_train_std+1e-8)

    Q_D_train = Q_D[0:num_of_group * (segment)]
    #Q_D_train_normolization = (Q_D_train)
    #if do pre-processing
    Q_D_train_mean = np.mean(Q_D_train, axis=0)
    Q_D_train_std = np.std(Q_D_train, axis=0)
    Q_D_train_std_tensor = torch.from_numpy(Q_D_train_std) 
    Q_D_train_mean_tensor = torch.from_numpy(Q_D_train_mean) 
    Q_D_train_normolization = (Q_D_train - Q_D_train_mean) / (Q_D_train_std+1e-8)

    D_train_normolization = np.concatenate((P_D_train_normolization,Q_D_train_normolization),axis = 1)    
    
    P_G_train = P_G[0:num_of_group * (segment)]/standval
    Lowbound_PG_training = np.tile(PG_Lowbound_Active,(len(P_G_train),1))
    Upbound_PG_training= np.tile(PG_Upbound_Active,(len(P_G_train),1))
    PG_train_normolization = (P_G_train-Lowbound_PG_training)/(Upbound_PG_training-Lowbound_PG_training)

    Volt_train = V[0:num_of_group * (segment)]
    Lowbound_Volt_training = np.tile(Lowbound_Volt,(len(Q_D_train),1))
    Upbound_Volt_training= np.tile(Upbound_Volt,(len(Q_D_train),1))
    Volt_train_normolization = (Volt_train-Lowbound_Volt_training)/(Upbound_Volt_training-Lowbound_Volt_training)

    Slack_V_train = Slack_V[0:num_of_group * (segment)].reshape(training_amout,1)
    Lowbound_Slack_V_training = np.tile(Lowbound_Slack_V,(len(Q_D_train),1)).reshape(training_amout,1) 
    Upbound_Slack_V_training= np.tile(Upbound_Slack_V,(len(Q_D_train),1)).reshape(training_amout,1)
    Slack_V_train_normolization = (Slack_V_train-Lowbound_Slack_V_training)/(Upbound_Slack_V_training-Lowbound_Slack_V_training)

    PV_train_temp = np.concatenate((PG_train_normolization,Slack_V_train_normolization),axis = 1)
    PV_train_normolization = np.concatenate((PV_train_temp, Volt_train_normolization),axis = 1)

    #init
    V_all_train_mean = np.mean(Vm[0:num_of_group * (segment)], axis=0)
    Theta_all_train_mean = np.zeros(NN_input_number, dtype='float32')
    P_G_all_train_mean = np.mean(P_G_all[0:num_of_group * (segment)], axis=0)
    Q_G_all_train_mean = np.mean(Q_G[0:num_of_group * (segment)], axis=0)
    V_G_all_train_mean = np.mean(Vm[0:num_of_group * (segment),Volt_PV_index_list], axis=0)

	#test data
    start_idx = P_D.shape[0]-test_amout
    P_D_test = P_D[start_idx:P_D.shape[0]]
    P_D_test_normolization = (P_D_test- P_D_train_mean) / (P_D_train_std+1e-8)
    #if do pre-processing
#    P_D_train_mean = np.mean(P_D_train, axis=0)
#    P_D_train_std = np.std(P_D_train, axis=0)
#    P_D_train_std_tensor = torch.from_numpy(P_D_train_std) 
#    P_D_train_mean_tensor = torch.from_numpy(P_D_train_mean) 
#    P_D_train_normolization = (P_D_train - P_D_train_mean) / (P_D_train_std+1e-8)

    Q_D_test = Q_D[start_idx:P_D.shape[0]]
    Q_D_test_normolization = (Q_D_test - Q_D_train_mean) / (Q_D_train_std+1e-8)
    #if do pre-processing
#    Q_D_train_mean = np.mean(Q_D_train, axis=0)
#    Q_D_train_std = np.std(Q_D_train, axis=0)
#    Q_D_train_std_tensor = torch.from_numpy(Q_D_train_std) 
#    Q_D_train_mean_tensor = torch.from_numpy(Q_D_train_mean) 
#    Q_D_train_normolization = (Q_D_train - Q_D_train_mean) / (Q_D_train_std+1e-8)

    D_test_normolization = np.concatenate((P_D_test_normolization,Q_D_test_normolization),axis = 1)

    P_G_test = P_G[start_idx:P_D.shape[0]]/standval
    Lowbound_PG_test = np.tile(PG_Lowbound_Active,(len(P_G_test),1))
    Upbound_PG_test = np.tile(PG_Upbound_Active,(len(P_G_test),1))
    PG_test_normolization = (P_G_test-Lowbound_PG_test)/(Upbound_PG_test-Lowbound_PG_test)

    Volt_test = V[start_idx:P_D.shape[0]]
    Lowbound_Volt_test = np.tile(Lowbound_Volt,(len(Q_D_test),1))
    Upbound_Volt_test = np.tile(Upbound_Volt,(len(Q_D_test),1))
    Volt_test_normolization = (Volt_test-Lowbound_Volt_test)/(Upbound_Volt_test-Lowbound_Volt_test)

    Slack_V_test = Slack_V[start_idx:P_D.shape[0]].reshape(test_amout,1)
    Lowbound_Slack_V_test = np.tile(Lowbound_Slack_V,(len(Q_D_test),1)).reshape(test_amout,1) 
    Upbound_Slack_V_test = np.tile(Upbound_Slack_V,(len(Q_D_test),1)).reshape(test_amout,1)
    Slack_V_test_normolization = (Slack_V_test-Lowbound_Slack_V_test)/(Upbound_Slack_V_test-Lowbound_Slack_V_test)

    PV_test_temp = np.concatenate((PG_test_normolization,Slack_V_test_normolization),axis = 1)
    PV_test_normolization = np.concatenate((PV_test_temp, Volt_test_normolization),axis = 1)
    
    PG1_difference = np.zeros((P_G_test.shape[0],P_G_test.shape[1]), dtype ='float32')
    Volt1_difference = np.zeros((Volt_test.shape[0],Volt_test.shape[1]), dtype ='float32')
    Slack1_V_difference = np.zeros((Slack_V_test.shape[0],1), dtype ='float32')
  
    criterion = nn.MSELoss(reduction='mean')
    ########################################
    model = Neuralnetwork(2*NN_input_number, neural_number, NN_output_number)
    model.load_state_dict(torch.load('target_dnn_file_name.pth',map_location='cpu'))
    
    # Test dataset
    D_test_normolization_tensor = torch.from_numpy(D_test_normolization) 
    PV_test_normolization_tensor = torch.from_numpy(PV_test_normolization)
    test_dataset = Data.TensorDataset(D_test_normolization_tensor, PV_test_normolization_tensor)
    test_loader = Data.DataLoader(
        dataset=test_dataset,      
        batch_size=1,#len(D_test_normolization),
        shuffle=False,
    )
    infeasible_volt_index_list = []
    infeasible_volt_number_list = []
    infeasible_qg_index_list =[]
    infeasible_qg_number_list = []
    infeasible_branch_index_list = []
    infeasible_branch_number_list = []
    feasible_number_list = []
    opf_time_list = []
    pf_time_list = []
    dnn_time_list = []
    Pre_voltage_list=[]
    Pre_slack_list=[]
    Pre_pg_list=[]
    cost_opf_list = []
    cost_pr2_list = []
    post_time_list = []
    ppopt = ppoption()

    for step,(test_x,test_y) in enumerate(test_loader):
        #PG       
        Lowbound_PG_nn_test = torch.from_numpy(np.tile(PG_Lowbound_Active, (len(test_x), 1)))
        Upbound_PG_nn_test = torch.from_numpy(np.tile(PG_Upbound_Active, (len(test_x), 1)))
        #Slack_bus_Voltage
        Lowbound_Slack_V_nn_test = torch.from_numpy(np.tile(Lowbound_Slack_V, (len(test_x), 1)))
        Upbound_Slack_V_nn_test = torch.from_numpy(np.tile(Upbound_Slack_V, (len(test_x), 1)))
        #Voltage
        Lowbound_Volt_nn_test = torch.from_numpy(np.tile(Lowbound_Volt, (len(test_x), 1)))
        Upbound_Volt_nn_test = torch.from_numpy(np.tile(Upbound_Volt, (len(test_x), 1)))		
        totalcost = 0
        #feedforward
        start_time = time.time()
        test_out, AC_PF_test = model(test_x)
        delta_time = time.time()-start_time
        if AC_PF_test.detach().numpy() == 0:
            feasible_number_list.append(step)
        
        #start_time2 = time.time()
        Pred_PG_test_tensor = Lowbound_PG_nn_test+(Upbound_PG_nn_test-Lowbound_PG_nn_test)*test_out[0,0:PG_Lowbound_Active.shape[1]]
        Pred_Slack_V_test_tensor = Lowbound_Slack_V_nn_test+(Upbound_Slack_V_nn_test-Lowbound_Slack_V_nn_test)*test_out[0,PG_Lowbound_Active.shape[1]]
        Pred_Volt_test_tensor = Lowbound_Volt_nn_test+(Upbound_Volt_nn_test-Lowbound_Volt_nn_test)*test_out[0,PG_Lowbound_Active.shape[1]+1:NN_output_number]   
		
        Pred_PG_test_np = Pred_PG_test_tensor.cpu().detach().numpy()
        Pred_Slack_V_test_np = Pred_Slack_V_test_tensor.cpu().detach().numpy()
        Pred_Volt_test_np = Pred_Volt_test_tensor.cpu().detach().numpy()
        #delta_time = delta_time+time.time()-start_time2
        
        Pre_pg_list.append(Pred_PG_test_np)
        Pre_slack_list.append(Pred_Slack_V_test_np)
        Pre_voltage_list.append(Pred_Volt_test_np)

        if AC_PF_test.detach().numpy() != 0:
            #solving PF
            mpc_pf_test = loadcase('pglib_opf_case2000_goc_rev.py')
            #pd,qd
            mpc_pf_test["bus"][:,2] =  P_D_test[step]
            mpc_pf_test["bus"][:,3] =  Q_D_test[step]
            Qg_index_zero_limit = np.where((mpc_pf_test['gen'][:,3]==0) & (mpc_pf_test['gen'][:,4] == 0))
            Qg_index_no_zero_limit = np.where((mpc_pf_test['gen'][:,3]!=0) & (mpc_pf_test['gen'][:,4] != 0))
            #modify QG
            mpc_pf_test['gen'][Qg_index_zero_limit,3] = 10
            mpc_pf_test['gen'][Qg_index_zero_limit,4] = -10
            mpc_pf_test['gen'][Qg_index_no_zero_limit,3] = mpc_pf_test['gen'][Qg_index_no_zero_limit,3]*5
            mpc_pf_test['gen'][Qg_index_no_zero_limit,4] = mpc_pf_test['gen'][Qg_index_no_zero_limit,4]*5          
            
            #pg,vg
            mpc_pf_test["gen"][SYNC_GEN_index_list,1] = 0
            mpc_pf_test["gen"][FIX_GEN_index_list,1] = mpc_pf_test["gen"][FIX_GEN_index_list,8]   
            mpc_pf_test["gen"][Active_GEN_index_list,1] = Pred_PG_test_np*standval
            Pred_Volt_test_np  = np.dot(Pred_Volt_test_np,Bus2Gen)                                   
            mpc_pf_test["gen"][:,5] = Pred_Volt_test_np
            #slack vg
            mpc_pf_test["gen"][Slack_GEN_index_list,5] = Pred_Slack_V_test_np   
        
            #run power flow
            ppopt = ppoption(ppopt, OUT_ALL=0, VERBOSE=False, ENFORCE_Q_LIMS=False)         
            r1_pf = runpf1(mpc_pf_test,ppopt)          
            pf_time_list.append(r1_pf[0]['et'])
            dnn_time_list.append(delta_time)          
            
            #solving OPF
            mpc_opf_test = loadcase('pglib_opf_case2000_goc_rev.py')
            [m,n]=mpc_opf_test["bus"].shape
            #pd,qd
            mpc_opf_test["bus"][:,2] =  P_D_test[step]
            mpc_opf_test["bus"][:,3] =  Q_D_test[step]
            mpc_opf_test["bus"][:,7] =  r1_pf[0]["bus"][:,7]
            mpc_opf_test["bus"][:,8] =  r1_pf[0]["bus"][:,8]
            #pg,vg
            mpc_opf_test["gen"][:,1] = r1_pf[0]["gen"][:,1]
            mpc_opf_test["gen"][:,2] = r1_pf[0]["gen"][:,2]                                      
            mpc_opf_test["gen"][:,5] = r1_pf[0]["gen"][:,5]
    
            #run optimal power flow
            ppopt = ppoption(ppopt, OUT_ALL=0, VERBOSE=False)             
            r1_opf = runopf(mpc_opf_test,ppopt)
            cost_pr2_list.append(r1_opf['f'])
            post_time_list.append(r1_opf['et'])
        else:
            #solving PF
            mpc_pf_test = loadcase('pglib_opf_case2000_goc_rev.py')
            mpc_pf_test["bus"][:,2] =  P_D_test[step]
            mpc_pf_test["bus"][:,3] =  Q_D_test[step]
            #modify QG
            Qg_index_zero_limit = np.where((mpc_pf_test['gen'][:,3]==0) & (mpc_pf_test['gen'][:,4] == 0))
            Qg_index_no_zero_limit = np.where((mpc_pf_test['gen'][:,3]!=0) & (mpc_pf_test['gen'][:,4] != 0))            
            mpc_pf_test['gen'][Qg_index_zero_limit,3] = 10
            mpc_pf_test['gen'][Qg_index_zero_limit,4] = -10
            mpc_pf_test['gen'][Qg_index_no_zero_limit,3] = mpc_pf_test['gen'][Qg_index_no_zero_limit,3]*5
            mpc_pf_test['gen'][Qg_index_no_zero_limit,4] = mpc_pf_test['gen'][Qg_index_no_zero_limit,4]*5  
            #pg,vg
            mpc_pf_test["gen"][SYNC_GEN_index_list,1] = 0
            mpc_pf_test["gen"][FIX_GEN_index_list,1] = mpc_pf_test["gen"][FIX_GEN_index_list,8]   
            mpc_pf_test["gen"][Active_GEN_index_list,1] = Pred_PG_test_np*standval
            Pred_Volt_test_np  = np.dot(Pred_Volt_test_np,Bus2Gen)                                   
            mpc_pf_test["gen"][:,5] = Pred_Volt_test_np
            #slack vg
            mpc_pf_test["gen"][Slack_GEN_index_list,5] = Pred_Slack_V_test_np     


            #run power flow
            ppopt = ppoption(ppopt, OUT_ALL=0, VERBOSE=False, ENFORCE_Q_LIMS=False)         
            r1_pf = runpf1(mpc_pf_test,ppopt)          
            pf_time_list.append(r1_pf[0]['et'])
            dnn_time_list.append(delta_time)
            post_time_list.append(0)            
            #compute cost for DNN
            On_index = np.where(r1_pf[0]["gen"][:,7]==1)[0]   
            PG = r1_pf[0]['gen'][On_index,1]
            Gen_cost = r1_pf[0]['gencost'][On_index,4:7]
            cost = np.multiply(Gen_cost[:,0], np.power(PG, 2)) + np.multiply(Gen_cost[:,1],PG) + Gen_cost[:,2] 
            totalcost = np.sum(cost, axis=0)            
            cost_pr2_list.append(totalcost)
    
    filename1='1_0_time.txt'
    fid1=open(filename1,'a')
    for i in range(0,len(cost_pr2_list)):
        fid1.write("%9.5f;%9.5f;%9.5f;%9.5f\n" % (0,pf_time_list[i],dnn_time_list[i],post_time_list[i]))
#        fid1.write("%9.5f;%9.5f;%9.5f\n" % (cost_list[i],dnn_time[i],cplex_time[i]))
    fid1.close()
    feasible_number_list =np.array(feasible_number_list)
    np.savetxt("feasible_list1_0.txt", feasible_number_list)    
    cost_opf_list =np.array(cost_opf_list).reshape(-1,1)
    cost_pr2_list =np.array(cost_pr2_list).reshape(-1,1)
    np.savetxt("costopf1_0.txt", cost_opf_list)
    np.savetxt("costpr2_1_0.txt", cost_pr2_list)
