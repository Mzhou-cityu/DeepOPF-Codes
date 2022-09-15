# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:26:56 2019

@author: xpan
"""
#AC-OPF-Case30
import time

import os
from pypower.runpf import runpf
from pypower.runpf1 import runpf1
#from pypower.runopf import runopf
from scipy.sparse import csr_matrix as sparse
from pypower.loadcase import loadcase
from pypower.ext2int import ext2int1

from pypower.ppoption import ppoption

import numpy as np

#pytorch package
import torch
from torch import nn, optim
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Function
#import matplotlib.pyplot as plt
import random
import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

torch.manual_seed(1)    # reproducible
random.seed(12343)

# parameter
training_cand_amout = 14000
training_amout = 10000
NN_input_number = 0 
NN_output_number = 0   
neural_number = 32
batch_size_training = 32  
batch_size_valtest = 1  
#learning parameters
learning_rate = 1e-3
num_epoches = 400
#global variables
Bus_number = 0
All_Gen_number = 0
Slack_bus = 0
standval = 0

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


#solving AC Power flow equation
def solve_pf(P_D_temp_train,Q_D_temp_train,Pred_PG_train,Pred_Volt_train,Pred_Slack_V_train,SYNC_GEN_index_list,FIX_GEN_index_list,Active_GEN_index_list,Volt_PV_GEN_index_list,Slack_GEN_index_list,standval):
	#solving PF
    mpc_pf_train = loadcase('pglib_opf_case30_ieee_rev.py')
    #pd,qd
    mpc_pf_train["bus"][:,2] =  P_D_temp_train[:]
    mpc_pf_train["bus"][:,3] =  Q_D_temp_train[:]

    #pg,vg
    mpc_pf_train["gen"][SYNC_GEN_index_list,1] = 0
    mpc_pf_train["gen"][Active_GEN_index_list,1] = Pred_PG_train[:]*standval                                        
    mpc_pf_train["gen"][Volt_PV_GEN_index_list,5] = Pred_Volt_train[:]
    #slack vg
    mpc_pf_train["gen"][Slack_GEN_index_list,5] = Pred_Slack_V_train[:]   
    #run power flow
    ppopt = ppoption()
    ppopt = ppoption(ppopt, OUT_ALL=0, VERBOSE=False, ENFORCE_Q_LIMS=False) 

    #run power flow
    r1_pf = runpf1(mpc_pf_train,ppopt)
    penalty = zero_order_penalty_abs(r1_pf)
    #end_time = time.time()
    #print(end_time-start)
    #print('\n')
    return penalty


#computing penalty gradient by zero-order optimization technique
def compute_pf_gradient(index,P_D_temp_train,Q_D_temp_train,vector_h,Pred_PG_train_plus_h,Pred_Volt_train_plus_h,Pred_Slack_V_train_plus_h,Pred_PG_train_minus_h,Pred_Volt_train_minus_h,Pred_Slack_V_train_minus_h,SYNC_GEN_index_list,FIX_GEN_index_list,Active_GEN_index_list,Gen_index_list,Slack_GEN_index_list,standval):
    h=1e-4
    mpc_pf_train_puls_h = loadcase('pglib_opf_case30_ieee_rev.py')
    #pd,qd
    mpc_pf_train_puls_h["bus"][:,2] =  P_D_temp_train[:]
    mpc_pf_train_puls_h["bus"][:,3] =  Q_D_temp_train[:]        
    #pg,vg
    mpc_pf_train_puls_h["gen"][SYNC_GEN_index_list,1] =  0
    mpc_pf_train_puls_h["gen"][Active_GEN_index_list,1] =  Pred_PG_train_plus_h[:]*standval                                         
    mpc_pf_train_puls_h["gen"][Gen_index_list,5] =  Pred_Volt_train_plus_h[:]
    #slack vg
    mpc_pf_train_puls_h["gen"][Slack_GEN_index_list,5] =  Pred_Slack_V_train_plus_h[:]
    ppopt = ppoption()
    ppopt = ppoption(ppopt, OUT_ALL=0, VERBOSE=False, ENFORCE_Q_LIMS=False)     
    #run power flow
    r1_plus_h_pf = runpf1(mpc_pf_train_puls_h,ppopt)
    penalty_plus_h = zero_order_penalty_abs(r1_plus_h_pf)                    
    #################################################################################################################
    #solving PF for minus h
    mpc_pf_train_minus_h = loadcase('pglib_opf_case30_ieee_rev.py')
    #pd,qd
    mpc_pf_train_minus_h["bus"][:,2] =  P_D_temp_train[:]
    mpc_pf_train_minus_h["bus"][:,3] =  Q_D_temp_train[:]        
    #pg,vg
    mpc_pf_train_minus_h["gen"][SYNC_GEN_index_list,1] =  0
    mpc_pf_train_minus_h["gen"][Active_GEN_index_list,1] = Pred_PG_train_minus_h[:]*standval                                         
    mpc_pf_train_minus_h["gen"][Gen_index_list,5] = Pred_Volt_train_minus_h[:]
    #slack vg
    mpc_pf_train_minus_h["gen"][Slack_GEN_index_list,5] = Pred_Slack_V_train_minus_h[:]
    #run power flow
    ppopt = ppoption()
    ppopt = ppoption(ppopt, OUT_ALL=0, VERBOSE=False, ENFORCE_Q_LIMS=False)        
    r1_minus_h_pf = runpf1(mpc_pf_train_minus_h,ppopt)
    penalty_minus_h = zero_order_penalty_abs(r1_minus_h_pf)
    #gradient descent update
    gradient_estimate = (penalty_plus_h - penalty_minus_h)*vector_h*(len(Active_GEN_index_list)+len(Gen_index_list)+len(Slack_GEN_index_list))/(2*h)
    #end_time = time.time()
    #print(end_time-start)
    #print('\n')
    return index,gradient_estimate


#class for last NN layer
class Penalty_ACPF(Function):    
    @staticmethod
    def forward(ctx, NN_input, load):
        ctx.save_for_backward(NN_input, load)
        NN_input, load = NN_input.cpu().detach(), load.cpu().detach()
        NN_input_np = NN_input.numpy()
        #PG      
        Pred_PG_train = PG_Lowbound_Active+(PG_Upbound_Active-PG_Lowbound_Active)*NN_input_np[:,0:PG_Lowbound_Active.shape[1]]
        #Slack_bus_Voltage
        Pred_Slack_V_train = Lowbound_Slack_V+(Upbound_Slack_V-Lowbound_Slack_V)*NN_input_np[:,PG_Lowbound_Active.shape[1]]
        Pred_Slack_V_train = Pred_Slack_V_train.reshape(NN_input_np.shape[0],-1)
        #Voltage
        Pred_Volt_train = Lowbound_Volt+(Upbound_Volt-Lowbound_Volt)*NN_input_np[:,PG_Lowbound_Active.shape[1]+1:NN_output_number]
        #input
        train_x_np = load.numpy()
        #P_D_temp_train =  train_x_np[:,0:NN_input_number]
        #Q_D_temp_train =  train_x_np[:,NN_input_number:2*NN_input_number] 
        P_D_temp_train =  (P_D_train_std+1e-8)*train_x_np[:,0:NN_input_number]+P_D_train_mean
        Q_D_temp_train =  (Q_D_train_std+1e-8)*train_x_np[:,NN_input_number:2*NN_input_number]+(Q_D_train_mean)                    

        result = []
        pool = mp.Pool(cores)
#        start = time.time()
        
        for i in range(0,NN_input_np.shape[0]):
#           temp_result =  solve_pf(P_D_temp_train[i],Q_D_temp_train[i],Pred_PG_train[i],Pred_Volt_train[i],Pred_Slack_V_train[i],SYNC_GEN_index_list,Active_GEN_index_list,Gen_index_list,Slack_GEN_index_list)            
            temp_result = pool.apply_async(solve_pf, args=(P_D_temp_train[i],Q_D_temp_train[i],Pred_PG_train[i],Pred_Volt_train[i],Pred_Slack_V_train[i],SYNC_GEN_index_list,FIX_GEN_index_list,Active_GEN_index_list,Volt_PV_GEN_index_list,Slack_GEN_index_list,standval))
            result.append(temp_result)
        pool.close()
        pool.join()
        result = [r.get() for r in result]
        result = np.array(result).sum()
#        end = time.time()
#        print(end - start)

        return torch.as_tensor(result).cuda()

    @staticmethod
    def backward(ctx, grad_output):
        NN_input, load = ctx.saved_tensors
        NN_input_temp = NN_input.cpu().detach().numpy()
        NN_input_modified_temp = NN_input_temp.copy()
        #sample random unit vector
        vec = np.random.randn(NN_input.shape[0], NN_input.shape[1])
        vec_norm = np.linalg.norm(vec, axis=1) .reshape(-1,1)
        vector_h = vec/vec_norm      

        h=1e-4
        NN_input_modified_plus_h = NN_input_modified_temp + vector_h*h
        #plus h
        #PG      
        Pred_PG_train_plus_h = PG_Lowbound_Active+(PG_Upbound_Active-PG_Lowbound_Active)*NN_input_modified_plus_h[:,0:PG_Lowbound_Active.shape[1]]
        #Slack_bus_Voltage
        Pred_Slack_V_train_plus_h = Lowbound_Slack_V+(Upbound_Slack_V-Lowbound_Slack_V)*NN_input_modified_plus_h[:,PG_Lowbound_Active.shape[1]]
        Pred_Slack_V_train_plus_h = Pred_Slack_V_train_plus_h.reshape(NN_input_temp.shape[0],-1)
        #Voltage
        Pred_Volt_train_plus_h = Lowbound_Volt+(Upbound_Volt-Lowbound_Volt)*NN_input_modified_plus_h[:,PG_Lowbound_Active.shape[1]+1:NN_output_number]

        #minus h
        NN_input_modified_minus_h = NN_input_modified_temp - vector_h*h
        #PG      
        Pred_PG_train_minus_h = PG_Lowbound_Active+(PG_Upbound_Active-PG_Lowbound_Active)*NN_input_modified_minus_h[:,0:PG_Lowbound_Active.shape[1]]
        #Slack_bus_Voltage
        Pred_Slack_V_train_minus_h = Lowbound_Slack_V+(Upbound_Slack_V-Lowbound_Slack_V)*NN_input_modified_minus_h[:,PG_Lowbound_Active.shape[1]]
        Pred_Slack_V_train_minus_h = Pred_Slack_V_train_plus_h.reshape(NN_input_temp.shape[0],-1)
        #Voltage
        Pred_Volt_train_minus_h = Lowbound_Volt+(Upbound_Volt-Lowbound_Volt)*NN_input_modified_minus_h[:,PG_Lowbound_Active.shape[1]+1:NN_output_number]

        #input
        train_x_np = load.cpu().detach().numpy()        
        #P_D_temp_train =  train_x_np[:,0:NN_input_number]
        #Q_D_temp_train =  train_x_np[:,NN_input_number:2*NN_input_number] 
        P_D_temp_train =  (P_D_train_std+1e-8)*train_x_np[:,0:NN_input_number]+P_D_train_mean
        Q_D_temp_train =  (Q_D_train_std+1e-8)*train_x_np[:,NN_input_number:2*NN_input_number]+(Q_D_train_mean)                    
        
        Est_grad = np.zeros([NN_input.shape[0],NN_input.shape[1]],dtype ='float32')
        
        pool = mp.Pool(cores)
#        start = time.time()
        result = []
        #index_list =[]
        #result_list =[]
        for i in range(0,NN_input.shape[0]):
            temp_result = pool.apply_async(compute_pf_gradient, args=(i,P_D_temp_train[i],Q_D_temp_train[i],vector_h[i],Pred_PG_train_plus_h[i],Pred_Volt_train_plus_h[i],Pred_Slack_V_train_plus_h[i],Pred_PG_train_minus_h[i],Pred_Volt_train_minus_h[i],Pred_Slack_V_train_minus_h[i],SYNC_GEN_index_list,FIX_GEN_index_list,Active_GEN_index_list,Volt_PV_GEN_index_list,Slack_GEN_index_list,standval))
            result.append(temp_result)
        pool.close()
        pool.join()

        for r in result:
            ind, rst = r.get()
            Est_grad[int(ind)] = rst

#        end = time.time()
#        print(end - start)
        return torch.from_numpy(Est_grad).cuda()*grad_output, None

# neural network definition
class Neuralnetwork(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, 2*n_hidden)
        #self.layer2 = nn.Linear(4*n_hidden, 2*n_hidden)
        self.layer3 = nn.Linear(2*n_hidden, n_hidden)
        self.layer4 = nn.Linear(n_hidden, out_dim)
        #Layer for AC-PF
        self.layer6 = Penalty_ACPF.apply

    def forward(self, x):
        x0 = F.relu(self.layer1(x))
        #x1 = F.relu(self.layer2(x0))
        x2 = F.relu(self.layer3(x0))
        x_sol = torch.sigmoid(self.layer4(x2))
        #Layer for AC-PF
        x_penalty = self.layer6(x_sol,x)
        return x_sol, x_penalty


#penalty term (absolute value)
def zero_order_penalty_abs(pf_results=None):
    standval = pf_results[0]["baseMVA"]
    ctol = 1e-4
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
    Ff_temp[Ff_temp< ctol] = 0
    Ft_temp = (Ft[Branch_index]-Branch_bound)/standval
    Ft_temp[Ft_temp< ctol] = 0    
    Ff_penalty = np.abs(Ff_temp)
    Ft_penalty = np.abs(Ft_temp)
#    
    return np.sum(PG_penalty)+np.sum(QG_penalty)+np.sum(V_penalty)+np.sum(Ff_penalty)+np.sum(Ft_penalty)

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
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

    global PG_Lowbound
    global PG_Upbound

    PG_Lowbound = np.zeros([1, mG-1],dtype='float32')
    PG_Upbound = np.zeros([1, mG-1],dtype='float32')

    Load_index_list=[]
    global Gen_index_list
    global SYNC_GEN_index_list
    global FIX_GEN_index_list
    global Active_GEN_index_list
    global Slack_GEN_index_list
    global Volt_PV_index_list
    global Volt_PV_GEN_index_list
    global All_gen_list
    #index transfer
    global Slack_bus
    global slack_bus_index
    for j in range(0,bus.shape[0],1):
#        if (bus[j,2] != 0 or bus[j,3] != 0):
        Load_index_list.append(bus[j,0])   
        if (bus[j,1] == 3):
            Slack_bus = int(bus[j,0])
            slack_bus_index = j
    
    gen_index = 0
    for j in range(0,mpc['gen'].shape[0],1):       
        Gen_index_list.append(int(mpc['gen'][int(j),0])) 
        if Slack_bus!= mpc['gen'][j,0] and mpc['gen'][j,7] != 0:
            Volt_PV_GEN_index_list.append(j)
            Volt_PV_index_list.append(int(mpc['gen'][int(j),0]))
            PG_Upbound[0,gen_index] = mpc['gen'][j,8]/standval
            PG_Lowbound[0,gen_index] = mpc['gen'][j,9]/standval
            gen_index = gen_index +1
            if (mpc['gen'][j,8] == 0 and mpc['gen'][j,9] == 0):
                SYNC_GEN_index_list.append(j)
            elif (mpc['gen'][j,8] > 0 and mpc['gen'][j,8] == mpc['gen'][j,9]):
                FIX_GEN_index_list.append(j)
            else:     
                Active_GEN_index_list.append(j)
        else:
            slack_bus_index = j
            Slack_GEN_index_list.append(j)

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
    Gen_index = np.array(Gen_index_list)
    Load_index = np.array(Load_index_list)      

    global NN_input_number
    global NN_output_number
    #input/output of NN    
    NN_output_number = len(Active_GEN_index_list)+mG
    NN_input_number = Load_index.shape[0]
    global All_Gen_number
    All_Gen_number = mG
    
    global Lowbound_Volt
    global Upbound_Volt
    global Lowbound_Slack_V
    global Upbound_Slack_V
    Lowbound_Volt = np.zeros((mpc_gen.shape[0]-1,),dtype='float32')
    Upbound_Volt = np.zeros((mpc_gen.shape[0]-1,),dtype='float32')
    Lowbound_Volt = Lowbound_Volt.T
    Upbound_Volt = Upbound_Volt.T
    index = 0
    for i in range(mpc_gen.shape[0]):
        bus_idx = np.where(bus[:,0]==mpc_gen[i,0])[0][0]
        if bus[bus_idx,1]!= 3:
            Lowbound_Volt[index] = np.power(bus[bus_idx,12],1)
            Upbound_Volt[index] = np.power(bus[bus_idx,11],1)
            index = index + 1
        else:
            Lowbound_Slack_V[0,0] = np.power(bus[bus_idx,12],1)
            Upbound_Slack_V[0,0] = np.power(bus[bus_idx,11],1)
    
            
if __name__ == '__main__': 
    cores = mp.cpu_count()
    Volt_PV_index_list = []
    Volt_PV_GEN_index_list = []
    SYNC_GEN_index_list = []
    FIX_GEN_index_list = []    
    Active_GEN_index_list = []
    Slack_GEN_index_list = []
    Active_GEN_index_list2PG_train = []
    Gen_index_list = []
    #load case
    mpc = loadcase('pglib_opf_case30_ieee_rev.py')
    model_generation(mpc)        
    dir=''
    # del data,line,odom
    P_D = np.load(dir+"P_D.npy").astype('float32')
    Q_D = np.load(dir+"Q_D.npy").astype('float32')    
    V = np.load(dir+"Vm.npy").astype('float32')
    Slack_V = V[:,int(Slack_bus)]#np.load(dir+"Slack_V.npy").astype('float32')
    P_G_all = np.load(dir+"P_G_all.npy").astype('float32')
    P_G = P_G_all[:,Active_GEN_index_list]    
    Q_G = np.load(dir+"Q_G.npy").astype('float32')       
    
    ############################################## 
    # Preprocessing training data
    idx_sample = random.sample(range(0, training_cand_amout), training_amout)
    idx_train = idx_sample[0:training_amout]
    idx_train = np.asarray(idx_train)    
    num_of_group = 1000
    segment = int(training_amout/1000)
    P_D_train = P_D[0:training_cand_amout][idx_train]#[0:num_of_group * (segment)]
    P_D_train_normolization = (P_D_train)
    #if do pre-processing
    P_D_train_mean = np.mean(P_D_train, axis=0)
    P_D_train_std = np.std(P_D_train, axis=0)
    P_D_train_std_tensor = torch.from_numpy(P_D_train_std).cuda()
    P_D_train_mean_tensor = torch.from_numpy(P_D_train_mean).cuda()
    P_D_train_normolization = (P_D_train - P_D_train_mean) / (P_D_train_std+1e-8)

    Q_D_train = Q_D[0:training_cand_amout][idx_train]#[0:num_of_group * (segment)]
    Q_D_train_normolization = (Q_D_train)
    #if do pre-processing
    Q_D_train_mean = np.mean(Q_D_train, axis=0)
    Q_D_train_std = np.std(Q_D_train, axis=0)
    Q_D_train_std_tensor = torch.from_numpy(Q_D_train_std).cuda() 
    Q_D_train_mean_tensor = torch.from_numpy(Q_D_train_mean).cuda() 
    Q_D_train_normolization = (Q_D_train - Q_D_train_mean) / (Q_D_train_std+1e-8)

    D_train_normolization = np.concatenate((P_D_train_normolization,Q_D_train_normolization),axis = 1)

    P_G_train = P_G[0:training_cand_amout][idx_train]/standval#[0:num_of_group * (segment)]
    Lowbound_PG_training = np.tile(PG_Lowbound_Active,(len(P_G_train),1))
    Upbound_PG_training= np.tile(PG_Upbound_Active,(len(P_G_train),1))
    PG_train_normolization = (P_G_train-Lowbound_PG_training)/(Upbound_PG_training-Lowbound_PG_training)

    Volt_train = V[0:training_cand_amout][idx_train][:,PV_index]
    Lowbound_Volt_training = np.tile(Lowbound_Volt,(len(Q_D_train),1))
    Upbound_Volt_training= np.tile(Upbound_Volt,(len(Q_D_train),1))
    Volt_train_normolization = (Volt_train-Lowbound_Volt_training)/(Upbound_Volt_training-Lowbound_Volt_training)

    Slack_V_train = Slack_V[0:training_cand_amout][idx_train]
    Lowbound_Slack_V_training = np.tile(Lowbound_Slack_V,(len(Q_D_train),))
    Upbound_Slack_V_training= np.tile(Upbound_Slack_V,(len(Q_D_train),))
    Slack_V_train_normolization = (Slack_V_train-Lowbound_Slack_V_training).squeeze()/(Upbound_Slack_V_training-Lowbound_Slack_V_training)
    Slack_V_train_normolization = Slack_V_train_normolization.T

    PV_train_temp = np.concatenate((PG_train_normolization,Slack_V_train_normolization),axis = 1)
    PV_train_normolization = np.concatenate((PV_train_temp, Volt_train_normolization),axis = 1)

    # Training dataset
    D_train_normolization_tensor = torch.from_numpy(D_train_normolization).cuda()  
    PV_train_normolization_tensor = torch.from_numpy(PV_train_normolization).cuda()
    training_dataset = Data.TensorDataset(D_train_normolization_tensor, PV_train_normolization_tensor)
    training_loader = Data.DataLoader(
        dataset=training_dataset,
        batch_size=batch_size_training,
        shuffle=True,
    )


    training_err1 = []
    training_err2 = []
    training_index = [] 

    ###############################################################
    #training DNN
    #parameter setting  
    criterion = nn.MSELoss(reduction='mean')
    ########################################
    if os.path.exists('ACOPF_case30_dnn_zop.pth') == 0:
        model = Neuralnetwork(2*NN_input_number, neural_number, NN_output_number)
        if torch.cuda.is_available():
            model = model.cuda()              
        optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=(0.9,0.99))
        #neural network parameters setting
        ##########################################
        #Training process
        total_time_forward = 0        
        total_time_backward = 0
        for epoch in range(num_epoches):
            print('epoch {}'.format(epoch + 1))
            print('*' * 10)
            running_loss = 0.0
            
            total_penalty = 0
            total_predict_error = 0  
            
            for step, (train_x, train_y) in enumerate(training_loader):
                #feedforward
                start_time = time.time()
                train_out, AC_PF = model(train_x)
                total_time_forward += time.time()-start_time

                train_loss1 = criterion(train_out, train_y)
                loss = 1*(train_loss1)+0.1*(AC_PF)                
                #zero-order penalty           
                running_loss =  running_loss+loss.item()
                
                total_predict_error+= train_loss1
                total_penalty += AC_PF 
                
               # backproprogate
                optimizer.zero_grad()
                start_time = time.time()
                loss.backward()                        
                total_time_backward += time.time()-start_time
                optimizer.step()                 
#                scheduler.step()

            training_err1.append(total_predict_error)
            training_err2.append(total_penalty)    
            training_index.append(epoch)  

            print('Finish {} epoch; Loss: {:.6f} ;Penalty : {:.6f}\n'.format(epoch + 1,total_predict_error,total_penalty))
                                                                                                                                                       
            if (epoch+1) % 50 == 0:
                target_file = 'ACOPF_case_30_dnn_'+str(epoch+1)+'.pth'
                torch.save(model.state_dict(), target_file)
        print("\n")       
        print(total_time_forward)
        print("\n")
        print(total_time_backward)
        print("\n")
