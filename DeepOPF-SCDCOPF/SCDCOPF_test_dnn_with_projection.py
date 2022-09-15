# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:10:25 2019

@author: xpan
"""

import numpy as np
#import matplotlib.pyplot as plt
from pypower.loadcase import loadcase
import time
from scipy.sparse import csr_matrix as sparse
from pypower.ext2int import ext2int1
from pypower.makeBdc import makeBdc
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
torch.manual_seed(1)    # reproducible
from gurobipy import *

# Create a new model
obj = QuadExpr()
# parameter
NN_input_number = 0 
NN_output_number = 0   
neural_number = 8

################################
#DCOPF
Bus_number = 0

Slack_bus = 1
Slack_angle = 0
standval = 0
Load_index_list=[]
Gen_index_list=[]
Power_Lowbound_Gen=np.zeros([1,1],dtype='float32')
Power_Upbound_Gen=np.zeros([1,1],dtype='float32')
Bus_admittance_full_NN2 = np.zeros([1,1],dtype='float32')
Bus_admittance_full = np.zeros([1,1],dtype='float32')
Bus_admittance_full_1 = np.zeros([1,1],dtype='float32')
Node_index = np.zeros([1,1],dtype='float32')
Node_angel = np.zeros([1,1],dtype='float32')
Load_index = np.zeros([1,1],dtype='float32')
Gen_index = np.zeros([1,1],dtype='float32')
Gen_Price = np.zeros([1,1],dtype='float32')
Branch_limit_ori = np.zeros([1,1],dtype='float32')
Branch_limit_sc = np.zeros([1,1],dtype='float32')

#SCOPF
SC_list = []
conflict_list = []
filter_list = []
NumofContigengcy = 0
SC_Bus_admittance_full_1_array = np.zeros((1,1,1),dtype='float32')
SC_Bus_admittance_line_1_array = np.zeros((1,1,1),dtype='float32')

# neural network definition
class Neuralnetwork(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, 4*n_hidden)
        self.layer2 = nn.Linear(4*n_hidden, 2*n_hidden)
        self.layer3 = nn.Linear(2*n_hidden, n_hidden)
        #self.layer4 = nn.Linear(n_hidden, n_hidden)
        self.layer5 = nn.Linear(n_hidden, out_dim)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        #x = F.relu(self.layer4(x))
        x = torch.sigmoid(self.layer5(x))
        return x


#gurobi_solv
def gurobi_solv(m=None,mpc=None,Round = None, Gen_initial=None, Init_default_theta = None, Init_SC_theta = None,Dstart_ori= None,Dstart_sc = None):
    #read admittance matrix from pypower
    global standval
    
    standval, mpc_bus, mpc_gen, mpc_branch = \
    mpc['baseMVA'], mpc['bus'], mpc['gen'], mpc['branch']

    ## switch to internal bus numbering and build admittance matrices
    _, bus, gen, branch = ext2int1(mpc_bus, mpc_gen, mpc_branch)
    
    [mL,nL] = bus.shape
    [mB,nB] = branch.shape
    global Bus_number
    Bus_number = mL
    ## check that all buses have a valid BUS_TYPE
    bt = mpc_bus[:, 1]    
    ## determine which buses, branches, gens are connected and
    ## in-service
    n2i = sparse((range(mL), (mpc_bus[:, 0], np.zeros(mL))),
        shape=(max(mpc_bus[:, 0].astype(int)) + 1, 1))
    n2i = np.array( n2i.todense().flatten() )[0, :] # as 1D array
    bs = (bt != None)                               ## bus status
    gs = ( (mpc_gen[:, 7] > 0) &          ## gen status
    bs[ n2i[mpc_gen[:, 0].astype(int)] ] )
    mG = len(gs.ravel().nonzero()[0])    ## on and connected
    nG = mpc_gen.shape[1]

    #price
    mpc_price = mpc["gencost"]
    npri = mpc_price.shape[1]
    Gen_Price = np.zeros((mG, npri-4),dtype = 'float64')   
    Gen_index = np.zeros((1, mG),dtype='float64')
    
    index = 0
    for j in range(0,gen.shape[0],1):
        if gen[j,7] != 0:
            Gen_Price[index,:] = mpc_price[j,4:npri]
            Gen_index[0,index] = gen[j,0]
            index = index + 1
       
    global NumofContigengcy
    
    if Round==0:  
        #global Branch_number
        Bus_admittance_full = np.zeros([mL,mL+mG],dtype='float64')
        Bus_admittance_line1 = np.zeros([mB,mL+mG],dtype='float64') 
        ## power mismatch constraints
        B, Bf, Pbusinj, Pfinj = makeBdc(standval, bus, branch)
        Bus_admittance_full[:,0:mL] = B.toarray()
        Bus_admittance_line1[:,0:mL] = Bf.toarray()   
    
        SC_list = []
        #SC-OPF
        if mpc["scopf"] == 1:
        	#different topology
            if Bus_number == 30:
                SC_list = [0,1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40]
                NumofContigengcy = len(SC_list)
            
        u = np.ones((mL+mB,), dtype = 'float64')
        l = -u
        v0 = np.zeros(((1+NumofContigengcy)*mL+2*mG,), dtype = 'float64')   
        vu = np.zeros(((1+NumofContigengcy)*mL+2*mG,), dtype = 'float64')   
        vl = np.zeros(((1+NumofContigengcy)*mL+2*mG,), dtype = 'float64')
        
        for j in range(0,mL,1):
            u[j] = -bus[j,2]/standval
            l[j] = -bus[j,2]/standval
            if (bus[j,1] == 3):
                v0[j] = bus[j,8]*np.pi/180
                vl[j] = bus[j,8]*np.pi/180
                vu[j] = bus[j,8]*np.pi/180
            else:
                v0[j] = bus[j,8]*np.pi/180
                vl[j] = -2*np.pi#/standval
                vu[j] = 2*np.pi#/standval
        for j in range(0,mG,1):
            Bus_admittance_full[int(Gen_index[0,j]),mL+j] = -1
            v0[mL+j] = gen[j,1]/standval
            vl[mL+j] = gen[j,9]/standval
            vu[mL+j] = gen[j,8]/standval

        #SC-OPF
        for i in range(0,NumofContigengcy,1):
            for k in range(0,mL,1):
                if (bus[k,1] == 3):
                    v0[mG+(i+1)*mL+k] = bus[k,8]*np.pi/180
                    vl[mG+(i+1)*mL+k] = bus[k,8]*np.pi/180
                    vu[mG+(i+1)*mL+k] = bus[k,8]*np.pi/180
                else:
                    v0[mG+(i+1)*mL+k] = bus[k,8]*np.pi/180
                    vl[mG+(i+1)*mL+k] = -2*np.pi#/standval
                    vu[mG+(i+1)*mL+k] = 2*np.pi#/standval

        for j in range(0,mG,1):
            v0[(1+NumofContigengcy)*mL+mG+j] = 0
            vl[(1+NumofContigengcy)*mL+mG+j] = 0
            vu[(1+NumofContigengcy)*mL+mG+j] = GRB.INFINITY                  
                    
        for j in range(0,mB,1):
            u[mL+j] = branch[j,5]/standval#-GRB.INFINITY
            l[mL+j] = -branch[j,5]/standval #-GRB.INFINITY
    
        # Create variables
        x = m.addVars((1+NumofContigengcy)*mL+2*mG, lb = vl, ub = vu, name="X")
        m.update()
        
        global obj
        # Set objective: cost of PG
        [mPrice,nPrice] = Gen_Price.shape
        for i in range(0,mPrice,1):
            obj.add(x[(1+NumofContigengcy)*mL+mG+i])
        m.setObjective(obj,sense=GRB.MINIMIZE)
        m.update()    
        
        # Add constraint:Default DC-OPF
        #flow balance
        for i in range(0,mL,1):
            EnqualityConsT = LinExpr()
            for j in range(0,mL+mG,1):
                EnqualityConsT.add(Bus_admittance_full[i][j]*x[j])
            m.addConstr(EnqualityConsT,GRB.EQUAL,u[i])
                          
        #branch transmission requirement
        for i in range(0,mB,1):
            InequallityConsT = LinExpr()                     
            for j in range(0,mL+mG,1):                                                                                                             
                InequallityConsT.add(Bus_admittance_line1[i][j]*x[j])
            m.addConstr(InequallityConsT,GRB.LESS_EQUAL,u[i+mL])
            m.addConstr(InequallityConsT,GRB.GREATER_EQUAL,-u[i+mL])
        m.update()          
        del Bus_admittance_full, Bus_admittance_line1 

        #SC-OPF
        if mpc["scopf"] == 1:
            #SC-OPF
            start_index = mG+mL
            #col_start_index = mL+mG
            Bus_admittance_full_temp = np.zeros([mL,mL+mG],dtype='float64')
            Bus_admittance_line1_temp = np.zeros([mB-1,mL],dtype='float64')
            temp_branch = np.zeros([mB-1,nB],dtype='float64')
            u = np.ones((mL+mB-1,), dtype = 'float64')
            l = -u
    
            for i in range(0,NumofContigengcy,1):
                Bus_admittance_full_temp[:,:] = 0
                Bus_admittance_line1_temp[:,:] = 0
                #index = 0
                #construct
                temp_branch[0:SC_list[i]] = branch[0:SC_list[i]]
                temp_branch[SC_list[i]:mB-1] = branch[SC_list[i]+1:mB]
                ## power mismatch constraints                                                                         
                                                                                
                B, Bf, Pbusinj, Pfinj = makeBdc(standval, bus, temp_branch)
                Bus_admittance_full_temp[:,0:mL] = B.toarray()
                Bus_admittance_line1_temp[:,0:mL] = Bf.toarray()
                ############################################################
                for j in range(0,mG,1):
                    Bus_admittance_full_temp[int(Gen_index[0,j]),mL+j] = -1
                 
                for k in range(0,mL,1):
                    u[k] = -bus[k,2]/standval
                    l[k] = -bus[k,2]/standval

                for k in range(0,mB-1,1):
                    u[mL+k] = temp_branch[k,5]/standval
    
                # Add constraint:SC-OPF
                #flow balance                                     
                for j in range(0,mL,1):
                    EnqualityConsT = LinExpr()                               
                    for k in range(0,mL+mG,1):                                                                               
                        if k >= mL:
                           EnqualityConsT.add(Bus_admittance_full_temp[j][k]*x[mL+(k-mL)]) 
                        else:
                           EnqualityConsT.add(Bus_admittance_full_temp[j][k]*x[start_index+i*(mL)+k])
                        #print(start_index+i*(mL)+k)
                    m.addConstr(EnqualityConsT,GRB.EQUAL,u[j])           
                for j in range(0,mB-1,1):
                    InequallityConsT = LinExpr()
                    for k in range(0,mL,1):
                        InequallityConsT.add(Bus_admittance_line1_temp[j][k]*x[start_index+i*(mL)+k])
                    m.addConstr(InequallityConsT,GRB.LESS_EQUAL,u[mL+j])
                    m.addConstr(InequallityConsT,GRB.GREATER_EQUAL,-u[mL+j])
                m.update()
        #absolute constraint
        for i in range(0,mG,1):
            InequallityConsT1 = LinExpr()                                                                                                                               
            InequallityConsT1.add(x[mL+i]-x[(1+NumofContigengcy)*mL+mG+i])
            m.addConstr(InequallityConsT1,GRB.LESS_EQUAL,Gen_initial[i])            
            InequallityConsT2 = LinExpr()                                                                                                                               
            InequallityConsT2.add(x[mL+i]+x[(1+NumofContigengcy)*mL+mG+i])
            m.addConstr(InequallityConsT2,GRB.GREATER_EQUAL,Gen_initial[i]) 
        m.update()

    else:
        #DC-OPF
        for constr in range(0,mL):
            m.getConstrs()[constr].RHS = -bus[constr,2]/standval
        #SC DC-OPF
        start_index = mL+2*mB
        for i in range(0,NumofContigengcy):
            for constr in range(0,mL):
                m.getConstrs()[start_index+i*(mL+2*(mB-1))+constr].RHS = -bus[constr,2]/standval
        start_index = mL+2*mB+NumofContigengcy*(mL+2*(mB-1))
        for constr in range(0,mG):
            m.getConstrs()[start_index+2*constr].RHS = Gen_initial[constr]
            m.getConstrs()[start_index+2*constr+1].RHS = Gen_initial[constr]
        m.update()
     
    
    m.optimize()
                
    Solution = []

    for v in range(mL,mL+mG,1):
        Solution.append(m.getVars()[v].x)

    print('Obj: %g' % obj.getValue())

    Solutionarr = np.array(Solution)
    #del m
    return Solutionarr,m.Runtime

#parameter generation
def para_generation(mpc=None):
    #read admittance matrix from pypower
    global standval
    
    standval, mpc_bus, mpc_gen, mpc_branch = \
    mpc['baseMVA'], mpc['bus'], mpc['gen'], mpc['branch']

    ## switch to internal bus numbering and build admittance matrices
    _, bus, gen, branch = ext2int1(mpc_bus, mpc_gen, mpc_branch)
    
    [mL,nL] = bus.shape
    [mB,nB] = branch.shape
    global Bus_number
    Bus_number = mL
    ## check that all buses have a valid BUS_TYPE
    bt = bus[:, 1]    
    ## determine which buses, branches, gens are connected and
    ## in-service
    n2i = sparse((range(mL), (bus[:, 0], np.zeros(mL))),
        shape=(max(bus[:, 0].astype(int)) + 1, 1))
    n2i = np.array( n2i.todense().flatten() )[0, :] # as 1D array
    bs = (bt != None)                               ## bus status
    gs = ( (gen[:, 7] > 0) &          ## gen status
    bs[ n2i[gen[:, 0].astype(int)] ] )
    mG = len(gs.ravel().nonzero()[0])    ## on and connected
    # nG = gen.shape[1]

    #price
    global Gen_index
    global Load_index
    global Load_index_list
    global Gen_index_list  
    global Slack_bus
    global Slack_angle
    global Gen_Price
    #genrator
    global Power_Lowbound_Gen
    global Power_Upbound_Gen
    mpc_price = mpc["gencost"]
    npri = mpc_price.shape[1]
    Gen_Price = np.zeros((mG, npri-4),dtype = 'float64')   
    Gen_index = np.zeros((1, mG),dtype='float64')
    Power_Lowbound_Gen = np.zeros((1, mG),dtype='float32')
    Power_Upbound_Gen = np.zeros((1, mG),dtype='float32')    
    index = 0
    for j in range(0,gen.shape[0],1):
        if gen[j,7] != 0:
            Gen_Price[index,:] = mpc_price[j,4:npri]
            Gen_index_list.append(gen[j,0])
            Power_Upbound_Gen[0,index] = gen[j,8]/standval
            Power_Lowbound_Gen[0,index] = gen[j,9]/standval
            index = index + 1

    for j in range(0,mL,1):
        if mpc_bus[j,1] == 3:
           Slack_bus = j
           Slack_angle = mpc_bus[j,8]*np.pi/180
        if (mpc_bus[j,2] != 0):
            Load_index_list.append(j) 

    Gen_index = np.array(Gen_index_list)
    Load_index = np.array(Load_index_list)

    global NN_input_number
    global NN_output_number  
    NN_output_number = mG-1
    NN_input_number = Load_index.shape[0]

    global Bus_admittance_full
    Bus_admittance_full = np.zeros([mL,mL],dtype='float32')
    global Bus_admittance_full_NN2
    Bus_admittance_full_NN2 = np.zeros([mB,mL],dtype='float32') 
    
    #DC-OPF
    ## power mismatch constraints
    B, Bf, Pbusinj, Pfinj = makeBdc(standval, bus, branch)   
    Bus_admittance_full[:,0:mL] = B.toarray()
    Bus_admittance_full_NN2[:,0:mL] = Bf.toarray()

    for i in range(0,mB,1):
        Bus_admittance_full_NN2[i,:] = Bus_admittance_full_NN2[i,:] / (branch[i,5]/standval)

    global Bus_admittance_full_1
    Bus_admittance_full_1 = Bus_admittance_full[0:Slack_bus,0:Slack_bus]
    Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Bus_admittance_full[0:Slack_bus,Slack_bus+1:Bus_number]),axis=1)
    Temp = Bus_admittance_full[Slack_bus+1:Bus_number,0:Slack_bus]
    Temp = np.concatenate((Temp,Bus_admittance_full[Slack_bus+1:Bus_number,Slack_bus+1:Bus_number]),axis=1)
    Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Temp),axis=0)    
    
    global Branch_limit_ori
    Branch_limit_ori = branch[:,5]/standval
    Branch_limit_ori = Branch_limit_ori.reshape(1, mB)

	 #SC-OPF
    global NumofContigengcy
    global SC_list
    global filter_list
    global conflict_list
    #SC-OPF
    if mpc["scopf"] == 1:
        #different topology
        if Bus_number == 30:
            SC_list = [0,1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40]
            NumofContigengcy = len(SC_list)

    global Branch_limit_sc
    Branch_limit_sc = np.zeros([NumofContigengcy,mB-1],dtype='float32')

    #SC-OPF
    if mpc["scopf"] == 1:
        global SC_Bus_admittance_full_1_array, SC_Bus_admittance_line_1_array
        SC_Bus_admittance_full_1_array = np.resize(SC_Bus_admittance_full_1_array,(NumofContigengcy,mL-1,mL-1))
        SC_Bus_admittance_line_1_array = np.resize(SC_Bus_admittance_line_1_array,(NumofContigengcy,mL,mB-1))
        SC_Bus_admittance_line_1_temp_array = np.zeros((NumofContigengcy,mB-1,mL),dtype='float32')
            
        temp_branch = np.zeros((mB-1,nB),dtype='float32')
        for i in range(0,NumofContigengcy,1):
            temp_branch[0:SC_list[i]] = branch[0:SC_list[i]]
            temp_branch[SC_list[i]:mB-1] = branch[SC_list[i]+1:mB]
            ##power flow constraints
            B, Bf, Pbusinj, Pfinj = makeBdc(standval, bus, temp_branch)
            B = B.toarray()
            SC_Bus_admittance_full_1 = B[0:Slack_bus,0:Slack_bus]
            SC_Bus_admittance_full_1 = np.concatenate((SC_Bus_admittance_full_1,B[0:Slack_bus,Slack_bus+1:Bus_number]),axis=1)
            Temp = B[Slack_bus+1:Bus_number,0:Slack_bus]
            Temp = np.concatenate((Temp,B[Slack_bus+1:Bus_number,Slack_bus+1:Bus_number]),axis=1)
            SC_Bus_admittance_full_1 = np.concatenate((SC_Bus_admittance_full_1,Temp),axis=0)   
            SC_Bus_admittance_full_1_array[i,:,:] = np.linalg.inv(SC_Bus_admittance_full_1)
            SC_Bus_admittance_line_1_temp_array[i,:,:] = Bf.toarray()

            Branch_limit_sc[:,] = temp_branch[:,5]/standval
            
            for j in range(0,mB-1,1):
                SC_Bus_admittance_line_1_temp_array[i,j,:] = SC_Bus_admittance_line_1_temp_array[i,j,:] / (temp_branch[j,5]/standval)
        for i in range(0,NumofContigengcy,1):
            SC_Bus_admittance_line_1_array[i,:,:] = SC_Bus_admittance_line_1_temp_array[i,:,:].T

if __name__ == '__main__':
    m = Model("qp")
    m.Params.Method = 1
    m.Params.Crossover =0
    m.Params.Threads = 1
    #read case file
    mpc = loadcase('pglib_opf_case30_ieee.py')
#    mpc['branch'][:,5] = mpc['branch'][:,5]*1.2
    #reading parameters
    para_generation(mpc)
    P_D = np.load("PD_train.npy").astype("float32")
    P_G = np.load("PG_train.npy").astype("float32")
    num_of_group = int(np.floor(P_D.shape[0] / 10))  
    # Preprocessing training data
    P_D_train = P_D[0:num_of_group * 9]
    P_D_train_mean = np.mean(P_D_train, axis=0)
    P_D_train_std = np.std(P_D_train, axis=0)
    P_D_train_std_tensor = torch.from_numpy(P_D_train_std) 
    P_D_train_mean_tensor = torch.from_numpy(P_D_train_mean) 
    del P_D_list, P_G_list
    #test inversve matrix of B (default DCOPF)
    Bus_1_Coeff = np.linalg.inv(Bus_admittance_full_1)
    Bus_1_Coeff_tensor = torch.from_numpy(Bus_1_Coeff) 
    ################################################################
    # load net
    model = Neuralnetwork(NN_input_number, neural_number, NN_output_number)
    model.load_state_dict(torch.load('SCOPF_case30_trained.pth',map_location='cpu'))#trained-DNN
    #if torch.cuda.is_available():
    #model = model.cuda()
    #read test data
    Input_data = np.load("PD_test").astype("float32")
    Output_data = np.load("PG_test").astype("float32")        
    # Preprocessing training data
    P_D_test = Input_data[0:1000]
    P_D_test_normolization = (P_D_test - P_D_train_mean) / (P_D_train_std+1e-8)
    P_G_test = Output_data[0:1000]       
    # Preprocessing training data

    Power_Lowbound_Gen_test = np.tile(Power_Lowbound_Gen,(len(P_D_test),1))
    Power_Upbound_Gen_test = np.tile(Power_Upbound_Gen,(len(P_D_test),1))
    P_G_test_normolization = (P_G_test-Power_Lowbound_Gen_test)/(Power_Upbound_Gen_test-Power_Lowbound_Gen_test)
    del P_D_list_test,P_G_list_test

    ########################################
    #Test dataset
    P_D_test_normolization_tensor = torch.from_numpy(P_D_test_normolization)
    P_G_test_normolization_tensor = torch.from_numpy(P_G_test_normolization)
    test_dataset = Data.TensorDataset(P_D_test_normolization_tensor, P_G_test_normolization_tensor)
    test_loader = Data.DataLoader(
        dataset=test_dataset,      # torch TensorDataset format
        batch_size=1,#len(P_D_test_normolization_tensor),      
        shuffle=False,               
        #num_workers=2,              
    )

    Gen_Price = Gen_Price.T
    totalcost = 0
    feasible = 0
    SC_feasible = 0
    Slack_Gen_feasible = 0 
    PDM_test = torch.zeros([NN_input_number,Bus_number])
    PGM_test = torch.zeros([NN_output_number+1,Bus_number])
    Bus_admittance_line_tensor = torch.from_numpy(Bus_admittance_full_NN2)
    SC_Bus_admittance_line_tensor = torch.from_numpy(SC_Bus_admittance_line_1_array)
    test_index = 0
    cost_list=[]
    dnn_time=[]
    cplex_time=[]
    flag1_list=[]
    flag2_list=[]
    flag3_list=[]    
    solution_list=[]                  
    for (test_x, test_y) in test_loader:
        total = 0

        Slack_index = np.where(np.array(Gen_index) == Slack_bus)[0][0]
        Power_Lowbound_Gen_nn_test = torch.from_numpy(np.tile(Power_Lowbound_Gen, (len(test_x), 1)))
        Power_Lowbound_Gen_nn_test_1 = Power_Lowbound_Gen_nn_test[:, 0:Slack_index]
        Power_Lowbound_Gen_nn_test_1 = torch.cat([Power_Lowbound_Gen_nn_test_1, Power_Lowbound_Gen_nn_test[:,Slack_index+1:Gen_index.shape[0]]], 1)

        Power_Upbound_Gen_nn_test = torch.from_numpy(np.tile(Power_Upbound_Gen, (len(test_x), 1)))
        Power_Upbound_Gen_nn_test_1 = Power_Upbound_Gen_nn_test[:, 0:Slack_index]
        Power_Upbound_Gen_nn_test_1 = torch.cat([Power_Upbound_Gen_nn_test_1, Power_Upbound_Gen_nn_test[:, Slack_index+1:Gen_index.shape[0]]], 1)
        test_D = (test_x * (P_D_train_std_tensor + 1e-8) + P_D_train_mean_tensor)

        #PG,PG
        for i in range(len(Gen_index)):
            PGM_test[i,int(Gen_index[i])]= 1
        for i in range(len(Load_index)):
            PDM_test[i,int(Load_index[i])]= 1
        detault_fea_num = 0

        SC_fea_num = 0
        #use neural network to solve the problem
        start = time.time()

        test_out = model(test_x)

        #Pg calculation
        Pred_Pg_test = Power_Lowbound_Gen_nn_test_1 + (Power_Upbound_Gen_nn_test_1 - Power_Lowbound_Gen_nn_test_1) * test_out
        Pred_Pg_test_rev = torch.unsqueeze(torch.sum(test_D[:, 0:Load_index.shape[0]], 1),1) - torch.unsqueeze(torch.sum(Pred_Pg_test, 1), 1)
        if Pred_Pg_test_rev < 0:             
            Pred_Pg_test = Power_Lowbound_Gen_nn_test_1 + (Power_Upbound_Gen_nn_test_1 - Power_Lowbound_Gen_nn_test_1) * test_out * torch.sum(test_D[:, 0:Load_index.shape[0]])/torch.sum(Pred_Pg_test, 1)
            Pred_Pg_test_rev = torch.zeros(len(test_x), 1)
        elif Pred_Pg_test_rev > Power_Upbound_Gen[0,Slack_index]:
            Pred_Pg_test = Power_Lowbound_Gen_nn_test_1 + (Power_Upbound_Gen_nn_test_1 - Power_Lowbound_Gen_nn_test_1) * test_out * (torch.sum(test_D[:, 0:Load_index.shape[0]])-Power_Upbound_Gen_nn_test[0,Slack_index])/torch.sum(Pred_Pg_test, 1)           
            Pred_Pg_test_rev = torch.ones(len(test_x), 1)*Power_Upbound_Gen_nn_test[0,Slack_index]

        Pred_Pg_test_rev = torch.cat([Pred_Pg_test[:, 0:Slack_index], Pred_Pg_test_rev], 1)
        Pred_Pg_test_rev = torch.cat([Pred_Pg_test_rev, Pred_Pg_test[:, Slack_index:Gen_index.shape[0]]], 1)

        predictionG_test = Pred_Pg_test_rev.mm(PGM_test)
        predictionD_test = test_D.mm(PDM_test)

        predictionInj_test = predictionG_test - predictionD_test
        predictionInj_test_1 = predictionInj_test[:,0:Slack_bus]
        predictionInj_test_1 = torch.cat([predictionInj_test_1, predictionInj_test[:,Slack_bus+1:Bus_number]],1)

        # theta calculation
        theta_1_result_output_test = (Bus_1_Coeff_tensor.mm((predictionInj_test_1).t()))
        theta_1_result_output_test = theta_1_result_output_test + Slack_angle

        slack_angle_array_test = np.ones([1,len(test_x)], dtype='float32')*Slack_angle
        slack_angle_array_tensor_test = torch.from_numpy(slack_angle_array_test)

        theta_result_output_test = theta_1_result_output_test[0:Slack_bus,:]
        theta_result_output_test = torch.cat([theta_result_output_test, slack_angle_array_tensor_test],0)
        theta_result_output_test = torch.cat([theta_result_output_test, theta_1_result_output_test[Slack_bus:Bus_number-1,:]],0)

        trans_result_output_test = (Bus_admittance_line_tensor.mm(theta_result_output_test).t())        
        trans_result_output_test_np = trans_result_output_test.cpu().detach().numpy()
        Dstart_ori = trans_result_output_test_np*Branch_limit_ori
        infeasible_index = np.where((trans_result_output_test_np > 1) | (trans_result_output_test_np < -1))


        detault_fea_num = len(infeasible_index[0])
#        if detault_fea_num <= 0:
#            feasible = feasible + 1

        if mpc["scopf"] == 1:
        #inversve matrix of B (SCOPF)
            SC_Bus_1_Coeff_tensor = torch.from_numpy(SC_Bus_admittance_full_1_array)
            start2 = time.time()
            SC_theta_1_output_test = (SC_Bus_1_Coeff_tensor.matmul(predictionInj_test_1.t()))
            SC_theta_1_output_test = SC_theta_1_output_test + Slack_angle
            
            SC_theta_1_output_test_T = SC_theta_1_output_test.permute(0, 2, 1)
            slack_angle_array = np.ones([NumofContigengcy,len(test_x),1], dtype='float32')*Slack_angle
            slack_angle_array_tensor = torch.from_numpy(slack_angle_array)

            SC_theta_result_output_test = SC_theta_1_output_test_T[:,:,0:Slack_bus]
            SC_theta_result_output_test = torch.cat([SC_theta_result_output_test, slack_angle_array_tensor],2)
            SC_theta_result_output_test = torch.cat([SC_theta_result_output_test, SC_theta_1_output_test_T[:,:,Slack_bus:Bus_number-1]],2)

            #transmission line inequality constraint
            SC_trans_result_output_test = SC_theta_result_output_test.matmul(SC_Bus_admittance_line_tensor)
            SC_trans_result_output_test_rev = SC_trans_result_output_test.squeeze()
            SC_trans_result_output_test_rev_np = SC_trans_result_output_test_rev.cpu().detach().numpy()

            Dstart_sc = SC_trans_result_output_test_rev_np*Branch_limit_sc
            SC_infeasible_index = np.where((SC_trans_result_output_test_rev_np> 1) | (SC_trans_result_output_test_rev_np< -1))
            
            SC_fea_num = len(SC_infeasible_index[0])
#            if SC_fea_num <= 0:
#                SC_feasible = SC_feasible + 1
        
        # print('P_g_Test_result:\n')
        total = total+(time.time()-start)
        Pred_Pg_test_np = Pred_Pg_test_rev.detach().numpy()
        Pred_Pg_test_np =Pred_Pg_test_np.squeeze()
        Actual_PG_np = test_y.mul((Power_Upbound_Gen_nn_test - Power_Lowbound_Gen_nn_test) + Power_Lowbound_Gen_nn_test).numpy()
        #ACC_PG = abs((Actual_PG_np - Pred_Pg_test_np))# / (Actual_PG_np + 1e-8))
        #pg_error = pg_error+(np.mean(ACC_PG,axis=1))
        
        #check feasibility
        flag = 1
        start_4 = time.time()      
        if detault_fea_num <= 0 and SC_fea_num <= 0 and (Pred_Pg_test_np[Slack_index] <=  Power_Upbound_Gen[0,Slack_index]) and (Pred_Pg_test_np[Slack_index] >=  Power_Lowbound_Gen[0,Slack_index]):
            feasible += 1
            flag = 0
        total = total+(time.time()-start_4)
        if detault_fea_num <= 0:
            flag1_list.append(0)
            flag1 = 0
        else:
            flag1_list.append(1)
            flag1 = 1
        if SC_fea_num <= 0:
            flag2_list.append(0)
            flag2 = 0
        else:
            flag2_list.append(0)
            flag2 = 1
        if (Pred_Pg_test_np[Slack_index] >  Power_Upbound_Gen[0,Slack_index]) or (Pred_Pg_test_np[Slack_index] <  Power_Lowbound_Gen[0,Slack_index]):
            total = total+(time.time()-start_4)
            flag3_list.append(1)
            Slack_Gen_feasible += 1
            flag3 = 1
        else:
            flag3_list.append(0)
            flag3 = 0
#        cost = 0
        #post-processing
        if flag > 0:
            mpc_post = loadcase('pglib_opf_case30_ieee.py')
#            mpc_post['branch'][:,5] = mpc_post['branch'][:,5]*1.2
            iteration_time = 0
            load_index = 0
            for j in range(0,mpc_post['bus'].shape[0],1):
                if(mpc_post['bus'][j,2] != 0):
                    mpc_post['bus'][j,2] = Input_data[test_index,load_index]*standval
                    load_index = load_index + 1

            theta_result_output_test_np = theta_result_output_test.detach().numpy()
            theta_result_output_test_np = theta_result_output_test_np.squeeze()
            SC_theta_result_output_test_np = SC_theta_result_output_test.detach().numpy()
            SC_theta_result_output_test_np = SC_theta_result_output_test_np.squeeze()
            sol,iteration_time = gurobi_solv(m,mpc_post,test_index,Pred_Pg_test_np,theta_result_output_test_np,SC_theta_result_output_test_np,Dstart_ori,Dstart_sc)

            Pred_Pg_test_np = sol[0:mpc_post['gen'].shape[0]]     
            cost = np.multiply(Gen_Price[0, :]*standval*standval, np.power(Pred_Pg_test_np, 2)) + np.multiply(Gen_Price[1, :]*standval,Pred_Pg_test_np) + Gen_Price[2, :]
            cost_list.append(np.sum(cost))
            dnn_time.append(total)
            cplex_time.append(iteration_time)
            solution_list.append(Pred_Pg_test_np)
        else:    
            cost = np.multiply(Gen_Price[0, :]*standval*standval, np.power(Pred_Pg_test_np, 2)) + np.multiply(Gen_Price[1, :]*standval,Pred_Pg_test_np) + Gen_Price[2, :]        
            cost_list.append(np.sum(cost))
            dnn_time.append(total)
            cplex_time.append(0.0)
            solution_list.append(Pred_Pg_test_np)
        
        totalcost = totalcost + np.sum(cost)
        test_index = test_index+1

    filename1='SC_DCOPF_test_case30-DNN.txt'
    fid1=open(filename1,'a')
    for i in range(0,len(cost_list)):
        fid1.write("%d;%d;%d;" % (flag1_list[i],flag2_list[i],flag3_list[i]))
        fid1.write("%9.5f;%9.5f;%9.5f\n" % (cost_list[i],dnn_time[i],cplex_time[i]))
    fid1.close()