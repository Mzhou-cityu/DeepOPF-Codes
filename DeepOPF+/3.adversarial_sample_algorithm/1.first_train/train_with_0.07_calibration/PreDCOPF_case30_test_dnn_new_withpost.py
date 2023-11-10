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
from pypower.makeBdc import makeBdc
total = 0
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
torch.manual_seed(1)    # reproducible
from gurobipy import *


# parameter
NN_input_number = 0 
NN_output_number = 0   
neural_number = 8

################################
#DCOPF
Bus_number = 0
Gen_number = 0
Branch_number = 1
Slack_bus = 1
Slack_angle = 0
standval = 0
Load_index_list=[]
Gen_index_list=[]
Power_Lowbound_Gen=np.zeros([1,1],dtype='float32')
Power_Upbound_Gen=np.zeros([1,1],dtype='float32')
Bus_admittance_line = np.zeros([1,1],dtype='float32')
Bus_admittance_full = np.zeros([1,1],dtype='float32')
Bus_admittance_full_1 = np.zeros([1,1],dtype='float32')
Node_index = np.zeros([1,1],dtype='float32')
Node_angel = np.zeros([1,1],dtype='float32')
Load_index = np.zeros([1,1],dtype='float32')
Gen_index = np.zeros([1,1],dtype='float32')
Gen_Price = np.zeros([1,1],dtype='float32')

mL = 0
mG = 0
#gurobi_solv
def gurobi_solv(Pred_Pg=None,Pred_theta=None,mpc=None):
    ite_time = 0
    #read admittance matrix from pypower
    global standval
    global Bus_number
    global mL
    global mG
    #system MVA base
    standval = mpc["baseMVA"]
    #bus+load
    mpc_bus=mpc["bus"]
    [mL,nL]=mpc_bus.shape  
    Bus_number = mL
    #branch
    mpc_branch = mpc["branch"]
    [mB,nB]=mpc_branch.shape
    #genrator
    mpc_gen=mpc["gen"]
    
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
    
    bus = mpc_bus
    branch = mpc_branch      
            
    Load_index = []
    Load_amount = []
    Node_index = np.zeros((mL,),dtype='float64')
    Node_angel = np.zeros((mL,),dtype='float64')
        
    Gen_index = np.zeros((1, mG),dtype='float64')
    gen = np.zeros((mG, nG),dtype='float64')
            
    #NumOfLoad = mL
    for j in range(0,mL,1):
        Node_index[j] = int(mpc_bus[j,0])
        Node_angel[j] = mpc_bus[j,8]
        
    #index transfer
    for j in range(0,mL,1):
        ind = np.where(Node_index == mpc_bus[j,0])[0][0]
        bus[j,0] = ind
        if (mpc_bus[j,2] != 0):
            Load_index.append(ind)      
            Load_amount.append(mpc_bus[j,2])
        
    index = 0
    for j in range(0,mpc_gen.shape[0],1):
        if mpc_gen[j,7] != 0:
            gen[index,:] = mpc_gen[j,:]
            Gen_Price[index,:] = mpc_price[j,4:npri]
            ind = np.where(Node_index == mpc_gen[j,0])[0][0]
            Gen_index[0,index] = ind
            index = index + 1
        
    for j in range(0,mB,1):
        ind0 = np.where(Node_index == mpc_branch[j,0])[0][0]
        branch[j,0] = ind0  
        ind1 = np.where(Node_index == mpc_branch[j,1])[0][0]
        branch[j,1] = ind1          
#    global NumofContigengcy
    
    #global Branch_number
    Bus_admittance_full = np.zeros([mL,mL+mG],dtype='float64')
    Bus_admittance_line1 = np.zeros([mB,mL+mG],dtype='float64') 
        
    ## power mismatch constraints
    B, Bf, Pbusinj, Pfinj = makeBdc(standval, bus, branch)
        
    Bus_admittance_full[:,0:mL] = B.toarray()
    Bus_admittance_line1[:,0:mL] = Bf.toarray()   
            
    u = np.ones((mL+mB,), dtype = 'float64')
    l = -u
    v0 = np.zeros(((1)*mL+2*mG,), dtype = 'float64')   
    vu = np.zeros(((1)*mL+2*mG,), dtype = 'float64')   
    vl = np.zeros(((1)*mL+2*mG,), dtype = 'float64')
    
    for j in range(0,mG,1):
        Bus_admittance_full[int(Gen_index[0,j]),mL+j] = -1
        v0[mL+j] = gen[j,1]/standval
        vl[mL+j] = gen[j,9]/standval
        vu[mL+j] = gen[j,8]/standval
            
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
            v0[(1)*mL+mG+j] = 0
            vl[(1)*mL+mG+j] = 0
            vu[(1)*mL+mG+j] = GRB.INFINITY   

    for j in range(0,mB,1):
        u[mL+j] = branch[j,5]/standval#-GRB.INFINITY
        l[mL+j] = -branch[j,5]/standval #-GRB.INFINITY

    # Create variables
    x = m.addVars((1)*mL+2*mG, lb = vl, ub = vu, name="X")
    m.update()
        
    # Create a new model
    # obj = QuadExpr()
    # for i in range(0,mL,1):
    #     obj.add((x[i]-Pred_theta[i,0])*(x[i]-Pred_theta[i,0]))    
    # for i in range(0,mG,1):
    #     obj.add((x[mL+i]-Pred_Pg[0,i])*(x[mL+i]-Pred_Pg[0,i]))
    # m.setObjective(obj,sense=GRB.MINIMIZE)
    # m.update()    
    global obj
    obj = QuadExpr()
        # Set objective: cost of PG
#    [mPrice,nPrice] = Gen_Price.shape
    for i in range(0,mG,1):
            obj.add(x[(1)*mL+mG+i])
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
        #absolute constraint
    for i in range(0,mG,1):
        InequallityConsT1 = LinExpr()                                                                                                                               
        InequallityConsT1.add(x[mL+i]-x[(1)*mL+mG+i])
        m.addConstr(InequallityConsT1,GRB.LESS_EQUAL,Pred_Pg[0,i])            
        InequallityConsT2 = LinExpr()                                                                                                                               
        InequallityConsT2.add(x[mL+i]+x[(1)*mL+mG+i])
        m.addConstr(InequallityConsT2,GRB.GREATER_EQUAL,Pred_Pg[0,i]) 
    m.update()    
    del Bus_admittance_full, Bus_admittance_line1
    
    it_time = time.time()
    m.optimize()   
    ite_time = ite_time+(time.time()-it_time)

#    print('Obj: %g' % obj.getValue()
    Solution = []
    for v in range(0,mL+mG,1):
        Solution.append(m.getVars()[v].x)
    Solutionarr = np.array(Solution)

    return Solutionarr,ite_time

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

#parameter generation
def para_generation(mpc=None):
    #read admittance matrix from pypower
    global standval
    global Bus_number
    #system MVA base
    standval = mpc["baseMVA"]
    #bus(load)
    mpc_bus=mpc["bus"]
    [mL,nL]=mpc_bus.shape  
    Bus_number = mL
    #branch
    mpc_branch = mpc["branch"]
    [mB,nB]=mpc_branch.shape
    #genrator
    mpc_gen=mpc["gen"]

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

    bus = mpc_bus
    branch = mpc_branch      

    global Slack_bus
    global Slack_angle
    global NumOfLoad
    global Bus_index
    global Node_index
    global Node_angel
    #genrator
    global Gen_index
    global Power_Lowbound_Gen
    global Power_Upbound_Gen
    
    Node_index = np.zeros((mL,),dtype='float32')
    Node_angel = np.zeros((mL,),dtype='float32')

    Power_Lowbound_Gen = np.zeros((1, mG),dtype='float32')
    Power_Upbound_Gen = np.zeros((1, mG),dtype='float32')

    gen = np.zeros((mG, nG),dtype='float32')
           
    for j in range(0,mL,1):
        Node_index[j] = int(mpc_bus[j,0])
        Node_angel[j] = mpc_bus[j,8]
        if mpc_bus[j,1] == 3:
           Slack_bus = j
           Slack_angle = mpc_bus[j,8]*np.pi/180


    Load_index_list = []
    Gen_index_list = []    
    #index transfer
    for j in range(0,mL,1):
        ind = np.where(Node_index == mpc_bus[j,0])[0][0]
        bus[j,0] = ind
        if (mpc_bus[j,2] != 0):
            Load_index_list.append(ind)      
    
    index = 0
    for j in range(0,mpc_gen.shape[0],1):
        if mpc_gen[j,7] != 0:
            gen[index,:] = mpc_gen[j,:]
            ind = np.where(Node_index == mpc_gen[j,0])[0][0]
            Gen_index_list.append(ind)
            Power_Upbound_Gen[0,index] = mpc_gen[j,8]/standval
            Power_Lowbound_Gen[0,index] = mpc_gen[j,9]/standval
            index = index + 1

    
    for j in range(0,mB,1):
        ind0 = np.where(Node_index == mpc_branch[j,0])[0][0]
        branch[j,0] = ind0  
        ind1 = np.where(Node_index == mpc_branch[j,1])[0][0]
        branch[j,1] = ind1          

    global Gen_index
    global Load_index
    Gen_index = np.array(Gen_index_list)
    Load_index = np.array(Load_index_list)

    global NN_input_number
    global NN_output_number  
    NN_output_number = mG-1
    NN_input_number = Load_index.shape[0]
    
    global Branch_number
    Branch_number = mB
    global Bus_admittance_full
    Bus_admittance_full = np.zeros([mL,mL],dtype='float32')
    global Bus_admittance_line
    Bus_admittance_line = np.zeros([mB,mL],dtype='float32') 
    
    #DC-OPF
    ## power mismatch constraints
    B, Bf, Pbusinj, Pfinj = makeBdc(standval, bus, branch)   
    Bus_admittance_full[:,0:mL] = B.toarray()
    Bus_admittance_line[:,0:mL] = Bf.toarray()

    for i in range(0,mB,1):
        Bus_admittance_line[i,:] = Bus_admittance_line[i,:] / (branch[i,5]/standval)

    global Bus_admittance_full_1
    Bus_admittance_full_1 = Bus_admittance_full[0:Slack_bus,0:Slack_bus]
    Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Bus_admittance_full[0:Slack_bus,Slack_bus+1:Bus_number]),axis=1)
    Temp = Bus_admittance_full[Slack_bus+1:Bus_number,0:Slack_bus]
    Temp = np.concatenate((Temp,Bus_admittance_full[Slack_bus+1:Bus_number,Slack_bus+1:Bus_number]),axis=1)
    Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Temp),axis=0)    
   
    mpc_price = mpc['gencost']
    [mpri,npri] = mpc_price.shape
    global Gen_Price
    Gen_Price = np.resize(Gen_Price, (mpri, npri-4))
    for i in range(0, mpri, 1):
        for j in range(0,npri-4,1):
            Gen_Price[i,j] = mpc_price[i,4+j]

if __name__ == '__main__':
    #read case file
    mpc = loadcase('case30_rev_100.py')
    mpc_branch=mpc['branch']
    mpc_branch[:,5]=mpc_branch[:,5]*1.017
    #reading parameters
    para_generation(mpc)
    # global data
    P_D_list = []
    P_G_list = []
    P_D_list_test = []
    P_G_list_test = []    
    A_post=[]
    #read training data
    numOfdata = 0
    dir='./training_data/'
    for index in range(0, 5, 1):
        filename= dir+'Pre_DC_training_case30-sampling_part_0.93_'+str(index)+'.txt'
        with open(filename, 'r') as f:
            data = f.readlines()
            numOfdata = numOfdata + int(len(data)/2) 
            j = 1
            for line in data:
                    line = line.strip('\n')
                    if j % 2 == 1:
                        odom = line.split(';')  # load
                        for i in range(len(odom)):
                            P_D_list.append(float(odom[i])/standval)
                    else:
                        odom  =line.split(';')  # supply
                        for i in range(len(odom)):
                            Gen_data = odom[i].split(':')
                            P_G_list.append(float(Gen_data[1]))
                    j = j + 1
    ################################################################
    del data
    num_of_group = int(np.floor(numOfdata / 10))
    P_D = np.array(P_D_list, dtype=np.float32)
    P_D = P_D.reshape(-1, NN_input_number)
    P_G = np.array(P_G_list, dtype=np.float32)
    P_G = P_G.reshape(-1, NN_output_number+1)
    # Preprocessing training data
    P_D_train = P_D[0:num_of_group * 8]
    P_D_train_mean = np.mean(P_D_train, axis=0)
    P_D_train_std = np.std(P_D_train, axis=0)
    P_D_train_std_tensor = torch.from_numpy(P_D_train_std) 
    P_D_train_mean_tensor = torch.from_numpy(P_D_train_mean) 
    del P_D_list, P_G_list
    #test inversve matrix of B (default DCOPF)
    Bus_1_Coeff = np.linalg.inv(Bus_admittance_full_1)
    Bus_1_Coeff_tensor = torch.from_numpy(Bus_1_Coeff) 
    ################################################################
    # restore net
    model = Neuralnetwork(NN_input_number, neural_number, NN_output_number)
    model.load_state_dict(torch.load('PreDCOPF_case30_0.93_dnn.pth',map_location='cpu'))
    #if torch.cuda.is_available():
    #model = model.cuda()
    #read test data
    dir = './test_data/'
    for index in range(0,1,1):
        filename= dir+'Pre_DC_test_case30-sampling'+'.txt'
        fid=open(filename,'r')
        data = fid.readlines()
        j = 1
        for line in data:
            line = line.strip('\n')
            if j % 2 == 1:
                odom = line.split(';')
                if j ==1:
                    NumOfLoad = len(odom)
                for i in range(len(odom)):
                    P_D_list_test.append(float(odom[i]))
            else:
                odom  =line.split(';')
                if j == 2:
                    NumOfGen = len(odom)
                for i in range(len(odom)):
                    Gen_data = odom[i].split(':')
                    P_G_list_test.append(float(Gen_data[1]))                 
            j = j + 1
        fid.close()
    ################################################################
    del data
    Input_data = np.array(P_D_list_test, dtype=np.float32).reshape(-1, NumOfLoad)
    Output_data = np.array(P_G_list_test, dtype=np.float32).reshape(-1, NumOfGen)         
    # Preprocessing training data
    P_D_test = Input_data/standval
    P_D_test_normolization = (P_D_test - P_D_train_mean) / (P_D_train_std+1e-8)
    P_G_test = Output_data
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
    pg_error = 0
    PDM_test = torch.zeros([NN_input_number,Bus_number])
    PGM_test = torch.zeros([NN_output_number+1,Bus_number])
    Bus_admittance_line_tensor = torch.from_numpy(Bus_admittance_line)
    cost_list = []
    test_index = 0
    for (test_x, test_y) in test_loader:
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
            PGM_test[i,Gen_index[i]]= 1
        for i in range(len(Load_index)):
            PDM_test[i,Load_index[i]]= 1

        #use neural network to solve the problem
        start = time.time()

        test_out = model(test_x)

        Pred_Pg_test = Power_Lowbound_Gen_nn_test_1 + (Power_Upbound_Gen_nn_test_1 - Power_Lowbound_Gen_nn_test_1) * test_out
        Pred_Pg_test_rev = torch.unsqueeze(torch.sum(test_D[:, 0:Load_index.shape[0]], 1),1) - torch.unsqueeze(torch.sum(Pred_Pg_test, 1), 1)

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
        time_consumption1 = (time.time()-start)     
        total = total+time_consumption1

        slack_angle_array_test = np.ones([1,len(test_x)], dtype='float32')*Slack_angle
        slack_angle_array_tensor_test = torch.from_numpy(slack_angle_array_test)

        start1 = time.time()
        theta_result_output_test = theta_1_result_output_test[0:Slack_bus,:]
        theta_result_output_test = torch.cat([theta_result_output_test, slack_angle_array_tensor_test],0)
        theta_result_output_test = torch.cat([theta_result_output_test, theta_1_result_output_test[Slack_bus:Bus_number-1,:]],0)

        trans_result_output_test = (Bus_admittance_line_tensor.mm(theta_result_output_test).t()) 
        trans_result_output_test_np = trans_result_output_test.cpu().detach().numpy()

        infeasible_index = np.where((trans_result_output_test_np > 1) | (trans_result_output_test_np < -1))
        time_consumption2 = (time.time()-start1)
        total = total+time_consumption2

        filename1='Pre_DC_test_case30-sampling_DNN_0.93_withpost.txt'
        fid1=open(filename1,'a') 
        # feasible = feasible + len(test_x) - len(infeasible_index[0])
        if len(infeasible_index[0]) >0:
            flag_1 = 1
            fid1.write("%d;" % (1))
            feasible = feasible+0
        else:
            flag_1 = 0
            fid1.write("%d;" % (0))
            feasible = feasible+1
        Pred_Pg_test_np = Pred_Pg_test_rev.detach().numpy()
        theta_result_output_test_np = theta_result_output_test.detach().numpy()
        if (Pred_Pg_test_np[0,Slack_index] >  Power_Upbound_Gen[0,Slack_index]) or (Pred_Pg_test_np[0,Slack_index] <  Power_Lowbound_Gen[0,Slack_index]):
            flag_2 = 1
            fid1.write("%d;" % (1))
        else:
            flag_2 = 0
            fid1.write("%d;" % (0))
     
        #for post-processing
        if flag_1 == 1 or flag_2 ==1:
            mpc = loadcase('case30_rev_100.py')
            mpc_branch=mpc['branch']
            mpc_branch[:,5]=mpc_branch[:,5]*1.017
            mpc_bus=mpc['bus']
            [m_bus,n_bus]=mpc_bus.shape 
            iteration_time = 0
            load_index = 0
            for j in range(0,m_bus,1):
                if(mpc_bus[j,2] != 0):
                    mpc_bus[j,2] = Input_data[test_index,load_index]
                    load_index = load_index + 1
            start_time = time.time()
            m = Model("qp")
            sol,temp_ite_time = gurobi_solv(Pred_Pg_test_np,theta_result_output_test_np,mpc)
            iteration_time = iteration_time + temp_ite_time
            A_post.append(sol)
            fid1.write("%9.5f;" % (time_consumption1+time_consumption2+iteration_time))
            Pred_Pg_test_np = sol

            cost = np.multiply(Gen_Price[0, :]*standval*standval, np.power(sol[mL:mL+mG], 2)) + np.multiply(Gen_Price[1, :]*standval,sol[mL:mL+mG]) + Gen_Price[2, :]
            #print(cost)
            cost_list.append(np.sum(cost))
            totalcost = totalcost + np.sum(cost)
        else:
            fid1.write("%9.5f;" % (time_consumption1+time_consumption2))
            cost = np.multiply(Gen_Price[0, :]*standval*standval, np.power(Pred_Pg_test_np, 2)) + np.multiply(Gen_Price[1, :]*standval,Pred_Pg_test_np) + Gen_Price[2, :]
            #print(cost)
            cost_list.append(np.sum(cost))
            totalcost = totalcost + np.sum(cost)
        filename2='cost.txt'
        fid2=open(filename2,'a')
        fid2.write("%9.5f;" % (np.sum(cost)))
        
        fid2.write("\n")
        fid1.write("\n")  
        fid1.close()   
        fid2.close() 

        test_index = test_index+1

    print('\ntotal time:%.3f\n' % total)
    print('\naverage cost:%.3f\n' % (totalcost/len(P_D_test_normolization)))
    print("\nfeasible transmission percentage is %.3f %%:\n" % (feasible/len(P_D_test_normolization)*100))
    
