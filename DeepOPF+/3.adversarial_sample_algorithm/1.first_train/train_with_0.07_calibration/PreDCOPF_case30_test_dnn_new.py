# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:10:25 2019

@author: xpan
"""

import numpy as np
import matplotlib.pyplot as plt
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
Time1=[]
Time2=[]
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
    dir = './/test_data//'
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
                    P_D_list_test.append(float(odom[i])/standval)
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
    P_D_test = Input_data
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

#        #PG
#        for i in range(Bus_number):
#            if i == 0:
#                if i in Gen_index:
#                    index = np.where(np.array(Gen_index)==i)[0]
#                    predictionG_test = Pred_Pg_test_rev[:, index]
#                else:
#                    predictionG_test = torch.zeros([len(test_x), 1])
#
#            else:
#                if i in Gen_index:
#                    index = np.where(np.array(Gen_index)==i)[0]
#                    predictionG_test = torch.cat([predictionG_test, Pred_Pg_test_rev[:, index]],1)
#                else:
#                    predictionG_test = torch.cat([predictionG_test, torch.zeros([len(test_x), 1])],1)
#
#        #PD
#        for i in range(Bus_number):
#            if i == 0:
#                if i in Load_index:
#                    index = np.where(np.array(Load_index)==i)[0]
#                    predictionD_test = test_D[:, index]
#                else:
#                    predictionD_test = torch.zeros([len(test_x), 1])
#            else:
#                if i in Load_index:
#                    index = np.where(np.array(Load_index)==i)[0]
#                    predictionD_test = torch.cat([predictionD_test, test_D[:, index]],1)
#                else:
#                    predictionD_test = torch.cat([predictionD_test, torch.zeros([len(test_x), 1])],1)

        predictionInj_test = predictionG_test - predictionD_test
        predictionInj_test_1 = predictionInj_test[:,0:Slack_bus]
        predictionInj_test_1 = torch.cat([predictionInj_test_1, predictionInj_test[:,Slack_bus+1:Bus_number]],1)

        # theta calculation
        theta_1_result_output_test = (Bus_1_Coeff_tensor.mm((predictionInj_test_1).t()))
        theta_1_result_output_test = theta_1_result_output_test + Slack_angle
        time_consumption1 = (time.time()-start)     
        Time1.append(time_consumption1)
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
        Time2.append(time_consumption2)
        total = total+time_consumption2

        filename1='Pre_DC_test_case30-sampling_DNN_0.93.txt'
        fid1=open(filename1,'a')
        fid1.write("%9.5f;" % (time_consumption1+time_consumption2))  

        if len(infeasible_index[0]) >0:
            fid1.write("%d;" % (1))
            feasible = feasible+0
        else:
            fid1.write("%d;" % (0))
            feasible = feasible+1
        Pred_Pg_test_np = Pred_Pg_test_rev.detach().numpy()
        if (Pred_Pg_test_np[0,Slack_index] >  Power_Upbound_Gen[0,Slack_index]) or (Pred_Pg_test_np[0,Slack_index] <  Power_Lowbound_Gen[0,Slack_index]):
            fid1.write("%d;" % (1))
        else:
            fid1.write("%d;" % (0))
        fid1.write("\n")  
        fid1.close()
#        Actual_PG_np = test_y.mul((Power_Upbound_Gen_nn_test - Power_Lowbound_Gen_nn_test) + Power_Lowbound_Gen_nn_test).numpy()
#        ACC_PG = abs((Actual_PG_np - Pred_Pg_test_np))# / (Actual_PG_np + 1e-8))
#        pg_error = pg_error+(np.mean(ACC_PG,axis=1))
        cost = np.multiply(Gen_Price[0, :]*standval*standval, np.power(Pred_Pg_test_np, 2)) + np.multiply(Gen_Price[1, :]*standval,Pred_Pg_test_np) + Gen_Price[2, :]
        cost_list.append(np.sum(cost, axis=1))
        totalcost = totalcost + np.sum(cost, axis=1)

    print('\ntotal time:%.3f\n' % total)
    print('\naverage cost:%.3f\n' % (totalcost/len(P_D_test_normolization)))
    print("\nfeasible transmission percentage is %.3f %%:\n" % (feasible/len(P_D_test_normolization)*100))
#    print("\nSC_feasible transmission percentage is %.3f %%:\n" % (SC_feasible/len(P_D_test_normolization)*100))
#    print("\npg error is %.6f:\n" % (pg_error/len(P_D_test_normolization)))
    
