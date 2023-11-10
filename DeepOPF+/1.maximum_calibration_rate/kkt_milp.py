# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:49:38 2021

@author: Tianyu_Zhao
"""

#DC-OPF
from gekko import GEKKO
from scipy.sparse import csr_matrix as sparse
from pypower.loadcase import loadcase
from pypower.makeBdc import makeBdc
import numpy as np
from pypower.ext2int import ext2int1
from torch import nn, torch, optim
import torch.nn.functional as F
import math
import threading
import time
import torch.utils.data as Data
import random
import os
from gurobipy import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
default=1.15
input_range = 0.15


    
   #ipopt_solve
   
def total_calibration(m=None,mpc=None):
    # global Lower_bound
    # global Upper_bound
    
    ###################################
    #variables for DC-OPF
    start=time.time()    
    
    
    Input_lower=np.zeros((len(Load_index_list),),dtype='float32')
    Input_upper=np.zeros((len(Load_index_list),),dtype='float32')
    for i in range(0,len(Load_index_list),1):
        if bus[Load_index_list[i],2]/standval >=0:
            Input_lower[i] = bus[Load_index_list[i],2]/standval*(default-input_range)
            Input_upper[i] = bus[Load_index_list[i],2]/standval*(default+input_range)
        else:
            Input_lower[i] = bus[Load_index_list[i],2]/standval*(default+input_range)
            Input_upper[i] = bus[Load_index_list[i],2]/standval*(default-input_range)
    Input = m.addVars(len(Load_index_list),lb = Input_lower, ub = Input_upper, name="")

	
    pd_upper=np.zeros((mL,),dtype='float32')
    pd_lower=np.zeros((mL,),dtype='float32')
    load_index = 0
    for i in range(0,mL,1):
        if i in Load_index_list:
            pd_upper[i] = Input_upper[load_index]
            pd_lower[i] = Input_lower[load_index]
            load_index += 1
        else:
            pd_upper[i] =0
            pd_lower[i] = 0            
    pd = m.addVars(mL, lb=pd_lower,ub=pd_upper,name="")
    for i in range(0, len(pd)):
        pd[i].start=pd_input[i]
    for i in range(0, len(Input)):
        Input[i].start=Input_initial[i]
	
	
	
	
    load_index = 0
    for i in range(0,mL,1):
        if i in Load_index_list:
            m.addConstr(pd[i] == Input[load_index])
            load_index += 1
        

    #pg_feasible
    pg_feasible_upper=np.zeros((mL,),dtype='float32')
    pg_feasible_lower=np.zeros((mL,),dtype='float32')
    
    gen_index =0       
    for i in range(0,mL,1):
        if bus[i,1] == 3:
            pg_feasible_lower[i] = Power_Lowbound_Gen_Slack
            pg_feasible_upper[i] = Power_Upbound_Gen_Slack
        elif bus[i,1] == 2 and i in Gen_index_list:
            pg_feasible_lower[i] = gen[gen_index,9]/standval
            pg_feasible_upper[i] = gen[gen_index,8]/standval
            gen_index += 1
        else:
            pg_feasible_lower[i] = 0
            pg_feasible_upper[i] = 0
        
    pg_feasible = m.addVars(mL, lb=pg_feasible_lower,ub=pg_feasible_upper,name="")
    m.addConstr(pg_feasible.sum() == pd.sum())
    
   
    
    
    Branch_feasible_upper=np.zeros((mB,),dtype='float32')
    Branch_feasible_lower=np.zeros((mB,),dtype='float32')
    for i in range(0, mB):
        
        Branch_feasible_lower[i]=-Power_bound_Branch[i]
        Branch_feasible_upper[i]=Power_bound_Branch[i]
		
    Branch_feasible=m.addVars(mB, lb=Branch_feasible_lower,ub=Branch_feasible_upper,name="")
    for i in range(0, mB):
        m.addConstr(Branch_feasible[i]==sum([PTDF_ori[i,j]*(pg_feasible[j]-pd[j]) for j in range(0,mL)]))

  #####
  #  DUAL PROBLEM
    Dual_variable_lower=np.zeros((2*len(branch_list)+2*mG+2+2+2*mB,),dtype='float32')
    Dual_variable_upper=np.zeros((2*len(branch_list)+2*mG+2+2+2*mB,),dtype='float32')
    for i in range(0,len(Dual_variable_lower),1):
        Dual_variable_lower[i] = 0
        Dual_variable_upper[i] = GRB.INFINITY
    Dual_variable=m.addVars(2*len(branch_list)+2*mG+2+2+2*mB,lb=Dual_variable_lower,ub=Dual_variable_upper,name="")
    Primal_variable_lower=np.zeros((mG+1,),dtype='float32')
    Primal_variable_upper=np.zeros((mG+1,),dtype='float32')
    for i in range(0,len(Primal_variable_lower),1):
        Primal_variable_lower[i] = -GRB.INFINITY
        Primal_variable_upper[i] = GRB.INFINITY
		
    Primal_variable=m.addVars(mG+1,lb=Primal_variable_lower,ub=Primal_variable_upper,name="")
    ##matrix A
    ####branch_magnitude
    matrix_temp1=PTDF_pg/Power_bound_Branch[:,None]
    PTDF_pg_normalized=PTDF_pg/Power_bound_Branch[:,None]
    matrix_temp2=np.zeros((mB, 1),dtype='float32')
    for i in range(0,mB):
        matrix_temp2[i]=1    
    matrix_block1=np.hstack((matrix_temp1,matrix_temp2))
    matrix_block2=np.hstack((-matrix_temp1,matrix_temp2))
    TEMP1=[];
    TEMP2=[];
    for i in range(0,mB):
        if i in branch_list:
            TEMP1.append(matrix_block1[i])
            TEMP2.append(matrix_block2[i])
    matrix_branch_calibration=np.vstack((TEMP1,TEMP2))
    
    #####generator_feasible
    matrix_temp3=np.zeros((2*mG, mG),dtype='float32')
    for i in range(0,mG):
        for j in range(0,mG):
            if i==j:
                matrix_temp3[i,j]=1
    for i in range(mG,2*mG):
        for j in range(0,mG):
            if i==j+mG:
                matrix_temp3[i,j]=-1
    matrix_temp4=np.zeros((2*mG, 1),dtype='float32')
    matrix_block3=np.hstack((matrix_temp3,matrix_temp4))
    
    
    #####net power balance
    matrix_temp5=np.zeros((1, mG+1),dtype='float32')
    for i in range(0,mG):
        matrix_temp5[0,i]=1
    matrix_temp6=np.zeros((1, mG+1),dtype='float32')
    for i in range(0,mG):
        matrix_temp6[0,i]=-1
    matrix_block4=np.vstack((matrix_temp5,matrix_temp6))
    
#####slack_calibration
    matrix_temp7=np.zeros((1, mG+1),dtype='float32')
    for i in range(0,mG):
        if i==Slack_gen_order:
            matrix_temp7[0,i]=0
        else:
            matrix_temp7[0,i]=-1
    matrix_temp7[0,mG]=Power_Upbound_Gen_Slack  
    matrix_temp8=np.zeros((1, mG+1),dtype='float32')
    for i in range(0,mG):
        if i==Slack_gen_order:
            matrix_temp8[0,i]=0
        else:
            matrix_temp8[0,i]=1
    matrix_temp8[0,mG]=Power_Upbound_Gen_Slack     
    matrix_block5=np.vstack((matrix_temp7,matrix_temp8))
    
 #####power flow feasible
    matrix_tempzero=np.zeros((mB, 1),dtype='float32')
    matrix_temp9=np.hstack((matrix_temp1,matrix_tempzero))
    matrix_temp10=np.hstack((-matrix_temp1,matrix_tempzero))    
    matrix_block6=np.vstack((matrix_temp9,matrix_temp10))
 
    Coefficient_matrix=np.vstack((matrix_branch_calibration,matrix_block3,matrix_block4,matrix_block5,matrix_block6))
    
    #object_vector
    C=np.zeros((mG+1, 1),dtype='float32')
    C[mG]=1
    C=C.flatten()
    
    #coefficient_vector
    PTDF_pd_normalized=PTDF_pd/Power_bound_Branch[:,None]
    Coefficient_vector_lower=np.zeros((2*len(branch_list)+2*mG+2+2+2*mB,),dtype='float32')
    Coefficient_vector_upper=np.zeros((2*len(branch_list)+2*mG+2+2+2*mB,),dtype='float32')
    for i in range(0,len(Coefficient_vector_lower),1):
        Coefficient_vector_lower[i] = -GRB.INFINITY
        Coefficient_vector_upper[i] = GRB.INFINITY
		
    Coefficient_vector=m.addVars(2*len(branch_list)+2*mG+2+2+2*mB,lb=Coefficient_vector_lower,ub=Coefficient_vector_upper,name="")

    # Coefficient_vector=m.Array(m.Var,(2*len(branch_list)+2*mG+2+2+2*mB,))
    
    index=0
    for i in range(0,mB):
        if i in branch_list:
            m.addConstr(Coefficient_vector[index]==sum([PTDF_pd_normalized[i,j]*Input[j] for j in range(0,len(Input))])+1)
            index=index+1
    index=len(branch_list)
    for i in range(0,mB):
        if i in branch_list:
            m.addConstr(Coefficient_vector[index]==sum([-PTDF_pd_normalized[i,j]*Input[j] for j in range(0,len(Input))])+1)
            index=index+1
            
    for i in range(2*len(branch_list),2*len(branch_list)+mG):
        m.addConstr(Coefficient_vector[i]==pgmax[i-2*len(branch_list)])
    for i in range(2*len(branch_list)+mG,2*len(branch_list)+2*mG):
        m.addConstr(Coefficient_vector[i]==-pgmin[i-2*len(branch_list)-mG])
    m.addConstr(Coefficient_vector[2*len(branch_list)+2*mG]==pd.sum())
    m.addConstr(Coefficient_vector[2*len(branch_list)+2*mG+1]==-pd.sum())
    m.addConstr(Coefficient_vector[2*len(branch_list)+2*mG+2]==Power_Upbound_Gen_Slack-pd.sum())
    m.addConstr(Coefficient_vector[2*len(branch_list)+2*mG+3]==pd.sum())
    for i in range(0,mB):
        m.addConstr(Coefficient_vector[2*len(branch_list)+2*mG+4+i]==sum([PTDF_pd_normalized[i,j]*Input[j] for j in range(0,len(Input))])+1)
    for i in range(0,mB):
        m.addConstr(Coefficient_vector[2*len(branch_list)+2*mG+4+mB+i]==sum([-PTDF_pd_normalized[i,j]*Input[j] for j in range(0,len(Input))])+1)
    
    #dual variable
    #dual>=0
    
        ##stationary condition
    for i in range(0,mG+1):
        m.addConstr(C[i]-sum([Coefficient_matrix[j,i]*Dual_variable[j] for j in range(0,len(Dual_variable))])==0)
		
		
    slack_integer = m.addVars(len(Dual_variable), vtype=GRB.BINARY, name="")
    for i in range(0,len(Dual_variable)):
        m.addConstr(Dual_variable[i]<=(1-slack_integer[i])*Upper_bound)
        m.addConstr((sum([Coefficient_matrix[i,j]*Primal_variable[j] for j in range(0,len(Primal_variable))])-Coefficient_vector[i])>=slack_integer[i]*Lower_bound)
# 		
		
    # for i in range(0,len(Dual_variable)):
        # m.addConstr((sum([Coefficient_matrix[i,j]*Primal_variable[j] for j in range(0,len(Primal_variable))])-Coefficient_vector[i])*Dual_variable[i]==0)
		
		
		
		
		
    # TEMP_primal=m.Array(m.Var,(len(Dual_variable),))
    for i in range(0,len(Dual_variable)):
        m.addConstr(sum([Coefficient_matrix[i,j]*Primal_variable[j] for j in range(0,len(Primal_variable))])-Coefficient_vector[i]<=0)
        # TEMP_primal[i].upper=0
    #objective
    # Objective=m.Array(m.Var,(1,))
    
    # m.Equation(Objective[0]==m.sum([C[j]*Primal_variable[j] for j in range(0,len(Primal_variable))]))
    
    
    
    
    
        

        
    #branch flow violation
    # Branch_index = 1
    
    m.setObjective(Primal_variable[len(Primal_variable)-1], sense=GRB.MINIMIZE)
    # m.Obj(Primal_variable[len(Primal_variable)-1])
    m.update()
    m.optimize()
    # minimize objective
    print(m.status)
    if m.status==2:
		
        for i in range(len(Input),len(Input)+len(pd)):
            A.append(m.getVars()[i].x)
        
        print(m.objVal)
        
        
        Result[increase]=np.array(m.objVal)
    else:
        A.append(pd_input)
        Result[increase]=1000000

        



        
            
        



    
if __name__ == '__main__':
    # Initialize Model

    epsilon=0.0001
    alpha=0.001
    # branch_list=[  3,   6,   8,  40,  42,  44,  48,  49,  54,  55,  56,  57,  58,
    #     60,  70,  73,  79,  80,  81,  82,  83,  84,  85,  86,  88,  89,
    #     90,  91,  92,  93,  94,  96,  97,  98,  99, 100, 101, 102, 103,
    #     104, 105, 106, 108, 109, 110, 111, 112, 114, 115, 118, 119, 127,
    #     128, 132, 137, 138, 139, 140, 141, 142, 144, 145, 146, 148, 149,
    #     150, 153, 156, 158, 159, 163, 166, 168, 169, 170, 171, 174, 177,
    #     178, 181, 183, 185, 187, 188, 189, 190, 191, 192, 193, 195, 196,
    #     198, 202, 203, 204, 210, 211, 213, 217, 218, 219, 220, 221, 222,
    #     224, 225, 226, 227, 228, 229, 231, 232, 233, 243, 246, 248, 249,
    #     250, 251, 254, 267, 271, 273, 288, 289, 290, 291, 296, 299, 306,
    #     308, 309, 335, 336, 338, 342, 346, 347, 348, 355, 359, 360, 364,
    #     378, 402]
    # branch_list=[ 29,  30,  37,  40,  44,  53,  58,  64,  65,  66,  95,  97,  98,
    #     103, 104, 105, 106, 107, 108, 109, 110, 111, 115, 118, 119, 120,
    #     122, 125, 126, 127, 128, 140, 147, 149, 150, 151, 152, 154, 155,
    #     157, 162, 184, 185]
    branch_list=[0,  9,  28,  29,  30,  31,  32,  34]

    

    
    
    
    
    
    
    #build model

        
    START=time.time()
    
    mpc = loadcase('case30_rev_100.py')
    #read admittance matrix from pypower    
    standval, mpc_bus, mpc_gen, mpc_branch = \
    mpc['baseMVA'], mpc['bus'], mpc['gen'], mpc['branch']

    ## switch to internal bus numbering and build admittance matrices
    _, bus, gen, branch = ext2int1(mpc_bus, mpc_gen, mpc_branch)
    
    [mL,nL] = bus.shape
    [mB,nB] = branch.shape

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

    Load_index_list = []
    Gen_index_list = []  
    Gen_index = np.zeros((1, mG),dtype='float64')
    Power_Lowbound_Gen = np.zeros((1, mG-1),dtype='float32')
    Power_Upbound_Gen = np.zeros((1, mG-1),dtype='float32')    
    Power_Lowbound_Gen_Slack = 0
    Power_Upbound_Gen_Slack = 0  

    for j in range(0,mL,1):
        if mpc_bus[j,1] == 3:
           Slack_bus = j
        if (mpc_bus[j,2] != 0):
            Load_index_list.append(j)

    index = 0
    for j in range(0,gen.shape[0],1):
        if gen[j,7] != 0 and gen[j,0] != Slack_bus:
            Gen_index_list.append(gen[j,0])
            Power_Upbound_Gen[0,index] = gen[j,8]/standval
            Power_Lowbound_Gen[0,index] = gen[j,9]/standval
            index = index + 1
        else:
            Power_Lowbound_Gen_Slack = gen[j,9]/standval
            Power_Upbound_Gen_Slack = gen[j,8]/standval

    Gen_index = np.array(Gen_index_list)
    Load_index = np.array(Load_index_list)
    ###########################################################33
    Power_bound_Branch = np.zeros((mB,),dtype='float32')
    Power_bound_Branch = branch[:,5]/standval*1.017
    
    Power_Upbound_Gen=Power_Upbound_Gen.flatten()
    Power_Lowbound_Gen=Power_Lowbound_Gen.flatten()
        
            
    Bus_admittance_full = np.zeros([mL,mL],dtype='float32')
    Bus_admittance_line1 = np.zeros([mB,mL],dtype='float32')    
    B, Bf, Pbusinj, Pfinj = makeBdc(standval, bus, branch)   
    Bus_admittance_full[:,0:mL] = B.toarray()
    Bus_admittance_line1[:,0:mL] = Bf.toarray()

#    global Bus_admittance_full_1
    Bus_admittance_full_1 = Bus_admittance_full[0:Slack_bus,0:Slack_bus]
    Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Bus_admittance_full[0:Slack_bus,Slack_bus+1:mL]),axis=1)
    Temp = Bus_admittance_full[Slack_bus+1:mL,0:Slack_bus]
    Temp = np.concatenate((Temp,Bus_admittance_full[Slack_bus+1:mL,Slack_bus+1:mL]),axis=1)
    Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Temp),axis=0)    

    Bus_admittance_full_1_inv = np.linalg.inv(Bus_admittance_full_1)
    Bus_admittance_full_part1 = Bus_admittance_full_1_inv[0:Slack_bus,0:Slack_bus]
    Bus_admittance_full_part1 = np.concatenate((Bus_admittance_full_part1,np.zeros([Bus_admittance_full_part1.shape[0],1],dtype='float32')),axis=1)
    Bus_admittance_full_part1 = np.concatenate((Bus_admittance_full_part1,Bus_admittance_full_1_inv[0:Slack_bus,Slack_bus:bus.shape[0]]),axis=1)
    Bus_admittance_full_part2 = Bus_admittance_full_1_inv[Slack_bus:bus.shape[0],0:Slack_bus]
    Bus_admittance_full_part2 = np.concatenate((Bus_admittance_full_part2,np.zeros([Bus_admittance_full_part2.shape[0],1],dtype='float32')),axis=1)
    Bus_admittance_full_part2 = np.concatenate((Bus_admittance_full_part2,Bus_admittance_full_1_inv[Slack_bus:bus.shape[0],Slack_bus:bus.shape[0]]),axis=1)
    Bus_admittance_full = np.concatenate((Bus_admittance_full_part1,np.zeros([1,bus.shape[0]],dtype='float32')),axis=0)
    Bus_admittance_full = np.concatenate((Bus_admittance_full,Bus_admittance_full_part2),axis=0)    

    PTDF_ori = np.dot(Bf.toarray(),Bus_admittance_full).astype(np.float32)
    
    
    Tempvecpg=np.zeros((mL,),dtype='float32')
    for i in range(0,mL):
        if bus[i,1] == 3 or (bus[i,1] == 2 and i in Gen_index_list):
            Tempvecpg[i]=1
        else:
            Tempvecpg[i]=0
    Tempvecpd=np.zeros((mL,),dtype='float32')
    for i in range(0,mL):
        if i in Load_index_list:
            Tempvecpd[i]=1
        else:
            Tempvecpd[i]=0
            
    Pg_matrix=np.zeros((mL, mG), int);
    Pd_matrix=np.zeros((mL, len(Load_index_list)), int);
    
    count=0
    for i in range(0,mL):
        if (bus[i,1] == 3 or (bus[i,1] == 2 and i in Gen_index_list)):
            for j in range(0,mG):
                if j==count:
                    Pg_matrix[i,j]=1
                else:
                    Pg_matrix[i,j]=0
            count=count+1
    count=0
    for i in range(0,mL):
        if i in Load_index_list:
            for j in range(0,len(Load_index_list)):
                if j==count:
                    Pd_matrix[i,j]=1
                else:
                    Pd_matrix[i,j]=0
            count=count+1
            
    Slack_gen_order=0
    
    for i in range(0,mG-1):
        if Slack_bus>Gen_index[i]:
            Slack_gen_order=Slack_gen_order+1
        else:
            Slack_gen_order=Slack_gen_order
            
    #pdmin pdmax
    pdmin=np.zeros((len(Load_index_list),),dtype='float32')
    pdmax=np.zeros((len(Load_index_list),),dtype='float32')
    for i in range(0,len(Load_index_list),1):
        if bus[Load_index_list[i],2]/standval >=0:
            pdmin[i] = bus[Load_index_list[i],2]/standval*(default-input_range)
            pdmax[i] = bus[Load_index_list[i],2]/standval*(default+input_range)
        else:
            pdmin[i] = bus[Load_index_list[i],2]/standval*(default+input_range)
            pdmax[i] = bus[Load_index_list[i],2]/standval*(default-input_range)
            
    
    
    pgmax = np.zeros((1, mG),dtype='float32')
    pgmin = np.zeros((1, mG),dtype='float32')    


    index = 0
    for j in range(0,gen.shape[0],1):
        if gen[j,7] != 0:
            pgmax[0,index] = gen[j,8]/standval
            pgmin[0,index] = gen[j,9]/standval
            index = index + 1
    pgmax=pgmax.flatten()
    pgmin=pgmin.flatten()
    
    
    PTDF_pd=np.dot(PTDF_ori,Pd_matrix).astype(np.float32)
    PTDF_pg=np.dot(PTDF_ori,Pg_matrix).astype(np.float32)
    PTDF_pg_exceptslack=np.hstack((PTDF_pg[:,0:Slack_gen_order],PTDF_pg[:,Slack_gen_order+1:mG])).astype(np.float32)
    PTDF_pg_slack=PTDF_pg[:,Slack_gen_order:Slack_gen_order+1].flatten().astype(np.float32)
  
    ###warm start of solver as initial point
    f = open('pd30_test.txt')
    line = f.readline()
    data_list = []
    while line:
        num = list(map(float,line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    pd = np.array(data_list)
    pd = pd.reshape(len(pd),)
    pd_input = pd
	
	
    Input_initial=np.zeros((len(Load_index_list),),dtype='float32')
    for i in range(0,len(Load_index_list),1):
        Input_initial[i]= pd_input[Load_index_list[i]]
       

    Result=np.zeros((1,),dtype='float32')
    K=[]
    for increase in range(0,1):
        A=[];
        
        Lower_bound = -1e3-increase*increase*1e1*120
        Upper_bound = 1e3+increase*increase*1e1*120
        K.append(Lower_bound)
        m = Model()
    # m.setParam('NonConvex', 2)
        m.setParam('MIPGap', 0)
        m.setParam('IntFeasTol', 1e-09)
        m.setParam('FeasibilityTol', 1e-09)
        total_calibration(m,mpc)
        A=np.array(A)
        np.savetxt('pd30_'+str(increase)+'.txt', np.array(A), fmt="%s")
    np.savetxt("Result30.txt", np.array(Result), fmt="%s")
    np.savetxt("Bound.txt", np.array(K), fmt="%s")

    # mpc = loadcase('case118_rev_100.py')

    
 
    
    del m
    
    FINISH=time.time()-START
        

        