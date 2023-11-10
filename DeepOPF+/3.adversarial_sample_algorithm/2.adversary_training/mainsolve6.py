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
from random import randint

default=1.15
input_range = 0.15
Lower_bound = -1e6
Upper_bound = 1e6

class Neuralnetwork3(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim):
        super(Neuralnetwork3, self).__init__()
        self.layer1 = nn.Linear(in_dim, 4*n_hidden)
        self.layer2 = nn.Linear(4*n_hidden, 2*n_hidden)
        self.layer3 = nn.Linear(2*n_hidden, n_hidden)
        self.layer4 = nn.Linear(n_hidden, out_dim)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        #x = F.relu(self.layer4(x))
        x = torch.sigmoid(self.layer4(x))
        return x
    
    

# neural network definition
class Neuralnetwork(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, 4*n_hidden)
        self.layer2 = nn.Linear(4*n_hidden, 2*n_hidden)
        self.layer3 = nn.Linear(2*n_hidden, n_hidden)
        self.layer5 = nn.Linear(n_hidden, out_dim)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        #x = F.relu(self.layer4(x))
        x = torch.sigmoid(self.layer5(x))
        return x
    

    
    
   #ipopt_solve
def total_branch36(m=None,mpc=None,initial=None):
    global Lower_bound
    global Upper_bound
    
    ###################################
    #variables for DC-OPF
    #Pd
    start=time.time()
    numberOfInput = layers[0]    
    #create variables
    Input = m.Array(m.Var,(len(Load_index_list),))        
    for i in range(0,len(Load_index_list),1):
        if bus[Load_index_list[i],2]/standval >=0:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)

            Input[i].value=initial[i]
        else:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            
            Input[i].value=initial[i]
    
    #number of variables for DNN    
    numberOfWeight = 0
    for i in range(0,len(layers)-1,1):
        numberOfWeight += layers[i]*layers[i+1]
    numberOfBias = sum(layers[1:len(layers)])
    numberOfStateVariable = numberOfBias
    numberOfNeuronOuput = numberOfStateVariable
    #create variables
    weight = weight_paras
    bias = bias_paras
    
    NeuronOutput = m.Array(m.Var,(numberOfNeuronOuput,))
    

    for i in range(0,layers[1]):

        m.Equation(NeuronOutput[i] ==m.max2(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])+bias[i],0))

    weight_index = layers[0]*layers[1]
    #for hidden layers
    for layer_index in range(2,len(layers)-1,1):
        previous_index = sum(layers[1:layer_index-1])
        current_index = sum(layers[1:layer_index])
        for i in range(0,layers[layer_index]):
            
            m.Equation(NeuronOutput[current_index+i] ==m.max2(sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])+bias[current_index+i],0))
            weight_index += layers[layer_index-1]

    #from hidden layer to output
    previous_index = sum(layers[1:len(layers)-2])
    current_index = sum(layers[1:len(layers)-1])   
    
    for i in range(0,layers[len(layers)-1]):
       
        m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] == sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])]))
       
        weight_index += layers[len(layers)-2]
   
    current_index = sum(layers[1:len(layers)-1])    
    #NN output
    NNOutput = m.Array(m.Var,(layers[len(layers)-1],))
    for i in range(0,layers[len(layers)-1]):
        NNOutput[i].lower = 0
        NNOutput[i].upper = 1
        #sigmoid
        m.Equation(NNOutput[i] == 1/(1+ m.exp(-NeuronOutput[current_index+i])))

    #Pg
    Pg = m.Array(m.Var,(len(Gen_index_list),))
    for i in range(0,len(Gen_index_list)):
          
        m.Equation(NNOutput[i]*(Power_Upbound_Gen[i]-Power_Lowbound_Gen[i])+Power_Lowbound_Gen[i]==Pg[i])

    #DC-PF equations
    # Create variables
    pg = m.Array(m.Var,(mL,))
    pd = m.Array(m.Var,(mL,))
    gen_index = 0
    load_index = 0
    for i in range(0,mL,1):
        if i in Load_index_list:
            m.Equation(pd[i] == Input[load_index]*((0.3*abs(bus[i,2])/standval)/12**0.5)+bus[i,2]/standval*(default))
            load_index += 1
        else:
            pd[i].lower = 0
            pd[i].upper = 0            
#            m.Equation(pd[i] == 0)
        if bus[i,1] == 3:
            pg[i].lower = Lower_bound
            pg[i].upper = Upper_bound
        elif bus[i,1] == 2 and i in Gen_index_list:
            # pg[i].lower = gen[gen_index,9]/standval
            # pg[i].upper = gen[gen_index,8]/standval
            m.Equation(pg[i] == Pg[gen_index])
            gen_index += 1                
        else:
            pg[i].lower = 0
            pg[i].upper = 0

    #DC-PF equations
    m.Equation(m.sum(pg) == m.sum(pd))
    
    #pg_feasible
    
    pg_feasible = m.Array(m.Var,(mL,))
    gen_index =0       
    for i in range(0,mL,1):
        if bus[i,1] == 3:
            pg_feasible[i].lower = Power_Lowbound_Gen_Slack
            pg_feasible[i].upper = Power_Upbound_Gen_Slack
        elif bus[i,1] == 2 and i in Gen_index_list:
            pg_feasible[i].lower = gen[gen_index,9]/standval
            pg_feasible[i].upper = gen[gen_index,8]/standval
            gen_index += 1
        else:
            pg_feasible[i].lower = 0
            pg_feasible[i].upper = 0
        
    
    m.Equation(m.sum(pg_feasible) == m.sum(pd))
    
   
    
    Branch_feasible=m.Array(m.Var,(mB,))
    for i in range(0, mB):
        m.Equation(Branch_feasible[i]==sum([PTDF_ori[i,j]*(pg_feasible[j]-pd[j]) for j in range(0,mL)]))
        Branch_feasible[i].lower=-Power_bound_Branch[i]
        Branch_feasible[i].upper=Power_bound_Branch[i]
        
    #branch flow violation
    # Branch_index = 1
    totalbranch_temp=m.Array(m.Var,(len(branch_list)+2,))
    for i in range(0,len(branch_list)):
        m.Equation(totalbranch_temp[i]==(abs(sum([PTDF_ori[branch_list[i],j]*(pg[j]-pd[j]) for j in range(0,mL)]))-Power_bound_Branch[branch_list[i]])/Power_bound_Branch[branch_list[i]])
    
    m.Equation(totalbranch_temp[len(branch_list)]==(sum(Pg)-sum(pd))/Power_Upbound_Gen_Slack)
    m.Equation(totalbranch_temp[len(branch_list)+1]==(sum(pd)-sum(Pg)-Power_Upbound_Gen_Slack)/Power_Upbound_Gen_Slack)
    
    compare_temp=m.Array(m.Var,(len(branch_list)+2,))
    m.Equation(compare_temp[0]==totalbranch_temp[0])
    for i in range(1, len(branch_list)+2):
        m.Equation(compare_temp[i]==m.max2(compare_temp[i-1],totalbranch_temp[i]))

     
    m.Obj(-compare_temp[len(branch_list)+1])
    
    # minimize objective
    m.solve()
    
        
    # for i in range(0,len(Load_index_list)):
    #     pd_temp[i]=Input[i].value[0]*((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)+bus[Load_index_list[i],2]/standval*(default)
    # for i in range(0,mL):
    #     pd_temp2[i]=pd[i].value[0]
    for i in range(0,len(Load_index_list)):
        pd_temp336[i]=Input[i].value[0]
        
    pd_save36.append(pd_temp336)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL36.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver36=time.time()-start
    print(solver36)
    solver_time36.append(solver36)
    
    filenamesolver36= dir+'s-solvertime36.txt'
    fidsolver36=open(filenamesolver36,'at')
    fidsolver36.write("%9.4f;" % (solver36))
    fidsolver36.write('\n')
    fidsolver36.close()
    
    filenamesolution36= dir+'s-total30_all-36.txt'
    fidsolution36=open(filenamesolution36,'at')
    fidsolution36.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution36.write('\n')
    fidsolution36.close()
    
def total_branch37(m=None,mpc=None,initial=None):
    global Lower_bound
    global Upper_bound
    
    ###################################
    #variables for DC-OPF
    #Pd
    start=time.time()
    numberOfInput = layers[0]    
    #create variables
    Input = m.Array(m.Var,(len(Load_index_list),))        
    for i in range(0,len(Load_index_list),1):
        if bus[Load_index_list[i],2]/standval >=0:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)

            Input[i].value=initial[i]
        else:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            
            Input[i].value=initial[i]
    
    #number of variables for DNN    
    numberOfWeight = 0
    for i in range(0,len(layers)-1,1):
        numberOfWeight += layers[i]*layers[i+1]
    numberOfBias = sum(layers[1:len(layers)])
    numberOfStateVariable = numberOfBias
    numberOfNeuronOuput = numberOfStateVariable
    #create variables
    weight = weight_paras
    bias = bias_paras
    
    NeuronOutput = m.Array(m.Var,(numberOfNeuronOuput,))
    

    for i in range(0,layers[1]):

        m.Equation(NeuronOutput[i] ==m.max2(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])+bias[i],0))

    weight_index = layers[0]*layers[1]
    #for hidden layers
    for layer_index in range(2,len(layers)-1,1):
        previous_index = sum(layers[1:layer_index-1])
        current_index = sum(layers[1:layer_index])
        for i in range(0,layers[layer_index]):
            
            m.Equation(NeuronOutput[current_index+i] ==m.max2(sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])+bias[current_index+i],0))
            weight_index += layers[layer_index-1]

    #from hidden layer to output
    previous_index = sum(layers[1:len(layers)-2])
    current_index = sum(layers[1:len(layers)-1])   
    
    for i in range(0,layers[len(layers)-1]):
       
        m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] == sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])]))
       
        weight_index += layers[len(layers)-2]
   
    current_index = sum(layers[1:len(layers)-1])    
    #NN output
    NNOutput = m.Array(m.Var,(layers[len(layers)-1],))
    for i in range(0,layers[len(layers)-1]):
        NNOutput[i].lower = 0
        NNOutput[i].upper = 1
        #sigmoid
        m.Equation(NNOutput[i] == 1/(1+ m.exp(-NeuronOutput[current_index+i])))

    #Pg
    Pg = m.Array(m.Var,(len(Gen_index_list),))
    for i in range(0,len(Gen_index_list)):
          
        m.Equation(NNOutput[i]*(Power_Upbound_Gen[i]-Power_Lowbound_Gen[i])+Power_Lowbound_Gen[i]==Pg[i])

    #DC-PF equations
    # Create variables
    pg = m.Array(m.Var,(mL,))
    pd = m.Array(m.Var,(mL,))
    gen_index = 0
    load_index = 0
    for i in range(0,mL,1):
        if i in Load_index_list:
            m.Equation(pd[i] == Input[load_index]*((0.3*abs(bus[i,2])/standval)/12**0.5)+bus[i,2]/standval*(default))
            load_index += 1
        else:
            pd[i].lower = 0
            pd[i].upper = 0            
#            m.Equation(pd[i] == 0)
        if bus[i,1] == 3:
            pg[i].lower = Lower_bound
            pg[i].upper = Upper_bound
        elif bus[i,1] == 2 and i in Gen_index_list:
            # pg[i].lower = gen[gen_index,9]/standval
            # pg[i].upper = gen[gen_index,8]/standval
            m.Equation(pg[i] == Pg[gen_index])
            gen_index += 1                
        else:
            pg[i].lower = 0
            pg[i].upper = 0

    #DC-PF equations
    m.Equation(m.sum(pg) == m.sum(pd))
    
    #pg_feasible
    
    pg_feasible = m.Array(m.Var,(mL,))
    gen_index =0       
    for i in range(0,mL,1):
        if bus[i,1] == 3:
            pg_feasible[i].lower = Power_Lowbound_Gen_Slack
            pg_feasible[i].upper = Power_Upbound_Gen_Slack
        elif bus[i,1] == 2 and i in Gen_index_list:
            pg_feasible[i].lower = gen[gen_index,9]/standval
            pg_feasible[i].upper = gen[gen_index,8]/standval
            gen_index += 1
        else:
            pg_feasible[i].lower = 0
            pg_feasible[i].upper = 0
        
    
    m.Equation(m.sum(pg_feasible) == m.sum(pd))
    
   
    
    Branch_feasible=m.Array(m.Var,(mB,))
    for i in range(0, mB):
        m.Equation(Branch_feasible[i]==sum([PTDF_ori[i,j]*(pg_feasible[j]-pd[j]) for j in range(0,mL)]))
        Branch_feasible[i].lower=-Power_bound_Branch[i]
        Branch_feasible[i].upper=Power_bound_Branch[i]
        
    #branch flow violation
    # Branch_index = 1
    totalbranch_temp=m.Array(m.Var,(len(branch_list)+2,))
    for i in range(0,len(branch_list)):
        m.Equation(totalbranch_temp[i]==(abs(sum([PTDF_ori[branch_list[i],j]*(pg[j]-pd[j]) for j in range(0,mL)]))-Power_bound_Branch[branch_list[i]])/Power_bound_Branch[branch_list[i]])
    
    m.Equation(totalbranch_temp[len(branch_list)]==(sum(Pg)-sum(pd))/Power_Upbound_Gen_Slack)
    m.Equation(totalbranch_temp[len(branch_list)+1]==(sum(pd)-sum(Pg)-Power_Upbound_Gen_Slack)/Power_Upbound_Gen_Slack)
    
    compare_temp=m.Array(m.Var,(len(branch_list)+2,))
    m.Equation(compare_temp[0]==totalbranch_temp[0])
    for i in range(1, len(branch_list)+2):
        m.Equation(compare_temp[i]==m.max2(compare_temp[i-1],totalbranch_temp[i]))

     
    m.Obj(-compare_temp[len(branch_list)+1])
    
    # minimize objective
    m.solve()
    
        
    # for i in range(0,len(Load_index_list)):
    #     pd_temp[i]=Input[i].value[0]*((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)+bus[Load_index_list[i],2]/standval*(default)
    # for i in range(0,mL):
    #     pd_temp2[i]=pd[i].value[0]
    for i in range(0,len(Load_index_list)):
        pd_temp337[i]=Input[i].value[0]
        
    pd_save37.append(pd_temp337)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL37.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver37=time.time()-start
    print(solver37)
    solver_time37.append(solver37)
    
    filenamesolver37= dir+'s-solvertime37.txt'
    fidsolver37=open(filenamesolver37,'at')
    fidsolver37.write("%9.4f;" % (solver37))
    fidsolver37.write('\n')
    fidsolver37.close()
    
    filenamesolution37= dir+'s-total30_all-37.txt'
    fidsolution37=open(filenamesolution37,'at')
    fidsolution37.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution37.write('\n')
    fidsolution37.close()
    
def total_branch38(m=None,mpc=None,initial=None):
    global Lower_bound
    global Upper_bound
    
    ###################################
    #variables for DC-OPF
    #Pd
    start=time.time()
    numberOfInput = layers[0]    
    #create variables
    Input = m.Array(m.Var,(len(Load_index_list),))        
    for i in range(0,len(Load_index_list),1):
        if bus[Load_index_list[i],2]/standval >=0:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)

            Input[i].value=initial[i]
        else:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            
            Input[i].value=initial[i]
    
    #number of variables for DNN    
    numberOfWeight = 0
    for i in range(0,len(layers)-1,1):
        numberOfWeight += layers[i]*layers[i+1]
    numberOfBias = sum(layers[1:len(layers)])
    numberOfStateVariable = numberOfBias
    numberOfNeuronOuput = numberOfStateVariable
    #create variables
    weight = weight_paras
    bias = bias_paras
    
    NeuronOutput = m.Array(m.Var,(numberOfNeuronOuput,))
    

    for i in range(0,layers[1]):

        m.Equation(NeuronOutput[i] ==m.max2(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])+bias[i],0))

    weight_index = layers[0]*layers[1]
    #for hidden layers
    for layer_index in range(2,len(layers)-1,1):
        previous_index = sum(layers[1:layer_index-1])
        current_index = sum(layers[1:layer_index])
        for i in range(0,layers[layer_index]):
            
            m.Equation(NeuronOutput[current_index+i] ==m.max2(sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])+bias[current_index+i],0))
            weight_index += layers[layer_index-1]

    #from hidden layer to output
    previous_index = sum(layers[1:len(layers)-2])
    current_index = sum(layers[1:len(layers)-1])   
    
    for i in range(0,layers[len(layers)-1]):
       
        m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] == sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])]))
       
        weight_index += layers[len(layers)-2]
   
    current_index = sum(layers[1:len(layers)-1])    
    #NN output
    NNOutput = m.Array(m.Var,(layers[len(layers)-1],))
    for i in range(0,layers[len(layers)-1]):
        NNOutput[i].lower = 0
        NNOutput[i].upper = 1
        #sigmoid
        m.Equation(NNOutput[i] == 1/(1+ m.exp(-NeuronOutput[current_index+i])))

    #Pg
    Pg = m.Array(m.Var,(len(Gen_index_list),))
    for i in range(0,len(Gen_index_list)):
          
        m.Equation(NNOutput[i]*(Power_Upbound_Gen[i]-Power_Lowbound_Gen[i])+Power_Lowbound_Gen[i]==Pg[i])

    #DC-PF equations
    # Create variables
    pg = m.Array(m.Var,(mL,))
    pd = m.Array(m.Var,(mL,))
    gen_index = 0
    load_index = 0
    for i in range(0,mL,1):
        if i in Load_index_list:
            m.Equation(pd[i] == Input[load_index]*((0.3*abs(bus[i,2])/standval)/12**0.5)+bus[i,2]/standval*(default))
            load_index += 1
        else:
            pd[i].lower = 0
            pd[i].upper = 0            
#            m.Equation(pd[i] == 0)
        if bus[i,1] == 3:
            pg[i].lower = Lower_bound
            pg[i].upper = Upper_bound
        elif bus[i,1] == 2 and i in Gen_index_list:
            # pg[i].lower = gen[gen_index,9]/standval
            # pg[i].upper = gen[gen_index,8]/standval
            m.Equation(pg[i] == Pg[gen_index])
            gen_index += 1                
        else:
            pg[i].lower = 0
            pg[i].upper = 0

    #DC-PF equations
    m.Equation(m.sum(pg) == m.sum(pd))
    
    #pg_feasible
    
    pg_feasible = m.Array(m.Var,(mL,))
    gen_index =0       
    for i in range(0,mL,1):
        if bus[i,1] == 3:
            pg_feasible[i].lower = Power_Lowbound_Gen_Slack
            pg_feasible[i].upper = Power_Upbound_Gen_Slack
        elif bus[i,1] == 2 and i in Gen_index_list:
            pg_feasible[i].lower = gen[gen_index,9]/standval
            pg_feasible[i].upper = gen[gen_index,8]/standval
            gen_index += 1
        else:
            pg_feasible[i].lower = 0
            pg_feasible[i].upper = 0
        
    
    m.Equation(m.sum(pg_feasible) == m.sum(pd))
    
   
    
    Branch_feasible=m.Array(m.Var,(mB,))
    for i in range(0, mB):
        m.Equation(Branch_feasible[i]==sum([PTDF_ori[i,j]*(pg_feasible[j]-pd[j]) for j in range(0,mL)]))
        Branch_feasible[i].lower=-Power_bound_Branch[i]
        Branch_feasible[i].upper=Power_bound_Branch[i]
        
    #branch flow violation
    # Branch_index = 1
    totalbranch_temp=m.Array(m.Var,(len(branch_list)+2,))
    for i in range(0,len(branch_list)):
        m.Equation(totalbranch_temp[i]==(abs(sum([PTDF_ori[branch_list[i],j]*(pg[j]-pd[j]) for j in range(0,mL)]))-Power_bound_Branch[branch_list[i]])/Power_bound_Branch[branch_list[i]])
    
    m.Equation(totalbranch_temp[len(branch_list)]==(sum(Pg)-sum(pd))/Power_Upbound_Gen_Slack)
    m.Equation(totalbranch_temp[len(branch_list)+1]==(sum(pd)-sum(Pg)-Power_Upbound_Gen_Slack)/Power_Upbound_Gen_Slack)
    
    compare_temp=m.Array(m.Var,(len(branch_list)+2,))
    m.Equation(compare_temp[0]==totalbranch_temp[0])
    for i in range(1, len(branch_list)+2):
        m.Equation(compare_temp[i]==m.max2(compare_temp[i-1],totalbranch_temp[i]))

     
    m.Obj(-compare_temp[len(branch_list)+1])
    
    # minimize objective
    m.solve()
    
        
    # for i in range(0,len(Load_index_list)):
    #     pd_temp[i]=Input[i].value[0]*((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)+bus[Load_index_list[i],2]/standval*(default)
    # for i in range(0,mL):
    #     pd_temp2[i]=pd[i].value[0]
    for i in range(0,len(Load_index_list)):
        pd_temp338[i]=Input[i].value[0]
        
    pd_save38.append(pd_temp338)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL38.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver38=time.time()-start
    print(solver38)
    solver_time38.append(solver38)
    
    filenamesolver38= dir+'s-solvertime38.txt'
    fidsolver38=open(filenamesolver38,'at')
    fidsolver38.write("%9.4f;" % (solver38))
    fidsolver38.write('\n')
    fidsolver38.close()
    
    filenamesolution38= dir+'s-total30_all-38.txt'
    fidsolution38=open(filenamesolution38,'at')
    fidsolution38.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution38.write('\n')
    fidsolution38.close()
    
def total_branch39(m=None,mpc=None,initial=None):
    global Lower_bound
    global Upper_bound
    
    ###################################
    #variables for DC-OPF
    #Pd
    start=time.time()
    numberOfInput = layers[0]    
    #create variables
    Input = m.Array(m.Var,(len(Load_index_list),))        
    for i in range(0,len(Load_index_list),1):
        if bus[Load_index_list[i],2]/standval >=0:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)

            Input[i].value=initial[i]
        else:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            
            Input[i].value=initial[i]
    
    #number of variables for DNN    
    numberOfWeight = 0
    for i in range(0,len(layers)-1,1):
        numberOfWeight += layers[i]*layers[i+1]
    numberOfBias = sum(layers[1:len(layers)])
    numberOfStateVariable = numberOfBias
    numberOfNeuronOuput = numberOfStateVariable
    #create variables
    weight = weight_paras
    bias = bias_paras
    
    NeuronOutput = m.Array(m.Var,(numberOfNeuronOuput,))
    

    for i in range(0,layers[1]):

        m.Equation(NeuronOutput[i] ==m.max2(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])+bias[i],0))

    weight_index = layers[0]*layers[1]
    #for hidden layers
    for layer_index in range(2,len(layers)-1,1):
        previous_index = sum(layers[1:layer_index-1])
        current_index = sum(layers[1:layer_index])
        for i in range(0,layers[layer_index]):
            
            m.Equation(NeuronOutput[current_index+i] ==m.max2(sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])+bias[current_index+i],0))
            weight_index += layers[layer_index-1]

    #from hidden layer to output
    previous_index = sum(layers[1:len(layers)-2])
    current_index = sum(layers[1:len(layers)-1])   
    
    for i in range(0,layers[len(layers)-1]):
       
        m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] == sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])]))
       
        weight_index += layers[len(layers)-2]
   
    current_index = sum(layers[1:len(layers)-1])    
    #NN output
    NNOutput = m.Array(m.Var,(layers[len(layers)-1],))
    for i in range(0,layers[len(layers)-1]):
        NNOutput[i].lower = 0
        NNOutput[i].upper = 1
        #sigmoid
        m.Equation(NNOutput[i] == 1/(1+ m.exp(-NeuronOutput[current_index+i])))

    #Pg
    Pg = m.Array(m.Var,(len(Gen_index_list),))
    for i in range(0,len(Gen_index_list)):
          
        m.Equation(NNOutput[i]*(Power_Upbound_Gen[i]-Power_Lowbound_Gen[i])+Power_Lowbound_Gen[i]==Pg[i])

    #DC-PF equations
    # Create variables
    pg = m.Array(m.Var,(mL,))
    pd = m.Array(m.Var,(mL,))
    gen_index = 0
    load_index = 0
    for i in range(0,mL,1):
        if i in Load_index_list:
            m.Equation(pd[i] == Input[load_index]*((0.3*abs(bus[i,2])/standval)/12**0.5)+bus[i,2]/standval*(default))
            load_index += 1
        else:
            pd[i].lower = 0
            pd[i].upper = 0            
#            m.Equation(pd[i] == 0)
        if bus[i,1] == 3:
            pg[i].lower = Lower_bound
            pg[i].upper = Upper_bound
        elif bus[i,1] == 2 and i in Gen_index_list:
            # pg[i].lower = gen[gen_index,9]/standval
            # pg[i].upper = gen[gen_index,8]/standval
            m.Equation(pg[i] == Pg[gen_index])
            gen_index += 1                
        else:
            pg[i].lower = 0
            pg[i].upper = 0

    #DC-PF equations
    m.Equation(m.sum(pg) == m.sum(pd))
    
    #pg_feasible
    
    pg_feasible = m.Array(m.Var,(mL,))
    gen_index =0       
    for i in range(0,mL,1):
        if bus[i,1] == 3:
            pg_feasible[i].lower = Power_Lowbound_Gen_Slack
            pg_feasible[i].upper = Power_Upbound_Gen_Slack
        elif bus[i,1] == 2 and i in Gen_index_list:
            pg_feasible[i].lower = gen[gen_index,9]/standval
            pg_feasible[i].upper = gen[gen_index,8]/standval
            gen_index += 1
        else:
            pg_feasible[i].lower = 0
            pg_feasible[i].upper = 0
        
    
    m.Equation(m.sum(pg_feasible) == m.sum(pd))
    
   
    
    Branch_feasible=m.Array(m.Var,(mB,))
    for i in range(0, mB):
        m.Equation(Branch_feasible[i]==sum([PTDF_ori[i,j]*(pg_feasible[j]-pd[j]) for j in range(0,mL)]))
        Branch_feasible[i].lower=-Power_bound_Branch[i]
        Branch_feasible[i].upper=Power_bound_Branch[i]
        
    #branch flow violation
    # Branch_index = 1
    totalbranch_temp=m.Array(m.Var,(len(branch_list)+2,))
    for i in range(0,len(branch_list)):
        m.Equation(totalbranch_temp[i]==(abs(sum([PTDF_ori[branch_list[i],j]*(pg[j]-pd[j]) for j in range(0,mL)]))-Power_bound_Branch[branch_list[i]])/Power_bound_Branch[branch_list[i]])
    
    m.Equation(totalbranch_temp[len(branch_list)]==(sum(Pg)-sum(pd))/Power_Upbound_Gen_Slack)
    m.Equation(totalbranch_temp[len(branch_list)+1]==(sum(pd)-sum(Pg)-Power_Upbound_Gen_Slack)/Power_Upbound_Gen_Slack)
    
    compare_temp=m.Array(m.Var,(len(branch_list)+2,))
    m.Equation(compare_temp[0]==totalbranch_temp[0])
    for i in range(1, len(branch_list)+2):
        m.Equation(compare_temp[i]==m.max2(compare_temp[i-1],totalbranch_temp[i]))

     
    m.Obj(-compare_temp[len(branch_list)+1])
    
    # minimize objective
    m.solve()
    
        
    # for i in range(0,len(Load_index_list)):
    #     pd_temp[i]=Input[i].value[0]*((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)+bus[Load_index_list[i],2]/standval*(default)
    # for i in range(0,mL):
    #     pd_temp2[i]=pd[i].value[0]
    for i in range(0,len(Load_index_list)):
        pd_temp339[i]=Input[i].value[0]
        
    pd_save39.append(pd_temp339)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL39.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver39=time.time()-start
    print(solver39)
    solver_time39.append(solver39)
    
    filenamesolver39= dir+'s-solvertime39.txt'
    fidsolver39=open(filenamesolver39,'at')
    fidsolver39.write("%9.4f;" % (solver39))
    fidsolver39.write('\n')
    fidsolver39.close()
    
    filenamesolution39= dir+'s-total30_all-39.txt'
    fidsolution39=open(filenamesolution39,'at')
    fidsolution39.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution39.write('\n')
    fidsolution39.close()
    
def total_branch40(m=None,mpc=None,initial=None):
    global Lower_bound
    global Upper_bound
    
    ###################################
    #variables for DC-OPF
    #Pd
    start=time.time()
    numberOfInput = layers[0]    
    #create variables
    Input = m.Array(m.Var,(len(Load_index_list),))        
    for i in range(0,len(Load_index_list),1):
        if bus[Load_index_list[i],2]/standval >=0:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)

            Input[i].value=initial[i]
        else:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            
            Input[i].value=initial[i]
    
    #number of variables for DNN    
    numberOfWeight = 0
    for i in range(0,len(layers)-1,1):
        numberOfWeight += layers[i]*layers[i+1]
    numberOfBias = sum(layers[1:len(layers)])
    numberOfStateVariable = numberOfBias
    numberOfNeuronOuput = numberOfStateVariable
    #create variables
    weight = weight_paras
    bias = bias_paras
    
    NeuronOutput = m.Array(m.Var,(numberOfNeuronOuput,))
    

    for i in range(0,layers[1]):

        m.Equation(NeuronOutput[i] ==m.max2(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])+bias[i],0))

    weight_index = layers[0]*layers[1]
    #for hidden layers
    for layer_index in range(2,len(layers)-1,1):
        previous_index = sum(layers[1:layer_index-1])
        current_index = sum(layers[1:layer_index])
        for i in range(0,layers[layer_index]):
            
            m.Equation(NeuronOutput[current_index+i] ==m.max2(sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])+bias[current_index+i],0))
            weight_index += layers[layer_index-1]

    #from hidden layer to output
    previous_index = sum(layers[1:len(layers)-2])
    current_index = sum(layers[1:len(layers)-1])   
    
    for i in range(0,layers[len(layers)-1]):
       
        m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] == sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])]))
       
        weight_index += layers[len(layers)-2]
   
    current_index = sum(layers[1:len(layers)-1])    
    #NN output
    NNOutput = m.Array(m.Var,(layers[len(layers)-1],))
    for i in range(0,layers[len(layers)-1]):
        NNOutput[i].lower = 0
        NNOutput[i].upper = 1
        #sigmoid
        m.Equation(NNOutput[i] == 1/(1+ m.exp(-NeuronOutput[current_index+i])))

    #Pg
    Pg = m.Array(m.Var,(len(Gen_index_list),))
    for i in range(0,len(Gen_index_list)):
          
        m.Equation(NNOutput[i]*(Power_Upbound_Gen[i]-Power_Lowbound_Gen[i])+Power_Lowbound_Gen[i]==Pg[i])

    #DC-PF equations
    # Create variables
    pg = m.Array(m.Var,(mL,))
    pd = m.Array(m.Var,(mL,))
    gen_index = 0
    load_index = 0
    for i in range(0,mL,1):
        if i in Load_index_list:
            m.Equation(pd[i] == Input[load_index]*((0.3*abs(bus[i,2])/standval)/12**0.5)+bus[i,2]/standval*(default))
            load_index += 1
        else:
            pd[i].lower = 0
            pd[i].upper = 0            
#            m.Equation(pd[i] == 0)
        if bus[i,1] == 3:
            pg[i].lower = Lower_bound
            pg[i].upper = Upper_bound
        elif bus[i,1] == 2 and i in Gen_index_list:
            # pg[i].lower = gen[gen_index,9]/standval
            # pg[i].upper = gen[gen_index,8]/standval
            m.Equation(pg[i] == Pg[gen_index])
            gen_index += 1                
        else:
            pg[i].lower = 0
            pg[i].upper = 0

    #DC-PF equations
    m.Equation(m.sum(pg) == m.sum(pd))
    
    #pg_feasible
    
    pg_feasible = m.Array(m.Var,(mL,))
    gen_index =0       
    for i in range(0,mL,1):
        if bus[i,1] == 3:
            pg_feasible[i].lower = Power_Lowbound_Gen_Slack
            pg_feasible[i].upper = Power_Upbound_Gen_Slack
        elif bus[i,1] == 2 and i in Gen_index_list:
            pg_feasible[i].lower = gen[gen_index,9]/standval
            pg_feasible[i].upper = gen[gen_index,8]/standval
            gen_index += 1
        else:
            pg_feasible[i].lower = 0
            pg_feasible[i].upper = 0
        
    
    m.Equation(m.sum(pg_feasible) == m.sum(pd))
    
   
    
    Branch_feasible=m.Array(m.Var,(mB,))
    for i in range(0, mB):
        m.Equation(Branch_feasible[i]==sum([PTDF_ori[i,j]*(pg_feasible[j]-pd[j]) for j in range(0,mL)]))
        Branch_feasible[i].lower=-Power_bound_Branch[i]
        Branch_feasible[i].upper=Power_bound_Branch[i]
        
    #branch flow violation
    # Branch_index = 1
    totalbranch_temp=m.Array(m.Var,(len(branch_list)+2,))
    for i in range(0,len(branch_list)):
        m.Equation(totalbranch_temp[i]==(abs(sum([PTDF_ori[branch_list[i],j]*(pg[j]-pd[j]) for j in range(0,mL)]))-Power_bound_Branch[branch_list[i]])/Power_bound_Branch[branch_list[i]])
    
    m.Equation(totalbranch_temp[len(branch_list)]==(sum(Pg)-sum(pd))/Power_Upbound_Gen_Slack)
    m.Equation(totalbranch_temp[len(branch_list)+1]==(sum(pd)-sum(Pg)-Power_Upbound_Gen_Slack)/Power_Upbound_Gen_Slack)
    
    compare_temp=m.Array(m.Var,(len(branch_list)+2,))
    m.Equation(compare_temp[0]==totalbranch_temp[0])
    for i in range(1, len(branch_list)+2):
        m.Equation(compare_temp[i]==m.max2(compare_temp[i-1],totalbranch_temp[i]))

     
    m.Obj(-compare_temp[len(branch_list)+1])
    
    # minimize objective
    m.solve()
    
        
    # for i in range(0,len(Load_index_list)):
    #     pd_temp[i]=Input[i].value[0]*((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)+bus[Load_index_list[i],2]/standval*(default)
    # for i in range(0,mL):
    #     pd_temp2[i]=pd[i].value[0]
    for i in range(0,len(Load_index_list)):
        pd_temp340[i]=Input[i].value[0]
        
    pd_save40.append(pd_temp340)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL40.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver40=time.time()-start
    print(solver40)
    solver_time40.append(solver40)
    
    filenamesolver40= dir+'s-solvertime40.txt'
    fidsolver40=open(filenamesolver40,'at')
    fidsolver40.write("%9.4f;" % (solver40))
    fidsolver40.write('\n')
    fidsolver40.close()
    
    filenamesolution40= dir+'s-total30_all-40.txt'
    fidsolution40=open(filenamesolution40,'at')
    fidsolution40.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution40.write('\n')
    fidsolution40.close()
    
def total_branch41(m=None,mpc=None,initial=None):
    global Lower_bound
    global Upper_bound
    
    ###################################
    #variables for DC-OPF
    #Pd
    start=time.time()
    numberOfInput = layers[0]    
    #create variables
    Input = m.Array(m.Var,(len(Load_index_list),))        
    for i in range(0,len(Load_index_list),1):
        if bus[Load_index_list[i],2]/standval >=0:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)

            Input[i].value=initial[i]
        else:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            
            Input[i].value=initial[i]
    
    #number of variables for DNN    
    numberOfWeight = 0
    for i in range(0,len(layers)-1,1):
        numberOfWeight += layers[i]*layers[i+1]
    numberOfBias = sum(layers[1:len(layers)])
    numberOfStateVariable = numberOfBias
    numberOfNeuronOuput = numberOfStateVariable
    #create variables
    weight = weight_paras
    bias = bias_paras
    
    NeuronOutput = m.Array(m.Var,(numberOfNeuronOuput,))
    

    for i in range(0,layers[1]):

        m.Equation(NeuronOutput[i] ==m.max2(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])+bias[i],0))

    weight_index = layers[0]*layers[1]
    #for hidden layers
    for layer_index in range(2,len(layers)-1,1):
        previous_index = sum(layers[1:layer_index-1])
        current_index = sum(layers[1:layer_index])
        for i in range(0,layers[layer_index]):
            
            m.Equation(NeuronOutput[current_index+i] ==m.max2(sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])+bias[current_index+i],0))
            weight_index += layers[layer_index-1]

    #from hidden layer to output
    previous_index = sum(layers[1:len(layers)-2])
    current_index = sum(layers[1:len(layers)-1])   
    
    for i in range(0,layers[len(layers)-1]):
       
        m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] == sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])]))
       
        weight_index += layers[len(layers)-2]
   
    current_index = sum(layers[1:len(layers)-1])    
    #NN output
    NNOutput = m.Array(m.Var,(layers[len(layers)-1],))
    for i in range(0,layers[len(layers)-1]):
        NNOutput[i].lower = 0
        NNOutput[i].upper = 1
        #sigmoid
        m.Equation(NNOutput[i] == 1/(1+ m.exp(-NeuronOutput[current_index+i])))

    #Pg
    Pg = m.Array(m.Var,(len(Gen_index_list),))
    for i in range(0,len(Gen_index_list)):
          
        m.Equation(NNOutput[i]*(Power_Upbound_Gen[i]-Power_Lowbound_Gen[i])+Power_Lowbound_Gen[i]==Pg[i])

    #DC-PF equations
    # Create variables
    pg = m.Array(m.Var,(mL,))
    pd = m.Array(m.Var,(mL,))
    gen_index = 0
    load_index = 0
    for i in range(0,mL,1):
        if i in Load_index_list:
            m.Equation(pd[i] == Input[load_index]*((0.3*abs(bus[i,2])/standval)/12**0.5)+bus[i,2]/standval*(default))
            load_index += 1
        else:
            pd[i].lower = 0
            pd[i].upper = 0            
#            m.Equation(pd[i] == 0)
        if bus[i,1] == 3:
            pg[i].lower = Lower_bound
            pg[i].upper = Upper_bound
        elif bus[i,1] == 2 and i in Gen_index_list:
            # pg[i].lower = gen[gen_index,9]/standval
            # pg[i].upper = gen[gen_index,8]/standval
            m.Equation(pg[i] == Pg[gen_index])
            gen_index += 1                
        else:
            pg[i].lower = 0
            pg[i].upper = 0

    #DC-PF equations
    m.Equation(m.sum(pg) == m.sum(pd))
    
    #pg_feasible
    
    pg_feasible = m.Array(m.Var,(mL,))
    gen_index =0       
    for i in range(0,mL,1):
        if bus[i,1] == 3:
            pg_feasible[i].lower = Power_Lowbound_Gen_Slack
            pg_feasible[i].upper = Power_Upbound_Gen_Slack
        elif bus[i,1] == 2 and i in Gen_index_list:
            pg_feasible[i].lower = gen[gen_index,9]/standval
            pg_feasible[i].upper = gen[gen_index,8]/standval
            gen_index += 1
        else:
            pg_feasible[i].lower = 0
            pg_feasible[i].upper = 0
        
    
    m.Equation(m.sum(pg_feasible) == m.sum(pd))
    
   
    
    Branch_feasible=m.Array(m.Var,(mB,))
    for i in range(0, mB):
        m.Equation(Branch_feasible[i]==sum([PTDF_ori[i,j]*(pg_feasible[j]-pd[j]) for j in range(0,mL)]))
        Branch_feasible[i].lower=-Power_bound_Branch[i]
        Branch_feasible[i].upper=Power_bound_Branch[i]
        
    #branch flow violation
    # Branch_index = 1
    totalbranch_temp=m.Array(m.Var,(len(branch_list)+2,))
    for i in range(0,len(branch_list)):
        m.Equation(totalbranch_temp[i]==(abs(sum([PTDF_ori[branch_list[i],j]*(pg[j]-pd[j]) for j in range(0,mL)]))-Power_bound_Branch[branch_list[i]])/Power_bound_Branch[branch_list[i]])
    
    m.Equation(totalbranch_temp[len(branch_list)]==(sum(Pg)-sum(pd))/Power_Upbound_Gen_Slack)
    m.Equation(totalbranch_temp[len(branch_list)+1]==(sum(pd)-sum(Pg)-Power_Upbound_Gen_Slack)/Power_Upbound_Gen_Slack)
    
    compare_temp=m.Array(m.Var,(len(branch_list)+2,))
    m.Equation(compare_temp[0]==totalbranch_temp[0])
    for i in range(1, len(branch_list)+2):
        m.Equation(compare_temp[i]==m.max2(compare_temp[i-1],totalbranch_temp[i]))

     
    m.Obj(-compare_temp[len(branch_list)+1])
    
    # minimize objective
    m.solve()
    
        
    # for i in range(0,len(Load_index_list)):
    #     pd_temp[i]=Input[i].value[0]*((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)+bus[Load_index_list[i],2]/standval*(default)
    # for i in range(0,mL):
    #     pd_temp2[i]=pd[i].value[0]
    for i in range(0,len(Load_index_list)):
        pd_temp341[i]=Input[i].value[0]
        
    pd_save41.append(pd_temp341)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL41.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver41=time.time()-start
    print(solver41)
    solver_time41.append(solver41)
    
    filenamesolver41= dir+'s-solvertime41.txt'
    fidsolver41=open(filenamesolver41,'at')
    fidsolver41.write("%9.4f;" % (solver41))
    fidsolver41.write('\n')
    fidsolver41.close()
    
    filenamesolution41= dir+'s-total30_all-41.txt'
    fidsolution41=open(filenamesolution41,'at')
    fidsolution41.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution41.write('\n')
    fidsolution41.close()
    
def total_branch42(m=None,mpc=None,initial=None):
    global Lower_bound
    global Upper_bound
    
    ###################################
    #variables for DC-OPF
    #Pd
    start=time.time()
    numberOfInput = layers[0]    
    #create variables
    Input = m.Array(m.Var,(len(Load_index_list),))        
    for i in range(0,len(Load_index_list),1):
        if bus[Load_index_list[i],2]/standval >=0:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)

            Input[i].value=initial[i]
        else:
            Input[i].lower = (bus[Load_index_list[i],2]/standval*(default-input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            Input[i].upper = (bus[Load_index_list[i],2]/standval*(default+input_range)-bus[Load_index_list[i],2]/standval*(default))/((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)
            
            Input[i].value=initial[i]
    
    #number of variables for DNN    
    numberOfWeight = 0
    for i in range(0,len(layers)-1,1):
        numberOfWeight += layers[i]*layers[i+1]
    numberOfBias = sum(layers[1:len(layers)])
    numberOfStateVariable = numberOfBias
    numberOfNeuronOuput = numberOfStateVariable
    #create variables
    weight = weight_paras
    bias = bias_paras
    
    NeuronOutput = m.Array(m.Var,(numberOfNeuronOuput,))
    

    for i in range(0,layers[1]):

        m.Equation(NeuronOutput[i] ==m.max2(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])+bias[i],0))

    weight_index = layers[0]*layers[1]
    #for hidden layers
    for layer_index in range(2,len(layers)-1,1):
        previous_index = sum(layers[1:layer_index-1])
        current_index = sum(layers[1:layer_index])
        for i in range(0,layers[layer_index]):
            
            m.Equation(NeuronOutput[current_index+i] ==m.max2(sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])+bias[current_index+i],0))
            weight_index += layers[layer_index-1]

    #from hidden layer to output
    previous_index = sum(layers[1:len(layers)-2])
    current_index = sum(layers[1:len(layers)-1])   
    
    for i in range(0,layers[len(layers)-1]):
       
        m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] == sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])]))
       
        weight_index += layers[len(layers)-2]
   
    current_index = sum(layers[1:len(layers)-1])    
    #NN output
    NNOutput = m.Array(m.Var,(layers[len(layers)-1],))
    for i in range(0,layers[len(layers)-1]):
        NNOutput[i].lower = 0
        NNOutput[i].upper = 1
        #sigmoid
        m.Equation(NNOutput[i] == 1/(1+ m.exp(-NeuronOutput[current_index+i])))

    #Pg
    Pg = m.Array(m.Var,(len(Gen_index_list),))
    for i in range(0,len(Gen_index_list)):
          
        m.Equation(NNOutput[i]*(Power_Upbound_Gen[i]-Power_Lowbound_Gen[i])+Power_Lowbound_Gen[i]==Pg[i])

    #DC-PF equations
    # Create variables
    pg = m.Array(m.Var,(mL,))
    pd = m.Array(m.Var,(mL,))
    gen_index = 0
    load_index = 0
    for i in range(0,mL,1):
        if i in Load_index_list:
            m.Equation(pd[i] == Input[load_index]*((0.3*abs(bus[i,2])/standval)/12**0.5)+bus[i,2]/standval*(default))
            load_index += 1
        else:
            pd[i].lower = 0
            pd[i].upper = 0            
#            m.Equation(pd[i] == 0)
        if bus[i,1] == 3:
            pg[i].lower = Lower_bound
            pg[i].upper = Upper_bound
        elif bus[i,1] == 2 and i in Gen_index_list:
            # pg[i].lower = gen[gen_index,9]/standval
            # pg[i].upper = gen[gen_index,8]/standval
            m.Equation(pg[i] == Pg[gen_index])
            gen_index += 1                
        else:
            pg[i].lower = 0
            pg[i].upper = 0

    #DC-PF equations
    m.Equation(m.sum(pg) == m.sum(pd))
    
    #pg_feasible
    
    pg_feasible = m.Array(m.Var,(mL,))
    gen_index =0       
    for i in range(0,mL,1):
        if bus[i,1] == 3:
            pg_feasible[i].lower = Power_Lowbound_Gen_Slack
            pg_feasible[i].upper = Power_Upbound_Gen_Slack
        elif bus[i,1] == 2 and i in Gen_index_list:
            pg_feasible[i].lower = gen[gen_index,9]/standval
            pg_feasible[i].upper = gen[gen_index,8]/standval
            gen_index += 1
        else:
            pg_feasible[i].lower = 0
            pg_feasible[i].upper = 0
        
    
    m.Equation(m.sum(pg_feasible) == m.sum(pd))
    
   
    
    Branch_feasible=m.Array(m.Var,(mB,))
    for i in range(0, mB):
        m.Equation(Branch_feasible[i]==sum([PTDF_ori[i,j]*(pg_feasible[j]-pd[j]) for j in range(0,mL)]))
        Branch_feasible[i].lower=-Power_bound_Branch[i]
        Branch_feasible[i].upper=Power_bound_Branch[i]
        
    #branch flow violation
    # Branch_index = 1
    totalbranch_temp=m.Array(m.Var,(len(branch_list)+2,))
    for i in range(0,len(branch_list)):
        m.Equation(totalbranch_temp[i]==(abs(sum([PTDF_ori[branch_list[i],j]*(pg[j]-pd[j]) for j in range(0,mL)]))-Power_bound_Branch[branch_list[i]])/Power_bound_Branch[branch_list[i]])
    
    m.Equation(totalbranch_temp[len(branch_list)]==(sum(Pg)-sum(pd))/Power_Upbound_Gen_Slack)
    m.Equation(totalbranch_temp[len(branch_list)+1]==(sum(pd)-sum(Pg)-Power_Upbound_Gen_Slack)/Power_Upbound_Gen_Slack)
    
    compare_temp=m.Array(m.Var,(len(branch_list)+2,))
    m.Equation(compare_temp[0]==totalbranch_temp[0])
    for i in range(1, len(branch_list)+2):
        m.Equation(compare_temp[i]==m.max2(compare_temp[i-1],totalbranch_temp[i]))

     
    m.Obj(-compare_temp[len(branch_list)+1])
    
    # minimize objective
    m.solve()
    
        
    # for i in range(0,len(Load_index_list)):
    #     pd_temp[i]=Input[i].value[0]*((0.3*bus[Load_index_list[i],2]/standval)/12**0.5)+bus[Load_index_list[i],2]/standval*(default)
    # for i in range(0,mL):
    #     pd_temp2[i]=pd[i].value[0]
    for i in range(0,len(Load_index_list)):
        pd_temp342[i]=Input[i].value[0]
        
    pd_save42.append(pd_temp342)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL42.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver42=time.time()-start
    print(solver42)
    solver_time42.append(solver42)
    
    filenamesolver42= dir+'s-solvertime42.txt'
    fidsolver42=open(filenamesolver42,'at')
    fidsolver42.write("%9.4f;" % (solver42))
    fidsolver42.write('\n')
    fidsolver42.close()
    
    filenamesolution42= dir+'s-total30_all-42.txt'
    fidsolution42=open(filenamesolution42,'at')
    fidsolution42.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution42.write('\n')
    fidsolution42.close()
        
            
        



    
if __name__ == '__main__':
    # Initialize Model

    epsilon=0.0001
    alpha=0.001
    branch_list=[0, 9, 28, 29, 30, 31, 32, 34]
    
    MEAN = np.array([0.24955, 0.0276 , 0.0874 , 0.2622 , 0.345  , 0.0667 , 0.1288 ,
       0.0713 , 0.0943 , 0.04025, 0.1035 , 0.0368 , 0.10925, 0.0253 ,
       0.20125, 0.0368 , 0.10005, 0.04025, 0.0276 , 0.1219 ]).astype(np.float32)
    
    STD = np.array([0.01879275, 0.00207846, 0.00658179, 0.01974538, 0.02598076,
       0.00502295, 0.00969948, 0.00536936, 0.00710141, 0.00303109,
       0.00779423, 0.00277128, 0.00822724, 0.00190526, 0.01515544,
       0.00277128, 0.00753442, 0.00303109, 0.00207846, 0.00917987]).astype(np.float32)
    
    branch_capacity=np.array([1.3221 , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
       0.     , 0.     , 0.32544, 0.     , 0.     , 0.     , 0.     ,
       0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
       0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
       0.32544, 0.16272, 0.16272, 0.16272, 0.16272, 0.     , 0.16272,
       0.     , 0.     , 0.     , 0.     , 0.     , 0.     ]);
    
    
    
    
    
    
    

    
    
    
    
    FINAL36=[];
    FINAL37=[];
    FINAL38=[];
    FINAL39=[];
    FINAL40=[];
    FINAL41=[];
    FINAL42=[];
    FF=[];
    
    solver_time36=[];
    solver_time37=[];
    solver_time38=[];
    solver_time39=[];
    solver_time40=[];
    solver_time41=[];
    solver_time42=[];
    P_update=[];
    
    WEIGHT=[];
    BIAS=[];
    
    
    
    
    pd_save36=[];
    pd_save37=[];
    pd_save38=[];
    pd_save39=[];
    pd_save40=[];
    pd_save41=[];
    pd_save42=[];
    pd_final=[];
    PD_FINAL=[];
    indexk=[];
    
    P_D_test=[];
    P_G_test=[];
    dir = './test_data/'
    for index in range(0,1,1):
        filename= dir+'Pre_DC_test_case30-sampling.txt'
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
                    P_D_test.append(float(odom[i])/100)
            else:
                odom  =line.split(';')  # supply
                for i in range(len(odom)):
                    Gen_data = odom[i].split(':')
                    # ind = np.where(mpc['gen'][:,0] == int(Gen_data[0]))[0][0]
                    # if mpc['gen'][ind,7] != 0:
                    P_G_test.append(float(Gen_data[1])/100)                
            j = j + 1
        fid.close()
    ################################################################
    del data
    Input_data = np.array(P_D_test, dtype=np.float32).reshape(-1, 20)
    P_D_test_normolization = (Input_data - MEAN) / (STD+1e-8)
    P_D_normalization=P_D_test_normolization.tolist()
    pd_temp0=np.ones(20)*1.7320508;
    for i in range(0,21):
        P_D_normalization.append(pd_temp0*(0.1*i-1))
        
    PREVIOUS=[];
    dir = './TEST/'
    for SOME in range(0,400*40):
        if os.access(dir+'TEST_'+str(SOME)+'.txt', os.F_OK):
            f = open(dir+'TEST_'+str(SOME)+'.txt')
            line = f.readline()
            data_list = []
            while line:
                num = list(map(float,line.split()))
                data_list.append(num)
                line = f.readline()
            f.close()
            pd = np.array(data_list)
            pd = pd.reshape(len(pd),)
            PREVIOUS.append(pd)               
            
    print(len(PREVIOUS))
        
    
    Point_index=0
    indexinitial=0   
    for IRound in range(0,400):
        increase=0
        dir = './TEST/'
        for SOME in range(Point_index,400*42):
            if os.access(dir+'TEST_'+str(SOME)+'.txt', os.F_OK):
                increase=increase+1
                f = open(dir+'TEST_'+str(SOME)+'.txt')
                line = f.readline()
                data_list = []
                while line:
                    num = list(map(float,line.split()))
                    data_list.append(num)
                    line = f.readline()
                f.close()
                pd = np.array(data_list)
                pd = pd.reshape(len(pd),)
                P_D_normalization.append(pd)               
                Point_index=Point_index+1
                
        #build model       
        if IRound==0:
            dir ='./S-PTH/'
            model = Neuralnetwork(20, 8, 5)
            model.load_state_dict(torch.load(dir+'PreDCOPF_case30_0.93_dnn.pth',map_location='cpu'))     
        
            params = list(model.parameters())
            
            layers = []
            layers.append(20)
            for i in params:
                if len(i.size()) == 2:
                    neuron_number = i.size()[0]
                    layers.append(neuron_number)
            
            #number of variables for DNN    
            numberOfWeight = 0
            for i in range(0,len(layers)-1,1):
                numberOfWeight += layers[i]*layers[i+1]
            numberOfBias = sum(layers[1:len(layers)])
            numberOfStateVariable = numberOfBias
            numberOfNeuronOuput = numberOfStateVariable
            
            #read NN's parameters(weights and bias)
            weight_paras = np.zeros((numberOfWeight,),dtype='float32')
            bias_paras = np.zeros((numberOfBias,),dtype='float32')
            weight_index = 0
            bias_index = 0
            for i in params:
                i_np = i.detach().numpy()
                if len(list(i.size()))==2:
                    dim1 = list(i.size())[0]
                    dim2 = list(i.size())[1]
                    for j in range(0,dim1):
                        for k in range(0,dim2):
                            weight_paras[weight_index] = i_np[j,k]
                            weight_index += 1
                elif len(list(i.size()))==1:
                    dim1 = list(i.size())[0]
                    for j in range(0,dim1):
                        bias_paras[bias_index] = i_np[j]
                        bias_index += 1
            WEIGHT.append(weight_paras)
            BIAS.append(bias_paras)
        else:
            dir ='./S-PTH/'
            model = Neuralnetwork(20, 8, 5)
            model.load_state_dict(torch.load(dir+'PreDCOPF_case30_0.93_dnn_'+str(IRound)+'.pth',map_location='cpu'))     
        
            params = list(model.parameters())
            
            layers = []
            layers.append(20)
            for i in params:
                if len(i.size()) == 2:
                    neuron_number = i.size()[0]
                    layers.append(neuron_number)
            
            #number of variables for DNN    
            numberOfWeight = 0
            for i in range(0,len(layers)-1,1):
                numberOfWeight += layers[i]*layers[i+1]
            numberOfBias = sum(layers[1:len(layers)])
            numberOfStateVariable = numberOfBias
            numberOfNeuronOuput = numberOfStateVariable
            
            #read NN's parameters(weights and bias)
            weight_paras = np.zeros((numberOfWeight,),dtype='float32')
            bias_paras = np.zeros((numberOfBias,),dtype='float32')
            weight_index = 0
            bias_index = 0
            for i in params:
                i_np = i.detach().numpy()
                if len(list(i.size()))==2:
                    dim1 = list(i.size())[0]
                    dim2 = list(i.size())[1]
                    for j in range(0,dim1):
                        for k in range(0,dim2):
                            weight_paras[weight_index] = i_np[j,k]
                            weight_index += 1
                elif len(list(i.size()))==1:
                    dim1 = list(i.size())[0]
                    for j in range(0,dim1):
                        bias_paras[bias_index] = i_np[j]
                        bias_index += 1
            WEIGHT.append(weight_paras)
            BIAS.append(bias_paras)
                        
                        
                        
                               
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
      
        weight_paras=WEIGHT[IRound]
        bias_paras=BIAS[IRound]
            
        print(IRound)
        
        pd_temp0=np.ones(20)*1.7320508;
        pd_temp= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp2= np.zeros((30,),dtype='float32')
        
        pd_temp336= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp337= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp338= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp339= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp340= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp341= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp342= np.zeros((len(pd_temp0),),dtype='float32')
        # pd_temp2= np.zeros((len(pd_temp),),dtype='float32')
        # for j in range(0,1):
        
        
             
        if random.randint(0, 1)==1 and indexinitial<=len(PREVIOUS)-1:
            load36=PREVIOUS[indexinitial]
            indexinitial=indexinitial+1
        else:
            load36=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
            
        if random.randint(0, 1)==1 and indexinitial<=len(PREVIOUS)-1:
            load37=PREVIOUS[indexinitial]
            indexinitial=indexinitial+1
        else:
            load37=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
            
        if random.randint(0, 1)==1 and indexinitial<=len(PREVIOUS)-1:
            load38=PREVIOUS[indexinitial]
            indexinitial=indexinitial+1
        else:
            load38=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
        # load30=np.random.uniform(-1,1,len(load1))*pd_temp0
        # load31=np.random.uniform(-1,1,len(load1))*pd_temp0
        
        # load32=2*(random.random()-0.5)*pd_temp0
        # load33=2*(random.random()-0.5)*pd_temp0
        if increase>=1:
            load39=P_D_normalization[len(P_D_normalization)-1]
        else:
            load39=2*(random.random()-0.5)*pd_temp0
            
        if increase>=2:
            load40=P_D_normalization[len(P_D_normalization)-2]
        else:
            load40=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0

        # load40=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
        load41=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
        load42=2*(random.random()-0.5)*pd_temp0
        
        print(increase)
        print('***********')
        
        
        
        dir = './S-SOLVE/'
        
        
        
        
        m36 = GEKKO(remote=False)
        m36.options.SOLVER = 1
        m36.options.MAX_ITER = 1000
        m37 = GEKKO(remote=False)
        m37.options.SOLVER = 1
        m37.options.MAX_ITER = 1000
        m38 = GEKKO(remote=False)
        m38.options.SOLVER = 1
        m38.options.MAX_ITER = 1000
        m39 = GEKKO(remote=False)
        m39.options.SOLVER = 1
        m39.options.MAX_ITER = 1000
        m40 = GEKKO(remote=False)
        m40.options.SOLVER = 1
        m40.options.MAX_ITER = 1000
        m41 = GEKKO(remote=False)
        m41.options.SOLVER = 1
        m41.options.MAX_ITER = 1000
        m42 = GEKKO(remote=False)
        m42.options.SOLVER = 1
        m42.options.MAX_ITER = 1000
        # mpc = loadcase('case118_rev_100.py')
        threads = []
        
        
        
        
        t36 = threading.Thread(target=total_branch36, args=(m36, mpc, load36))
        threads.append(t36)
        t36.start()
        t37 = threading.Thread(target=total_branch37, args=(m37, mpc, load37))
        threads.append(t37)
        t37.start()
        t38 = threading.Thread(target=total_branch38, args=(m38, mpc, load38))
        threads.append(t38)
        t38.start()
        t39 = threading.Thread(target=total_branch39, args=(m39, mpc, load39))
        threads.append(t39)
        t39.start()
        t40 = threading.Thread(target=total_branch40, args=(m40, mpc, load40))
        threads.append(t40)
        t40.start()
        t41 = threading.Thread(target=total_branch41, args=(m41, mpc, load41))
        threads.append(t41)
        t41.start()
        t42 = threading.Thread(target=total_branch42, args=(m42, mpc, load42))
        threads.append(t42)
        t42.start()
        
        for t in threads:
            t.join()
            
        
            
        
        
            
        if len(FINAL36)<IRound+1:
            FINAL36.append(-100)
            filenamesolver36= dir+'s-solvertime36.txt'
            fidsolver36=open(filenamesolver36,'at')
            fidsolver36.write("%9.4f;" % (1000000))
            fidsolver36.write('\n')
            fidsolver36.close()
    
            filenamesolution36= dir+'s-total30_all-36.txt'
            fidsolution36=open(filenamesolution36,'at')
            fidsolution36.write("%9.4f;" % (-100))
            fidsolution36.write('\n')
            fidsolution36.close()
            
            pd_save36.append(pd_temp0)
            
        if len(FINAL37)<IRound+1:
            FINAL37.append(-100)
            filenamesolver37= dir+'s-solvertime37.txt'
            fidsolver37=open(filenamesolver37,'at')
            fidsolver37.write("%9.4f;" % (1000000))
            fidsolver37.write('\n')
            fidsolver37.close()
    
            filenamesolution37= dir+'s-total30_all-37.txt'
            fidsolution37=open(filenamesolution37,'at')
            fidsolution37.write("%9.4f;" % (-100))
            fidsolution37.write('\n')
            fidsolution37.close()
            
            pd_save37.append(pd_temp0)
            
        if len(FINAL38)<IRound+1:
            FINAL38.append(-100)
            filenamesolver38= dir+'s-solvertime38.txt'
            fidsolver38=open(filenamesolver38,'at')
            fidsolver38.write("%9.4f;" % (1000000))
            fidsolver38.write('\n')
            fidsolver38.close()
    
            filenamesolution38= dir+'s-total30_all-38.txt'
            fidsolution38=open(filenamesolution38,'at')
            fidsolution38.write("%9.4f;" % (-100))
            fidsolution38.write('\n')
            fidsolution38.close()
            
            pd_save38.append(pd_temp0)
            
        if len(FINAL39)<IRound+1:
            FINAL39.append(-100)
            filenamesolver39= dir+'s-solvertime39.txt'
            fidsolver39=open(filenamesolver39,'at')
            fidsolver39.write("%9.4f;" % (1000000))
            fidsolver39.write('\n')
            fidsolver39.close()
    
            filenamesolution39= dir+'s-total30_all-39.txt'
            fidsolution39=open(filenamesolution39,'at')
            fidsolution39.write("%9.4f;" % (-100))
            fidsolution39.write('\n')
            fidsolution39.close()
            
            pd_save39.append(pd_temp0)
            
        if len(FINAL40)<IRound+1:
            FINAL40.append(-100)
            filenamesolver40= dir+'s-solvertime40.txt'
            fidsolver40=open(filenamesolver40,'at')
            fidsolver40.write("%9.4f;" % (1000000))
            fidsolver40.write('\n')
            fidsolver40.close()
    
            filenamesolution40= dir+'s-total30_all-40.txt'
            fidsolution40=open(filenamesolution40,'at')
            fidsolution40.write("%9.4f;" % (-100))
            fidsolution40.write('\n')
            fidsolution40.close()
            
            pd_save40.append(pd_temp0)
            
        if len(FINAL41)<IRound+1:
            FINAL41.append(-100)
            filenamesolver41= dir+'s-solvertime41.txt'
            fidsolver41=open(filenamesolver41,'at')
            fidsolver41.write("%9.4f;" % (1000000))
            fidsolver41.write('\n')
            fidsolver41.close()
    
            filenamesolution41= dir+'s-total30_all-41.txt'
            fidsolution41=open(filenamesolution41,'at')
            fidsolution41.write("%9.4f;" % (-100))
            fidsolution41.write('\n')
            fidsolution41.close()
            
            pd_save41.append(pd_temp0)
            
        if len(FINAL42)<IRound+1:
            FINAL42.append(-100)
            filenamesolver42= dir+'s-solvertime42.txt'
            fidsolver42=open(filenamesolver42,'at')
            fidsolver42.write("%9.4f;" % (1000000))
            fidsolver42.write('\n')
            fidsolver42.close()
    
            filenamesolution42= dir+'s-total30_all-42.txt'
            fidsolution42=open(filenamesolution42,'at')
            fidsolution42.write("%9.4f;" % (-100))
            fidsolution42.write('\n')
            fidsolution42.close()
            
            pd_save42.append(pd_temp0)
            
            
        
        
        P_D_normalization.append(pd_save36[IRound])
        P_D_normalization.append(pd_save37[IRound])
        P_D_normalization.append(pd_save38[IRound])
        P_D_normalization.append(pd_save39[IRound])
        P_D_normalization.append(pd_save40[IRound])
        P_D_normalization.append(pd_save41[IRound])
        P_D_normalization.append(pd_save42[IRound])
        
        dir='./S-TEST/'
        for i in range(0,7):
            np.savetxt(dir+'s-TEST_'+str(i+IRound*42+35)+'.txt', np.array(P_D_normalization[len(P_D_normalization)-i-1]), fmt="%s")
 

        
        
        
        del m36
        del m37
        del m38
        del m39
        del m40
        del m41
        del m42
        
        FINISH=time.time()-START
        
        dir = './S-SOLVE_TIME/'
        filenametime6= dir+'time6.txt'
        fidtime6=open(filenametime6,'at')
        fidtime6.write("%9.4f;" % (FINISH))
        fidtime6.write('\n')
        fidtime6.close()
        
        
        wait1=time.time()
        dir = './S-PTH/'
        for SOME in range(0,1000):            
            if IRound==0:
                if os.access(dir+'PreDCOPF_case30_0.93_dnn_'+str(IRound+1)+'.pth', os.F_OK) and os.access('testtrainfinish.txt', os.F_OK):
                    time.sleep(0)
                else:
                    time.sleep(100)
            else:     
                filename= 'testtrainfinish.txt'
                fid=open(filename,'r')
                data = fid.readlines()
                
                if os.access(dir+'PreDCOPF_case30_0.93_dnn_'+str(IRound+1)+'.pth', os.F_OK) and len(data)==IRound+1:
                    time.sleep(0)
                else:
                    time.sleep(100)
                
        dir = './S-WAIT/'
        filenamewait= dir+'WAIT6.txt'
        fidwait=open(filenamewait,'at')
        fidwait.write("%9.4f;" % (time.time()-wait1))
        fidwait.write('\n')
        fidwait.close()
                
        if os.access('FINISH.txt', os.F_OK):
            break
                
        
