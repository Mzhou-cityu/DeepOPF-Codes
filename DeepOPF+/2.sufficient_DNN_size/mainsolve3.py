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

 
    
def total_branch15(m=None,mpc=None,initial=None):
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
        pd_temp315[i]=Input[i].value[0]
        
    pd_save15.append(pd_temp315)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL15.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver15=time.time()-start
    print(solver15)
    solver_time15.append(solver15)
    
    filenamesolver15= dir+'s-solvertime15.txt'
    fidsolver15=open(filenamesolver15,'at')
    fidsolver15.write("%9.4f;" % (solver15))
    fidsolver15.write('\n')
    fidsolver15.close()
    
    filenamesolution15= dir+'s-total30_all-15.txt'
    fidsolution15=open(filenamesolution15,'at')
    fidsolution15.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution15.write('\n')
    fidsolution15.close()
    
def total_branch16(m=None,mpc=None,initial=None):
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
        pd_temp316[i]=Input[i].value[0]
        
    pd_save16.append(pd_temp316)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL16.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver16=time.time()-start
    print(solver16)
    solver_time16.append(solver16)
    
    filenamesolver16= dir+'s-solvertime16.txt'
    fidsolver16=open(filenamesolver16,'at')
    fidsolver16.write("%9.4f;" % (solver16))
    fidsolver16.write('\n')
    fidsolver16.close()
    
    filenamesolution16= dir+'s-total30_all-16.txt'
    fidsolution16=open(filenamesolution16,'at')
    fidsolution16.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution16.write('\n')
    fidsolution16.close()
    
def total_branch17(m=None,mpc=None,initial=None):
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
        pd_temp317[i]=Input[i].value[0]
        
    pd_save17.append(pd_temp317)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL17.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver17=time.time()-start
    print(solver17)
    solver_time17.append(solver17)
    
    filenamesolver17= dir+'s-solvertime17.txt'
    fidsolver17=open(filenamesolver17,'at')
    fidsolver17.write("%9.4f;" % (solver17))
    fidsolver17.write('\n')
    fidsolver17.close()
    
    filenamesolution17= dir+'s-total30_all-17.txt'
    fidsolution17=open(filenamesolution17,'at')
    fidsolution17.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution17.write('\n')
    fidsolution17.close()
    
def total_branch18(m=None,mpc=None,initial=None):
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
        pd_temp318[i]=Input[i].value[0]
        
    pd_save18.append(pd_temp318)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL18.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver18=time.time()-start
    print(solver18)
    solver_time18.append(solver18)
    
    filenamesolver18= dir+'s-solvertime18.txt'
    fidsolver18=open(filenamesolver18,'at')
    fidsolver18.write("%9.4f;" % (solver18))
    fidsolver18.write('\n')
    fidsolver18.close()
    
    filenamesolution18= dir+'s-total30_all-18.txt'
    fidsolution18=open(filenamesolution18,'at')
    fidsolution18.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution18.write('\n')
    fidsolution18.close()
    
def total_branch19(m=None,mpc=None,initial=None):
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
        pd_temp319[i]=Input[i].value[0]
        
    pd_save19.append(pd_temp319)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL19.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver19=time.time()-start
    print(solver19)
    solver_time19.append(solver19)
    
    filenamesolver19= dir+'s-solvertime19.txt'
    fidsolver19=open(filenamesolver19,'at')
    fidsolver19.write("%9.4f;" % (solver19))
    fidsolver19.write('\n')
    fidsolver19.close()
    
    filenamesolution19= dir+'s-total30_all-19.txt'
    fidsolution19=open(filenamesolution19,'at')
    fidsolution19.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution19.write('\n')
    fidsolution19.close()
    
def total_branch20(m=None,mpc=None,initial=None):
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
        pd_temp320[i]=Input[i].value[0]
        
    pd_save20.append(pd_temp320)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL20.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver20=time.time()-start
    print(solver20)
    solver_time20.append(solver20)
    
    filenamesolver20= dir+'s-solvertime20.txt'
    fidsolver20=open(filenamesolver20,'at')
    fidsolver20.write("%9.4f;" % (solver20))
    fidsolver20.write('\n')
    fidsolver20.close()
    
    filenamesolution20= dir+'s-total30_all-20.txt'
    fidsolution20=open(filenamesolution20,'at')
    fidsolution20.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution20.write('\n')
    fidsolution20.close()
    
def total_branch21(m=None,mpc=None,initial=None):
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
        pd_temp321[i]=Input[i].value[0]
        
    pd_save21.append(pd_temp321)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL21.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver21=time.time()-start
    print(solver21)
    solver_time21.append(solver21)
    
    filenamesolver21= dir+'s-solvertime21.txt'
    fidsolver21=open(filenamesolver21,'at')
    fidsolver21.write("%9.4f;" % (solver21))
    fidsolver21.write('\n')
    fidsolver21.close()
    
    filenamesolution21= dir+'s-total30_all-21.txt'
    fidsolution21=open(filenamesolution21,'at')
    fidsolution21.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution21.write('\n')
    fidsolution21.close()
        
            
        



    
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
    
    
    
    
    
    
    

    
    
    FINAL15=[];
    FINAL16=[];
    FINAL17=[];
    FINAL18=[];
    FINAL19=[];
    FINAL20=[];
    FINAL21=[];
    
    FF=[];
    
    solver_time15=[];
    solver_time16=[];
    solver_time17=[];
    solver_time18=[];
    solver_time19=[];
    solver_time20=[];
    solver_time21=[];
    
    P_update=[];
    
    WEIGHT=[];
    BIAS=[];
    
    
    pd_save15=[];
    pd_save16=[];
    pd_save17=[];
    pd_save18=[];
    pd_save19=[];
    pd_save20=[];
    pd_save21=[];
    
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
        
    

    for IRound in range(0,400):
        
             #build model       
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
            model = Neuralnetwork3(20, 8, 5)
            model.load_state_dict(torch.load(dir+'s-TEST_'+str(IRound)+'.pth',map_location='cpu'))     
        
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
        
        pd_temp315= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp316= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp317= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp318= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp319= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp320= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp321= np.zeros((len(pd_temp0),),dtype='float32')
        
        # pd_temp2= np.zeros((len(pd_temp),),dtype='float32')
        # for j in range(0,1):
        
        load15=-pd_temp0*4/10
        load16=-pd_temp0*5/10
        load17=-pd_temp0*6/10
        load18=-pd_temp0*7/10
        load19=-pd_temp0*8/10
        load20=-pd_temp0*9/10
        load21=-pd_temp0
        
        
        
        
        
        dir = './S-SOLVE/'
        
        
        m15 = GEKKO(remote=False)
        m15.options.SOLVER = 1
        m15.options.MAX_ITER = 1000
        m16 = GEKKO(remote=False)
        m16.options.SOLVER = 1
        m16.options.MAX_ITER = 1000
        m17 = GEKKO(remote=False)
        m17.options.SOLVER = 1
        m17.options.MAX_ITER = 1000
        m18 = GEKKO(remote=False)
        m18.options.SOLVER = 1
        m18.options.MAX_ITER = 1000
        m19 = GEKKO(remote=False)
        m19.options.SOLVER = 1
        m19.options.MAX_ITER = 1000
        m20 = GEKKO(remote=False)
        m20.options.SOLVER = 1
        m20.options.MAX_ITER = 1000
        m21 = GEKKO(remote=False)
        m21.options.SOLVER = 1
        m21.options.MAX_ITER = 1000
        
        # mpc = loadcase('case118_rev_100.py')
        threads = []
        
        
        t15 = threading.Thread(target=total_branch15, args=(m15, mpc, load15))
        threads.append(t15)
        t15.start()
        t16 = threading.Thread(target=total_branch16, args=(m16, mpc, load16))
        threads.append(t16)
        t16.start()
        t17 = threading.Thread(target=total_branch17, args=(m17, mpc, load17))
        threads.append(t17)
        t17.start()
        t18 = threading.Thread(target=total_branch18, args=(m18, mpc, load18))
        threads.append(t18)
        t18.start()
        t19 = threading.Thread(target=total_branch19, args=(m19, mpc, load19))
        threads.append(t19)
        t19.start()
        t20 = threading.Thread(target=total_branch20, args=(m20, mpc, load20))
        threads.append(t20)
        t20.start()
        t21 = threading.Thread(target=total_branch21, args=(m21, mpc, load21))
        threads.append(t21)
        t21.start()
        
        
        for t in threads:
            t.join()
            
        
            
        if len(FINAL15)<IRound+1:
            FINAL15.append(-100)
            filenamesolver15= dir+'s-solvertime15.txt'
            fidsolver15=open(filenamesolver15,'at')
            fidsolver15.write("%9.4f;" % (1000000))
            fidsolver15.write('\n')
            fidsolver15.close()
    
            filenamesolution15= dir+'s-total30_all-15.txt'
            fidsolution15=open(filenamesolution15,'at')
            fidsolution15.write("%9.4f;" % (-100))
            fidsolution15.write('\n')
            fidsolution15.close()
            
            pd_save15.append(pd_temp0)
            
        if len(FINAL16)<IRound+1:
            FINAL16.append(-100)
            filenamesolver16= dir+'s-solvertime16.txt'
            fidsolver16=open(filenamesolver16,'at')
            fidsolver16.write("%9.4f;" % (1000000))
            fidsolver16.write('\n')
            fidsolver16.close()
    
            filenamesolution16= dir+'s-total30_all-16.txt'
            fidsolution16=open(filenamesolution16,'at')
            fidsolution16.write("%9.4f;" % (-100))
            fidsolution16.write('\n')
            fidsolution16.close()
            
            pd_save16.append(pd_temp0)
            
        if len(FINAL17)<IRound+1:
            FINAL17.append(-100)
            filenamesolver17= dir+'s-solvertime17.txt'
            fidsolver17=open(filenamesolver17,'at')
            fidsolver17.write("%9.4f;" % (1000000))
            fidsolver17.write('\n')
            fidsolver17.close()
    
            filenamesolution17= dir+'s-total30_all-17.txt'
            fidsolution17=open(filenamesolution17,'at')
            fidsolution17.write("%9.4f;" % (-100))
            fidsolution17.write('\n')
            fidsolution17.close()
            
            pd_save17.append(pd_temp0)
            
        if len(FINAL18)<IRound+1:
            FINAL18.append(-100)
            filenamesolver18= dir+'s-solvertime18.txt'
            fidsolver18=open(filenamesolver18,'at')
            fidsolver18.write("%9.4f;" % (1000000))
            fidsolver18.write('\n')
            fidsolver18.close()
    
            filenamesolution18= dir+'s-total30_all-18.txt'
            fidsolution18=open(filenamesolution18,'at')
            fidsolution18.write("%9.4f;" % (-100))
            fidsolution18.write('\n')
            fidsolution18.close()
            
            pd_save18.append(pd_temp0)
            
        if len(FINAL19)<IRound+1:
            FINAL19.append(-100)
            filenamesolver19= dir+'s-solvertime19.txt'
            fidsolver19=open(filenamesolver19,'at')
            fidsolver19.write("%9.4f;" % (1000000))
            fidsolver19.write('\n')
            fidsolver19.close()
    
            filenamesolution19= dir+'s-total30_all-19.txt'
            fidsolution19=open(filenamesolution19,'at')
            fidsolution19.write("%9.4f;" % (-100))
            fidsolution19.write('\n')
            fidsolution19.close()
            
            pd_save19.append(pd_temp0)
            
        if len(FINAL20)<IRound+1:
            FINAL20.append(-100)
            filenamesolver20= dir+'s-solvertime20.txt'
            fidsolver20=open(filenamesolver20,'at')
            fidsolver20.write("%9.4f;" % (1000000))
            fidsolver20.write('\n')
            fidsolver20.close()
    
            filenamesolution20= dir+'s-total30_all-20.txt'
            fidsolution20=open(filenamesolution20,'at')
            fidsolution20.write("%9.4f;" % (-100))
            fidsolution20.write('\n')
            fidsolution20.close()
            
            pd_save20.append(pd_temp0)
            
        if len(FINAL21)<IRound+1:
            FINAL21.append(-100)
            filenamesolver21= dir+'s-solvertime21.txt'
            fidsolver21=open(filenamesolver21,'at')
            fidsolver21.write("%9.4f;" % (1000000))
            fidsolver21.write('\n')
            fidsolver21.close()
    
            filenamesolution21= dir+'s-total30_all-21.txt'
            fidsolution21=open(filenamesolution21,'at')
            fidsolution21.write("%9.4f;" % (-100))
            fidsolution21.write('\n')
            fidsolution21.close()
            
            pd_save21.append(pd_temp0)
            
        
            
            
        
        
        P_D_normalization.append(pd_save15[IRound])
        P_D_normalization.append(pd_save16[IRound])
        P_D_normalization.append(pd_save17[IRound])
        P_D_normalization.append(pd_save18[IRound])
        P_D_normalization.append(pd_save19[IRound])
        P_D_normalization.append(pd_save20[IRound])
        P_D_normalization.append(pd_save21[IRound])
        
        
        dir='./S-TEST/'
        for i in range(0,7):
            np.savetxt(dir+'s-TEST_'+str(i+IRound*42+14)+'.txt', np.array(P_D_normalization[len(P_D_normalization)-i-1]), fmt="%s")
 

        
        del m15
        del m16
        del m17
        del m18
        del m19
        del m20
        del m21
        
        
        FINISH=time.time()-START
        
        dir = './S-SOLVE_TIME/'
        filenametime3= dir+'time3.txt'
        fidtime3=open(filenametime3,'at')
        fidtime3.write("%9.4f;" % (FINISH))
        fidtime3.write('\n')
        fidtime3.close()
        
        
        wait1=time.time()
        dir = './S-PTH/'
        for SOME in range(0,1000):
            if os.access(dir+'s-TEST_'+str(IRound+1)+'.pth', os.F_OK):
                time.sleep(0)
            else:
                time.sleep(1)
                
            
                
        dir = './S-WAIT/'
        filenamewait= dir+'WAIT3.txt'
        fidwait=open(filenamewait,'at')
        fidwait.write("%9.4f;" % (time.time()-wait1))
        fidwait.write('\n')
        fidwait.close()
                
        if os.access('FINISH.txt', os.F_OK):
            break
                
        
