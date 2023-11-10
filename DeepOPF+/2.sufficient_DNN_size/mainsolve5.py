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
    
def total_branch29(m=None,mpc=None,initial=None):
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
        pd_temp329[i]=Input[i].value[0]
        
    pd_save29.append(pd_temp329)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL29.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver29=time.time()-start
    print(solver29)
    solver_time29.append(solver29)
    
    filenamesolver29= dir+'s-solvertime29.txt'
    fidsolver29=open(filenamesolver29,'at')
    fidsolver29.write("%9.4f;" % (solver29))
    fidsolver29.write('\n')
    fidsolver29.close()
    
    filenamesolution29= dir+'s-total30_all-29.txt'
    fidsolution29=open(filenamesolution29,'at')
    fidsolution29.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution29.write('\n')
    fidsolution29.close()
    
def total_branch30(m=None,mpc=None,initial=None):
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
        pd_temp330[i]=Input[i].value[0]
        
    pd_save30.append(pd_temp330)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL30.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver30=time.time()-start
    print(solver30)
    solver_time30.append(solver30)
    
    filenamesolver30= dir+'s-solvertime30.txt'
    fidsolver30=open(filenamesolver30,'at')
    fidsolver30.write("%9.4f;" % (solver30))
    fidsolver30.write('\n')
    fidsolver30.close()
    
    filenamesolution30= dir+'s-total30_all-30.txt'
    fidsolution30=open(filenamesolution30,'at')
    fidsolution30.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution30.write('\n')
    fidsolution30.close()
        
    
def total_branch31(m=None,mpc=None,initial=None):
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
        pd_temp331[i]=Input[i].value[0]
        
    pd_save31.append(pd_temp331)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL31.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver31=time.time()-start
    print(solver31)
    solver_time31.append(solver31)
    
    filenamesolver31= dir+'s-solvertime31.txt'
    fidsolver31=open(filenamesolver31,'at')
    fidsolver31.write("%9.4f;" % (solver31))
    fidsolver31.write('\n')
    fidsolver31.close()
    
    filenamesolution31= dir+'s-total30_all-31.txt'
    fidsolution31=open(filenamesolution31,'at')
    fidsolution31.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution31.write('\n')
    fidsolution31.close()
    
def total_branch32(m=None,mpc=None,initial=None):
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
        pd_temp332[i]=Input[i].value[0]
        
    pd_save32.append(pd_temp332)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL32.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver32=time.time()-start
    print(solver32)
    solver_time32.append(solver32)
    
    filenamesolver32= dir+'s-solvertime32.txt'
    fidsolver32=open(filenamesolver32,'at')
    fidsolver32.write("%9.4f;" % (solver32))
    fidsolver32.write('\n')
    fidsolver32.close()
    
    filenamesolution32= dir+'s-total30_all-32.txt'
    fidsolution32=open(filenamesolution32,'at')
    fidsolution32.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution32.write('\n')
    fidsolution32.close()
    
def total_branch33(m=None,mpc=None,initial=None):
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
        pd_temp333[i]=Input[i].value[0]
        
    pd_save33.append(pd_temp333)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL33.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver33=time.time()-start
    print(solver33)
    solver_time33.append(solver33)
    
    filenamesolver33= dir+'s-solvertime33.txt'
    fidsolver33=open(filenamesolver33,'at')
    fidsolver33.write("%9.4f;" % (solver33))
    fidsolver33.write('\n')
    fidsolver33.close()
    
    filenamesolution33= dir+'s-total30_all-33.txt'
    fidsolution33=open(filenamesolution33,'at')
    fidsolution33.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution33.write('\n')
    fidsolution33.close()
    
def total_branch34(m=None,mpc=None,initial=None):
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
        pd_temp334[i]=Input[i].value[0]
        
    pd_save34.append(pd_temp334)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL34.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver34=time.time()-start
    print(solver34)
    solver_time34.append(solver34)
    
    filenamesolver34= dir+'s-solvertime34.txt'
    fidsolver34=open(filenamesolver34,'at')
    fidsolver34.write("%9.4f;" % (solver34))
    fidsolver34.write('\n')
    fidsolver34.close()
    
    filenamesolution34= dir+'s-total30_all-34.txt'
    fidsolution34=open(filenamesolution34,'at')
    fidsolution34.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution34.write('\n')
    fidsolution34.close()
    
def total_branch35(m=None,mpc=None,initial=None):
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
        pd_temp335[i]=Input[i].value[0]
        
    pd_save35.append(pd_temp335)

        
    # print(Slack_Pg_violation[0].value)
    print(compare_temp[len(branch_list)+1].value[0])
    
    FINAL35.append(compare_temp[len(branch_list)+1].value[0])
    # Final.append(compare_temp[len(branch_list)-1].value[0])
    solver35=time.time()-start
    print(solver35)
    solver_time35.append(solver35)
    
    filenamesolver35= dir+'s-solvertime35.txt'
    fidsolver35=open(filenamesolver35,'at')
    fidsolver35.write("%9.4f;" % (solver35))
    fidsolver35.write('\n')
    fidsolver35.close()
    
    filenamesolution35= dir+'s-total30_all-35.txt'
    fidsolution35=open(filenamesolution35,'at')
    fidsolution35.write("%9.4f;" % (compare_temp[len(branch_list)+1].value[0]))
    fidsolution35.write('\n')
    fidsolution35.close()
    




    
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
    
    
    
    
    
    
    

    
    
    
    
    FINAL29=[];
    FINAL30=[];
    FINAL31=[];
    FINAL32=[];
    FINAL33=[];
    FINAL34=[];
    FINAL35=[];
    
    FF=[];
    
    
    solver_time29=[];
    solver_time30=[];
    solver_time31=[];
    solver_time32=[];
    solver_time33=[];
    solver_time34=[];
    solver_time35=[];
    
    P_update=[];
    
    WEIGHT=[];
    BIAS=[];
    
    
    
    
    pd_save29=[];
    pd_save30=[];
    pd_save31=[];
    pd_save32=[];
    pd_save33=[];
    pd_save34=[];
    pd_save35=[];
    
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
        
        
        
        pd_temp329= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp330= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp331= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp332= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp333= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp334= np.zeros((len(pd_temp0),),dtype='float32')
        pd_temp335= np.zeros((len(pd_temp0),),dtype='float32')
        
        # pd_temp2= np.zeros((len(pd_temp),),dtype='float32')
        # for j in range(0,1):
        
        
        
        
        
        
            
        if IRound<=3:
            load28=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
            load29=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
        else:
            dir = './S-pdfinal/'                  
            f = open(dir+'pd_final_'+str(IRound-4)+'.txt')
            line = f.readline()
            data_list = []
            while line:
                num = list(map(float,line.split()))
                data_list.append(num)
                line = f.readline()
            f.close()
            pd = np.array(data_list)
            load28 = pd.reshape(len(pd),)
            
            dir = './LPDFINAL/'                  
            f = open(dir+'PD_FINAL_'+str(IRound-4)+'.txt')
            line = f.readline()
            data_list = []
            while line:
                num = list(map(float,line.split()))
                data_list.append(num)
                line = f.readline()
            f.close()
            pd = np.array(data_list)
            loadtest = pd.reshape(len(pd),)
             
            if (load28==loadtest).all():
                load29=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
            else:
                load29=loadtest
            
        if IRound<=4:
            load30=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
            load31=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
        else:
            dir = './S-pdfinal/'                  
            f = open(dir+'pd_final_'+str(IRound-5)+'.txt')
            line = f.readline()
            data_list = []
            while line:
                num = list(map(float,line.split()))
                data_list.append(num)
                line = f.readline()
            f.close()
            pd = np.array(data_list)
            load30 = pd.reshape(len(pd),)
            
            dir = './LPDFINAL/'                  
            f = open(dir+'PD_FINAL_'+str(IRound-5)+'.txt')
            line = f.readline()
            data_list = []
            while line:
                num = list(map(float,line.split()))
                data_list.append(num)
                line = f.readline()
            f.close()
            pd = np.array(data_list)
            loadtest = pd.reshape(len(pd),)
             
            if (load30==loadtest).all():
                load31=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
            else:
                load31=loadtest
            
        if IRound<=5:
            load32=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
            load33=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
        else:
            dir = './S-pdfinal/'                  
            f = open(dir+'pd_final_'+str(IRound-6)+'.txt')
            line = f.readline()
            data_list = []
            while line:
                num = list(map(float,line.split()))
                data_list.append(num)
                line = f.readline()
            f.close()
            pd = np.array(data_list)
            load32 = pd.reshape(len(pd),)
            
            dir = './LPDFINAL/'                  
            f = open(dir+'PD_FINAL_'+str(IRound-6)+'.txt')
            line = f.readline()
            data_list = []
            while line:
                num = list(map(float,line.split()))
                data_list.append(num)
                line = f.readline()
            f.close()
            pd = np.array(data_list)
            loadtest = pd.reshape(len(pd),)
             
            if (load32==loadtest).all():
                load33=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
            else:
                load33=loadtest
                
        if IRound<=6:
            load34=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
            load35=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
        else:
            dir = './S-pdfinal/'                  
            f = open(dir+'pd_final_'+str(IRound-7)+'.txt')
            line = f.readline()
            data_list = []
            while line:
                num = list(map(float,line.split()))
                data_list.append(num)
                line = f.readline()
            f.close()
            pd = np.array(data_list)
            load34 = pd.reshape(len(pd),)
            
            dir = './LPDFINAL/'                  
            f = open(dir+'PD_FINAL_'+str(IRound-7)+'.txt')
            line = f.readline()
            data_list = []
            while line:
                num = list(map(float,line.split()))
                data_list.append(num)
                line = f.readline()
            f.close()
            pd = np.array(data_list)
            loadtest = pd.reshape(len(pd),)
             
            if (load34==loadtest).all():
                load35=np.random.uniform(-1,1,len(pd_temp0))*pd_temp0
            else:
                load35=loadtest


        
        
        
        
        dir = './S-SOLVE/'
        
        
        
        
        m29 = GEKKO(remote=False)
        m29.options.SOLVER = 1
        m29.options.MAX_ITER = 1000
        m30 = GEKKO(remote=False)
        m30.options.SOLVER = 1
        m30.options.MAX_ITER = 1000
        m31 = GEKKO(remote=False)
        m31.options.SOLVER = 1
        m31.options.MAX_ITER = 1000
        m32 = GEKKO(remote=False)
        m32.options.SOLVER = 1
        m32.options.MAX_ITER = 1000
        m33 = GEKKO(remote=False)
        m33.options.SOLVER = 1
        m33.options.MAX_ITER = 1000
        m34 = GEKKO(remote=False)
        m34.options.SOLVER = 1
        m34.options.MAX_ITER = 1000
        m35 = GEKKO(remote=False)
        m35.options.SOLVER = 1
        m35.options.MAX_ITER = 1000
        
        # mpc = loadcase('case118_rev_100.py')
        threads = []
        
        
        
        
        t29 = threading.Thread(target=total_branch29, args=(m29, mpc, load29))
        threads.append(t29)
        t29.start()
        t30 = threading.Thread(target=total_branch30, args=(m30, mpc, load30))
        threads.append(t30)
        t30.start()
        t31 = threading.Thread(target=total_branch31, args=(m31, mpc, load31))
        threads.append(t31)
        t31.start()
        t32 = threading.Thread(target=total_branch32, args=(m32, mpc, load32))
        threads.append(t32)
        t32.start()
        t33 = threading.Thread(target=total_branch33, args=(m33, mpc, load33))
        threads.append(t33)
        t33.start()
        t34 = threading.Thread(target=total_branch34, args=(m34, mpc, load34))
        threads.append(t34)
        t34.start()
        t35 = threading.Thread(target=total_branch35, args=(m35, mpc, load35))
        threads.append(t35)
        t35.start()
        
        
        for t in threads:
            t.join()
            
        
            
        
        
            
        if len(FINAL29)<IRound+1:
            FINAL29.append(-100)
            filenamesolver29= dir+'s-solvertime29.txt'
            fidsolver29=open(filenamesolver29,'at')
            fidsolver29.write("%9.4f;" % (1000000))
            fidsolver29.write('\n')
            fidsolver29.close()
    
            filenamesolution29= dir+'s-total30_all-29.txt'
            fidsolution29=open(filenamesolution29,'at')
            fidsolution29.write("%9.4f;" % (-100))
            fidsolution29.write('\n')
            fidsolution29.close()
            
            pd_save29.append(pd_temp0)
            
        if len(FINAL30)<IRound+1:
            FINAL30.append(-100)
            filenamesolver30= dir+'s-solvertime30.txt'
            fidsolver30=open(filenamesolver30,'at')
            fidsolver30.write("%9.4f;" % (1000000))
            fidsolver30.write('\n')
            fidsolver30.close()
    
            filenamesolution30= dir+'s-total30_all-30.txt'
            fidsolution30=open(filenamesolution30,'at')
            fidsolution30.write("%9.4f;" % (-100))
            fidsolution30.write('\n')
            fidsolution30.close()
            
            pd_save30.append(pd_temp0)
            
        if len(FINAL31)<IRound+1:
            FINAL31.append(-100)
            filenamesolver31= dir+'s-solvertime31.txt'
            fidsolver31=open(filenamesolver31,'at')
            fidsolver31.write("%9.4f;" % (1000000))
            fidsolver31.write('\n')
            fidsolver31.close()
    
            filenamesolution31= dir+'s-total30_all-31.txt'
            fidsolution31=open(filenamesolution31,'at')
            fidsolution31.write("%9.4f;" % (-100))
            fidsolution31.write('\n')
            fidsolution31.close()
            
            pd_save31.append(pd_temp0)
            
        if len(FINAL32)<IRound+1:
            FINAL32.append(-100)
            filenamesolver32= dir+'s-solvertime32.txt'
            fidsolver32=open(filenamesolver32,'at')
            fidsolver32.write("%9.4f;" % (1000000))
            fidsolver32.write('\n')
            fidsolver32.close()
    
            filenamesolution32= dir+'s-total30_all-32.txt'
            fidsolution32=open(filenamesolution32,'at')
            fidsolution32.write("%9.4f;" % (-100))
            fidsolution32.write('\n')
            fidsolution32.close()
            
            pd_save32.append(pd_temp0)
            
        if len(FINAL33)<IRound+1:
            FINAL33.append(-100)
            filenamesolver33= dir+'s-solvertime33.txt'
            fidsolver33=open(filenamesolver33,'at')
            fidsolver33.write("%9.4f;" % (1000000))
            fidsolver33.write('\n')
            fidsolver33.close()
    
            filenamesolution33= dir+'s-total30_all-33.txt'
            fidsolution33=open(filenamesolution33,'at')
            fidsolution33.write("%9.4f;" % (-100))
            fidsolution33.write('\n')
            fidsolution33.close()
            
            pd_save33.append(pd_temp0)
            
        if len(FINAL34)<IRound+1:
            FINAL34.append(-100)
            filenamesolver34= dir+'s-solvertime34.txt'
            fidsolver34=open(filenamesolver34,'at')
            fidsolver34.write("%9.4f;" % (1000000))
            fidsolver34.write('\n')
            fidsolver34.close()
    
            filenamesolution34= dir+'s-total30_all-34.txt'
            fidsolution34=open(filenamesolution34,'at')
            fidsolution34.write("%9.4f;" % (-100))
            fidsolution34.write('\n')
            fidsolution34.close()
            
            pd_save34.append(pd_temp0)
            
        if len(FINAL35)<IRound+1:
            FINAL35.append(-100)
            filenamesolver35= dir+'s-solvertime35.txt'
            fidsolver35=open(filenamesolver35,'at')
            fidsolver35.write("%9.4f;" % (1000000))
            fidsolver35.write('\n')
            fidsolver35.close()
    
            filenamesolution35= dir+'s-total30_all-35.txt'
            fidsolution35=open(filenamesolution35,'at')
            fidsolution35.write("%9.4f;" % (-100))
            fidsolution35.write('\n')
            fidsolution35.close()
            
            pd_save35.append(pd_temp0)
            
        
            
            
        
        
        P_D_normalization.append(pd_save29[IRound])
        P_D_normalization.append(pd_save30[IRound])
        P_D_normalization.append(pd_save31[IRound])
        P_D_normalization.append(pd_save32[IRound])
        P_D_normalization.append(pd_save33[IRound])
        P_D_normalization.append(pd_save34[IRound])
        P_D_normalization.append(pd_save35[IRound])
        
        
        dir='./S-TEST/'
        for i in range(0,7):
            np.savetxt(dir+'s-TEST_'+str(i+IRound*42+28)+'.txt', np.array(P_D_normalization[len(P_D_normalization)-i-1]), fmt="%s")
 

        
        
        
        del m29
        del m30
        del m31
        del m32
        del m33
        del m34
        del m35
        
        
        FINISH=time.time()-START
        
        dir = './S-SOLVE_TIME/'
        filenametime5= dir+'time5.txt'
        fidtime5=open(filenametime5,'at')
        fidtime5.write("%9.4f;" % (FINISH))
        fidtime5.write('\n')
        fidtime5.close()
        
        
        wait1=time.time()
        dir = './S-PTH/'
        for SOME in range(0,1000):
            if os.access(dir+'s-TEST_'+str(IRound+1)+'.pth', os.F_OK):
                time.sleep(0)
            else:
                time.sleep(1)
                
            
                
        dir = './S-WAIT/'
        filenamewait= dir+'WAIT5.txt'
        fidwait=open(filenamewait,'at')
        fidwait.write("%9.4f;" % (time.time()-wait1))
        fidwait.write('\n')
        fidwait.close()
                
        if os.access('FINISH.txt', os.F_OK):
            break
                
        
