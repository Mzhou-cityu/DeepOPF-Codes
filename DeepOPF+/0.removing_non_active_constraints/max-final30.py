#DC-OPF
from gekko import GEKKO
from scipy.sparse import csr_matrix as sparse
from pypower.loadcase import loadcase
from pypower.makeBdc import makeBdc
import numpy as np
from pypower.ext2int import ext2int1
from torch import nn
import torch.nn.functional as F

input_range = 0.15
default=1.15
Lower_bound = -1e5
Upper_bound = 1e5

# neural network definition
# class Neuralnetwork(nn.Module):
#     def __init__(self, in_dim, n_hidden, out_dim):
#         super(Neuralnetwork, self).__init__()
#         self.layer1 = nn.Linear(in_dim, 4*n_hidden)
#         self.layer2 = nn.Linear(4*n_hidden, 2*n_hidden)
#         self.layer3 = nn.Linear(2*n_hidden, n_hidden)
#         self.layer4 = nn.Linear(n_hidden, out_dim)

#     def forward(self, x):
#         x0 = F.relu(self.layer1(x))
#         x1 = F.relu(self.layer2(x0))
#         x2 = F.relu(self.layer3(x1))
#         x3 = F.relu(self.layer4(x1))
#         return x3


#ipopt_solve
def worst_case_prediction_error_branch1(m=None,mpc=None):
    global Lower_bound
    global Upper_bound
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
    #build model
    # model = Neuralnetwork(Load_index.shape[0], 8, Gen_index.shape[0])
    # params = list(model.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k)) 
    # ##############################
    # layers = []
    # layers.append(Load_index.shape[0])
    # for i in params:
    #     if len(i.size()) == 2:
    #         neuron_number = i.size()[0]
    #         layers.append(neuron_number)
    ###################################
    #variables for DC-OPF
    #Pd
    # numberOfInput = layers[0]  
    # z=[]
    #create variables
    Input = m.Array(m.Var,(len(Load_index_list),))        
    for i in range(0,len(Load_index_list),1):
        Input[i].lower = bus[Load_index_list[i],2]/standval*(default-input_range)
        Input[i].upper = bus[Load_index_list[i],2]/standval*(default+input_range)
    # for i in range(0,len(Load_index_list),1):
    #     z.append(bus[Load_index_list[i],2]/standval)
        
    
    # z=np.zeros((1, len(Load_index_list)),dtype='float32') 
    # for i in range(0,len(Load_index_list),1):
    #     z[0,i]=bus[Load_index_list[i],2]/standval
    
    
    #number of variables for DNN    
    # numberOfWeight = 0
    # for i in range(0,len(layers)-1,1):
    #     numberOfWeight += layers[i]*layers[i+1]
    # numberOfBias = sum(layers[1:len(layers)])
    # numberOfStateVariable = numberOfBias
    # numberOfNeuronOuput = numberOfStateVariable
    # #create variables
    # weight = m.Array(m.Var,(numberOfWeight,))
    # bias = m.Array(m.Var,(numberOfBias,))
    # # State = m.Array(m.Var,(numberOfBias,),lb=0,ub=1,integer=True)
    # NeuronOutput = m.Array(m.Var,(numberOfNeuronOuput,))
    
#    for i in range(0,numberOfWeight,1):    
#        weight[i].lower = -1e5
#        weight[i].upper = 1e5

    # for i in range(0,numberOfBias,1):
    #     State[i].lower = 0
    #     State[i].upper = 1
#        bias[i].lower = -1e5
#        bias[i].upper = 1e5     
#        NeuronOutput[i].lower = -1e5
#        NeuronOutput[i].upper = 1e5

    #from input to first hidden layer
#    First_layer_output = [None]*layers[1]
#     for i in range(0,layers[1]):
# #        First_layer_output[i] = m.Intermediate(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)]))
#         # m.Equation(NeuronOutput[i] >=0)
#         m.Equation(NeuronOutput[i] == m.max2(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])+bias[i],0))
#         # m.Equation(NeuronOutput[i]-bias[i] <= sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])-Lower_bound*(1-State[i]))        
#         # m.Equation(NeuronOutput[i] <= Upper_bound*State[i])
    
#     weight_index = layers[0]*layers[1]
#     #for hidden layers
#     for layer_index in range(2,len(layers)-1,1):
#         previous_index = sum(layers[1:layer_index-1])
#         current_index = sum(layers[1:layer_index])
#         for i in range(0,layers[layer_index]):
#             # m.Equation(NeuronOutput[current_index+i] >=0)
#             m.Equation(NeuronOutput[current_index+i]==m.max2(sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])+bias[current_index+i],0))
#             # m.Equation(NeuronOutput[current_index+i]-bias[current_index+i]<=sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])-Lower_bound*(1-State[current_index+i]))        
#             # m.Equation(NeuronOutput[current_index+i]<=Upper_bound*State[current_index+i])
#             weight_index += layers[layer_index-1]

#     #from hidden layer to output
#     previous_index = sum(layers[1:len(layers)-2])
#     current_index = sum(layers[1:len(layers)-1])   
#     # for i in range(0,layers[len(layers)-1]):
#     #     m.Equation(NeuronOutput[current_index+i] >=0)
#     #     m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] >= sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])]))
#     #     m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] <= sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])])-Lower_bound*(1-State[current_index+i]))        
#     #     m.Equation(NeuronOutput[current_index+i] <= Upper_bound*State[current_index+i])
#     #     weight_index += layers[len(layers)-2]
#     for i in range(0,layers[len(layers)-1]):
#         # m.Equation(NeuronOutput[current_index+i] >=0)
#         m.Equation(NeuronOutput[current_index+i] == sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])])+bias[current_index+i])
#         # m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] <= sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])])-Lower_bound*(1-State[current_index+i]))        
#         # m.Equation(NeuronOutput[current_index+i] <= Upper_bound*State[current_index+i])
#         weight_index += layers[len(layers)-2]
   
#     current_index = sum(layers[1:len(layers)-1])    
#     #NN output
#     NNOutput = m.Array(m.Var,(layers[len(layers)-1],))
#     for i in range(0,layers[len(layers)-1]):
#         # NNOutput[i].lower = 0
#         # NNOutput[i].upper = 1
#         #sigmoid
#         # m.Equation(NNOutput[i] == 1/(1+ m.exp(-NeuronOutput[current_index+i])))
#         m.Equation(NNOutput[i] == NeuronOutput[current_index+i])

    #Pg
    Pg = m.Array(m.Var,(len(Gen_index_list),))
    
    for i in range(0,len(Gen_index_list)):
      #  Pg[i].lower = gen[int(Gen_index_list[i]),9]/standval
       # Pg[i].upper = gen[int(Gen_index_list[i]),8]/standval      
        # m.Equation(Pg[i]==NNOutput[i])
        Pg[i].lower = Power_Lowbound_Gen[0,i]
        Pg[i].upper = Power_Upbound_Gen[0,i]
    #DC-PF equations
    # Create variables
    pg = m.Array(m.Var,(mL,))
    pg_feasible = m.Array(m.Var,(mL,))
    pd = m.Array(m.Var,(mL,))
    gen_index = 0
    load_index = 0
    for i in range(0,mL,1):
        if i in Load_index_list:
            m.Equation(pd[i] == Input[load_index])
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
            # pg[i].lower = 0
            # pg[i].upper = 0
            m.Equation(pg[i] == Pg[gen_index])
            gen_index += 1                
        else:
            pg[i].lower = 0
            pg[i].upper = 0
            
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

    #DC-PF equations
    m.Equation(m.sum(pg) == m.sum(pd))
    m.Equation(m.sum(pg_feasible) == m.sum(pd))
    # m.Equation(m.sum(pg) == m.sum(pd))
    
    #slack bus violation
    # Slack_Pg_violation = m.Array(m.Var,(1,))
    # Slack_Pg_violation[0].lower = 0
    # Slack_Pg_violation[0].upper = 1e9
    
    # Slack_Pg_Violation_temp = [None]*2
    # Slack_Pg_Violation_temp[0] = m.Intermediate(m.max2(pg[Slack_bus]-Power_Upbound_Gen_Slack,0))
    # Slack_Pg_Violation_temp[1] = m.Intermediate(m.max2(Power_Lowbound_Gen_Slack-pg[Slack_bus],0))
    # m.Equation(Slack_Pg_violation[0] == Slack_Pg_Violation_temp[0]+Slack_Pg_Violation_temp[1])   
    # m.Obj(-Slack_Pg_violation[0])
#
#    # minimize objective
#    m.solve()
#    pd_solution = []
#    for i in range(0,mL):
#        pd_solution.append(pd[i].value[0])
#    pg_solution = []
#    for i in range(0,mL):
#        pg_solution.append(pg[i].value[0])
#
#    print(Slack_Pg_violation[0].value)
#    print('\n')    
#    print(pg[Slack_bus].value)
#    print('\n') 
#    print(Pg[0].value)
    ##########################################################
    #Branch limit
    Power_bound_Branch = np.zeros((mB,),dtype='float32')
    Power_bound_Branch = branch[:,5]/standval*1.017

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

    PTDF_ori = np.dot(Bf.toarray(),Bus_admittance_full)
    Branch_feasible=m.Array(m.Var,(mB,))
    for i in range(0, mB):
        m.Equation(Branch_feasible[i]==sum([PTDF_ori[i,j]*(pg_feasible[j]-pd[j]) for j in range(0,mL)]))
        Branch_feasible[i].lower=-Power_bound_Branch[i]
        Branch_feasible[i].upper=Power_bound_Branch[i]
        
        
    
    #branch flow violation
    # Branch_index = 1
    Branch_flow_violation = m.Array(m.Var,(1,))
    Branch_flow_violation[0].lower = Lower_bound
    Branch_flow_violation[0].upper = Upper_bound
    
    Branch_flow_Violation_temp = [None]*2
    # Branch_flow_Violation_temp[0] = m.Intermediate(m.max2(sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)])-Power_bound_Branch[Branch_index],0))
    # Branch_flow_Violation_temp[1] = m.Intermediate(m.max2(-Power_bound_Branch[Branch_index]-sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)]),0))
    Branch_flow_Violation_temp[0] = m.Intermediate(sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)]))
    Branch_flow_Violation_temp[1] = m.Intermediate(-sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)]))
    m.Equation(Branch_flow_violation[0] == Branch_flow_Violation_temp[0])   
    m.Obj(-Branch_flow_violation[0])

    # minimize objective
    m.solve()
    pd_solution = []
    for i in range(0,mL):
        pd_solution.append(pd[i].value[0])
    pg_solution = []
    for i in range(0,mL):
        pg_solution.append(pg[i].value[0])
        
    # print(Slack_Pg_violation[0].value)
    print(Branch_flow_violation[0].value)
    print(pg_solution)
    print(pd_solution)
    # print(State)
    print('\n')
    Branchresult1.append(Branch_flow_violation[0].value)
    
    Actual_branch_flow = np.zeros((1,),dtype='float32')
    for i in range(0,mL):
       Actual_branch_flow[0] += PTDF_ori[Branch_index,i]*(pg[i].value[0]-pd[i].value[0])
    print(Actual_branch_flow[0])
    print('\n')    
    


    
def worst_case_prediction_error_branch2(m=None,mpc=None):
    global Lower_bound
    global Upper_bound
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
    #build model
    # model = Neuralnetwork(Load_index.shape[0], 8, Gen_index.shape[0])
    # params = list(model.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k)) 
    # ##############################
    # layers = []
    # layers.append(Load_index.shape[0])
    # for i in params:
    #     if len(i.size()) == 2:
    #         neuron_number = i.size()[0]
    #         layers.append(neuron_number)
    ###################################
    #variables for DC-OPF
    #Pd
    # numberOfInput = layers[0]  
    # z=[]
    #create variables
    Input = m.Array(m.Var,(len(Load_index_list),))        
    for i in range(0,len(Load_index_list),1):
        Input[i].lower = bus[Load_index_list[i],2]/standval*(default-input_range)
        Input[i].upper = bus[Load_index_list[i],2]/standval*(default+input_range)
    # for i in range(0,len(Load_index_list),1):
    #     z.append(bus[Load_index_list[i],2]/standval)
        
    
    # z=np.zeros((1, len(Load_index_list)),dtype='float32') 
    # for i in range(0,len(Load_index_list),1):
    #     z[0,i]=bus[Load_index_list[i],2]/standval
    
    
    #number of variables for DNN    
    # numberOfWeight = 0
    # for i in range(0,len(layers)-1,1):
    #     numberOfWeight += layers[i]*layers[i+1]
    # numberOfBias = sum(layers[1:len(layers)])
    # numberOfStateVariable = numberOfBias
    # numberOfNeuronOuput = numberOfStateVariable
    # #create variables
    # weight = m.Array(m.Var,(numberOfWeight,))
    # bias = m.Array(m.Var,(numberOfBias,))
    # # State = m.Array(m.Var,(numberOfBias,),lb=0,ub=1,integer=True)
    # NeuronOutput = m.Array(m.Var,(numberOfNeuronOuput,))
    
#    for i in range(0,numberOfWeight,1):    
#        weight[i].lower = -1e5
#        weight[i].upper = 1e5

    # for i in range(0,numberOfBias,1):
    #     State[i].lower = 0
    #     State[i].upper = 1
#        bias[i].lower = -1e5
#        bias[i].upper = 1e5     
#        NeuronOutput[i].lower = -1e5
#        NeuronOutput[i].upper = 1e5

    #from input to first hidden layer
#    First_layer_output = [None]*layers[1]
#     for i in range(0,layers[1]):
# #        First_layer_output[i] = m.Intermediate(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)]))
#         # m.Equation(NeuronOutput[i] >=0)
#         m.Equation(NeuronOutput[i] == m.max2(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])+bias[i],0))
#         # m.Equation(NeuronOutput[i]-bias[i] <= sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])-Lower_bound*(1-State[i]))        
#         # m.Equation(NeuronOutput[i] <= Upper_bound*State[i])
    
#     weight_index = layers[0]*layers[1]
#     #for hidden layers
#     for layer_index in range(2,len(layers)-1,1):
#         previous_index = sum(layers[1:layer_index-1])
#         current_index = sum(layers[1:layer_index])
#         for i in range(0,layers[layer_index]):
#             # m.Equation(NeuronOutput[current_index+i] >=0)
#             m.Equation(NeuronOutput[current_index+i]==m.max2(sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])+bias[current_index+i],0))
#             # m.Equation(NeuronOutput[current_index+i]-bias[current_index+i]<=sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])-Lower_bound*(1-State[current_index+i]))        
#             # m.Equation(NeuronOutput[current_index+i]<=Upper_bound*State[current_index+i])
#             weight_index += layers[layer_index-1]

#     #from hidden layer to output
#     previous_index = sum(layers[1:len(layers)-2])
#     current_index = sum(layers[1:len(layers)-1])   
#     # for i in range(0,layers[len(layers)-1]):
#     #     m.Equation(NeuronOutput[current_index+i] >=0)
#     #     m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] >= sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])]))
#     #     m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] <= sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])])-Lower_bound*(1-State[current_index+i]))        
#     #     m.Equation(NeuronOutput[current_index+i] <= Upper_bound*State[current_index+i])
#     #     weight_index += layers[len(layers)-2]
#     for i in range(0,layers[len(layers)-1]):
#         # m.Equation(NeuronOutput[current_index+i] >=0)
#         m.Equation(NeuronOutput[current_index+i] == sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])])+bias[current_index+i])
#         # m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] <= sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])])-Lower_bound*(1-State[current_index+i]))        
#         # m.Equation(NeuronOutput[current_index+i] <= Upper_bound*State[current_index+i])
#         weight_index += layers[len(layers)-2]
   
#     current_index = sum(layers[1:len(layers)-1])    
#     #NN output
#     NNOutput = m.Array(m.Var,(layers[len(layers)-1],))
#     for i in range(0,layers[len(layers)-1]):
#         # NNOutput[i].lower = 0
#         # NNOutput[i].upper = 1
#         #sigmoid
#         # m.Equation(NNOutput[i] == 1/(1+ m.exp(-NeuronOutput[current_index+i])))
#         m.Equation(NNOutput[i] == NeuronOutput[current_index+i])

    #Pg
    Pg = m.Array(m.Var,(len(Gen_index_list),))
    for i in range(0,len(Gen_index_list)):
      #  Pg[i].lower = gen[int(Gen_index_list[i]),9]/standval
       # Pg[i].upper = gen[int(Gen_index_list[i]),8]/standval      
        # m.Equation(Pg[i]==NNOutput[i])
        Pg[i].lower = Power_Lowbound_Gen[0,i]
        Pg[i].upper = Power_Upbound_Gen[0,i]
    #DC-PF equations
    # Create variables
    pg = m.Array(m.Var,(mL,))
    pg_feasible = m.Array(m.Var,(mL,))
    pd = m.Array(m.Var,(mL,))
    gen_index = 0
    load_index = 0
    for i in range(0,mL,1):
        if i in Load_index_list:
            m.Equation(pd[i] == Input[load_index])
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
            # pg[i].lower = 0
            # pg[i].upper = 0
            m.Equation(pg[i] == Pg[gen_index])
            gen_index += 1                
        else:
            pg[i].lower = 0
            pg[i].upper = 0
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

    #DC-PF equations
    m.Equation(m.sum(pg) == m.sum(pd))
    m.Equation(m.sum(pg_feasible) == m.sum(pd))    
    # m.Equation(m.sum(pg) == m.sum(pd))
    
    #slack bus violation
    # Slack_Pg_violation = m.Array(m.Var,(1,))
    # Slack_Pg_violation[0].lower = 0
    # Slack_Pg_violation[0].upper = 1e9
    
    # Slack_Pg_Violation_temp = [None]*2
    # Slack_Pg_Violation_temp[0] = m.Intermediate(m.max2(pg[Slack_bus]-Power_Upbound_Gen_Slack,0))
    # Slack_Pg_Violation_temp[1] = m.Intermediate(m.max2(Power_Lowbound_Gen_Slack-pg[Slack_bus],0))
    # m.Equation(Slack_Pg_violation[0] == Slack_Pg_Violation_temp[0]+Slack_Pg_Violation_temp[1])   
    # m.Obj(-Slack_Pg_violation[0])
#
#    # minimize objective
#    m.solve()
#    pd_solution = []
#    for i in range(0,mL):
#        pd_solution.append(pd[i].value[0])
#    pg_solution = []
#    for i in range(0,mL):
#        pg_solution.append(pg[i].value[0])
#
#    print(Slack_Pg_violation[0].value)
#    print('\n')    
#    print(pg[Slack_bus].value)
#    print('\n') 
#    print(Pg[0].value)
    ##########################################################
    #Branch limit
    Power_bound_Branch = np.zeros((mB,),dtype='float32')
    Power_bound_Branch = branch[:,5]/standval*1.017

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

    PTDF_ori = np.dot(Bf.toarray(),Bus_admittance_full)
    Branch_feasible=m.Array(m.Var,(mB,))
    for i in range(0, mB):
        m.Equation(Branch_feasible[i]==sum([PTDF_ori[i,j]*(pg_feasible[j]-pd[j]) for j in range(0,mL)]))
        Branch_feasible[i].lower=-Power_bound_Branch[i]
        Branch_feasible[i].upper=Power_bound_Branch[i]
    #branch flow violation
    # Branch_index = 1
    Branch_flow_violation = m.Array(m.Var,(1,))
    Branch_flow_violation[0].lower = Lower_bound
    Branch_flow_violation[0].upper = Upper_bound
    
    Branch_flow_Violation_temp = [None]*2
    # Branch_flow_Violation_temp[0] = m.Intermediate(m.max2(sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)])-Power_bound_Branch[Branch_index],0))
    # Branch_flow_Violation_temp[1] = m.Intermediate(m.max2(-Power_bound_Branch[Branch_index]-sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)]),0))
    Branch_flow_Violation_temp[0] = m.Intermediate(sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)]))
    Branch_flow_Violation_temp[1] = m.Intermediate(-sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)]))
    m.Equation(Branch_flow_violation[0] == Branch_flow_Violation_temp[0])   
    m.Obj(Branch_flow_violation[0])

    # minimize objective
    m.solve()
    pd_solution = []
    for i in range(0,mL):
        pd_solution.append(pd[i].value[0])
    pg_solution = []
    for i in range(0,mL):
        pg_solution.append(pg[i].value[0])
        
    Branchlimit.append(Power_bound_Branch[Branch_index])
    # print(Slack_Pg_violation[0].value)
    print(Branch_flow_violation[0].value)
    print(pg_solution)
    print(pd_solution)
    # print(State)
    print('\n')
    Branchresult2.append(Branch_flow_violation[0].value)
    
    Actual_branch_flow = np.zeros((1,),dtype='float32')
    for i in range(0,mL):
       Actual_branch_flow[0] += PTDF_ori[Branch_index,i]*(pg[i].value[0]-pd[i].value[0])
    print(Actual_branch_flow[0])
    print('\n')    
    

    
    
    
def worst_case_prediction_error_slack1(m=None,mpc=None):
    global Lower_bound
    global Upper_bound
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
    #build model
    # model = Neuralnetwork(Load_index.shape[0], 8, Gen_index.shape[0])
    # params = list(model.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k)) 
    # ##############################
    # layers = []
    # layers.append(Load_index.shape[0])
    # for i in params:
    #     if len(i.size()) == 2:
    #         neuron_number = i.size()[0]
    #         layers.append(neuron_number)
    ###################################
    #variables for DC-OPF
    #Pd
    # numberOfInput = layers[0]  
    # z=[]
    #create variables
    Input = m.Array(m.Var,(len(Load_index_list),))        
    for i in range(0,len(Load_index_list),1):
        Input[i].lower = bus[Load_index_list[i],2]/standval*(default-input_range)
        Input[i].upper = bus[Load_index_list[i],2]/standval*(default+input_range)
    # for i in range(0,len(Load_index_list),1):
    #     z.append(bus[Load_index_list[i],2]/standval)
        
    
    # z=np.zeros((1, len(Load_index_list)),dtype='float32') 
    # for i in range(0,len(Load_index_list),1):
    #     z[0,i]=bus[Load_index_list[i],2]/standval
    
    
    #number of variables for DNN    
    # numberOfWeight = 0
    # for i in range(0,len(layers)-1,1):
    #     numberOfWeight += layers[i]*layers[i+1]
    # numberOfBias = sum(layers[1:len(layers)])
    # numberOfStateVariable = numberOfBias
    # numberOfNeuronOuput = numberOfStateVariable
    # #create variables
    # weight = m.Array(m.Var,(numberOfWeight,))
    # bias = m.Array(m.Var,(numberOfBias,))
    # # State = m.Array(m.Var,(numberOfBias,),lb=0,ub=1,integer=True)
    # NeuronOutput = m.Array(m.Var,(numberOfNeuronOuput,))
    
#    for i in range(0,numberOfWeight,1):    
#        weight[i].lower = -1e5
#        weight[i].upper = 1e5

    # for i in range(0,numberOfBias,1):
    #     State[i].lower = 0
    #     State[i].upper = 1
#        bias[i].lower = -1e5
#        bias[i].upper = 1e5     
#        NeuronOutput[i].lower = -1e5
#        NeuronOutput[i].upper = 1e5

    #from input to first hidden layer
#    First_layer_output = [None]*layers[1]
#     for i in range(0,layers[1]):
# #        First_layer_output[i] = m.Intermediate(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)]))
#         # m.Equation(NeuronOutput[i] >=0)
#         m.Equation(NeuronOutput[i] == m.max2(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])+bias[i],0))
#         # m.Equation(NeuronOutput[i]-bias[i] <= sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])-Lower_bound*(1-State[i]))        
#         # m.Equation(NeuronOutput[i] <= Upper_bound*State[i])
    
#     weight_index = layers[0]*layers[1]
#     #for hidden layers
#     for layer_index in range(2,len(layers)-1,1):
#         previous_index = sum(layers[1:layer_index-1])
#         current_index = sum(layers[1:layer_index])
#         for i in range(0,layers[layer_index]):
#             # m.Equation(NeuronOutput[current_index+i] >=0)
#             m.Equation(NeuronOutput[current_index+i]==m.max2(sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])+bias[current_index+i],0))
#             # m.Equation(NeuronOutput[current_index+i]-bias[current_index+i]<=sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])-Lower_bound*(1-State[current_index+i]))        
#             # m.Equation(NeuronOutput[current_index+i]<=Upper_bound*State[current_index+i])
#             weight_index += layers[layer_index-1]

#     #from hidden layer to output
#     previous_index = sum(layers[1:len(layers)-2])
#     current_index = sum(layers[1:len(layers)-1])   
#     # for i in range(0,layers[len(layers)-1]):
#     #     m.Equation(NeuronOutput[current_index+i] >=0)
#     #     m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] >= sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])]))
#     #     m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] <= sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])])-Lower_bound*(1-State[current_index+i]))        
#     #     m.Equation(NeuronOutput[current_index+i] <= Upper_bound*State[current_index+i])
#     #     weight_index += layers[len(layers)-2]
#     for i in range(0,layers[len(layers)-1]):
#         # m.Equation(NeuronOutput[current_index+i] >=0)
#         m.Equation(NeuronOutput[current_index+i] == sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])])+bias[current_index+i])
#         # m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] <= sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])])-Lower_bound*(1-State[current_index+i]))        
#         # m.Equation(NeuronOutput[current_index+i] <= Upper_bound*State[current_index+i])
#         weight_index += layers[len(layers)-2]
   
#     current_index = sum(layers[1:len(layers)-1])    
#     #NN output
#     NNOutput = m.Array(m.Var,(layers[len(layers)-1],))
#     for i in range(0,layers[len(layers)-1]):
#         # NNOutput[i].lower = 0
#         # NNOutput[i].upper = 1
#         #sigmoid
#         # m.Equation(NNOutput[i] == 1/(1+ m.exp(-NeuronOutput[current_index+i])))
#         m.Equation(NNOutput[i] == NeuronOutput[current_index+i])

    #Pg
    Pg = m.Array(m.Var,(len(Gen_index_list),))
    for i in range(0,len(Gen_index_list)):
      #  Pg[i].lower = gen[int(Gen_index_list[i]),9]/standval
       # Pg[i].upper = gen[int(Gen_index_list[i]),8]/standval      
        # m.Equation(Pg[i]==NNOutput[i])
        Pg[i].lower = Power_Lowbound_Gen[0,i]
        Pg[i].upper = Power_Upbound_Gen[0,i]
    #DC-PF equations
    # Create variables
    pg = m.Array(m.Var,(mL,))
    pg_feasible = m.Array(m.Var,(mL,))
    pd = m.Array(m.Var,(mL,))
    gen_index = 0
    load_index = 0
    for i in range(0,mL,1):
        if i in Load_index_list:
            m.Equation(pd[i] == Input[load_index])
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
            # pg[i].lower = 0
            # pg[i].upper = 0
            m.Equation(pg[i] == Pg[gen_index])
            gen_index += 1                
        else:
            pg[i].lower = 0
            pg[i].upper = 0
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

    #DC-PF equations
    m.Equation(m.sum(pg) == m.sum(pd))
    m.Equation(m.sum(pg_feasible) == m.sum(pd))
    # m.Equation(m.sum(pg) == m.sum(pd))
    
    # slack bus violation
    Slack_Pg_violation = m.Array(m.Var,(1,))
    Slack_Pg_violation[0].lower = Lower_bound
    Slack_Pg_violation[0].upper = Upper_bound
    
    Slack_Pg_Violation_temp = [None]*2
    # Slack_Pg_Violation_temp[0] = m.Intermediate(m.max2(pg[Slack_bus]-Power_Upbound_Gen_Slack,0))
    # Slack_Pg_Violation_temp[1] = m.Intermediate(m.max2(Power_Lowbound_Gen_Slack-pg[Slack_bus],0))
    Slack_Pg_Violation_temp[0] = m.Intermediate(pg[Slack_bus])
    # Slack_Pg_Violation_temp[1] = m.Intermediate(-pg[Slack_bus])
    m.Equation(Slack_Pg_violation[0] == Slack_Pg_Violation_temp[0])   
    m.Obj(-Slack_Pg_violation[0])
#
#    # minimize objective
#    m.solve()
#    pd_solution = []
#    for i in range(0,mL):
#        pd_solution.append(pd[i].value[0])
#    pg_solution = []
#    for i in range(0,mL):
#        pg_solution.append(pg[i].value[0])
#
#    print(Slack_Pg_violation[0].value)
#    print('\n')    
#    print(pg[Slack_bus].value)
#    print('\n') 
#    print(Pg[0].value)
    ##########################################################
    #Branch limit
    Power_bound_Branch = np.zeros((mB,),dtype='float32')
    Power_bound_Branch = branch[:,5]/standval*1.017

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

    PTDF_ori = np.dot(Bf.toarray(),Bus_admittance_full)
    Branch_feasible=m.Array(m.Var,(mB,))
    for i in range(0, mB):
        m.Equation(Branch_feasible[i]==sum([PTDF_ori[i,j]*(pg_feasible[j]-pd[j]) for j in range(0,mL)]))
        Branch_feasible[i].lower=-Power_bound_Branch[i]
        Branch_feasible[i].upper=Power_bound_Branch[i]
    #branch flow violation
    # Branch_index = 1
    Branch_flow_violation = m.Array(m.Var,(1,))
    Branch_flow_violation[0].lower = Lower_bound
    Branch_flow_violation[0].upper = Upper_bound
    
    Branch_flow_Violation_temp = [None]*2
    # Branch_flow_Violation_temp[0] = m.Intermediate(m.max2(sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)])-Power_bound_Branch[Branch_index],0))
    # Branch_flow_Violation_temp[1] = m.Intermediate(m.max2(-Power_bound_Branch[Branch_index]-sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)]),0))
    Branch_flow_Violation_temp[0] = m.Intermediate(sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)]))
    Branch_flow_Violation_temp[1] = m.Intermediate(-sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)]))
    m.Equation(Branch_flow_violation[0] == Branch_flow_Violation_temp[0])   
    # m.Obj(Branch_flow_violation[0])

    # minimize objective
    m.solve()
    pd_solution = []
    for i in range(0,mL):
        pd_solution.append(pd[i].value[0])
    pg_solution = []
    for i in range(0,mL):
        pg_solution.append(pg[i].value[0])
        
    # Slacklimit1.append(Power_Upbound_Gen_Slack)
    # Slacklimit2.append(Power_Lowbound_Gen_Slack)
    print(Slack_Pg_violation[0].value)
    # print(Branch_flow_violation[0].value)
    print(pg_solution)
    print(pd_solution)
    # print(State)
    print('\n')
    # Branchresult2.append(Branch_flow_violation[0].value)
    Slackresult1.append(Slack_Pg_violation[0].value)
    # Actual_branch_flow = np.zeros((1,),dtype='float32')
    # for i in range(0,mL):
    #    Actual_branch_flow[0] += PTDF_ori[Branch_index,i]*(pg[i].value[0]-pd[i].value[0])
    # print(Actual_branch_flow[0])
    # print('\n')    
    
def worst_case_prediction_error_slack2(m=None,mpc=None):
    global Lower_bound
    global Upper_bound
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
    #build model
    # model = Neuralnetwork(Load_index.shape[0], 8, Gen_index.shape[0])
    # params = list(model.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k)) 
    # ##############################
    # layers = []
    # layers.append(Load_index.shape[0])
    # for i in params:
    #     if len(i.size()) == 2:
    #         neuron_number = i.size()[0]
    #         layers.append(neuron_number)
    ###################################
    #variables for DC-OPF
    #Pd
    # numberOfInput = layers[0]  
    # z=[]
    #create variables
    Input = m.Array(m.Var,(len(Load_index_list),))        
    for i in range(0,len(Load_index_list),1):
        Input[i].lower = bus[Load_index_list[i],2]/standval*(default-input_range)
        Input[i].upper = bus[Load_index_list[i],2]/standval*(default+input_range)
    # for i in range(0,len(Load_index_list),1):
    #     z.append(bus[Load_index_list[i],2]/standval)
        
    
    # z=np.zeros((1, len(Load_index_list)),dtype='float32') 
    # for i in range(0,len(Load_index_list),1):
    #     z[0,i]=bus[Load_index_list[i],2]/standval
    
    
    #number of variables for DNN    
    # numberOfWeight = 0
    # for i in range(0,len(layers)-1,1):
    #     numberOfWeight += layers[i]*layers[i+1]
    # numberOfBias = sum(layers[1:len(layers)])
    # numberOfStateVariable = numberOfBias
    # numberOfNeuronOuput = numberOfStateVariable
    # #create variables
    # weight = m.Array(m.Var,(numberOfWeight,))
    # bias = m.Array(m.Var,(numberOfBias,))
    # # State = m.Array(m.Var,(numberOfBias,),lb=0,ub=1,integer=True)
    # NeuronOutput = m.Array(m.Var,(numberOfNeuronOuput,))
    
#    for i in range(0,numberOfWeight,1):    
#        weight[i].lower = -1e5
#        weight[i].upper = 1e5

    # for i in range(0,numberOfBias,1):
    #     State[i].lower = 0
    #     State[i].upper = 1
#        bias[i].lower = -1e5
#        bias[i].upper = 1e5     
#        NeuronOutput[i].lower = -1e5
#        NeuronOutput[i].upper = 1e5

    #from input to first hidden layer
#    First_layer_output = [None]*layers[1]
#     for i in range(0,layers[1]):
# #        First_layer_output[i] = m.Intermediate(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)]))
#         # m.Equation(NeuronOutput[i] >=0)
#         m.Equation(NeuronOutput[i] == m.max2(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])+bias[i],0))
#         # m.Equation(NeuronOutput[i]-bias[i] <= sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])-Lower_bound*(1-State[i]))        
#         # m.Equation(NeuronOutput[i] <= Upper_bound*State[i])
    
#     weight_index = layers[0]*layers[1]
#     #for hidden layers
#     for layer_index in range(2,len(layers)-1,1):
#         previous_index = sum(layers[1:layer_index-1])
#         current_index = sum(layers[1:layer_index])
#         for i in range(0,layers[layer_index]):
#             # m.Equation(NeuronOutput[current_index+i] >=0)
#             m.Equation(NeuronOutput[current_index+i]==m.max2(sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])+bias[current_index+i],0))
#             # m.Equation(NeuronOutput[current_index+i]-bias[current_index+i]<=sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])-Lower_bound*(1-State[current_index+i]))        
#             # m.Equation(NeuronOutput[current_index+i]<=Upper_bound*State[current_index+i])
#             weight_index += layers[layer_index-1]

#     #from hidden layer to output
#     previous_index = sum(layers[1:len(layers)-2])
#     current_index = sum(layers[1:len(layers)-1])   
#     # for i in range(0,layers[len(layers)-1]):
#     #     m.Equation(NeuronOutput[current_index+i] >=0)
#     #     m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] >= sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])]))
#     #     m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] <= sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])])-Lower_bound*(1-State[current_index+i]))        
#     #     m.Equation(NeuronOutput[current_index+i] <= Upper_bound*State[current_index+i])
#     #     weight_index += layers[len(layers)-2]
#     for i in range(0,layers[len(layers)-1]):
#         # m.Equation(NeuronOutput[current_index+i] >=0)
#         m.Equation(NeuronOutput[current_index+i] == sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])])+bias[current_index+i])
#         # m.Equation(NeuronOutput[current_index+i]-bias[current_index+i] <= sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])])-Lower_bound*(1-State[current_index+i]))        
#         # m.Equation(NeuronOutput[current_index+i] <= Upper_bound*State[current_index+i])
#         weight_index += layers[len(layers)-2]
   
#     current_index = sum(layers[1:len(layers)-1])    
#     #NN output
#     NNOutput = m.Array(m.Var,(layers[len(layers)-1],))
#     for i in range(0,layers[len(layers)-1]):
#         # NNOutput[i].lower = 0
#         # NNOutput[i].upper = 1
#         #sigmoid
#         # m.Equation(NNOutput[i] == 1/(1+ m.exp(-NeuronOutput[current_index+i])))
#         m.Equation(NNOutput[i] == NeuronOutput[current_index+i])

    #Pg
    Pg = m.Array(m.Var,(len(Gen_index_list),))
    for i in range(0,len(Gen_index_list)):
      #  Pg[i].lower = gen[int(Gen_index_list[i]),9]/standval
       # Pg[i].upper = gen[int(Gen_index_list[i]),8]/standval      
        # m.Equation(Pg[i]==NNOutput[i])
        Pg[i].lower = Power_Lowbound_Gen[0,i]
        Pg[i].upper = Power_Upbound_Gen[0,i]
    #DC-PF equations
    # Create variables
    pg = m.Array(m.Var,(mL,))
    pg_feasible = m.Array(m.Var,(mL,))
    pd = m.Array(m.Var,(mL,))
    gen_index = 0
    load_index = 0
    for i in range(0,mL,1):
        if i in Load_index_list:
            m.Equation(pd[i] == Input[load_index])
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
            # pg[i].lower = 0
            # pg[i].upper = 0
            m.Equation(pg[i] == Pg[gen_index])
            gen_index += 1                
        else:
            pg[i].lower = 0
            pg[i].upper = 0
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

    #DC-PF equations
    m.Equation(m.sum(pg) == m.sum(pd))
    m.Equation(m.sum(pg_feasible) == m.sum(pd))
    # m.Equation(m.sum(pg) == m.sum(pd))
    
    # slack bus violation
    Slack_Pg_violation = m.Array(m.Var,(1,))
    Slack_Pg_violation[0].lower = Lower_bound
    Slack_Pg_violation[0].upper = Upper_bound
    
    Slack_Pg_Violation_temp = [None]*2
    # Slack_Pg_Violation_temp[0] = m.Intermediate(m.max2(pg[Slack_bus]-Power_Upbound_Gen_Slack,0))
    # Slack_Pg_Violation_temp[1] = m.Intermediate(m.max2(Power_Lowbound_Gen_Slack-pg[Slack_bus],0))
    Slack_Pg_Violation_temp[0] = m.Intermediate(pg[Slack_bus])
    # Slack_Pg_Violation_temp[1] = m.Intermediate(-pg[Slack_bus])
    m.Equation(Slack_Pg_violation[0] == Slack_Pg_Violation_temp[0])   
    m.Obj(Slack_Pg_violation[0])
#
#    # minimize objective
#    m.solve()
#    pd_solution = []
#    for i in range(0,mL):
#        pd_solution.append(pd[i].value[0])
#    pg_solution = []
#    for i in range(0,mL):
#        pg_solution.append(pg[i].value[0])
#
#    print(Slack_Pg_violation[0].value)
#    print('\n')    
#    print(pg[Slack_bus].value)
#    print('\n') 
#    print(Pg[0].value)
    ##########################################################
    #Branch limit
    Power_bound_Branch = np.zeros((mB,),dtype='float32')
    Power_bound_Branch = branch[:,5]/standval*1.017

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

    PTDF_ori = np.dot(Bf.toarray(),Bus_admittance_full)
    Branch_feasible=m.Array(m.Var,(mB,))
    for i in range(0, mB):
        m.Equation(Branch_feasible[i]==sum([PTDF_ori[i,j]*(pg_feasible[j]-pd[j]) for j in range(0,mL)]))
        Branch_feasible[i].lower=-Power_bound_Branch[i]
        Branch_feasible[i].upper=Power_bound_Branch[i]
    #branch flow violation
    # Branch_index = 1
    Branch_flow_violation = m.Array(m.Var,(1,))
    Branch_flow_violation[0].lower = Lower_bound
    Branch_flow_violation[0].upper = Upper_bound
    
    Branch_flow_Violation_temp = [None]*2
    # Branch_flow_Violation_temp[0] = m.Intermediate(m.max2(sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)])-Power_bound_Branch[Branch_index],0))
    # Branch_flow_Violation_temp[1] = m.Intermediate(m.max2(-Power_bound_Branch[Branch_index]-sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)]),0))
    Branch_flow_Violation_temp[0] = m.Intermediate(sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)]))
    Branch_flow_Violation_temp[1] = m.Intermediate(-sum([PTDF_ori[Branch_index,j]*(pg[j]-pd[j]) for j in range(0,mL)]))
    m.Equation(Branch_flow_violation[0] == Branch_flow_Violation_temp[0])   
    # m.Obj(Branch_flow_violation[0])

    # minimize objective
    m.solve()
    pd_solution = []
    for i in range(0,mL):
        pd_solution.append(pd[i].value[0])
    pg_solution = []
    for i in range(0,mL):
        pg_solution.append(pg[i].value[0])
        
    Slacklimit1.append(Power_Upbound_Gen_Slack)
    Slacklimit2.append(Power_Lowbound_Gen_Slack)
    print(Slack_Pg_violation[0].value)
    # print(Branch_flow_violation[0].value)
    print(pg_solution)
    print(pd_solution)
    # print(State)
    print('\n')
    # Branchresult2.append(Branch_flow_violation[0].value)
    Slackresult2.append(Slack_Pg_violation[0].value)
    # Actual_branch_flow = np.zeros((1,),dtype='float32')
    # for i in range(0,mL):
    #    Actual_branch_flow[0] += PTDF_ori[Branch_index,i]*(pg[i].value[0]-pd[i].value[0])
    # print(Actual_branch_flow[0])
    # print('\n')    
    

    
if __name__ == '__main__':
    # Initialize Model
#     m = GEKKO(remote=False)
# #    m.options.RTOL=1e-1
# #    m.options.OTOL=1e-1
# #    m.options.SCALING=0
# #    m.options.MAX_ITER = 500
#     m.options.SOLVER = 1
    Branchresult1=[];
    Branchresult2=[];
    Slackresult1=[];
    Slackresult2=[];
    Branchlimit=[];
    Slacklimit1=[];
    Slacklimit2=[];
    # branch violation
    for i in range(0,41):
        
        m = GEKKO(remote=False)
        m.options.SOLVER = 3
        m.options.MAX_ITER = 1000
        Branch_index=i
        mpc = loadcase('case30_rev_100.py')
        worst_case_prediction_error_branch1(m,mpc)
        del m, mpc
        
        m = GEKKO(remote=False)
        m.options.SOLVER = 3
        m.options.MAX_ITER = 1000
        Branch_index=i
        mpc = loadcase('case30_rev_100.py')
        worst_case_prediction_error_branch2(m,mpc)
        del m, mpc
    # slack violation    
    m = GEKKO(remote=False)
    m.options.SOLVER = 3
    m.options.MAX_ITER = 1000
    Branch_index=i
    mpc = loadcase('case30_rev_100.py')
    worst_case_prediction_error_slack1(m,mpc)
    del m, mpc
        
    m = GEKKO(remote=False)
    m.options.SOLVER = 3
    m.options.MAX_ITER = 1000
    Branch_index=i
    mpc = loadcase('case30_rev_100.py')
    worst_case_prediction_error_slack2(m,mpc)
    del m, mpc
        
        
    Slacklimit1=np.array(Slacklimit1).T
    Slacklimit2=np.array(Slacklimit2).T
    Branchlimit=np.array(Branchlimit).T
    Branchresult1=abs(np.array(Branchresult1)).flatten()-Branchlimit
    Branchresult2=abs(np.array(Branchresult2)).flatten()-Branchlimit
    Slackresult1=(np.array(Slackresult1))-Slacklimit1
    Slackresult2=(np.array(Slackresult2))-Slacklimit2
    Final_branch=np.zeros((len(Branchresult1),),dtype='float32')
    Final_slack=np.zeros((len(Slackresult1),2),dtype='float32')
    
    for i in range(0, len(Branchresult1)):
        if (Branchresult1[i]>0 or Branchresult2[i]>0):
            Final_branch[i]=max(Branchresult1[i],Branchresult2[i])
        else:
            Final_branch[i]=0
    for i in range(0, len(Slackresult1)):
        if Slackresult1[i]>0:
            Final_slack[i,0]=Slackresult1[i]
        else:
            Final_slack[i,0]=0
        if (Slackresult2[i]<0):
            Final_slack[i,1]=Slackresult2[i]
        else:
            Final_slack[i,1]=0
   
    
    # filename1='Finalbranch30.txt'
    # fid1=open(filename1,'a')
    # fid1.write("%9.5f;" % (Final_branch))       
    # fid1.write("\n")
    # fid1.close()
    
    # filename2='Finalbranch30.txt'
    # fid2=open(filename2,'a')
    # fid2.write("%9.5f;" % (Final_slack))       
    # fid2.write("\n")
    # fid2.close()
    
    np.savetxt("Finalbranch30.txt", Final_branch, fmt="%s")
    np.savetxt("Finalslack30.txt", Final_slack, fmt="%s")
    

    
#    m.options.IMODE = 3
    #preparing model
#     mpc = loadcase('pglib_opf_case30_ieee_rev.py')
# #    mpc['bus'][:, 2] = mpc['bus'][:, 2]*0.5
# #    mpc['bus'][:, 3] = mpc['bus'][:, 3]*0.5
# #    mpc['bus'][:, 4] = 0
# #    mpc['bus'][:, 5] = 0
#     worst_case_prediction_error_slack_bus_or_branch(m,mpc)
#     del m, mpc