#DC-OPF
from gekko import GEKKO
from scipy.sparse import csr_matrix as sparse
from pypower.loadcase import loadcase
from pypower.rundcopf import rundcopf
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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

default=1.15
input_range = 0.15
Lower_bound = -1e6
Upper_bound = 1e6
NUM=20000



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
    
    
class Neuralnetwork2(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim):
        super(Neuralnetwork2, self).__init__()
        self.layer1 = nn.Linear(in_dim, 4*n_hidden)
        self.layer1.weight = torch.nn.Parameter(torch.Tensor(weight_matrix[0]))
        self.layer1.bias = torch.nn.Parameter(torch.Tensor(bias_vector[0]))
        self.layer2 = nn.Linear(4*n_hidden, 2*n_hidden)
        self.layer2.weight = torch.nn.Parameter(torch.Tensor(weight_matrix[1]))
        self.layer2.bias = torch.nn.Parameter(torch.Tensor(bias_vector[1]))
        self.layer3 = nn.Linear(2*n_hidden, n_hidden)
        self.layer3.weight = torch.nn.Parameter(torch.Tensor(weight_matrix[2]))
        self.layer3.bias = torch.nn.Parameter(torch.Tensor(bias_vector[2]))
        self.layer4 = nn.Linear(n_hidden, out_dim)
        self.layer4.weight = torch.nn.Parameter(torch.Tensor(weight_matrix[3]))
        self.layer4.bias = torch.nn.Parameter(torch.Tensor(bias_vector[3]))


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
    
class Penalty(nn.Module):
    def __init__(self):
        super(Penalty, self).__init__()
    def forward(self, pred):
        hinge_loss1 = abs(pred[0:mB])-1
        hinge_loss2 = (pred[mB:mB+1])-1
        hinge_loss3 = (pred[mB+1:mB+2])
        hinge_loss=torch.cat((hinge_loss1,hinge_loss2,hinge_loss3), 0)
        # print(hinge_loss1)
        # print(hinge_loss2)
        # print(hinge_loss3)
        # print(hinge_loss)
        # hinge_loss[hinge_loss < 0] = 0
        return torch.max(hinge_loss)
    

    
if __name__ == '__main__':
    # Initialize Model

    epsilon=0.0001
    alpha=0.002
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
    
    
  
    FF=[];
    P_update=[];   
    WEIGHT=[];
    BIAS=[];
    WEIGHT2=[];
    BIAS2=[];
    pd_final=[];
    PD_FINAL=[];
    PD_FINAL2=[];
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
    P_D_normalization=[]
    P_D_test_normolization = (Input_data - MEAN) / (STD+1e-8)
    P_D_normalization=P_D_test_normolization.tolist()
    pd_temp0=np.ones(20)*1.7320508;
    for i in range(0,21):
        P_D_normalization.append(pd_temp0*(0.1*i-1))
        
    for i in range(0,500):
        P_D_normalization.append(2*(random.random()-0.5)*pd_temp0)
    
    for i in range(0,1200):
        P_D_normalization.append(pd_temp0*(2*(np.random.uniform(0,1,len(pd_temp0))-0.5)))

    FFALL=[];
    for IRound in range(0,400):
        if IRound>0:
	        dir = './S-PTH/'
	        for SOME in range(0,1000):
	            if os.access(dir+'s-TEST_'+str(IRound)+'.pth', os.F_OK):
	                time.sleep(0)
	            else:
	                time.sleep(1)
        
        
        pd_final_temp=[];
                           
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

        
        start2=time.time()
        
        layers=[20, 32, 16, 8, 5]
        
        weight=np.array(WEIGHT[IRound])
        bias=np.array(BIAS[IRound])
        
        weight_matrix=[];
        weight_matrix.append(np.array(weight[0:layers[0]*layers[1]]).reshape((layers[1],layers[0])))
        weight_matrix.append(np.array(weight[layers[0]*layers[1]:layers[0]*layers[1]+layers[1]*layers[2]]).reshape((layers[2],layers[1])))
        weight_matrix.append(np.array(weight[layers[0]*layers[1]+layers[1]*layers[2]:layers[0]*layers[1]+layers[1]*layers[2]+layers[2]*layers[3]]).reshape((layers[3],layers[2])))
        weight_matrix.append(np.array(weight[layers[0]*layers[1]+layers[1]*layers[2]+layers[2]*layers[3]:layers[0]*layers[1]+layers[1]*layers[2]+layers[2]*layers[3]+layers[3]*layers[4]]).reshape((layers[4],layers[3])))
            
        bias_vector=[];
        bias_vector.append(np.array(bias[0:layers[1]]))
        bias_vector.append(np.array(bias[layers[1]:layers[1]+layers[2]]))
        bias_vector.append(np.array(bias[layers[1]+layers[2]:layers[1]+layers[2]+layers[3]]))
        bias_vector.append(np.array(bias[layers[1]+layers[2]+layers[3]:layers[1]+layers[2]+layers[3]+layers[4]]))


        numberOfWeight = 0
        for i in range(0,len(layers)-1,1):
            numberOfWeight += layers[i]*layers[i+1]
        numberOfBias = sum(layers[1:len(layers)])
        numberOfStateVariable = numberOfBias
        numberOfNeuronOuput = numberOfStateVariable
        numberOfInput = layers[0]
        COMPARE=[]
        indexxx=0
        for kkk in range(2*int(len(P_D_normalization)/4),3*int(len(P_D_normalization)/4)):
            
            if kkk % 1000==0 and kkk !=0:
                indexxx=indexxx+1
                print(indexxx)
          
            NeuronOutput = np.zeros((numberOfNeuronOuput,),dtype='float32')
        
            Input=P_D_normalization[kkk]
            for i in range(0,layers[1]):
            
                NeuronOutput[i]=max(sum([Input[j]*weight[i*numberOfInput+j] for j in range(0,numberOfInput)])+bias[i],0)
            
            weight_index = layers[0]*layers[1]
            #for hidden layers
            for layer_index in range(2,len(layers)-1,1):
                previous_index = sum(layers[1:layer_index-1])
                current_index = sum(layers[1:layer_index])
                for i in range(0,layers[layer_index]):
                    
                    NeuronOutput[current_index+i] =max(sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index-1])])+bias[current_index+i],0)
                    weight_index += layers[layer_index-1]
            
            #from hidden layer to output
            previous_index = sum(layers[1:len(layers)-2])
            current_index = sum(layers[1:len(layers)-1])   
            
            for i in range(0,layers[len(layers)-1]):
               
                NeuronOutput[current_index+i]= sum([NeuronOutput[previous_index+j]*weight[weight_index+j] for j in range(0,layers[layer_index])])+bias[current_index+i] 
               
                weight_index += layers[len(layers)-2]
               
            current_index = sum(layers[1:len(layers)-1])    
            #NN output
            NNOutput = np.zeros((layers[len(layers)-1],),dtype='float32')
        
            for i in range(0,layers[len(layers)-1]):
    
                #sigmoid
                NNOutput[i] = 1/(1+ math.exp(-NeuronOutput[current_index+i]))
            
            #Pg
            Pg = np.zeros((len(Gen_index_list),),dtype='float32')
            
            for i in range(0,len(Gen_index_list)):
                  
                Pg[i]=NNOutput[i]*(Power_Upbound_Gen[i]-Power_Lowbound_Gen[i])+Power_Lowbound_Gen[i]
            
            #DC-PF equations
            # Create variables
            pg = np.zeros((mL,),dtype='float32')
            pd = np.zeros((mL,),dtype='float32')
            gen_index = 0
            load_index = 0
            for i in range(0,mL,1):
                if i in Load_index_list:
                    pd[i] = Input[load_index]*((0.3*abs(bus[i,2])/standval)/12**0.5)+bus[i,2]/standval*(default)
                    load_index += 1
                else:
                    pd[i] = 0
            #            m.Equation(pd[i] == 0)
                if bus[i,1] == 3:
                    pg[i]=0
                elif bus[i,1] == 2 and i in Gen_index_list:
                    # pg[i].lower = gen[gen_index,9]/standval
                    # pg[i].upper = gen[gen_index,8]/standval
                    pg[i] = Pg[gen_index]
                    gen_index += 1                
                else:
                    pg[i]= 0
                    
            slack=sum(pd)-sum(pg)
            
            for i in range(0,mL,1):
                if bus[i,1] == 3:
                    pg[i]=slack
                
                    
    
            totalbranch_temp=np.zeros((len(branch_list)+2,),dtype='float32')
            for i in range(0,len(branch_list)):
                totalbranch_temp[i]=(abs(sum([PTDF_ori[branch_list[i],j]*(pg[j]-pd[j]) for j in range(0,mL)]))-Power_bound_Branch[branch_list[i]])/Power_bound_Branch[branch_list[i]]
            
            totalbranch_temp[len(branch_list)]=(sum(Pg)-sum(pd))/Power_Upbound_Gen_Slack
            totalbranch_temp[len(branch_list)+1]=(sum(pd)-sum(Pg)-Power_Upbound_Gen_Slack)/Power_Upbound_Gen_Slack
            
            compare_temp=np.zeros((len(branch_list)+2,),dtype='float32')
            
            compare_temp[0]=totalbranch_temp[0]
            for i in range(1, len(branch_list)+2):
                compare_temp[i]=max(compare_temp[i-1],totalbranch_temp[i])
                
            COMPARE.append(compare_temp[len(branch_list)+1])
        
        dir='./S-COMPARE/' 
        np.savetxt(dir+'compare3.txt', np.array(COMPARE), fmt="%s")
        np.savetxt(dir+'Finish_3_'+str(IRound)+'.txt', np.array([1]), fmt="%s")
        
        for wait in range(0,1000):       
            dir = './S-TEST/'
            increase=0
            increasess=0
            for SOME in range(IRound*42,IRound*42+42):          
                if os.access(dir+'s-TEST_'+str(SOME)+'.txt', os.F_OK):
                    increase=increase+1  
            
            # dir = './TEST/'
            # for SOME in range(IRound*42,IRound*42+42):          
            #     if os.access(dir+'TEST_'+str(SOME)+'.txt', os.F_OK):
            #         increasess=increasess+1 
                    
                    
                    
            if increase==42:
                time.sleep(0)
            else:
                time.sleep(100)
            

        # dir = './TEST/'
        # increase2=0
        # for TEST in range(IRound*42,IRound*42+42):
        #     if os.access(dir+'TEST_'+str(TEST)+'.txt', os.F_OK):
        #         increase2=increase2+1
        #         f = open(dir+'TEST_'+str(TEST)+'.txt')
        #         line = f.readline()
        #         data_list = []
        #         while line:
        #             num = list(map(float,line.split()))
        #             data_list.append(num)
        #             line = f.readline()
        #         f.close()
        #         pd = np.array(data_list)
        #         pd = pd.reshape(len(pd),)
        #         P_D_normalization.append(pd)               
                
        
        dir = './S-TEST/' 
        increase3=0
        for STEST in range(IRound*42,IRound*42+42):          
            if os.access(dir+'s-TEST_'+str(STEST)+'.txt', os.F_OK):
                increase3=increase3+1                             
                f = open(dir+'s-TEST_'+str(STEST)+'.txt')
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
                pd_final_temp.append(pd)
        
        