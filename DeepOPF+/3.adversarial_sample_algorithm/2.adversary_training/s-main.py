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
default=1.15
iteration_time=20
input_range = 0.15
Lower_bound = -1e6
Upper_bound = 1e6
NN_input_number = 0 
NN_output_number = 0  

batch_size_training = 8
batch_size_valtest = 1  
neural_number = 8

#learning parameters
learning_rate = 1e-4
num_epoches = 10
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
Bus_admittance_full_NN2 = np.zeros([1,1],dtype='float32')
Bus_admittance_full = np.zeros([1,1],dtype='float32')
Bus_admittance_full_1 = np.zeros([1,1],dtype='float32')
Node_index = np.zeros([1,1],dtype='float32')
Node_angel = np.zeros([1,1],dtype='float32')
Load_index = np.zeros([1,1],dtype='float32')
Gen_index = np.zeros([1,1],dtype='float32')
Gen_Price = np.zeros([1,1],dtype='float32')


#SCOPF
SC_list = []
conflict_list = []
filter_list = []
NumofContigengcy = 0
SC_Bus_admittance_full_1_array = np.zeros((1,1,1),dtype='float32')
SC_Bus_admittance_line_1_array = np.zeros((1,1,1),dtype='float32')

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
        hinge_loss = pred**2-1
        hinge_loss[hinge_loss < 0] = 0
        return torch.mean(hinge_loss)
        #return torch.mean(max(0,pred**2-1))
        
class Penalty2(nn.Module):
    def __init__(self):
        super(Penalty2, self).__init__()
    def forward(self, pred):
        hinge_loss = pred
        hinge_loss[hinge_loss > 0] = 0
        # print(hinge_loss)
        hinge_loss=abs(hinge_loss)
        return torch.mean(hinge_loss)
            
        
class Penalty3(nn.Module):
    def __init__(self):
        super(Penalty3, self).__init__()
    def forward(self, pred):
        hinge_loss = pred-1
        hinge_loss[hinge_loss <0] = 0
        return torch.mean(hinge_loss)
    

    
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
    
    
  
    FF=[];
    P_update=[];   
    WEIGHT=[];
    BIAS=[];
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
        
    
    print(len(P_D_normalization))
    
    Feasible=[];
    FFALL=[];
    for IRound in range(0,400):
        
        pd_final_temp=[];
                
        
        for wait in range(0,1000):       
            dir = './S-TEST/'
            increase=0
            for SOME in range(IRound*42,IRound*42+42):          
                if os.access(dir+'s-TEST_'+str(SOME)+'.txt', os.F_OK):
                    increase=increase+1                
            if increase==42:
                time.sleep(0)
            else:
                time.sleep(100)
            
                
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
               
        filenameincrease= 's-increase.txt'
        fidin=open(filenameincrease,'at')
        fidin.write("%9.4f;" % (increase3))
        fidin.write('\n')
        fidin.close()

                
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
        # MEAN=[];
        # STD=[];
        # for i in range(0,30):
        #     if i in Load_index_list:
        #         MEAN.append(mpc_bus[i,2]*1.15/100)
                
        # for i in range(0,30):
        #     if i in Load_index_list:
        #         STD.append((mpc_bus[i,2]*0.3/12**0.5)/100)
                
        
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
        for kkk in range(0,len(P_D_normalization)):
            
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
            
            if kkk>=len(P_D_normalization)-42:
                filenametest= 's-max_test.txt'
                fidtest=open(filenametest,'at')
                fidtest.write("%9.4f;" % (compare_temp[len(branch_list)+1]))
                fidtest.write("%9.4f;" % (kkk))
                fidtest.write('\n')
                fidtest.close()
                
            if kkk==len(P_D_normalization)-1:
                filenametest= 's-max_test.txt'
                fidtest=open(filenametest,'at')
                fidtest.write('Finish'+str(IRound)+'************************')
                fidtest.write('\n')
                fidtest.close()
                
        
        PD_FINAL.append(P_D_normalization[COMPARE.index(max(COMPARE))])
        FFALL.append(max(COMPARE))
        
        
        
        
        COMPARE2=[]
        for kkk2 in range(0,len(pd_final_temp)):
            
          
            NeuronOutput = np.zeros((numberOfNeuronOuput,),dtype='float32')
        
            Input=pd_final_temp[kkk2]
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
                
            COMPARE2.append(compare_temp[len(branch_list)+1])
            
                
        
        pd_final.append(pd_final_temp[COMPARE2.index(max(COMPARE2))])
        dir='./S-pdfinal/'       
        np.savetxt(dir+'pd_final_'+str(IRound)+'.txt', np.array(pd_final[IRound]), fmt="%s")
        
        filenamesolution= 's-total30_all-final.txt'
        fidsolution=open(filenamesolution,'at')
        fidsolution.write("%9.4f;" % (max(COMPARE2)))
        fidsolution.write('\n')
        fidsolution.close()
    
    
        filenameindex= 's-index.txt'
        fidindex=open(filenameindex,'at')
        fidindex.write("%9.4f;" % (COMPARE2.index(max(COMPARE2))))
        fidindex.write('\n')
        fidindex.close()
        
        
        
        Input= np.array(P_D_normalization[COMPARE.index(max(COMPARE))]).astype(np.float32)
        
        filenamefind= 's-max_find.txt'
        fidfind=open(filenamefind,'at')
        fidfind.write("%9.4f;" % (COMPARE.index(max(COMPARE))))
        fidfind.write("%9.4f;" % (max(COMPARE)))
        fidfind.write("%9.4f;" % (len(P_D_normalization)))
        fidfind.write('\n')
        fidfind.close()
        
        pd = np.zeros((30,),dtype='float32')
            
        load_index = 0
        for i in range(0,mL,1):
            if i in Load_index_list:
                pd[i] = Input[load_index]*((0.3*abs(bus[i,2])/standval)/12**0.5)+bus[i,2]/standval*(default)
                load_index += 1
            else:
                pd[i] = 0

        mpc = loadcase('case30_rev_100.py')
        mpc_bus=mpc['bus']
        [m,n] = mpc_bus.shape        
        tempload=pd*100
        tempload2=np.zeros((30,),dtype='float32')                     
        mpc_bus[:,2] = tempload
        mpc_branch=mpc['branch']
        mpc['gen'][:,8]=mpc['gen'][:,8]
        mpc_branch[:,5]=mpc_branch[:,5]*1.017
        
        r = rundcopf(mpc)        
          
        if (r['success']== True):
            filenamesuccess= 'TEST_success.txt'
            fidsuccess=open(filenamesuccess,'at')
            fidsuccess.write("%9.4f;" % (0))
            fidsuccess.write('\n')
            fidsuccess.close()
            PD_FINAL2.append(Input)
            dir='./LPDFINAL/'       
            np.savetxt(dir+'PD_FINAL_'+str(IRound)+'.txt', np.array(PD_FINAL2[IRound]), fmt="%s")
        else:
            filenamesuccess= 'TEST_success.txt'
            fidsuccess=open(filenamesuccess,'at')
            fidsuccess.write("%9.4f;" % (1))
            fidsuccess.write('\n')
            fidsuccess.close()
            PD_FINAL2.append(pd_temp0)
            Input=PD_FINAL2[IRound]
            dir='./LPDFINAL/'       
            np.savetxt(dir+'PD_FINAL_'+str(IRound)+'.txt', np.array(PD_FINAL2[IRound]), fmt="%s")          
        r=[]
        
        
        pd2 = np.zeros((30,),dtype='float32')
            
        load_index = 0
        for i in range(0,mL,1):
            if i in Load_index_list:
                pd2[i] = Input[load_index]*((0.3*abs(bus[i,2])/standval)/12**0.5)+bus[i,2]/standval*(default)
                load_index += 1
            else:
                pd2[i] = 0
                
                
        
        for index in range(IRound+5,IRound+6,1):
            for kkk in range(0,100):
            
                filename7= 'Pre_DC_training_case30-sampling_part_0.93_'+str(index)+'.txt'
                fid7=open(filename7,'at')
                
                filename5= 'Success_0.93_'+str(index)+'.txt'
                fid5=open(filename5,'at')
    
                #########load:1.1-1.2,branch:0.9
                
                mpc7 = loadcase('case30_rev_070.py')
                mpc7_bus=mpc7['bus']
                [m,n] = mpc7_bus.shape
                if kkk<=98:
                    tempload=pd2*100*(0.02*(np.random.uniform(0,1,30)-0.5)+1)
                else:
                    tempload=pd2*100
                tempload2=np.zeros((30,),dtype='float32')
                for i in range(0,30):
                    if tempload[i]>=mpc7_bus[i,2]*1.3 and tempload[i]>=0:
                        tempload2[i]=mpc7_bus[i,2]*1.3
                    if tempload[i]<=mpc7_bus[i,2] and tempload[i]>=0:
                        tempload2[i]=mpc7_bus[i,2]
                    if tempload[i]<=mpc7_bus[i,2]*1.3 and tempload[i]<=0:
                        tempload2[i]=mpc7_bus[i,2]*1.3
                    if tempload[i]>=mpc7_bus[i,2] and tempload[i]<=0:
                        tempload2[i]=mpc7_bus[i,2]
                    if tempload[i]>=mpc7_bus[i,2] and tempload[i]<=mpc7_bus[i,2]*1.3 and tempload[i]>=0:
                        tempload2[i]=tempload[i]
                    if tempload[i]<=mpc7_bus[i,2] and tempload[i]>=mpc7_bus[i,2]*1.3 and tempload[i]<=0:
                        tempload2[i]=tempload[i]
                        
                        
                mpc7_bus[:,2] = tempload2
                mpc7_branch=mpc7['branch']
                mpc7['gen'][:,8]=mpc7['gen'][:,8]
                factor=0.07
                mpc7_branch[:,5]=mpc7_branch[:,5]*1.017-branch_capacity*factor*100
                
                r7 = rundcopf(mpc7)        
                
                r7_bus = r7['bus']
                for j in range(0,m,1):
                    
                    if(r7_bus[j,2]*r7_bus[j,2] > 0):
                        if j != m-1:
                            fid7.write("%9.4f;" % (r7_bus[j,2]))                   
                        else:
                            fid7.write("%9.4f\n" % (r7_bus[j,2]))
        
                
                r7_gen=r7['gen']
                m1 = len(r7_gen)
                
                for j in range(0,m1,1):
                    if j != m1-1:
                        fid7.write("%d:%9.4f;" % (int(r7_gen[j,0]),r7_gen[j,1]))
                    else:
                        fid7.write("%d:%9.4f\n" % (int(r7_gen[j,0]),r7_gen[j,1]))
                
                    
                if (r7['success']== True):
                    fid5.write("%d" % (0))
                else:
                    fid5.write("%d" % (1))
                    
                r7=[]
            
            fid5.close()
            fid7.close()
            
               # model = Neuralnetwork(Load_index.shape[0], 32, Gen_index.shape[0])
        # parameter
        torch.manual_seed(1)    # reproducible
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
        
        mpc = loadcase('case30_rev_100.py')
        mpc['gen'][:,8]=mpc['gen'][:,8]
        mpc['branch'][:,5]=mpc['branch'][:,5]*1.017
        #reading parameters

        #system MVA base
        standval = mpc["baseMVA"]
        #bus(load)
        mpc_bus=mpc["bus"]
         
        
        #branch
        mpc_branch = mpc["branch"]
        
        #genrator
        mpc_gen=mpc["gen"]
        
        ## switch to internal bus numbering and build admittance matrices
        _, bus, gen, branch = ext2int1(mpc_bus, mpc_gen, mpc_branch)
        
        [mL,nL]=bus.shape 
        [mB,nB]=branch.shape
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
        nG = gen.shape[1]
    
        # bus = mpc_bus
        # branch = mpc_branch      
    
        
        
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
    

        Gen_index = np.array(Gen_index_list)
        Load_index = np.array(Load_index_list)
    
         
        NN_output_number = mG-1
        NN_input_number = Load_index.shape[0]
        
        
        Branch_number = mB
        
        Bus_admittance_full = np.zeros([mL,mL],dtype='float32')
        
        Bus_admittance_line = np.zeros([mB,mL],dtype='float32') 
        
        #DC-OPF
        ## power mismatch constraints
        B, Bf, Pbusinj, Pfinj = makeBdc(standval, bus, branch)   
        Bus_admittance_full[:,0:mL] = B.toarray()
        Bus_admittance_line[:,0:mL] = Bf.toarray()
    
        for i in range(0,mB,1):
            Bus_admittance_line[i,:] = Bus_admittance_line[i,:] / (branch[i,5]/standval)
    
        
        Bus_admittance_full_1 = Bus_admittance_full[0:Slack_bus,0:Slack_bus]
        Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Bus_admittance_full[0:Slack_bus,Slack_bus+1:Bus_number]),axis=1)
        Temp = Bus_admittance_full[Slack_bus+1:Bus_number,0:Slack_bus]
        Temp = np.concatenate((Temp,Bus_admittance_full[Slack_bus+1:Bus_number,Slack_bus+1:Bus_number]),axis=1)
        Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Temp),axis=0)    
    
    
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
        
        mpc_price = mpc['gencost']
        [mpri,npri] = mpc_price.shape
        
        Gen_Price = np.zeros([mG,npri-4],dtype='float32')
        index = 0
        for i in range(0, mpc_gen.shape[0], 1):
            if mpc_gen[i,7] !=0:
                Gen_Price[index,0:npri-4] = mpc_price[i,4:4+npri-3]
                index = index +1
        Gen_Price
        # global data
        P_D_list = []
        P_G_list = []
        P_D_list_test = []
        P_G_list_test = []    
        Pre=[]
        #read training data
        numOfdata = 0
        
        for index in range(0, IRound+5, 1):
            filename= 'Pre_DC_training_case30-sampling_part_0.93_'+str(index)+'.txt'
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
                                # ind = np.where(mpc['gen'][:,0] == int(Gen_data[0]))[0][0]
                                # if mpc['gen'][ind,7] != 0:
                                P_G_list.append(float(Gen_data[1])/standval)
                        j = j + 1
        ################################################################
        del data
        num_of_group = int(np.floor(numOfdata / 10))
        P_D = np.array(P_D_list, dtype=np.float32)
        P_D = P_D.reshape(-1, NN_input_number)
        P_G = np.array(P_G_list, dtype=np.float32)
        P_G = P_G.reshape(-1, NN_output_number+1)
        
        NEW=np.hstack((P_D,P_G)).tolist()
        random.shuffle(NEW)
        NEW=np.array(NEW)
        P_D=NEW[:,0:NN_input_number].astype(np.float32)
        P_G=NEW[:,NN_input_number:NN_input_number+NN_output_number+1].astype(np.float32)
        
        # Preprocessing training data
        P_D_train = P_D[0:num_of_group * 8]
        # P_D_train_mean = np.mean(P_D_train, axis=0)
        # P_D_train_std = np.std(P_D_train, axis=0)
        P_D_train_mean = MEAN
        P_D_train_std = STD
        P_D_train_std_tensor = torch.from_numpy(P_D_train_std) 
        P_D_train_mean_tensor = torch.from_numpy(P_D_train_mean) 
    #    del P_D_list, P_G_list
        #test inversve matrix of B (default DCOPF)
        Bus_1_Coeff = np.linalg.inv(Bus_admittance_full_1)
        Bus_1_Coeff_tensor = torch.from_numpy(Bus_1_Coeff) 
        ################################################################
        # restore net
        if IRound==0:
            dir ='./S-PTH/'
            model = Neuralnetwork(20, 8, 5)
            model.load_state_dict(torch.load(dir+'PreDCOPF_case30_0.93_dnn.pth',map_location='cpu'))     
        else:
            dir ='./S-PTH/'
            model = Neuralnetwork(20, 8, 5)
            model.load_state_dict(torch.load(dir+'PreDCOPF_case30_0.93_dnn_'+str(IRound)+'.pth',map_location='cpu'))   
        #if torch.cuda.is_available():
        #model = model.cuda()
        #read test data
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
                        P_D_list_test.append(float(odom[i])/standval)
                else:
                    odom  =line.split(';')  # supply
                    for i in range(len(odom)):
                        Gen_data = odom[i].split(':')
                        # ind = np.where(mpc['gen'][:,0] == int(Gen_data[0]))[0][0]
                        # if mpc['gen'][ind,7] != 0:
                        P_G_list_test.append(float(Gen_data[1])/standval)                
                j = j + 1
            fid.close()
        ################################################################
        del data
        Input_data = np.array(P_D_list_test, dtype=np.float32).reshape(-1, NN_input_number)
        Output_data = np.array(P_G_list_test, dtype=np.float32).reshape(-1, NN_output_number+1)         
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
        
        pg_error = 0
        PDM_test = torch.zeros([NN_input_number,Bus_number])
        PGM_test = torch.zeros([NN_output_number+1,Bus_number])
        Bus_admittance_line_tensor = torch.from_numpy(Bus_admittance_line)
        cost_list = []
        feasible1_test=0
        feasible2_test=0
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

    
            slack_angle_array_test = np.ones([1,len(test_x)], dtype='float32')*Slack_angle
            slack_angle_array_tensor_test = torch.from_numpy(slack_angle_array_test)
    
            start1 = time.time()
            theta_result_output_test = theta_1_result_output_test[0:Slack_bus,:]
            theta_result_output_test = torch.cat([theta_result_output_test, slack_angle_array_tensor_test],0)
            theta_result_output_test = torch.cat([theta_result_output_test, theta_1_result_output_test[Slack_bus:Bus_number-1,:]],0)
    
            trans_result_output_test = (Bus_admittance_line_tensor.mm(theta_result_output_test).t()) 
            trans_result_output_test_np = trans_result_output_test.cpu().detach().numpy()
    
            infeasible_index = np.where((trans_result_output_test_np > 1) | (trans_result_output_test_np < -1))

            
            
            
            if len(infeasible_index[0]) >0:
                # fid1.write("%d;" % (1))
                feasible1_test = feasible1_test+0
            else:
                # fid1.write("%d;" % (0))
                feasible1_test = feasible1_test+1
            Pred_Pg_test_np = Pred_Pg_test_rev.detach().numpy()
            if (Pred_Pg_test_np[0,Slack_index] >  Power_Upbound_Gen[0,Slack_index]) or (Pred_Pg_test_np[0,Slack_index] <  Power_Lowbound_Gen[0,Slack_index]):
                feasible2_test = feasible2_test+0
                # fid1.write("%d;" % (1))
            else:
                feasible2_test = feasible2_test+1
                
            # print(feasible1_test+feasible2_test)
        
        Feasible.append(feasible1_test+feasible2_test)
        
        filenamefeasibleb= 'feasiblecheck_branch.txt'
        fidfeasibleb=open(filenamefeasibleb,'at')
        fidfeasibleb.write("%9.4f;" % (feasible1_test))
        fidfeasibleb.write('\n')
        fidfeasibleb.close()
        
        filenamefeasibles= 'feasiblecheck_slack.txt'
        fidfeasibles=open(filenamefeasibles,'at')
        fidfeasibles.write("%9.4f;" % (feasible2_test))
        fidfeasibles.write('\n')
        fidfeasibles.close()
            
        if  FFALL[IRound]<=0:
            weight_out=WEIGHT[IRound]
            bias_out=BIAS[IRound]
            break
            
        train_time=time.time()
        
        for iteration_train in range(0,iteration_time):
            
                   
            mpc = loadcase('case30_rev_070.py')
            mpc['gen'][:,8]=mpc['gen'][:,8]
            
            factor=0.07
        
            mpc['branch'][:,5]=mpc['branch'][:,5]*1.017-branch_capacity*factor*100
        
            #reading parameters
           
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
        
        
            Gen_index = np.array(Gen_index_list)
            Load_index = np.array(Load_index_list)
        
         
            NN_output_number = mG-1
            NN_input_number = Load_index.shape[0]
            
        
            Branch_number = mB
        
            Bus_admittance_full = np.zeros([mL,mL],dtype='float32')
        
            Bus_admittance_full_NN2 = np.zeros([mB,mL],dtype='float32') 
            
            #DC-OPF
            ## power equation constraints
            B, Bf, Pbusinj, Pfinj = makeBdc(standval, bus, branch)   
            Bus_admittance_full[:,0:mL] = B.toarray()
            Bus_admittance_full_NN2[:,0:mL] = Bf.toarray()
        
            for i in range(0,mB,1):
                Bus_admittance_full_NN2[i,:] = Bus_admittance_full_NN2[i,:] / (branch[i,5]/standval) #shrink 0%        
        
            
            Bus_admittance_full_1 = Bus_admittance_full[0:Slack_bus,0:Slack_bus]
            Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Bus_admittance_full[0:Slack_bus,Slack_bus+1:Bus_number]),axis=1)
            Temp = Bus_admittance_full[Slack_bus+1:Bus_number,0:Slack_bus]
            Temp = np.concatenate((Temp,Bus_admittance_full[Slack_bus+1:Bus_number,Slack_bus+1:Bus_number]),axis=1)
            Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Temp),axis=0) 
            # global data
            P_D_list = []
            P_G_list = []
            #read training data
            numOfdata = 0
            for index in range(0, IRound+6, 1):
                filename= 'Pre_DC_training_case30-sampling_part_0.93_'+str(index)+'.txt'
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
                                    ind = np.where(mpc['gen'][:,0] == int(Gen_data[0]))[0][0]
                                    if mpc['gen'][ind,7] != 0:
                                        P_G_list.append(float(Gen_data[1])/standval)
                            j = j + 1
            ################################################################
            del data
            num_of_group = int(np.floor(numOfdata / 10))
            P_D = np.array(P_D_list, dtype=np.float32)
            P_D = P_D.reshape(-1, NN_input_number)
            P_G = np.array(P_G_list, dtype=np.float32)
            P_G = P_G.reshape(-1, NN_output_number+1)
            
            NEW=np.hstack((P_D,P_G)).tolist()
            random.shuffle(NEW)
            NEW=np.array(NEW)
            P_D=NEW[:,0:NN_input_number].astype(np.float32)
            P_G=NEW[:,NN_input_number:NN_input_number+NN_output_number+1].astype(np.float32)
        #    theta = np.array(theta_list, dtype=np.float32).reshape(-1,(NumofContigengcy+1), Bus_number)
            # Preprocessing training data
            P_D_train = P_D[0:num_of_group * 8]
            # P_D_train_mean = np.mean(P_D_train, axis=0)
            # P_D_train_std = np.std(P_D_train, axis=0)
            P_D_train_mean = MEAN
            P_D_train_std = STD
            P_D_train_std_tensor = torch.from_numpy(P_D_train_std) 
            P_D_train_mean_tensor = torch.from_numpy(P_D_train_mean) 
    
            P_D_train_normolization = (P_D_train - P_D_train_mean) / (P_D_train_std+1e-8)
            P_G_train = P_G[0:num_of_group * 8]
            Power_Lowbound_Gen_training = np.tile(Power_Lowbound_Gen,(len(P_D_train),1))
            Power_Upbound_Gen_training= np.tile(Power_Upbound_Gen,(len(P_D_train),1))
            P_G_train_normolization = (P_G_train-Power_Lowbound_Gen_training)/(Power_Upbound_Gen_training-Power_Lowbound_Gen_training)
        
            
            #test inversve matrix of B (default DCOPF)
            Bus_1_Coeff = np.linalg.inv(Bus_admittance_full_1)
            Bus_1_Coeff_tensor = torch.from_numpy(Bus_1_Coeff)    
            
            ########################################
            # Preprocessing validation data
            P_D_val = P_D[num_of_group * 8:len(P_D)]
            P_D_val_normolization = (P_D_val - P_D_train_mean) / (P_D_train_std + 1e-8)
            P_G_val = P_G[num_of_group * 8:len(P_D)]
            Power_Lowbound_Gen_val = np.tile(Power_Lowbound_Gen,(len(P_D_val),1))
            Power_Upbound_Gen_val= np.tile(Power_Upbound_Gen,(len(P_D_val),1))
            P_G_val_normolization = (P_G_val-Power_Lowbound_Gen_val)/(Power_Upbound_Gen_val-Power_Lowbound_Gen_val)
       
        
            ########################################
            model = Neuralnetwork(NN_input_number, neural_number, NN_output_number)
            if IRound==0:
                if iteration_train==0:
                    dir ='./S-PTH/'
                    model = Neuralnetwork(20, 8, 5)
                    model.load_state_dict(torch.load(dir+'PreDCOPF_case30_0.93_dnn.pth',map_location='cpu'))     
                else:
                    dir ='./S-PTH/'
                    model = Neuralnetwork(20, 8, 5)
                    model.load_state_dict(torch.load(dir+'PreDCOPF_case30_0.93_dnn_'+str(IRound+1)+'.pth',map_location='cpu'))     
            else:
                if iteration_train==0:
                    dir ='./S-PTH/'
                    model = Neuralnetwork(20, 8, 5)
                    model.load_state_dict(torch.load(dir+'PreDCOPF_case30_0.93_dnn_'+str(IRound)+'.pth',map_location='cpu'))     
                else:
                    dir ='./S-PTH/'
                    model = Neuralnetwork(20, 8, 5)
                    model.load_state_dict(torch.load(dir+'PreDCOPF_case30_0.93_dnn_'+str(IRound+1)+'.pth',map_location='cpu'))
            
            penalty = Penalty()
            penalty2 = Penalty2()
            penalty3 = Penalty3()
            #neural network parameters setting
            criterion = nn.MSELoss(reduce=True,size_average = True)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            ##########################################
         
            # Training dataset
            P_D_train_normolization_tensor = torch.from_numpy(P_D_train_normolization) 
            P_G_train_normolization_tensor = torch.from_numpy(P_G_train_normolization) 
        #    theta_train_tensor = torch.from_numpy(theta_train) 
            training_dataset = Data.TensorDataset(P_D_train_normolization_tensor, P_G_train_normolization_tensor)
            training_loader = Data.DataLoader(
                dataset=training_dataset,      # torch TensorDataset format
                batch_size=batch_size_training,      # mini batch size
                shuffle=True,
                #num_workers=2,
            )
            
            #Validation dataset
            P_D_val_normolization_tensor = torch.from_numpy(P_D_val_normolization) 
            P_G_val_normolization_tensor = torch.from_numpy(P_G_val_normolization) 
        #    theta_val_tensor = torch.from_numpy(theta_val) 
            validation_dataset = Data.TensorDataset(P_D_val_normolization_tensor, P_G_val_normolization_tensor)
            validation_loader = Data.DataLoader(
                dataset=validation_dataset,      # torch TensorDataset format
                batch_size=len(P_D_val_normolization_tensor),#batch_size_valtest,      # mini batch size
                shuffle=True,                
                #num_workers=2,              
            )
        
            start = time.time()
            total_time = 0
            #Training + validation process
            for epoch in range(num_epoches):
                print('epoch {}'.format(epoch + 1))
                print('*' * 10)
                running_loss = 0.0
        
                for step, (train_x, train_y) in enumerate(training_loader):#, 
                    # feedforward
                    train_out = model(train_x)
                    # calculate the slack bus output
                    Slack_index = np.where(np.array(Gen_index) == Slack_bus)[0][0]
        
                    Power_Lowbound_Gen_nn_train = torch.from_numpy(np.tile(Power_Lowbound_Gen, (len(train_x), 1)))
                    Power_Lowbound_Gen_nn_train_1 = Power_Lowbound_Gen_nn_train[:, 0:Slack_index ]
                    Power_Lowbound_Gen_nn_train_1 = torch.cat([Power_Lowbound_Gen_nn_train_1, Power_Lowbound_Gen_nn_train[:, Slack_index+1:Gen_index.shape[0]]], 1)
        
                    Power_Upbound_Gen_nn_train = torch.from_numpy(np.tile(Power_Upbound_Gen, (len(train_x), 1)))
                    Power_Upbound_Gen_nn_train_1 = Power_Upbound_Gen_nn_train[:, 0:Slack_index]
                    Power_Upbound_Gen_nn_train_1 = torch.cat([Power_Upbound_Gen_nn_train_1, Power_Upbound_Gen_nn_train[:, Slack_index+1:Gen_index.shape[0]]], 1)
        
                    Pred_Pg_train = Power_Lowbound_Gen_nn_train_1 + (Power_Upbound_Gen_nn_train_1 - Power_Lowbound_Gen_nn_train_1) * train_out
        
        #            Actual_PG_train = train_y.mul(Power_Upbound_Gen_nn_train - Power_Lowbound_Gen_nn_train) + Power_Lowbound_Gen_nn_train            
        #            Pred_Pg_train_rev = torch.unsqueeze(torch.sum(Actual_PG_train[:, 0:Gen_index.shape[0]], 1),1) - torch.unsqueeze(torch.sum(Pred_Pg_train, 1), 1)
                    train_D = (train_x* (P_D_train_std_tensor+1e-8) + P_D_train_mean_tensor)
                    Pred_Pg_train_rev = torch.unsqueeze(torch.sum(train_D[:, 0:Load_index.shape[0]], 1),1) - torch.unsqueeze(torch.sum(Pred_Pg_train, 1), 1)
                    Pred_Pg_train_rev = torch.cat([Pred_Pg_train[:, 0:Slack_index], Pred_Pg_train_rev], 1)
                    Pred_Pg_train_rev = torch.cat([Pred_Pg_train_rev, Pred_Pg_train[:, Slack_index:Gen_index.shape[0]]], 1)
                    
                    train_out = (Pred_Pg_train_rev - Power_Lowbound_Gen_nn_train) / (Power_Upbound_Gen_nn_train - Power_Lowbound_Gen_nn_train)
                    kk=train_out*1
                    zz=train_out*1
                    
                    
                    # slack_loss=penalty2(train_out)+penalty3(train_out)
                    
                    train_loss1 = criterion(train_out, train_y)
                    #PG
                    for i in range(Bus_number):
                        if i == 0:
                            if i in Gen_index:
                                index = np.where(np.array(Gen_index)==i)[0]
                                predictionG_train = Pred_Pg_train_rev[:, index]
                            else:
                                predictionG_train = torch.zeros([len(train_x), 1])
                        else:
                            if i in Gen_index:
                                index = np.where(np.array(Gen_index)==i)[0]
                                predictionG_train = torch.cat([predictionG_train, Pred_Pg_train_rev[:, index]],1)
                            else:
                                predictionG_train = torch.cat([predictionG_train, torch.zeros([len(train_x), 1])], 1)
                            
                    #PD
                    for i in range(Bus_number):
                        if i == 0:
                            if i in Load_index:
                                index = np.where(np.array(Load_index)==i)[0]
                                predictionD_train = train_D[:, index]
                            else:
                                predictionD_train = torch.zeros([len(train_x), 1])
                        else:
                            if i in Load_index:
                                index = np.where(np.array(Load_index)==i)[0]
                                predictionD_train = torch.cat([predictionD_train, train_D[:, index]],1)
                            else:
                                predictionD_train = torch.cat([predictionD_train, torch.zeros([len(train_x), 1])],1)
        
                    predictionInj_train = predictionG_train - predictionD_train
                    predictionInj_train_1 = predictionInj_train[:, 0:Slack_bus]
                    predictionInj_train_1 = torch.cat([predictionInj_train_1, predictionInj_train[:, Slack_bus+1:Bus_number]], 1)
        
                   
        
                    #predicted theta calculation
                    theta_1_result_output_train = (Bus_1_Coeff_tensor.mm((predictionInj_train_1).t()))
                    theta_1_result_output_train = theta_1_result_output_train+Slack_angle
                    slack_angle_array = np.ones([1,len(train_x)], dtype='float32')*Slack_angle
                    slack_angle_array_tensor = torch.from_numpy(slack_angle_array)
        
                    theta_result_output_train = theta_1_result_output_train[0:Slack_bus,:]
                    theta_result_output_train = torch.cat([theta_result_output_train, slack_angle_array_tensor],0)
                    theta_result_output_train = torch.cat([theta_result_output_train, theta_1_result_output_train[Slack_bus:Bus_number-1,:]],0)
        
                    
            
                    #transmission line inequality constraint
                    Bus_admittance_full_NN2_tensor = torch.from_numpy(Bus_admittance_full_NN2)
                    trans_result_output_train = (Bus_admittance_full_NN2_tensor.mm(theta_result_output_train).t()) 
        #            trans_result_output_train_np = trans_result_output_train.cpu().detach().numpy()
        
                    
                    train_loss3 = penalty(trans_result_output_train)
                    
                    # train_loss4 = penalty2(train_out)
                    train_loss4 = penalty2(kk)+penalty3(zz)
        
                    # train_loss4 = penalty2(train_out)+penalty3(train_out)
                    # train_loss4=0
                    loss = 1*(train_loss1) + (1/1)*train_loss3 + (1)*train_loss4
                    running_loss += loss.item() #* train_y.size(0)
        
                    # backproprogate
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                if (epoch+1) % 10 == 0 and  epoch !=0 :
                    eval_loss = 0.
                    for (val_x, val_y) in validation_loader:#val_theta
                        val_out = model(val_x)
                        #calculate the slack bus output
                        Slack_index = np.where(np.array(Gen_index) == Slack_bus)[0][0]
        
                        Power_Lowbound_Gen_nn_val = torch.from_numpy(np.tile(Power_Lowbound_Gen, (len(val_x), 1)))
                        Power_Lowbound_Gen_nn_val_1 = Power_Lowbound_Gen_nn_val[:, 0:Slack_index]
                        Power_Lowbound_Gen_nn_val_1 = torch.cat([Power_Lowbound_Gen_nn_val_1, Power_Lowbound_Gen_nn_val[:, Slack_index+1:Gen_index.shape[0]]], 1)
        
                        Power_Upbound_Gen_nn_val = torch.from_numpy(np.tile(Power_Upbound_Gen, (len(val_x), 1)))
                        Power_Upbound_Gen_nn_val_1 = Power_Upbound_Gen_nn_val[:, 0:Slack_index]
                        Power_Upbound_Gen_nn_val_1 = torch.cat([Power_Upbound_Gen_nn_val_1, Power_Upbound_Gen_nn_val[:, Slack_index+1:Gen_index.shape[0]]], 1)
        
                        Pred_Pg_val = Power_Lowbound_Gen_nn_val_1 + (Power_Upbound_Gen_nn_val_1 - Power_Lowbound_Gen_nn_val_1) * val_out
        
        #                Actual_PG_val = val_y.mul((Power_Upbound_Gen_nn_val - Power_Lowbound_Gen_nn_val) + Power_Lowbound_Gen_nn_val)
        #                Pred_Pg_val_rev = torch.unsqueeze(torch.sum(Actual_PG_val[:, 0:Gen_index.shape[0]], 1),1) - torch.unsqueeze(torch.sum(Pred_Pg_val, 1), 1)
                        val_D = (val_x * (P_D_train_std_tensor + 1e-8) + P_D_train_mean_tensor)
                        Pred_Pg_val_rev = torch.unsqueeze(torch.sum(val_D[:, 0:Load_index.shape[0]], 1),1) - torch.unsqueeze(torch.sum(Pred_Pg_val, 1), 1)                
                        Pred_Pg_val_rev = torch.cat([Pred_Pg_val[:, 0:Slack_index], Pred_Pg_val_rev], 1)
                        Pred_Pg_val_rev = torch.cat([Pred_Pg_val_rev, Pred_Pg_val[:, Slack_index:Gen_index.shape[0]]], 1)
                        val_out = (Pred_Pg_val_rev - Power_Lowbound_Gen_nn_val) / (Power_Upbound_Gen_nn_val - Power_Lowbound_Gen_nn_val)
                        aa=val_out*1
                        bb=val_out*1
                        val_loss1 = criterion(val_out, val_y)
        
                        # PG
                        for i in range(Bus_number):
                            if i == 0:
                                if i in Gen_index:
                                    index = np.where(np.array(Gen_index) == i)[0]
                                    predictionG_val = Pred_Pg_val_rev[:, index]
                                    # print("a:{}".format(predictionG.shape))
                                else:
                                    predictionG_val = torch.zeros([len(val_x), 1])
                                    # print("b:{}".format(predictionG.shape))
                            else:
                                if i in Gen_index:
                                    index = np.where(np.array(Gen_index) == i)[0]
                                    predictionG_val = torch.cat([predictionG_val, Pred_Pg_val_rev[:, index]], 1)
                                else:
                                    predictionG_val = torch.cat([predictionG_val, torch.zeros([len(val_x), 1])], 1)
        
                        # PD
                        for i in range(Bus_number):
                            if i == 0:
                                if i in Load_index:
                                    index = np.where(np.array(Load_index) == i)[0]
                                    predictionD_val = val_D[:, index]
                                else:
                                    predictionD_val = torch.zeros([len(val_x), 1])
                            else:
                                if i in Load_index:
                                    index = np.where(np.array(Load_index) == i)[0]
                                    predictionD_val = torch.cat([predictionD_val, val_D[:, index]], 1)
                                else:
                                    predictionD_val = torch.cat([predictionD_val, torch.zeros([len(val_x), 1])], 1)
        
                        predictionInj_val = predictionG_val - predictionD_val  # shape
                        predictionInj_val_1 = predictionInj_val[:, 0:Slack_bus]
                        predictionInj_val_1 = torch.cat([predictionInj_val_1, predictionInj_val[:, Slack_bus+1:Bus_number]], 1)
        
                        # theta calculation
                        theta_1_output_val = (Bus_1_Coeff_tensor.mm((predictionInj_val_1).t()))
                        theta_1_output_val = theta_1_output_val + Slack_angle
                        slack_angle_array_val = np.ones([1,len(val_x)], dtype='float32')*Slack_angle
                        slack_angle_array_tensor_val = torch.from_numpy(slack_angle_array_val)
                            
                        theta_output_val = theta_1_output_val[0:Slack_bus,:]
                        theta_output_val = torch.cat([theta_output_val, slack_angle_array_tensor_val],0)
                        theta_output_val = torch.cat([theta_output_val, theta_1_output_val[Slack_bus:Bus_number-1,:]],0)
                            
                        theta_output_val = theta_output_val.t()
                
                        #transmission line inequality constraint
                        Bus_admittance_full_NN2_tensor = torch.from_numpy(Bus_admittance_full_NN2)
                        trans_result_output_val = (Bus_admittance_full_NN2_tensor.mm(theta_output_val.t())).t() 
        
                        val_loss3 = penalty(trans_result_output_val)#, target_trans_val)
                        #SC-OPF
                        # val_loss4 = penalty2(val_out)+penalty3(val_out)
                        val_loss4 = penalty2(aa)+penalty3(bb)
                        # val_loss4=0
                        
                    val_loss = 1*val_loss1+ (1/1)*val_loss3 + (1)*val_loss4
                    eval_loss += val_loss.item()
                    
                    print('Finish {} epoch, Validation_Loss: {:.6f}\nLoss1 : {:.6f}\nLoss3 : {:.6f} \n Loss4 : {:.6f}'.format(epoch + 1,eval_loss,val_loss1,val_loss3,val_loss4))
                scheduler.step()
                print('Finish {} epoch, Training_Loss: {:.6f}\nLoss1 : {:.6f} \nLoss3 : {:.6f} \nLoss4 : {:.6f}'.format(epoch + 1, running_loss/step,train_loss1,train_loss3,train_loss4))
        
            #current = time.time()
            total_time = total_time + (time.time() - start)
            print('\ntotal training time:%.3f\n' % total_time)
            dir = './S-PTH/'
            torch.save(model.state_dict(), dir+'PreDCOPF_case30_0.93_dnn_'+str(IRound+1)+'.pth')
            
     ###test
            #read case file
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
            
            mpc = loadcase('case30_rev_100.py')
            mpc['gen'][:,8]=mpc['gen'][:,8]
            mpc['branch'][:,5]=mpc['branch'][:,5]*1.017
            #reading parameters

            #system MVA base
            standval = mpc["baseMVA"]
            #bus(load)
            mpc_bus=mpc["bus"]
             
            
            #branch
            mpc_branch = mpc["branch"]
            
            #genrator
            mpc_gen=mpc["gen"]
            
            ## switch to internal bus numbering and build admittance matrices
            _, bus, gen, branch = ext2int1(mpc_bus, mpc_gen, mpc_branch)
            
            [mL,nL]=bus.shape 
            [mB,nB]=branch.shape
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
            nG = gen.shape[1]
        
            # bus = mpc_bus
            # branch = mpc_branch      
        

            
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
        

            Gen_index = np.array(Gen_index_list)
            Load_index = np.array(Load_index_list)
        
             
            NN_output_number = mG-1
            NN_input_number = Load_index.shape[0]
            
            
            Branch_number = mB
            
            Bus_admittance_full = np.zeros([mL,mL],dtype='float32')
            
            Bus_admittance_line = np.zeros([mB,mL],dtype='float32') 
            
            #DC-OPF
            ## power mismatch constraints
            B, Bf, Pbusinj, Pfinj = makeBdc(standval, bus, branch)   
            Bus_admittance_full[:,0:mL] = B.toarray()
            Bus_admittance_line[:,0:mL] = Bf.toarray()
        
            for i in range(0,mB,1):
                Bus_admittance_line[i,:] = Bus_admittance_line[i,:] / (branch[i,5]/standval)
        
            
            Bus_admittance_full_1 = Bus_admittance_full[0:Slack_bus,0:Slack_bus]
            Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Bus_admittance_full[0:Slack_bus,Slack_bus+1:Bus_number]),axis=1)
            Temp = Bus_admittance_full[Slack_bus+1:Bus_number,0:Slack_bus]
            Temp = np.concatenate((Temp,Bus_admittance_full[Slack_bus+1:Bus_number,Slack_bus+1:Bus_number]),axis=1)
            Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Temp),axis=0)    
        
        
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
            
            mpc_price = mpc['gencost']
            [mpri,npri] = mpc_price.shape
            
            Gen_Price = np.zeros([mG,npri-4],dtype='float32')
            index = 0
            for i in range(0, mpc_gen.shape[0], 1):
                if mpc_gen[i,7] !=0:
                    Gen_Price[index,0:npri-4] = mpc_price[i,4:4+npri-3]
                    index = index +1
            Gen_Price
            # global data
            P_D_list = []
            P_G_list = []
            P_D_list_test = []
            P_G_list_test = []    
            Pre=[]
            #read training data
            numOfdata = 0
            
            for index in range(0, IRound+6, 1):
                filename= 'Pre_DC_training_case30-sampling_part_0.93_'+str(index)+'.txt'
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
                                    # ind = np.where(mpc['gen'][:,0] == int(Gen_data[0]))[0][0]
                                    # if mpc['gen'][ind,7] != 0:
                                    P_G_list.append(float(Gen_data[1])/standval)
                            j = j + 1
            ################################################################
            del data
            num_of_group = int(np.floor(numOfdata / 10))
            P_D = np.array(P_D_list, dtype=np.float32)
            P_D = P_D.reshape(-1, NN_input_number)
            P_G = np.array(P_G_list, dtype=np.float32)
            P_G = P_G.reshape(-1, NN_output_number+1)
            
            NEW=np.hstack((P_D,P_G)).tolist()
            random.shuffle(NEW)
            NEW=np.array(NEW)
            P_D=NEW[:,0:NN_input_number].astype(np.float32)
            P_G=NEW[:,NN_input_number:NN_input_number+NN_output_number+1].astype(np.float32)
        
            # Preprocessing training data
            P_D_train = P_D[0:num_of_group * 8]
            # P_D_train_mean = np.mean(P_D_train, axis=0)
            # P_D_train_std = np.std(P_D_train, axis=0)
            P_D_train_mean = MEAN
            P_D_train_std = STD
            P_D_train_std_tensor = torch.from_numpy(P_D_train_std) 
            P_D_train_mean_tensor = torch.from_numpy(P_D_train_mean) 
        #    del P_D_list, P_G_list
            #test inversve matrix of B (default DCOPF)
            Bus_1_Coeff = np.linalg.inv(Bus_admittance_full_1)
            Bus_1_Coeff_tensor = torch.from_numpy(Bus_1_Coeff) 
            ################################################################
            # restore net
            dir = './S-PTH/'
            model = Neuralnetwork(NN_input_number, neural_number, NN_output_number)
            model.load_state_dict(torch.load(dir+'PreDCOPF_case30_0.93_dnn_'+str(IRound+1)+'.pth',map_location='cpu'))
            #if torch.cuda.is_available():
            #model = model.cuda()
            #read test data
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
                            P_D_list_test.append(float(odom[i])/standval)
                    else:
                        odom  =line.split(';')  # supply
                        for i in range(len(odom)):
                            Gen_data = odom[i].split(':')
                            # ind = np.where(mpc['gen'][:,0] == int(Gen_data[0]))[0][0]
                            # if mpc['gen'][ind,7] != 0:
                            P_G_list_test.append(float(Gen_data[1])/standval)                
                    j = j + 1
                fid.close()
            ################################################################
            del data
            Input_data = np.array(P_D_list_test, dtype=np.float32).reshape(-1, NN_input_number)
            Output_data = np.array(P_G_list_test, dtype=np.float32).reshape(-1, NN_output_number+1)         
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
            
            pg_error = 0
            PDM_test = torch.zeros([NN_input_number,Bus_number])
            PGM_test = torch.zeros([NN_output_number+1,Bus_number])
            Bus_admittance_line_tensor = torch.from_numpy(Bus_admittance_line)
            cost_list = []
            feasible1 = 0
            feasible2 = 0
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

        
                slack_angle_array_test = np.ones([1,len(test_x)], dtype='float32')*Slack_angle
                slack_angle_array_tensor_test = torch.from_numpy(slack_angle_array_test)
        
                start1 = time.time()
                theta_result_output_test = theta_1_result_output_test[0:Slack_bus,:]
                theta_result_output_test = torch.cat([theta_result_output_test, slack_angle_array_tensor_test],0)
                theta_result_output_test = torch.cat([theta_result_output_test, theta_1_result_output_test[Slack_bus:Bus_number-1,:]],0)
        
                trans_result_output_test = (Bus_admittance_line_tensor.mm(theta_result_output_test).t()) 
                trans_result_output_test_np = trans_result_output_test.cpu().detach().numpy()
        
                infeasible_index = np.where((trans_result_output_test_np > 1) | (trans_result_output_test_np < -1))

                
                
                if len(infeasible_index[0]) >0:
                    # fid1.write("%d;" % (1))
                    feasible1 = feasible1+0
                else:
                    # fid1.write("%d;" % (0))
                    feasible1 = feasible1+1
                Pred_Pg_test_np = Pred_Pg_test_rev.detach().numpy()
                if (Pred_Pg_test_np[0,Slack_index] >  Power_Upbound_Gen[0,Slack_index]) or (Pred_Pg_test_np[0,Slack_index] <  Power_Lowbound_Gen[0,Slack_index]):
                    feasible2 = feasible2+0
                    # fid1.write("%d;" % (1))
                else:
                    feasible2 = feasible2+1
                    
                # print(feasible1+feasible2)
                
            #read case file
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
            
            mpc = loadcase('case30_rev_100.py')
            mpc['gen'][:,8]=mpc['gen'][:,8]
            mpc['branch'][:,5]=mpc['branch'][:,5]*1.017
            #reading parameters

            #system MVA base
            standval = mpc["baseMVA"]
            #bus(load)
            mpc_bus=mpc["bus"]
             
            
            #branch
            mpc_branch = mpc["branch"]
            
            #genrator
            mpc_gen=mpc["gen"]
            
            ## switch to internal bus numbering and build admittance matrices
            _, bus, gen, branch = ext2int1(mpc_bus, mpc_gen, mpc_branch)
            
            [mL,nL]=bus.shape 
            [mB,nB]=branch.shape
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
            nG = gen.shape[1]
        
            # bus = mpc_bus
            # branch = mpc_branch      
        

            
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
        

            Gen_index = np.array(Gen_index_list)
            Load_index = np.array(Load_index_list)
          
            NN_output_number = mG-1
            NN_input_number = Load_index.shape[0]
            
            
            Branch_number = mB
            
            Bus_admittance_full = np.zeros([mL,mL],dtype='float32')
            
            Bus_admittance_line = np.zeros([mB,mL],dtype='float32') 
            
            #DC-OPF
            ## power mismatch constraints
            B, Bf, Pbusinj, Pfinj = makeBdc(standval, bus, branch)   
            Bus_admittance_full[:,0:mL] = B.toarray()
            Bus_admittance_line[:,0:mL] = Bf.toarray()
        
            for i in range(0,mB,1):
                Bus_admittance_line[i,:] = Bus_admittance_line[i,:] / (branch[i,5]/standval)
        
            
            Bus_admittance_full_1 = Bus_admittance_full[0:Slack_bus,0:Slack_bus]
            Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Bus_admittance_full[0:Slack_bus,Slack_bus+1:Bus_number]),axis=1)
            Temp = Bus_admittance_full[Slack_bus+1:Bus_number,0:Slack_bus]
            Temp = np.concatenate((Temp,Bus_admittance_full[Slack_bus+1:Bus_number,Slack_bus+1:Bus_number]),axis=1)
            Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Temp),axis=0)    
        
        
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
            
            mpc_price = mpc['gencost']
            [mpri,npri] = mpc_price.shape
            
            Gen_Price = np.zeros([mG,npri-4],dtype='float32')
            index = 0
            for i in range(0, mpc_gen.shape[0], 1):
                if mpc_gen[i,7] !=0:
                    Gen_Price[index,0:npri-4] = mpc_price[i,4:4+npri-3]
                    index = index +1
            Gen_Price
            # global data
            P_D_list = []
            P_G_list = []
            P_D_list_test = []
            P_G_list_test = []    
            Pre=[]
            #read training data
            numOfdata = 0
            
            for index in range(0, IRound+6, 1):
                filename= 'Pre_DC_training_case30-sampling_part_0.93_'+str(index)+'.txt'
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
                                    # ind = np.where(mpc['gen'][:,0] == int(Gen_data[0]))[0][0]
                                    # if mpc['gen'][ind,7] != 0:
                                    P_G_list.append(float(Gen_data[1])/standval)
                            j = j + 1
            ################################################################
            del data
            num_of_group = int(np.floor(numOfdata / 10))
            P_D = np.array(P_D_list, dtype=np.float32)
            P_D = P_D.reshape(-1, NN_input_number)
            P_G = np.array(P_G_list, dtype=np.float32)
            P_G = P_G.reshape(-1, NN_output_number+1)
            
            NEW=np.hstack((P_D,P_G)).tolist()
            random.shuffle(NEW)
            NEW=np.array(NEW)
            P_D=NEW[:,0:NN_input_number].astype(np.float32)
            P_G=NEW[:,NN_input_number:NN_input_number+NN_output_number+1].astype(np.float32)
            
            # Preprocessing training data
            P_D_train = P_D[0:num_of_group * 8]
            # P_D_train_mean = np.mean(P_D_train, axis=0)
            # P_D_train_std = np.std(P_D_train, axis=0)
            P_D_train_mean = MEAN
            P_D_train_std = STD
            P_D_train_std_tensor = torch.from_numpy(P_D_train_std) 
            P_D_train_mean_tensor = torch.from_numpy(P_D_train_mean) 
        #    del P_D_list, P_G_list
            #test inversve matrix of B (default DCOPF)
            Bus_1_Coeff = np.linalg.inv(Bus_admittance_full_1)
            Bus_1_Coeff_tensor = torch.from_numpy(Bus_1_Coeff) 
            ################################################################
            # restore net
            dir = './S-PTH/'
            model = Neuralnetwork(NN_input_number, neural_number, NN_output_number)
            model.load_state_dict(torch.load(dir+'PreDCOPF_case30_0.93_dnn_'+str(IRound+1)+'.pth',map_location='cpu'))
            #if torch.cuda.is_available():
            #model = model.cuda()
            #read test data
            for index in range(0,1,1):
                filename= 'Pre_DC_training_case30-sampling_part_0.93_'+str(IRound+5)+'.txt'
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
                        odom  =line.split(';')  # supply
                        for i in range(len(odom)):
                            Gen_data = odom[i].split(':')
                            # ind = np.where(mpc['gen'][:,0] == int(Gen_data[0]))[0][0]
                            # if mpc['gen'][ind,7] != 0:
                            P_G_list_test.append(float(Gen_data[1])/standval)                
                    j = j + 1
                fid.close()
            ################################################################
            del data
            Input_data = np.array(P_D_list_test, dtype=np.float32).reshape(-1, NN_input_number)
            Output_data = np.array(P_G_list_test, dtype=np.float32).reshape(-1, NN_output_number+1)         
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
            
            pg_error = 0
            PDM_test = torch.zeros([NN_input_number,Bus_number])
            PGM_test = torch.zeros([NN_output_number+1,Bus_number])
            Bus_admittance_line_tensor = torch.from_numpy(Bus_admittance_line)
            cost_list = []
            feasible3=0
            feasible4=0
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

                slack_angle_array_test = np.ones([1,len(test_x)], dtype='float32')*Slack_angle
                slack_angle_array_tensor_test = torch.from_numpy(slack_angle_array_test)
        
                start1 = time.time()
                theta_result_output_test = theta_1_result_output_test[0:Slack_bus,:]
                theta_result_output_test = torch.cat([theta_result_output_test, slack_angle_array_tensor_test],0)
                theta_result_output_test = torch.cat([theta_result_output_test, theta_1_result_output_test[Slack_bus:Bus_number-1,:]],0)
        
                trans_result_output_test = (Bus_admittance_line_tensor.mm(theta_result_output_test).t()) 
                trans_result_output_test_np = trans_result_output_test.cpu().detach().numpy()
        
                infeasible_index = np.where((trans_result_output_test_np > 1) | (trans_result_output_test_np < -1))

                
                
             
                if len(infeasible_index[0]) >0:
                    # fid1.write("%d;" % (1))
                    feasible3 = feasible3+0
                else:
                    # fid1.write("%d;" % (0))
                    feasible3 = feasible3+1
                Pred_Pg_test_np = Pred_Pg_test_rev.detach().numpy()
                if (Pred_Pg_test_np[0,Slack_index] >  Power_Upbound_Gen[0,Slack_index]) or (Pred_Pg_test_np[0,Slack_index] <  Power_Lowbound_Gen[0,Slack_index]):
                    feasible4 = feasible4+0
                    # fid1.write("%d;" % (1))
                else:
                    feasible4 = feasible4+1
                    
                # print(feasible3+feasible4)
            filenamefeasible_train= 'feasible_train.txt'
            fidfeasible_train=open(filenamefeasible_train,'at')
            fidfeasible_train.write("%9.4f;" % (feasible1))
            fidfeasible_train.write('\n')
            fidfeasible_train.write("%9.4f;" % (feasible2))
            fidfeasible_train.write('\n')
            fidfeasible_train.write("%9.4f;" % (feasible3))
            fidfeasible_train.write('\n')
            fidfeasible_train.write("%9.4f;" % (feasible4))
            fidfeasible_train.write('\n')
            fidfeasible_train.close()
        

            if (feasible3+feasible4==200*(iteration_train+1)) or iteration_train==iteration_time-1:
                
                filenamefeasible_train= 'feasible_train.txt'
                fidfeasible_train=open(filenamefeasible_train,'at')
                fidfeasible_train.write('Finish'+str(IRound)+'************************')
                fidfeasible_train.write('\n')
                fidfeasible_train.write("%9.4f;" % (feasible1+feasible2+feasible3+feasible4))
                fidfeasible_train.write('\n')
                fidfeasible_train.write("%9.4f;" % (iteration_train))
                fidfeasible_train.write('\n')
                fidfeasible_train.write('**********Finish'+str(IRound)+'************************')
                fidfeasible_train.write('\n')
                fidfeasible_train.close()
                # print(feasible1+feasible2)
                break
            else:
                pd2 = np.zeros((30,),dtype='float32')
            
                load_index = 0
                for i in range(0,mL,1):
                    if i in Load_index_list:
                        pd2[i] = Input[load_index]*((0.3*abs(bus[i,2])/standval)/12**0.5)+bus[i,2]/standval*(default)
                        load_index += 1
                    else:
                        pd2[i] = 0
                        
                        
                
                for index in range(IRound+5,IRound+6,1):
                    for kkk in range(0,100):
                    
                        filename7= 'Pre_DC_training_case30-sampling_part_0.93_'+str(index)+'.txt'
                        fid7=open(filename7,'at')
                        
                        filename5= 'Success_0.93_'+str(index)+'.txt'
                        fid5=open(filename5,'at')
            
                        #########load:1.1-1.2,branch:0.9
                        
                        mpc7 = loadcase('case30_rev_070.py')
                        mpc7_bus=mpc7['bus']
                        [m,n] = mpc7_bus.shape
                        if kkk<=98:
                            tempload=pd2*100*(0.02*(np.random.uniform(0,1,30)-0.5)+1)
                        else:
                            tempload=pd2*100
                        tempload2=np.zeros((30,),dtype='float32')
                        for i in range(0,30):
                            if tempload[i]>=mpc7_bus[i,2]*1.3 and tempload[i]>=0:
                                tempload2[i]=mpc7_bus[i,2]*1.3
                            if tempload[i]<=mpc7_bus[i,2] and tempload[i]>=0:
                                tempload2[i]=mpc7_bus[i,2]
                            if tempload[i]<=mpc7_bus[i,2]*1.3 and tempload[i]<=0:
                                tempload2[i]=mpc7_bus[i,2]*1.3
                            if tempload[i]>=mpc7_bus[i,2] and tempload[i]<=0:
                                tempload2[i]=mpc7_bus[i,2]
                            if tempload[i]>=mpc7_bus[i,2] and tempload[i]<=mpc7_bus[i,2]*1.3 and tempload[i]>=0:
                                tempload2[i]=tempload[i]
                            if tempload[i]<=mpc7_bus[i,2] and tempload[i]>=mpc7_bus[i,2]*1.3 and tempload[i]<=0:
                                tempload2[i]=tempload[i]
                                
                                
                        mpc7_bus[:,2] = tempload2
                        mpc7_branch=mpc7['branch']
                        mpc7['gen'][:,8]=mpc7['gen'][:,8]
                        factor=0.07
                        mpc7_branch[:,5]=mpc7_branch[:,5]*1.017-branch_capacity*factor*100
                        
                        r7 = rundcopf(mpc7)        
                        
                        r7_bus = r7['bus']
                        for j in range(0,m,1):
                            
                            if(r7_bus[j,2]*r7_bus[j,2] > 0):
                                if j != m-1:
                                    fid7.write("%9.4f;" % (r7_bus[j,2]))                   
                                else:
                                    fid7.write("%9.4f\n" % (r7_bus[j,2]))
                
                        
                        r7_gen=r7['gen']
                        m1 = len(r7_gen)
                        
                        for j in range(0,m1,1):
                            if j != m1-1:
                                fid7.write("%d:%9.4f;" % (int(r7_gen[j,0]),r7_gen[j,1]))
                            else:
                                fid7.write("%d:%9.4f\n" % (int(r7_gen[j,0]),r7_gen[j,1]))
                        
                            
                        if (r7['success']== True):
                            fid5.write("%d" % (0))
                        else:
                            fid5.write("%d" % (1))
                            
                        r7=[]
                    
                    fid5.close()
                    fid7.close()
                
                
            
        Finshtrain=time.time()
        
        filenametrain= 'traintime.txt'
        fidtrain=open(filenametrain,'at')
        fidtrain.write("%9.4f;" % (Finshtrain-train_time))
        fidtrain.write('\n')
        fidtrain.close()     
        
        filenametrain2= 'testtrainfinish.txt'
        fidtrain2=open(filenametrain2,'at')
        fidtrain2.write("%9.4f;" % (1))
        fidtrain2.write('\n')
        fidtrain2.close()  
            




    # np.savetxt("solvertime.txt", np.array(solver_time), fmt="%s")
    np.savetxt("s-weight.txt", np.array(weight_out), fmt="%s")
    np.savetxt("s-test.txt", np.array(len(WEIGHT)).reshape(-1,), fmt="%s")
    np.savetxt("s-bias.txt", np.array(bias_out), fmt="%s")
    

      
