import numpy as np
from pypower.loadcase import loadcase
from scipy.sparse import csr_matrix as sparse
from pypower.ext2int import ext2int1
# from pypower.makeYbus import makeYbus
from pypower.makeBdc import makeBdc
from torch import nn, optim
import torch.nn.functional as F
import time
import torch
import torch.utils.data as Data

torch.manual_seed(1)    #reproducible

# parameter
NN_input_number = 0 
NN_output_number = 0  

batch_size_training = 64  
batch_size_valtest = 1  
neural_number = 8

#learning parameters
learning_rate = 1e-4
num_epoches = 200
################################
#SCDCOPF
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

SC_list = []
conflict_list = []
filter_list = []
NumofContigengcy = 0
SC_Bus_admittance_full_1_array = np.zeros((1,1,1),dtype='float32')
SC_Bus_admittance_line_1_array = np.zeros((1,1,1),dtype='float32')


#penalty definition
class Penalty(nn.Module):
    def __init__(self):
        super(Penalty, self).__init__()
    def forward(self, pred):
        hinge_loss = pred**2-1
        hinge_loss[hinge_loss < 0] = 0
        return torch.mean(hinge_loss)
        #return torch.mean(max(0,pred**2-1))

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


#class for graph
class Gragh():
    def __init__(self,nodes,sides):
        self.sequense = {}
        self.side=[]
        for node in nodes:
            for side in sides:
                u,v=side
                if node ==u:
                    self.side.append(v)
                elif node == v:
                    self.side.append(u)
            self.sequense[node] = self.side
            self.side=[]

    # Depth-First-Search 
    def DFS(self,node0):
        queue,order=[],[]
        queue.append(node0)
        while queue:
            v = queue.pop()
            order.append(v)
            for w in self.sequense[v]:
                if w not in order and w not in queue: 
                    queue.append(w)
        return order

#     BFS
    def BFS(self,node0):
        queue,order = [],[]
        queue.append(node0)
        order.append(node0)
        while queue:
            v = queue.pop(0)
            for w in self.sequense[v]:
                if w not in order:
                    order.append(w)
                    queue.append(w)
        return order

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
    
    #Admittance for standard DC-OPF
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

	#SC-DCOPF
    global NumofContigengcy
    global SC_list
    global filter_list
    global conflict_list
    #SC-OPF
    if mpc["scopf"] == 1:
        #check different topologies (keep the power network is connected)
        if Bus_number == 30:
            SC_list = [0,1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40]
            NumofContigengcy = len(SC_list)
        else:               
            nodes = [i for i in range(Bus_number)]
            temp_branch = np.zeros((mB-1,nB),dtype='float32')
            for j in range(0,mB,1):
                temp_branch[:,:] = 0
                temp_branch[0:j] = branch[0:j]
                temp_branch[j:mB-1] = branch[j+1:mB]
                ## power flow constraints
                sides = []
                for i in range(0,mB-1,1):
                    sides.append((temp_branch[i,0], temp_branch[i,1]))
                G = Gragh(nodes,sides)
                res = G.BFS(temp_branch[0,0])
                del G
                if len(res) == Bus_number:
                    NumofContigengcy = NumofContigengcy + 1
                    SC_list.append(j)
    #            print(SC_list)
    
    
    #Admittance matrix for each contingency in SC-OPF
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
            for j in range(0,mB-1,1):
                SC_Bus_admittance_line_1_temp_array[i,j,:] = SC_Bus_admittance_line_1_temp_array[i,j,:] / (temp_branch[j,5]/standval)
        for i in range(0,NumofContigengcy,1):
            SC_Bus_admittance_line_1_array[i,:,:] = SC_Bus_admittance_line_1_temp_array[i,:,:].T


if __name__ == '__main__':
    #read case file
    mpc = loadcase('pglib_opf_case30_ieee.py')
    #reading parameters
    para_generation(mpc)    
    #read training data
    P_D = np.load("PD_train.npy").astype("float32")
    P_G = np.load("PG_train.npy").astype("float32")
    num_of_group = int(np.floor(P_D.shape[0] / 10))    
#    theta = np.array(theta_list, dtype=np.float32).reshape(-1,(NumofContigengcy+1), Bus_number)

    # Preprocessing training data
    P_D_train = P_D[0:num_of_group * 9]
    P_D_train_mean = np.mean(P_D_train, axis=0)
    P_D_train_std = np.std(P_D_train, axis=0)
    P_D_train_std_tensor = torch.from_numpy(P_D_train_std) 
    P_D_train_mean_tensor = torch.from_numpy(P_D_train_mean)
    P_D_train_normolization = (P_D_train - P_D_train_mean) / (P_D_train_std+1e-8)

    P_G_train = P_G[0:num_of_group * 9]
    Power_Lowbound_Gen_training = np.tile(Power_Lowbound_Gen,(len(P_D_train),1))
    Power_Upbound_Gen_training= np.tile(Power_Upbound_Gen,(len(P_D_train),1))
    P_G_train_normolization = (P_G_train-Power_Lowbound_Gen_training)/(Power_Upbound_Gen_training-Power_Lowbound_Gen_training)
    
    #test inversve matrix of B (default DCOPF)
    Bus_1_Coeff = np.linalg.inv(Bus_admittance_full_1)
    Bus_1_Coeff_tensor = torch.from_numpy(Bus_1_Coeff)    

    ########################################
    model = Neuralnetwork(NN_input_number, neural_number, NN_output_number)
    penalty = Penalty()
    #neural network parameters setting
    criterion = nn.MSELoss(reduce=True,size_average = True)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    ##########################################
    # Training dataset
    P_D_train_normolization_tensor = torch.from_numpy(P_D_train_normolization) 
    P_G_train_normolization_tensor = torch.from_numpy(P_G_train_normolization) 
    training_dataset = Data.TensorDataset(P_D_train_normolization_tensor, P_G_train_normolization_tensor)
    training_loader = Data.DataLoader(
        dataset=training_dataset,      # torch TensorDataset format
        batch_size=batch_size_training,      # mini batch size
        shuffle=True,
        #num_workers=2,
    )
    
    start = time.time()
    total_time = 0
    #Training process
    Bus_admittance_full_NN2_tensor = torch.from_numpy(Bus_admittance_full_NN2)
    if mpc["scopf"] == 1:
        SC_Bus_1_Coeff_tensor = torch.from_numpy(SC_Bus_admittance_full_1_array)
        #transmission line inequality constraint
        SC_Bus_admittance_line_tensor = torch.from_numpy(SC_Bus_admittance_line_1_array)
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

            #theta calculation
            #predicted theta
            theta_1_result_output_train = (Bus_1_Coeff_tensor.mm((predictionInj_train_1).t()))
            theta_1_result_output_train = theta_1_result_output_train+Slack_angle
            slack_angle_array = np.ones([1,len(train_x)], dtype='float32')*Slack_angle
            slack_angle_array_tensor = torch.from_numpy(slack_angle_array)

            theta_result_output_train = theta_1_result_output_train[0:Slack_bus,:]
            theta_result_output_train = torch.cat([theta_result_output_train, slack_angle_array_tensor],0)
            theta_result_output_train = torch.cat([theta_result_output_train, theta_1_result_output_train[Slack_bus:Bus_number-1,:]],0)
    
            #transmission line inequality constraint
            #Bus_admittance_full_NN2_tensor = torch.from_numpy(Bus_admittance_full_NN2)
            trans_result_output_train = (Bus_admittance_full_NN2_tensor.mm(theta_result_output_train).t()) 
#            trans_result_output_train_np = trans_result_output_train.cpu().detach().numpy()

            train_loss3 = penalty(trans_result_output_train)

            train_loss4 = 0
            #SC-OPF
            if mpc["scopf"] == 1:
                #inversve matrix of B (SCOPF)
                #SC_Bus_1_Coeff_tensor = torch.from_numpy(SC_Bus_admittance_full_1_array)
                SC_theta_1_output_train = (SC_Bus_1_Coeff_tensor.matmul(predictionInj_train_1.t())) 
                SC_gc_theta_1_output_train_T = SC_theta_1_output_train.permute(0, 2, 1)
                SC_gc_theta_1_output_train_T = SC_gc_theta_1_output_train_T + Slack_angle
                slack_angle_array = np.ones([NumofContigengcy,len(train_x),1], dtype='float32')*Slack_angle
                slack_angle_array_tensor = torch.from_numpy(slack_angle_array)

                theta_result_output_train = SC_gc_theta_1_output_train_T[:,:,0:Slack_bus]
                theta_result_output_train = torch.cat([theta_result_output_train, slack_angle_array_tensor],2)
                theta_result_output_train = torch.cat([theta_result_output_train, SC_gc_theta_1_output_train_T[:,:,Slack_bus:Bus_number-1]],2)

                #transmission line inequality constraint
                #SC_Bus_admittance_line_tensor = torch.from_numpy(SC_Bus_admittance_line_1_array)
                trans_result_output_train = theta_result_output_train.matmul(SC_Bus_admittance_line_tensor)
#                trans_result_output_train_T = trans_result_output_train.permute(0, 2, 1)
                trans_result_output_train_np = trans_result_output_train.cpu().detach().numpy()
                train_loss4 = penalty(trans_result_output_train)

            loss = 1*(train_loss1) + (1/1)*train_loss3 + (1/1)*train_loss4
            running_loss += loss.item() #* train_y.size(0)

            # backproprogate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('Finish {} epoch, Training_Loss: {:.6f}\nLoss1 : {:.6f} \nLoss3 : {:.6f} \nLoss4 : {:.6f}'.format(epoch + 1, running_loss/step,train_loss1,train_loss3,train_loss4))

    #current = time.time()
    total_time = total_time + (time.time() - start)
    print('\ntotal training time:%.3f\n' % total_time)
    torch.save(model.state_dict(), 'SCOPF_case30_dnn_typical.pth')