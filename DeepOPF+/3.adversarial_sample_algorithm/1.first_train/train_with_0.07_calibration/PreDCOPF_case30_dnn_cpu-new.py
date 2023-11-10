
import numpy as np
from pypower.loadcase import loadcase
from scipy.sparse import csr_matrix as sparse
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

batch_size_training = 8
batch_size_valtest = 1  
neural_number = 8

#learning parameters
learning_rate = 1e-4
num_epoches = 200
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
    global Bus_admittance_full_NN2
    Bus_admittance_full_NN2 = np.zeros([mB,mL],dtype='float32') 
    
    #DC-OPF
    ## power equation constraints
    B, Bf, Pbusinj, Pfinj = makeBdc(standval, bus, branch)   
    Bus_admittance_full[:,0:mL] = B.toarray()
    Bus_admittance_full_NN2[:,0:mL] = Bf.toarray()

    for i in range(0,mB,1):
        Bus_admittance_full_NN2[i,:] = Bus_admittance_full_NN2[i,:] / (branch[i,5]/standval) #shrink 5%
#        Bus_admittance_full_NN2[i,:] = Bus_admittance_full_NN2[i,:] / (branch[i,5]*0.9/standval) #shrink 10%
    global Bus_admittance_full_1
    Bus_admittance_full_1 = Bus_admittance_full[0:Slack_bus,0:Slack_bus]
    Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Bus_admittance_full[0:Slack_bus,Slack_bus+1:Bus_number]),axis=1)
    Temp = Bus_admittance_full[Slack_bus+1:Bus_number,0:Slack_bus]
    Temp = np.concatenate((Temp,Bus_admittance_full[Slack_bus+1:Bus_number,Slack_bus+1:Bus_number]),axis=1)
    Bus_admittance_full_1 = np.concatenate((Bus_admittance_full_1,Temp),axis=0)    
   

if __name__ == '__main__':
    #read case file
    mpc = loadcase('case30_rev_070.py')
    mpc_branch=mpc['branch']
    branch_capacity=np.array([1.3221 , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
       0.     , 0.     , 0.32544, 0.     , 0.     , 0.     , 0.     ,
       0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
       0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
       0.32544, 0.16272, 0.16272, 0.16272, 0.16272, 0.     , 0.16272,
       0.     , 0.     , 0.     , 0.     , 0.     , 0.     ])
    factor=0.07
    mpc_branch[:,5]=mpc_branch[:,5]*1.017-branch_capacity*factor*100
    #reading parameters
    para_generation(mpc)
    # global data
    P_D_list = []
    P_G_list = []
    #read training data
    numOfdata = 0
    dir='.//training_data//'
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
                            P_G_list.append(float(Gen_data[1])/standval)
                    j = j + 1
    ################################################################
    del data
    num_of_group = int(np.floor(numOfdata / 10))
    P_D = np.array(P_D_list, dtype=np.float32)
    P_D = P_D.reshape(-1, NN_input_number)
    P_G = np.array(P_G_list, dtype=np.float32)
    P_G = P_G.reshape(-1, NN_output_number+1)
#    theta = np.array(theta_list, dtype=np.float32).reshape(-1,(NumofContigengcy+1), Bus_number)
    # Preprocessing training data
    P_D_train = P_D[0:num_of_group * 8]
    P_D_train_mean = np.mean(P_D_train, axis=0)
    P_D_train_std = np.std(P_D_train, axis=0)
    P_D_train_std_tensor = torch.from_numpy(P_D_train_std) 
    P_D_train_mean_tensor = torch.from_numpy(P_D_train_mean) 
#    theta_mean = np.mean(theta1, axis=0)
#    theta_std = np.std(theta1, axis=0)
#    theta_std_tensor = torch.from_numpy(theta_std) 
#    theta_mean_tensor = torch.from_numpy(theta_mean) 

    P_D_train_normolization = (P_D_train - P_D_train_mean) / (P_D_train_std+1e-8)
    P_G_train = P_G[0:num_of_group * 8]
    Power_Lowbound_Gen_training = np.tile(Power_Lowbound_Gen,(len(P_D_train),1))
    Power_Upbound_Gen_training= np.tile(Power_Upbound_Gen,(len(P_D_train),1))
    P_G_train_normolization = (P_G_train-Power_Lowbound_Gen_training)/(Power_Upbound_Gen_training-Power_Lowbound_Gen_training)
#    theta_train = theta[0:num_of_group * 8]
#    theta_mean_training = np.tile(theta_mean,(len(x_train),1))
#    theta_std_training= np.tile(theta_std,(len(x_train),1))
#    theta_train_normolization = (theta_train - theta_mean_training) / (theta_std_training + 1e-8)
    
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
#    theta_val = theta[num_of_group * 8:num_of_group * 9]
#    theta_mean_val = np.tile(theta_mean,(len(x_val),1))
#    theta_std_val = np.tile(theta_std,(len(x_val),1))
#    theta_val_normolization = (theta_val - theta_mean_val) / (theta_std_val + 1e-8)

    ########################################
    # Preprocessing test data
#    P_D_test = P_D[num_of_group * 9:len(P_D)]
#    P_D_test_normolization = (P_D_test - P_D_train_mean) / (P_D_train_std+1e-8)
#    P_G_test = P_G[num_of_group * 9:len(P_D)]
#    Power_Lowbound_Gen_test = np.tile(Power_Lowbound_Gen,(len(P_D_test),1))
#    Power_Upbound_Gen_test = np.tile(Power_Upbound_Gen,(len(P_D_test),1))
#    P_G_test_normolization = (P_G_test-Power_Lowbound_Gen_test)/(Power_Upbound_Gen_test-Power_Lowbound_Gen_test)
#    theta_test = theta[num_of_group * 9 :len(P_D)]
#    theta_mean_test = np.tile(theta_mean,(len(x_test),1))
#    theta_std_test = np.tile(theta_std,(len(x_test),1))
#    theta_test_normolization = (theta_test - theta_mean_test) / (theta_std_test + 1e-8)   

    ########################################
    model = Neuralnetwork(NN_input_number, neural_number, NN_output_number)
    penalty = Penalty()
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
        scheduler.step()
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

            ##groundtruth
#            Actual_PG_gc = train_y.mul(Power_Upbound_Gen_nn_train - Power_Lowbound_Gen_nn_train) + Power_Lowbound_Gen_nn_train            
#            #PG
#            for i in range(Bus_number):
#                if i == 0:
#                    if i in Gen_index:
#                        index = np.where(np.array(Gen_index)==i)[0]
#                        predictionG_gc = Actual_PG_gc[:, index]
#                        #print("a:{}".format(predictionG.shape))
#                    else:
#                        predictionG_gc = torch.zeros([len(train_x), 1])
#                        #print("b:{}".format(predictionG.shape))
#
#                else:
#                    if i in Gen_index:
#                        index = np.where(np.array(Gen_index)==i)[0]
#                        predictionG_gc = torch.cat([predictionG_gc, Actual_PG_gc[:, index]],1)
#                    else:
#                        predictionG_gc = torch.cat([predictionG_gc, torch.zeros([len(train_x), 1])], 1)
#                    
#            predictionInj_gc = predictionG_gc - predictionD_train
#            predictionInj_gc_1 = predictionInj_gc[:, 0:Slack_index]
#            predictionInj_gc_1 = torch.cat([predictionInj_gc_1, predictionInj_gc[:, Slack_index+1:Bus_number]], 1)

            #predicted theta calculation
            theta_1_result_output_train = (Bus_1_Coeff_tensor.mm((predictionInj_train_1).t()))
            theta_1_result_output_train = theta_1_result_output_train+Slack_angle
            slack_angle_array = np.ones([1,len(train_x)], dtype='float32')*Slack_angle
            slack_angle_array_tensor = torch.from_numpy(slack_angle_array)

            theta_result_output_train = theta_1_result_output_train[0:Slack_bus,:]
            theta_result_output_train = torch.cat([theta_result_output_train, slack_angle_array_tensor],0)
            theta_result_output_train = torch.cat([theta_result_output_train, theta_1_result_output_train[Slack_bus:Bus_number-1,:]],0)

            #target theta
#            predictionInj_gc_1_np = predictionInj_gc_1.cpu().detach().numpy()
#            PG_PD_np = (torch.from_numpy(Bus_admittance_full).mm(train_theta[:,0,:].t())).t().cpu().detach().numpy()
#            theta_result_output_train_array = theta_result_output_train.t().cpu().detach().numpy()
#            gc_theta_array = (Bus_1_Coeff_tensor.mm((predictionInj_gc_1).t()))#.cpu().detach().numpy()
#            theta_result_output_gc = gc_theta_array[0:Slack_bus,:]
#            theta_result_output_gc = torch.cat([theta_result_output_gc, slack_angle_array_tensor],0)
#            theta_result_output_gc = torch.cat([theta_result_output_gc, gc_theta_array[Slack_bus:Bus_number-1,:]],0)
            
#            theta_result_output_gc_array = theta_result_output_gc.t().cpu().detach().numpy()
#            train_theta_array = train_theta[:,0,:].cpu().detach().numpy()
#            theta_result_output_train_norm = ((theta_result_output_train-theta_mean_nn_train)/(theta_std_nn_train+1e-8))
#            train_loss2 = criterion(theta_result_output_train_norm,train_theta)
    
            #transmission line inequality constraint
            Bus_admittance_full_NN2_tensor = torch.from_numpy(Bus_admittance_full_NN2)
            trans_result_output_train = (Bus_admittance_full_NN2_tensor.mm(theta_result_output_train).t()) 
#            trans_result_output_train_np = trans_result_output_train.cpu().detach().numpy()

            #trans benchmark
#            trans_result_output_benchmark = (Bus_admittance_full_NN2_tensor.mm(theta_result_output_gc).t()) 
#            trans_result_output_benchmark_np = trans_result_output_benchmark.cpu().detach().numpy()
#            trans_result_output_benchmark1 = (Bus_admittance_full_NN2_tensor.mm(train_theta[:,0,:].t()).t()) 
#            trans_result_output_benchmark1_np = trans_result_output_benchmark1.cpu().detach().numpy()  
            train_loss3 = penalty(trans_result_output_train)

            train_loss4 = 0
            loss = 1*(train_loss1) + (1/1)*train_loss3 + (1/1)*train_loss4
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
                val_loss4 = 0
            val_loss = 1*val_loss1+ (1/1)*val_loss3 + (1/1)*val_loss4
            eval_loss += val_loss.item()
            print('Finish {} epoch, Validation_Loss: {:.6f}\nLoss1 : {:.6f}\nLoss3 : {:.6f} \n Loss4 : {:.6f}'.format(epoch + 1,eval_loss,val_loss1,val_loss3,val_loss4))
        print('Finish {} epoch, Training_Loss: {:.6f}\nLoss1 : {:.6f} \nLoss3 : {:.6f} \nLoss4 : {:.6f}'.format(epoch + 1, running_loss/step,train_loss1,train_loss3,train_loss4))

    #current = time.time()
    total_time = total_time + (time.time() - start)
    print('\ntotal training time:%.3f\n' % total_time)
    torch.save(model.state_dict(), 'PreDCOPF_case30_0.93_dnn.pth')


    #Test dataset
#    P_D_test_normolization_tensor = torch.from_numpy(P_D_test_normolization)
#    P_G_test_normolization_tensor = torch.from_numpy(P_G_test_normolization)
##    theta_test_tensor = torch.from_numpy(theta_test)
#    test_dataset = Data.TensorDataset(P_D_test_normolization_tensor, P_G_test_normolization_tensor)
#    test_loader = Data.DataLoader(
#        dataset=test_dataset,      # torch TensorDataset format
#        batch_size=len(P_D_test_normolization_tensor),#batch_size_valtest,      # mini batch size
#        shuffle=False,               #
#        #num_workers=2,              #
#    )
#    
#    
#    for (test_x, test_y, test_theta) in test_loader:
#        test_out = model(test_x)
#        # calculate the slack bus output
#        Slack_index = np.where(np.array(Gen_index) == Slack_bus)[0][0]
#        Power_Lowbound_Gen_nn_test = torch.from_numpy(np.tile(Power_Lowbound_Gen, (len(test_x), 1)))
#        Power_Lowbound_Gen_nn_test_1 = Power_Lowbound_Gen_nn_test[:, 0:Slack_index]
#        Power_Lowbound_Gen_nn_test_1 = torch.cat([Power_Lowbound_Gen_nn_test_1, Power_Lowbound_Gen_nn_test[:,Slack_index+1:Gen_index.shape[0]]], 1)
#
#        Power_Upbound_Gen_nn_test = torch.from_numpy(np.tile(Power_Upbound_Gen, (len(test_x), 1)))
#        Power_Upbound_Gen_nn_test_1 = Power_Upbound_Gen_nn_test[:, 0:Slack_index]
#        Power_Upbound_Gen_nn_test_1 = torch.cat([Power_Upbound_Gen_nn_test_1, Power_Upbound_Gen_nn_test[:, Slack_index+1:Gen_index.shape[0]]], 1)
#
#        Pred_Pg_test = Power_Lowbound_Gen_nn_test_1 + (Power_Upbound_Gen_nn_test_1 - Power_Lowbound_Gen_nn_test_1) * test_out
#
##        Actual_PG_test = test_y.mul((Power_Upbound_Gen_nn_test - Power_Lowbound_Gen_nn_test) + Power_Lowbound_Gen_nn_test)
##        Pred_Pg_test_rev = torch.unsqueeze(torch.sum(Actual_PG_test[:, 0:Gen_index.shape[0]], 1),1) - torch.unsqueeze(torch.sum(Pred_Pg_test, 1), 1)
#        test_D = (test_x * (P_D_train_std_tensor + 1e-8) + P_D_train_mean_tensor)
#        Pred_Pg_test_rev = torch.unsqueeze(torch.sum(test_D[:, 0:Load_index.shape[0]], 1),1) - torch.unsqueeze(torch.sum(Pred_Pg_test, 1), 1)                
#        Pred_Pg_test_rev = torch.cat([Pred_Pg_test[:, 0:Slack_index], Pred_Pg_test_rev], 1)
#        Pred_Pg_test_rev = torch.cat([Pred_Pg_test_rev, Pred_Pg_test[:, Slack_index:Gen_index.shape[0]]], 1)
#
#        #PG
#        for i in range(Bus_number):
#            if i == 0:
#                if i in Gen_index:
#                    index = np.where(np.array(Gen_index)==i)[0]
#                    predictionG_test = Pred_Pg_test_rev[:, index]
#
#                else:
#                    predictionG_test = torch.zeros([len(test_x), 1])
#
#
#            else:
#                if i in Gen_index:
#                    index = np.where(np.array(Gen_index)==i)[0]
#                    predictionG_test = torch.cat([predictionG_test, Pred_Pg_test_rev[:, index]],1)
#                else:
#                    predictionG_test = torch.cat([predictionG_test, torch.zeros([len(test_x), 1])],1)
#
#        #PD
#
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
#
#        predictionInj_test = predictionG_test - predictionD_test  
#        predictionInj_test_1 = predictionInj_test[:,0:Slack_bus]
#        predictionInj_test_1 = torch.cat([predictionInj_test_1, predictionInj_test[:,Slack_bus+1:Bus_number]],1)
#
#        # theta calculation
#        theta_1_result_output_test = (Bus_1_Coeff_tensor.mm((predictionInj_test_1).t()))  
#        theta_result_output_test = torch.cat([torch.zeros([1, len(test_x)]), theta_1_result_output_test], 0)
#
#        slack_angle_array_test = np.ones([1,len(test_x)], dtype='float32')*Slack_angle
#        slack_angle_array_tensor_test = torch.from_numpy(slack_angle_array_test)
#                    
#        theta_result_output_test = theta_1_result_output_test[0:Slack_bus,:]
#        theta_result_output_test = torch.cat([theta_result_output_test, slack_angle_array_tensor_test],0).cuda()
#        theta_result_output_test = torch.cat([theta_result_output_test, theta_1_result_output_test[Slack_bus:Bus_number-1,:]],0).cuda()
#        theta_result_output_test = theta_result_output_test + Slack_angle
#
#
#        Bus_admittance_full_NN2_tensor = torch.from_numpy(Bus_admittance_full_NN2)
#        trans_result_output_test = (Bus_admittance_full_NN2_tensor.mm(theta_result_output_test).t()) 
#        trans_result_output_test_np = trans_result_output_test.cpu().detach().numpy()
#
#        if mpc["scopf"] == 1:
#        #inversve matrix of B (SCOPF)
#            SC_Bus_1_Coeff_tensor = torch.from_numpy(SC_Bus_admittance_full_1_array)
#            SC_theta_1_output_test = (predictionInj_test_1.matmul(SC_Bus_1_Coeff_tensor))
#            SC_theta_1_output_test = SC_theta_1_output_test + Slack_angle 
#            slack_angle_array = np.ones([NumofContigengcy,len(test_x),1], dtype='float32')*Slack_angle
#            slack_angle_array_tensor = torch.from_numpy(slack_angle_array)
#            
#            SC_theta_result_output_test = SC_theta_1_output_test[:,:,0:Slack_bus]
#            SC_theta_result_output_test = torch.cat([SC_theta_result_output_test, slack_angle_array_tensor],2)
#            SC_theta_result_output_test = torch.cat([SC_theta_result_output_test, SC_theta_1_output_test[:,:,Slack_bus:Bus_number-1]],2)
#                         
#            SC_theta_result_output_test_np = SC_theta_result_output_test.cpu().detach().numpy()
#            
##        test_theta_np = test_theta.cpu().numpy()
##        theta_result_output_cal_test_np = theta_result_output_test.detach().cpu().numpy()
#        
#        Pred_Pg_test_np = Pred_Pg_test_rev.detach().cpu().numpy()
#        Actual_PG_np = (test_y.mul(Power_Upbound_Gen_nn_test-Power_Lowbound_Gen_nn_test) + Power_Lowbound_Gen_nn_test.cpu()).numpy()
#        error = np.mean(abs((Actual_PG_np-Pred_Pg_test_np)/(Actual_PG_np+1e-8)))
#        print("\nPower test error is %.2f %%:\n" % (error*100))