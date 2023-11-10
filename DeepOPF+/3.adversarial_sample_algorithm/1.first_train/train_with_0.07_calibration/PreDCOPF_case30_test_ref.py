# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:48:14 2019

@author: xpan
"""
#PreDCOPF
from pypower.loadcase import loadcase
import numpy as np
from pypower.rundcopf import rundcopf

#processing time calculation
total = 0

if __name__ == '__main__':
    # global data
    P_D_list = []
    P_G_list = []
    NumOfGen = 0
    NumOfLoad = 0
    dir='.//test_data//'
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
                    P_D_list.append(float(odom[i]))
            else:
                odom  =line.split(';')
                if j == 2:
                    NumOfGen = len(odom)
                for i in range(len(odom)):
                    Gen_data = odom[i].split(':')
                    P_G_list.append(float(Gen_data[1]))                 
            j = j + 1
        fid.close()
    ################################################################
    Input_data = np.array(P_D_list, dtype=np.float32).reshape(-1, NumOfLoad)
    Output_data = np.array(P_G_list, dtype=np.float32).reshape(-1, NumOfGen)         
    #preparing model
    total_time = 0
    ave_cost = 0
    cost_list = []
    for i in range(0,Input_data.shape[0],1):
        mpc = loadcase('case30_rev_100.py')
        mpc_bus=mpc['bus']
        [m_bus,n_bus]=mpc_bus.shape
        mpc_branch=mpc['branch']
        mpc_branch[:,5]=mpc_branch[:,5]*1.017
        load_index = 0
        for j in range(0,m_bus,1):
            if(mpc_bus[j,2] != 0):
                mpc_bus[j,2] = Input_data[i,load_index]
                load_index = load_index + 1
#        start_time = time.time()
        r1 = rundcopf(mpc)
        total_time = total_time + r1['et']
        filename1='Pre_DC_test_case30-sampling_ref.txt'
        fid1=open(filename1,'a')
        filename2='cost_ref.txt'
        fid2=open(filename2,'a')
        fid2.write("%9.5f;\n" % (r1['f']))
        fid1.write("%9.5f;" % (r1['et']))  
        filename3='bind_ref.txt'
        fid3=open(filename3,'a')
        Ff = r1['branch'][:, 13]
        Ft = r1['branch'][:, 15]
        gen= r1['gen'][:, 1]
        ctol=5e-06
        ptol=0.0001
        if any((r1['branch'][:, 5] != 0) & (abs(Ff) > r1['branch'][:, 5] - ctol)) | any((r1['branch'][:, 5] != 0) & (abs(Ft) > r1['branch'][:, 5] - ctol)) | any(r1['branch'][:, 17] > ptol) | any(r1['branch'][:, 18] > ptol):
            fid1.write("line_active\n")
        else:
            fid1.write("\n")          
        bind_index = np.where((abs(Ff) > r1['branch'][:, 5] - ctol))
        bind2_index = np.where((gen > r1['gen'][:, 8] - ctol))
        bind3_index = np.where((gen < r1['gen'][:, 9] + ctol))
        fid3.write("%9.5f;%9.5f\n" % (len(bind_index[0]),len(bind2_index[0])+len(bind3_index[0]))) 
        fid1.close()
        fid2.close()
        ave_cost = ave_cost + r1['f']
        cost_list.append(r1['f'])
    print("Average total time :%.4f" % (total_time))
    print("Average cost :%.4f" % (ave_cost/Input_data.shape[0]))
    del Input_data, Output_data, mpc