import random
import numpy as np
#import time
#import pandapower as pp
from pypower.loadcase import loadcase
from pypower.rundcopf import rundcopf

total = 0
maxviolation=np.array([0.34567416, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.06340984,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.10419586, 0.10532518,
       0.03616018, 0.03713043, 0.09370235, 0.        , 0.13373151,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        ]);
branch_capacity=np.array([1.3221 , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
       0.     , 0.     , 0.32544, 0.     , 0.     , 0.     , 0.     ,
       0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
       0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
       0.32544, 0.16272, 0.16272, 0.16272, 0.16272, 0.     , 0.16272,
       0.     , 0.     , 0.     , 0.     , 0.     , 0.     ])

for index in range(0,5,1):
    for i in range(0,10000):
        filename0= 'Pre_DC_training_case30-sampling_part_default_'+str(index)+'.txt'
        fid0=open(filename0,'at')        
        filename1= 'Pre_DC_training_case30-sampling_part_0.995_'+str(index)+'.txt'
        fid1=open(filename1,'at')
        filename2= 'Pre_DC_training_case30-sampling_part_0.990_'+str(index)+'.txt'
        fid2=open(filename2,'at')
        filename3= 'Pre_DC_training_case30-sampling_part_0.965_'+str(index)+'.txt'
        fid3=open(filename3,'at')        
        filename4= 'Pre_DC_training_case30-sampling_part_0.95_'+str(index)+'.txt'
        fid4=open(filename4,'at')
        filename6= 'Pre_DC_training_case30-sampling_part_0.93_'+str(index)+'.txt'
        fid6=open(filename6,'at')
        # filename3= 'Success_default_'+str(index)+'.txt'
        # fid3=open(filename3,'at')        
        filename5= 'Success_0.93_'+str(index)+'.txt'
        fid5=open(filename5,'at')
        # filename5= 'Success_0.9_'+str(index)+'.txt'
        # fid5=open(filename5,'at')
        
        mpc1 = loadcase('case30_rev_005.py')
        mpc1_bus=mpc1['bus']
        [m,n] = mpc1_bus.shape
        #########load:1.1-1.2,branch:0.95
        mpc1_bus[:,2]=mpc1_bus[:,2]*(1.15+ 2*(np.random.uniform(0,1,m)-0.5)*0.15)
        # mpc1_bus[:,2]=mpc1_bus[:,2]*(1.3)
        mpc1_branch=mpc1['branch']
        factor=0.005
        mpc1_branch[:,5]=mpc1_branch[:,5]*1.017-branch_capacity*factor*100
        #########load:1.1-1.2,branch:0.9
        mpc2 = loadcase('case30_rev_010.py')
        mpc2_bus=mpc2['bus']
        mpc2_bus[:,] = mpc1_bus[:,]
        mpc2_branch=mpc2['branch']
        factor=0.01
        mpc2_branch[:,5]=mpc2_branch[:,5]*1.017-branch_capacity*factor*100
        #########load:1.1-1.2,branch:default        
        mpc0 = loadcase('case30_rev_100.py')
        mpc0_bus=mpc0['bus']
        mpc0_bus[:,] = mpc1_bus[:,]
        mpc0_branch=mpc0['branch']
        mpc0_branch[:,5]=mpc0_branch[:,5]*1.017
        
        mpc3 = loadcase('case30_rev_035.py')
        mpc3_bus=mpc3['bus']
        mpc3_bus[:,] = mpc1_bus[:,]
        mpc3_branch=mpc3['branch']
        factor=0.035
        mpc3_branch[:,5]=mpc3_branch[:,5]*1.017-branch_capacity*factor*100
        
        mpc4 = loadcase('case30_rev_050.py')
        mpc4_bus=mpc4['bus']
        mpc4_bus[:,] = mpc1_bus[:,]
        mpc4_branch=mpc4['branch']
        factor=0.05
        mpc4_branch[:,5]=mpc4_branch[:,5]*1.017-branch_capacity*factor*100
        
        mpc6 = loadcase('case30_rev_070.py')
        mpc6_bus=mpc6['bus']
        mpc6_bus[:,] = mpc1_bus[:,]
        mpc6_branch=mpc6['branch']
        factor=0.07
        mpc6_branch[:,5]=mpc6_branch[:,5]*1.017-branch_capacity*factor*100
        #################################        
        r1 = rundcopf(mpc1)
        r2 = rundcopf(mpc2)
        r0 = rundcopf(mpc0) 
        r3 = rundcopf(mpc3)
        r4 = rundcopf(mpc4) 
        r6 = rundcopf(mpc6)        
        #, process_time
        #total = total+(process_time)
        r1_bus = r1['bus']
        r2_bus = r2['bus']        
        r0_bus = r0['bus']   
        r3_bus = r3['bus']        
        r4_bus = r4['bus']   
        r6_bus = r6['bus']
        for j in range(0,m,1):
            if(r1_bus[j,2] > 0):
                if j != m-1:
                    fid1.write("%9.4f;" % (r1_bus[j,2]))                   
                else:
                    fid1.write("%9.4f\n" % (r1_bus[j,2]))
            if(r2_bus[j,2] > 0):
                if j != m-1:
                    fid2.write("%9.4f;" % (r2_bus[j,2]))                   
                else:
                    fid2.write("%9.4f\n" % (r2_bus[j,2]))
            if(r0_bus[j,2] > 0):
                if j != m-1:
                    fid0.write("%9.4f;" % (r0_bus[j,2]))                   
                else:
                    fid0.write("%9.4f\n" % (r0_bus[j,2]))  
            if(r3_bus[j,2] > 0):
                if j != m-1:
                    fid3.write("%9.4f;" % (r3_bus[j,2]))                   
                else:
                    fid3.write("%9.4f\n" % (r3_bus[j,2])) 
            if(r4_bus[j,2] > 0):
                if j != m-1:
                    fid4.write("%9.4f;" % (r4_bus[j,2]))                   
                else:
                    fid4.write("%9.4f\n" % (r4_bus[j,2])) 
            if(r6_bus[j,2] > 0):
                if j != m-1:
                    fid6.write("%9.4f;" % (r6_bus[j,2]))                   
                else:
                    fid6.write("%9.4f\n" % (r6_bus[j,2]))

        r1_gen=r1['gen']
        r2_gen=r2['gen']
        r0_gen=r0['gen']
        r3_gen=r3['gen']
        r4_gen=r4['gen']
        r6_gen=r6['gen']
        
        m1 = len(r1_gen)
        for j in range(0,m1,1):
            if j != m1-1:
                fid1.write("%d:%9.4f;" % (int(r1_gen[j,0]),r1_gen[j,1]))
            else:
                fid1.write("%d:%9.4f\n" % (int(r1_gen[j,0]),r1_gen[j,1]))
        for j in range(0,m1,1):
            if j != m1-1:
                fid2.write("%d:%9.4f;" % (int(r2_gen[j,0]),r2_gen[j,1]))
            else:
                fid2.write("%d:%9.4f\n" % (int(r2_gen[j,0]),r2_gen[j,1]))
        for j in range(0,m1,1):
            if j != m1-1:
                fid0.write("%d:%9.4f;" % (int(r0_gen[j,0]),r0_gen[j,1]))
            else:
                fid0.write("%d:%9.4f\n" % (int(r0_gen[j,0]),r0_gen[j,1]))
        for j in range(0,m1,1):
            if j != m1-1:
                fid3.write("%d:%9.4f;" % (int(r3_gen[j,0]),r3_gen[j,1]))
            else:
                fid3.write("%d:%9.4f\n" % (int(r3_gen[j,0]),r3_gen[j,1]))
        for j in range(0,m1,1):
            if j != m1-1:
                fid4.write("%d:%9.4f;" % (int(r4_gen[j,0]),r4_gen[j,1]))
            else:
                fid4.write("%d:%9.4f\n" % (int(r4_gen[j,0]),r4_gen[j,1]))
        for j in range(0,m1,1):
            if j != m1-1:
                fid6.write("%d:%9.4f;" % (int(r6_gen[j,0]),r6_gen[j,1]))
            else:
                fid6.write("%d:%9.4f\n" % (int(r6_gen[j,0]),r6_gen[j,1]))
        
        # if (r1['success']== True):
        #     fid4.write("%d" % (0))
        # else:
        #     fid4.write("%d" % (1))
            
        if (r6['success']== True):
            fid5.write("%d" % (0))
        else:
            fid5.write("%d" % (1))
            
        # if (r0['success']== True):
        #     fid3.write("%d" % (0))
        # else:
        #     fid3.write("%d" % (1))
#        fid4.write("%d:%9.4f\n" % (int(r0_gen[j,0]),r0_gen[j,1]))
#        fid5.write("%d:%9.4f\n" % (int(r0_gen[j,0]),r0_gen[j,1]))
        
        r1=[]
        r2=[]
        r0=[]
        r3=[]
        r4=[]
        r6=[]
    fid1.close()
    fid2.close()
    fid0.close()
    fid3.close()
    fid4.close()
    fid5.close()
    fid6.close()
#print("total_time %.4f \n" % total)
