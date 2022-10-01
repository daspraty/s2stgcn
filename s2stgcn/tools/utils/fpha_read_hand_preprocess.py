import scipy.io
import numpy as np
from scipy import signal
# from matplotlib import plot as plt
# file_path='/home/pratyusha/Pratyusha_workspace/project/DATA_SET/Hand_pose_annotation_ICL/action_sequences_normalized/sequences_proc.mat'
max_frame = 600

def read_xyz(file, max_body=2, num_joint=21):
    m=1
    numFrame,n=file.shape
    file1=file

    if max_frame>=numFrame:
        no_repeat=int(np.floor(max_frame/numFrame)-1)
        for i in range (no_repeat):
            file=np.concatenate((file,file1),axis=0)
    else:
        no_repeat=int(np.ceil(numFrame/max_frame))
        file=signal.decimate(file,no_repeat,axis=0)

    numFrame,nj=file.shape
    # print(file.shape)
    x=file[:,0::3]
    y=file[:,1::3]
    z=file[:,2::3]
    # print(x.shape)
    data = np.zeros((3, numFrame, num_joint, max_body))
    for n in range(numFrame):
        for j in range(num_joint):
            # if m < max_body and j < num_joint:
            # print(x[n,j])
            data[:, n, j, 0] = [x[n,j], y[n,j], z[n,j]]
            # else:
            #     pass
    # print(data.shape)
    # print(data[0,5,5,0])
    # error


    y=data
    y[:,:,1,:]=x.data[:,:,1,:]+x.data[:,:,5,:]
    y[:,:,5,:]=x.data[:,:,1,:]-x.data[:,:,5,:]
    y[:,:,2,:]=x.data[:,:,2,:]+x.data[:,:,4,:]
    y[:,:,4,:]=x.data[:,:,2,:]-x.data[:,:,4,:]
    y[:,:,6,:]=x.data[:,:,6,:]+x.data[:,:,10,:]
    y[:,:,10,:]=x.data[:,:,6,:]-x.data[:,:,10,:]
    y[:,:,7,:]=x.data[:,:,7,:]+x.data[:,:,9,:]
    y[:,:,9,:]=x.data[:,:,7,:]-x.data[:,:,9,:]

    y[:,:,11,:]=x.data[:,:,11,:]+x.data[:,:,15,:]
    y[:,:,15,:]=x.data[:,:,11,:]-x.data[:,:,15,:]
    y[:,:,12,:]=x.data[:,:,12,:]+x.data[:,:,14,:]
    y[:,:,14,:]=x.data[:,:,12,:]-x.data[:,:,14,:]
    y[:,:,16,:]=x.data[:,:,16,:]+x.data[:,:,20,:]
    y[:,:,20,:]=x.data[:,:,16,:]-x.data[:,:,20,:]
    y[:,:,17,:]=x.data[:,:,17,:]+x.data[:,:,19,:]
    y[:,:,19,:]=x.data[:,:,17,:]-x.data[:,:,19,:]
    y[:,:,0,:]=np.sqrt(2)*x.data[:,:,0,:]
    y[:,:,3,:]=np.sqrt(2)*x.data[:,:,3,:]
    y[:,:,8,:]=np.sqrt(2)*x.data[:,:,8,:]
    y[:,:,13,:]=np.sqrt(2)*x.data[:,:,13,:]
    y[:,:,18,:]=np.sqrt(2)*x.data[:,:,18,:]

    z=y
    z[:,:,1,:]=y[:,:,1,:]+y[:,:,2,:]
    z[:,:,2,:]=y[:,:,1,:]-y[:,:,2,:]
    z[:,:,6,:]=y[:,:,6,:]+y[:,:,7,:]
    z[:,:,7,:]=y[:,:,6,:]-y[:,:,7,:]
    z[:,:,11,:]=y[:,:,11,:]+y[:,:,12,:]
    z[:,:,12,:]=y[:,:,11,:]-y[:,:,12,:]
    z[:,:,16,:]=y[:,:,16,:]+y[:,:,17,:]
    z[:,:,17,:]=y[:,:,16,:]-y[:,:,17,:]
    z[:,:,0,:]=np.sqrt(2)*y[:,:,0,:]
    z[:,:,3,:]=np.sqrt(2)*y[:,:,3,:]
    z[:,:,8,:]=np.sqrt(2)*y[:,:,8,:]
    z[:,:,13,:]=np.sqrt(2)*y[:,:,13,:]
    z[:,:,18,:]=np.sqrt(2)*y[:,:,18,:]
    return z
