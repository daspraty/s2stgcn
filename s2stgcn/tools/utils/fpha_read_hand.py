import scipy.io
import numpy as np
from scipy import signal
# from matplotlib import plot as plt
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

    return data
