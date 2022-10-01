import os
import sys
import pickle
import scipy.io
import argparse
import numpy as np
from numpy.lib.format import open_memmap

from utils.fpha_read_hand import read_xyz


max_body = 1
num_joint = 21
max_frame = 600
toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            part='eval'):
    print(part)
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []

    i = input("Enter the test class : ")
    i=int(i)
    eos=[0,185,368,559,764,967,1175]
    test_mat=np.arange(eos[i-1],eos[i],1)
    train_mat=np.array([]).astype(int)
    for j in range(1,7):
        if j!=i:
            train_mat=np.concatenate((train_mat,np.array([i for i in range(eos[j-1],eos[j],1)])))

    data_mat_ = scipy.io.loadmat(data_path+'sequences_proc.mat')
    data_mat=data_mat_['sequences_proc'][0]

    label_mat_=scipy.io.loadmat(data_path+'labels.mat')
    label_mat=label_mat_['labels'][0]



    label_mat=label_mat-1



    if part == 'train':
        for q1, q2 in enumerate(train_mat):
            sample_name.append(str(q2))
        sample_label=list(label_mat[train_mat])
    elif part == 'val':
        for q1, q2 in enumerate(test_mat):
            sample_name.append(str(q2))
        sample_label=list(label_mat[test_mat])



    with open('{}/{}_label6.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(
        '{}/{}_data6.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))

    for i, s in enumerate(sample_name):
        # print_toolbar(i * 1.0 / len(sample_label),
        #               '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
        #                   i + 1, len(sample_name), part))
        data = read_xyz(
            data_mat[int(s)], max_body=max_body, num_joint=num_joint)
        # print(data.shape)
        fp[i, :, 0:data.shape[1], :, :] = data
    # end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ICL_hand_data')
    parser.add_argument(
        '--data_path', default='/DATA_SET/Hand_pose_annotation_ICL/action_sequences_normalized/'
)
    parser.add_argument(
        '--ignored_sample_path',
        default=None)
    parser.add_argument('--out_folder', default='data/fpha')

    part = ['train', 'val']
    arg = parser.parse_args()


    for p in part:
        out_path = arg.out_folder
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        gendata(
            arg.data_path,
            out_path,
            arg.ignored_sample_path,
            part=p)
