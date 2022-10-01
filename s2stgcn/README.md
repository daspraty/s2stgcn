# S2STGCN

## Introduction
This repository holds the codebase, dataset and models for the paper:

**Symmetric sub-graph spatio-temporal graph convolution and its application in complex activity recognition** Pratyusha Das, Antonio Ortega, ICASSP2021 (https://ieeexplore.ieee.org/abstract/document/9413833)


The code is written for First-Person Hand Action Benchmark 3D Hand Pose dataset (https://guiggh.github.io/publications/first-person-hands/)



## Prerequisites
- Python3 (>3.5)
- [PyTorch](http://pytorch.org/)

- Other Python libraries can be installed by `pip install -r requirements.txt`


### Installation
``` shell
git clone https://github.com/daspraty/s2stgcn.git; cd s2stgcn
cd torchlight; python setup.py install; cd ..
```


## Data Preparation

We experimented on tFPHA datasts:
Before training and testing,
the datasets should be preprocessed and converted to proper file structure.



Otherwise, for processing raw data by yourself,
please refer to below guidances.

#### FPHA
FPHA can be downloaded from [their website](https://guiggh.github.io/publications/first-person-hands/).
Only the **3D 3D Hand Pose** modality is required in our experiments. After that, this command should be used to build the database for training or evaluation:
```
python tools/fpha_gendata_hand_trtst.py --data_path <FPHA>
```



## Testing Pretrained Models

<!-- ### Evaluation
Once datasets ready, we can start the evaluation. -->

For **cross-subject** evaluation in **FPHA**, run
```
python main.py recognition -c config/s2stgcn/hand/test.yaml
```


To speed up evaluation by multi-gpu inference or modify batch size for reducing the memory cost, set ```--test_batch_size``` and ```--device``` like:
```
python main.py recognition -c <config file> --test_batch_size <batch size> --device <gpu0> <gpu1> ...
```



## Training
To train a new S2STGCN model, run


#hand
''' python main.py recognition -c config/s2stgcn/hand/train.yaml '''

where the ```<dataset>``` must be ```fpha-xsub```

The training results, including **model weights**, configurations and logging files, will be saved under the ```./work_dir``` by default or ```<work folder>``` if you appoint it.

You can modify the training parameters such as ```work_dir```, ```batch_size```, ```step```, ```base_lr``` and ```device``` in the command line or configuration files. The order of priority is:  command line > config file > default parameter. For more information, use ```main.py -h```.

Finally, custom model evaluation can be achieved by this command as we mentioned above:

#hand
''' python main.py recognition -c config/s2stgcn/hand/test.yaml '''

## Citation
Please cite the following paper if you use this repository in your research.
```
@INPROCEEDINGS{s2stgcn,
  author={Das, Pratyusha and Ortega, Antonio},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Symmetric Sub-graph Spatio-Temporal Graph Convolution and its application in Complex Activity Recognition},
  year={2021},
  volume={},
  number={},
  pages={3215-3219},
  doi={10.1109/ICASSP39728.2021.9413833}}
```

## Contact
For any question, feel free to contact
```
Pratyusha Das    : daspraty@usc.edu

```
