# CVGSAE
The implementation of 'Contrastive Variational Graph Symmetric Autoencoder for Full Extrapolation over Temporal Knowledge Graphs'.

[//]: # (This repository contains the experimental code for our IJCAI 2022 paper: [Meta-Learning Based Knowledge Extrapolation for Knowledge Graphs in the Federated Setting]&#40;https://arxiv.org/abs/2205.04692&#41;. We study the knowledge extrapolation problem to embed new components &#40;i.e., entities and relations&#41; that come with emerging knowledge graphs &#40;KGs&#41; in the federated setting. In this problem, a model trained on an existing KG needs to embed an emerging KG with unseen entities and relations. )

[//]: # ()
[//]: # ()
[//]: # (![method]&#40;./fig/method.png&#41;)

## Requirements
+ python 3.9
+ pytorch 1.12.1
+ dgl 1.1.1+cu113
+ numpy
+ lmdb 1.4.1
+ argparse

## Dataset

We put datasets uased in our paper in ``./data``, and each dataset is serialized by ``pickle``. You can load the data by:

```python
import pickle
data = pickle.load(open('.data/icews14/day/icews14_ext.pkl', 'rb'))
```
Each dataset is formulated as a python ``dict``, like:

```python
data = {
'train': {
        'triples': [[0, 1, 2, 3], [3, 4, 5, 6], ...] # list of facts in (h, r, t, \tau), denoted by corresponding indexes
        'ent2id': {'abc':0, 'def':1, ...} # map entity name from original dataset (e.g., ICEWS14) to the index of above facts
        'rel2id': {'xyz':0, 'ijk':1, ...} # map relation name from original dataset (e.g.,ICEWS14) to the index of above facts
        'time2id': {'2005-01-01':0, '2005-01-02':1, ...} # map timestamp name from original dataset (e.g., ICEWS14) to the index of above facts
}

'valid': {
        'support': # support facts
        'query': # query facts
        'ent_map_list': [0, -1, 4, -1, -1, ...] # map entity indexes to train entities, -1 denotes an unseen entity
        'rel_map_list': [-1, 2, -1, -1, -1, ...] # map relation indexes to train relation, -1 denotes an unseen relation
        'time_map_list': [-1, 1, 2, -1, -1, ...] # map timestamp indexes to train timestamps, -1 denotes an unseen timestamp
        'ent2id':
        'rel2id':
        'time2id':
}

'test': {
        'support': 
        'query_uent': # query facts only containing unseen entities
        'query_urel': # query facts only containing unseen relations
        'query_utime':  # query facts only containing unseen timestamps
        'query_uentrel': # query facts containing both unseen entities and relations
        'query_uenttime': # query facts containing both unseen entities and timestamps
        'query_ureltime': # query facts containing both unseen relations and timestamps
        'query_uall': # query facts simultaneously containing unseen entities, relations, and timestamps
        'ent_map_list': 
        'rel_map_list': 
        'time_map_list':
        'ent2id':
        'rel2id':
        'time2id':
}}
```

## Training
You can easily run our code. For example, the following is for running on ICEWS14_Ext.
```bash
python main_ic14.py 
```



[//]: # (You can try our code easily by runing the scripts in ``./script``, for example:)

[//]: # (```bash)

[//]: # (bash ./script/run_fb_ext.sh)

[//]: # (```)

[//]: # (The training losses and validation results will be printed and saved in the corresponding log file in ``./log``. You can check the log based on the ``task_name`` and the number of the current experiment; for example, for the first run of the task with name ``fb_ext_transe``, you can check the log in ``./log/fb_ext_transe_run0.log``. Furthermore, you can find more detail results in ``./log/fb_ext_transe_50_uent.csv``, which records the results of sampling 50 negative candidate triples for _urel_ query triples.)

[//]: # ()
[//]: # (We put the tensorboard log files in ``./tb_log`` and trained model state dicts in ``./state``.)

[//]: # (## Citation)

[//]: # (Please cite our paper if you use our model in your work:)

[//]: # ()
[//]: # (```latex)

[//]: # (@inproceedings{MaKEr,)

[//]: # (  title     = {Meta-Learning Based Knowledge Extrapolation for Knowledge Graphs in the Federated Setting},)

[//]: # (  author    = {Chen, Mingyang and Zhang, Wen and Yao, Zhen and Chen, Xiangnan and Ding, Mengxiao and Huang, Fei and Chen, Huajun},)

[//]: # (  booktitle = {Proceedings of the Thirty-First International Joint Conference on)

[//]: # (               Artificial Intelligence, {IJCAI-22}},)

[//]: # (  publisher = {International Joint Conferences on Artificial Intelligence Organization},)

[//]: # (  editor    = {Lud De Raedt},)

[//]: # (  pages     = {1966--1972},)

[//]: # (  year      = {2022},)

[//]: # (  month     = {7},)

[//]: # (  note      = {Main Track})

[//]: # (  doi       = {10.24963/ijcai.2022/273},)

[//]: # (  url       = {https://doi.org/10.24963/ijcai.2022/273},)

[//]: # (})

[//]: # (```)


