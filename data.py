
import torch
import dgl
import numpy as np
from scipy import sparse
from collections import defaultdict as ddict
from torch.utils.data import Dataset
import lmdb
from utils import deserialize
import random
import time


class TrainSubgraphDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.env = lmdb.open(args.db_path, readonly=True, max_dbs=1, lock=False)
        self.subgraphs_db = self.env.open_db("train_subgraphs".encode())

    def __len__(self):
        return self.args.num_train_subgraph

    @staticmethod
    def collate_fn(data):
        return data

    def get_train_g(self, sup_tri, ent_map_list, ent_mask):
        triples = torch.LongTensor(sup_tri)
        num_tri = triples.shape[0]
        g = dgl.graph((torch.cat([triples[:, 0].t(), triples[:, 2].t()]),
                       torch.cat([triples[:, 2].t(), triples[:, 0].t()])))
        #inverse relations
        g.edata['rel'] = torch.cat([triples[:, 1].t(), triples[:, 1].t()])
        g.edata['inv'] = torch.cat([torch.zeros(num_tri), torch.ones(num_tri)])
        g.edata['time'] = torch.cat([triples[:, 3].t(), triples[:, 3].t()])

        ent_mask_list = np.array(list(map(lambda x: x in ent_mask, np.arange(len(ent_map_list)))))
        ent_map_list = np.array(ent_map_list)
        ent_map_list[ent_mask_list] = -1

        """
        change type to long
        """
        g.ndata['ori_idx'] = torch.tensor(ent_map_list).long()

        return g

    def get_pattern_g(self, pattern_tri, rel_map_list, rel_mask):
        triples = torch.LongTensor(pattern_tri)
        num_tri = triples.shape[0]
        g = dgl.graph((torch.cat([triples[:, 0].t(), triples[:, 2].t()]),
                       torch.cat([triples[:, 2].t(), triples[:, 0].t()])))
        g.edata['rel'] = torch.cat([triples[:, 1].t(), triples[:, 1].t()])
        g.edata['inv'] = torch.cat([torch.zeros(num_tri), torch.ones(num_tri)])

        rel_mask_list = np.array(list(map(lambda x: x in rel_mask, np.arange(len(rel_map_list)))))
        rel_map_list = np.array(rel_map_list)
        rel_map_list[rel_mask_list] = -1

        """
        change type to long
        """
        g.ndata['ori_idx'] = torch.tensor(rel_map_list).long()

        return g

    def get_pattern_hr2t_rt2h(self, triples):
        hr2t = ddict(list)
        rt2h = ddict(list)
        for tri in triples:
            h, r, t = tri
            hr2t[(h,r)].append(t)
            rt2h[(r,t)].append(h)

        return hr2t, rt2h


    def __getitem__(self, idx):
        with self.env.begin(db=self.subgraphs_db) as txn:
            str_id = '{:08}'.format(idx).encode('ascii')
            sup_tri, pattern_tri, que_tri, hr2t, rt2h, ent_reidx_list, rel_reidx_list = deserialize(txn.get(str_id))

        nentity = len(ent_reidx_list)

        pattern_hr2t, pattern_rt2h = self.get_pattern_hr2t_rt2h(pattern_tri)
        nrelation =len(rel_reidx_list)
        t1 = time.time()
        que_neg_tail_ent = [np.random.choice(np.delete(np.arange(nentity), hr2t[(h, r, T)]),
                                        self.args.metatrain_num_neg) for h, r, t, T in que_tri]

        que_neg_head_ent = [np.random.choice(np.delete(np.arange(nentity), rt2h[(r, t, T)]),
                                        self.args.metatrain_num_neg) for h, r, t, T in que_tri]
        # print("neg_e_time:{}".format(time.time() - t1))

        # pattern_neg_tail = [np.random.choice(np.delete(np.arange(nrelation), pattern_hr2t[(h, r)]),
        #                                 self.args.metatrain_num_pattern_neg) for h, r, t in pattern_tri]
        # pattern_neg_tail = self.get_neg_tail(pattern_tri, pattern_hr2t, nrelation)
        # pattern_neg_head = self.get_neg_head(pattern_tri, pattern_rt2h, nrelation)
        # pattern_neg_tail = [self.get_neg_ent(pattern_hr2t[(h, r)], nrelation) for h,r,t in pattern_tri]
        # pattern_neg_head = [self.get_neg_ent(pattern_hr2t[(r, t)], nrelation) for h, r, t in pattern_tri]
        # pattern_neg_head = [np.random.choice(np.delete(np.arange(nrelation), pattern_rt2h[(r, t)]),
        #                                 self.args.metatrain_num_pattern_neg) for h, r, t in pattern_tri]
        # print("neg_r_time:{}".format(time.time() - t1))

        # During training, we randomly treat entities and relations as
        # unseen with the ratio of 30% ∼ 80% for each task.
        ent_mask = np.random.choice(np.arange(len(ent_reidx_list)),
                                    int(len(ent_reidx_list) * random.randint(3, 8) * 0.1), replace=False)
        rel_mask = np.random.choice(np.arange(len(rel_reidx_list)),
                                    int(len(rel_reidx_list) * random.randint(3, 8) * 0.1), replace=False)

        g = self.get_train_g(sup_tri, ent_reidx_list, ent_mask)
        pattern_g = self.get_pattern_g(pattern_tri, rel_reidx_list, rel_mask)

        return g, pattern_g, torch.tensor(np.array(que_tri)), \
               torch.tensor(np.array(que_neg_tail_ent)), torch.tensor(np.array(que_neg_head_ent))
    def get_neg_ent(self, pattern_hr2t, nrelation):
        # neg_list = list()
        # for h,r,t in triple:
        neg_ent = set()
        while len(neg_ent) < self.args.metatrain_num_pattern_neg:
            neg = np.random.randint(0, nrelation)
            if neg not in pattern_hr2t:
                neg_ent.add(neg)
        # neg_list.append(list(neg_ent))
            # neg_ent = np.int32(list(neg_ent))
        return list(neg_ent)
    # def get_neg_head(self, triple, pattern_hr2t, nrelation):
    #     neg_list = list()
    #     for h,r,t in triple:
    #         neg_ent = set()
    #         while len(neg_ent) < self.args.metatrain_num_pattern_neg:
    #             neg = np.random.randint(0, nrelation)
    #             if neg not in pattern_hr2t[r,t]:
    #                 neg_ent.add(neg)
    #         neg_list.append(list(neg_ent))
    #         # neg_ent = np.int32(list(neg_ent))
    #     return np.int32(neg_list)

class EvalDataset(Dataset):
    def __init__(self, args, data, que_triples):
        self.args = args

        self.hr2t = data.hr2t_all
        self.rt2h = data.rt2h_all
        self.triples = que_triples

        self.num_ent = data.num_ent

        self.num_cand = 'all'

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        pos_triple = self.triples[idx]
        h, r, t, T = pos_triple
        if self.num_cand == 'all':
            tail_label, head_label = self.get_label(self.hr2t[(h, r, T)], self.rt2h[(r, t, T)])
            pos_triple = torch.LongTensor(pos_triple)

            return pos_triple, tail_label, head_label
        else:
            neg_tail_cand = np.random.choice(np.delete(np.arange(self.num_ent), self.hr2t[(h, r, T)]),
                                             self.num_cand)

            neg_head_cand = np.random.choice(np.delete(np.arange(self.num_ent), self.rt2h[(r, t, T)]),
                                             self.num_cand)
            tail_cand = torch.from_numpy(np.concatenate((neg_tail_cand, [t])))
            head_cand = torch.from_numpy(np.concatenate((neg_head_cand, [h])))

            pos_triple = torch.LongTensor(pos_triple)

            return pos_triple, tail_cand, head_cand

    def get_label(self, true_tail, true_head):
        y_tail = np.zeros([self.num_ent], dtype=np.float32)
        for e in true_tail:
            y_tail[e] = 1.0
        y_head = np.zeros([self.num_ent], dtype=np.float32)
        for e in true_head:
            y_head[e] = 1.0

        return torch.FloatTensor(y_tail), torch.FloatTensor(y_head)

    @staticmethod
    def collate_fn(data):
        pos_triple = torch.stack([_[0] for _ in data], dim=0)
        tail_label_or_cand = torch.stack([_[1] for _ in data], dim=0)
        head_label_or_cand = torch.stack([_[2] for _ in data], dim=0)
        return pos_triple, tail_label_or_cand, head_label_or_cand


class Data(object):
    def __init__(self, args, data):
        self.args = args

        self.entity_dict = data['ent2id']
        self.relation_dict = data['rel2id']
        self.time_dict = data['time2id']

        self.num_ent = len(self.entity_dict)
        self.num_rel = len(self.relation_dict)
        self.num_time = len(self.time_dict)

    def get_train_g(self, sup_tri, ent_reidx_list=None):
        triples = torch.LongTensor(sup_tri)
        num_tri = triples.shape[0]
        g = dgl.graph((torch.cat([triples[:, 0].t(), triples[:, 2].t()]),
                       torch.cat([triples[:, 2].t(), triples[:, 0].t()])))
        g.edata['rel'] = torch.cat([triples[:, 1].t(), triples[:, 1].t()])
        g.edata['b_rel'] = torch.cat([triples[:, 1].t(), triples[:, 1].t()])
        g.edata['inv'] = torch.cat([torch.zeros(num_tri), torch.ones(num_tri)])
        g.edata['time'] = torch.cat([triples[:, 3].t(), triples[:, 3].t()])

        if ent_reidx_list is None:
            g.ndata['ori_idx'] = torch.tensor(np.arange(g.num_nodes()))
        else:
            g.ndata['ori_idx'] = torch.tensor(ent_reidx_list)

        return g

    def get_pattern_g(self, sup_facts, rel_reidx_list=None):
        rel_head = torch.zeros((self.num_rel, self.num_ent), dtype=torch.int)
        rel_tail = torch.zeros((self.num_rel, self.num_ent), dtype=torch.int)
        rel_time = torch.zeros((self.num_rel, self.num_time), dtype=torch.int)
        for tri in sup_facts:
            h, r, t, T = tri
            rel_head[r, h] += 1
            rel_tail[r, t] += 1
            rel_time[r, T] = T + 1

        # adjacency matrix for rel and rel of different pattern
        tail_head = torch.matmul(rel_tail, rel_head.t())
        head_tail = torch.matmul(rel_head, rel_tail.t())
        tail_tail = torch.matmul(rel_tail, rel_tail.t()) - torch.diag(torch.sum(rel_tail, axis=1))
        head_head = torch.matmul(rel_head, rel_head.t()) - torch.diag(torch.sum(rel_head, axis=1))

        rel_time_max = torch.max(rel_time, dim=1).values
        # rel_time_max = torch.min(rel_time, dim=1).values
        rel_time_max = rel_time_max.repeat((rel_time_max.shape[0], 1))
        rel_time_max_t = rel_time_max.t()

        forward = torch.gt(rel_time_max, rel_time_max_t).long()
        backward = 1 - torch.ge(rel_time_max, rel_time_max_t).long()
        equal = torch.eq(rel_time_max, rel_time_max_t).long() - torch.diag(torch.ones(rel_time_max.shape[0]))

        pattern_mat_list = []

        for idx1, mat1 in enumerate([tail_head, head_tail, tail_tail, head_head]):
            for idx2, mat2 in enumerate([forward, backward, equal]):
                pattern_mat_list.append(mat1 * mat2)

        # construct pattern graph from adjacency matrix
        src = torch.LongTensor([])
        dst = torch.LongTensor([])
        p_rel = torch.LongTensor([])
        for p_rel_idx, mat in enumerate(pattern_mat_list):
            sp_mat = sparse.coo_matrix(mat)
            src = torch.cat([src, torch.from_numpy(sp_mat.row)])
            dst = torch.cat([dst, torch.from_numpy(sp_mat.col)])
            p_rel = torch.cat([p_rel, torch.LongTensor([p_rel_idx] * len(sp_mat.data))])


        num_tri = src.shape[0]
        # g = dgl.graph((src, dst), num_nodes=self.num_rel)
        # g.edata['rel'] = p_rel

        g = dgl.graph((torch.cat([src, dst]),
                       torch.cat([dst, src])))
        g.edata['rel'] = torch.cat([p_rel, p_rel])
        g.edata['b_rel'] = torch.cat([p_rel, p_rel])
        g.edata['inv'] = torch.cat([torch.zeros(num_tri), torch.ones(num_tri)])

        if rel_reidx_list is None:
            g.ndata['ori_idx'] = torch.tensor(np.arange(g.num_nodes()))
        else:
            g.ndata['ori_idx'] = torch.tensor(rel_reidx_list)

        return g

    def get_hr2t_rt2h(self, triples):
        hr2t = ddict(list)
        rt2h = ddict(list)
        for tri in triples:
            h, r, t, T = tri
            hr2t[(h, r, T)].append(t)
            rt2h[(r, t, T)].append(h)

        return hr2t, rt2h


# class TrainData(Data):
#     def __init__(self, args, data):
#         super(TrainData, self).__init__(args, data)
#         self.train_triples = data['triples']
#
#         self.hr2t_train, self.rt2h_train = self.get_hr2t_rt2h(self.train_triples)
#
#         # g and pattern g
#         self.g = self.get_train_g(self.train_triples).to(args.gpu)
#
#         # self.pattern_tri = self.get_pattern_tri(self.train_triples)
#         self.pattern_g = self.get_pattern_g(self.train_triples).to(args.gpu)


class ValidData(Data):
    def __init__(self, args, data):
        super(ValidData, self).__init__(args, data)
        self.sup_triples = data['support']
        self.que_triples = data['query']

        self.ent_map_list = data['ent_map_list']
        self.rel_map_list = data['rel_map_list']

        self.hr2t_all, self.rt2h_all = self.get_hr2t_rt2h(self.sup_triples + self.que_triples)

        # g and pattern g
        self.g = self.get_train_g(self.sup_triples, ent_reidx_list=self.ent_map_list).to(args.gpu)

        # self.pattern_tri = self.get_pattern_tri(self.sup_triples)
        self.pattern_g = self.get_pattern_g(self.sup_triples, rel_reidx_list=self.rel_map_list).to(args.gpu)


class TestData(Data):
    def __init__(self, args, data):
        super(TestData, self).__init__(args, data)
        self.sup_triples = data['support']
        self.que_triples = data['query_uent'] + data['query_urel'] + data['query_utime'] + data['query_uentrel'] + data['query_uenttime']+ data['query_ureltime'] + data['query_uall']
        self.que_uent = data['query_uent']
        self.que_urel = data['query_urel']
        self.que_utime = data['query_utime']
        self.que_uentrel = data['query_uentrel']
        self.que_uenttime = data['query_uenttime']
        self.que_ureltime = data['query_ureltime']
        self.que_uall = data['query_uall']

        self.ent_map_list = data['ent_map_list']
        self.rel_map_list = data['rel_map_list']

        self.hr2t_all, self.rt2h_all = self.get_hr2t_rt2h(self.sup_triples + self.que_triples)

        # g and pattern g
        self.g = self.get_train_g(self.sup_triples, ent_reidx_list=self.ent_map_list).to(args.gpu)

        # self.pattern_tri = self.get_pattern_tri(self.sup_triples)
        self.pattern_g = self.get_pattern_g(self.sup_triples, rel_reidx_list=self.rel_map_list).to(args.gpu)


class TrainDatasetMode(Dataset):
    def __init__(self, args, data, mode):
        self.args = args
        self.triples = data.train_triples
        self.num_ent = data.num_ent
        self.num_neg = args.kge_num_neg
        self.hr2t = data.hr2t_train
        self.rt2h = data.rt2h_train
        self.mode = mode

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail, time = positive_sample

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.num_neg:
            negative_sample = np.random.randint(self.num_ent, size=self.num_neg * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.rt2h[(relation, tail, time)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.hr2t[(head, relation, time)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.num_neg]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, self.mode


    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, negative_sample, mode


class OneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)

    def __next__(self):
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a.md PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a.md PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
