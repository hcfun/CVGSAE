from model import Model
from data import *
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
import numpy as np
from trainer import Trainer
import os
import time

class MetaTrainer(Trainer):
    def __init__(self, args):
        super(MetaTrainer, self).__init__(args)
        # dataset
        self.train_subgraph_iter = OneShotIterator(DataLoader(TrainSubgraphDataset(args),
                                                   batch_size=self.args.train_bs,
                                                   shuffle=True,
                                                   collate_fn=TrainSubgraphDataset.collate_fn))

        # model
        self.model = Model(args).to(args.gpu)

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # args for controlling training
        self.num_step = args.num_step
        self.log_per_step = args.log_per_step
        self.check_per_step = args.check_per_step
        self.early_stop_patience = args.early_stop_patience

        self.ce_loss = torch.nn.CrossEntropyLoss()

    def get_curr_state(self):
        state = {'model': self.model.state_dict()}
        return state

    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        self.model.load_state_dict(state['model'])

    def get_loss_task(self, tri, neg_tail_ent, neg_head_ent, ent_emb, rel_emb, time_emb):
        t1 = time.time()
        neg_tail_score = self.kge_model((tri, neg_tail_ent), ent_emb, rel_emb, time_emb, mode='tail-batch')
        neg_head_score = self.kge_model((tri, neg_head_ent), ent_emb, rel_emb, time_emb, mode='head-batch')
        neg_score = torch.cat([neg_tail_score, neg_head_score])
        neg_score = (F.softmax(neg_score * self.args.adv_temp, dim=1).detach()
                     * F.logsigmoid(-neg_score)).sum(dim=1)

        pos_score = self.kge_model(tri, ent_emb, rel_emb, time_emb)
        pos_score = F.logsigmoid(pos_score).squeeze(dim=1)

        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()
        loss = (positive_sample_loss + negative_sample_loss) / 2
        # print("task_time:{}".format(time.time()-t1))
        return loss

    def get_loss_secl_e(self, tri, neg_tail_ent, neg_head_ent, ent_emb, rel_emb, time_emb):
        temp = self.args.cl_temp
        #use z and zg (two representations)
        ##for intra entity cl
        neg_tail_score = self.kge_model((tri, neg_tail_ent), ent_emb, rel_emb, time_emb, mode='tail-batch')
        neg_head_score = self.kge_model((tri, neg_head_ent), ent_emb, rel_emb, time_emb, mode='head-batch')
        pos_score = self.kge_model(tri, ent_emb, rel_emb, time_emb)
        score = torch.cat([pos_score / temp, neg_tail_score / temp, neg_head_score / temp], dim=1)
        labels = torch.zeros(score.size(0)).long().to(self.args.gpu)
        loss_e = self.ce_loss(score, labels)
        ##for intra relation cl
        # neg_tail_score_r = self.pattern_kge_model((pattern_tri, pattern_neg_tail), rel_emb, metarel_emb, mode='tail-batch')
        # neg_head_score_r = self.pattern_kge_model((pattern_tri, pattern_neg_head), rel_emb, metarel_emb, mode='head-batch')
        # pos_score_r = self.pattern_kge_model(pattern_tri, rel_emb, metarel_emb)
        # score_r = torch.cat([pos_score_r / temp, neg_tail_score_r / temp, neg_head_score_r / temp], dim=1)
        # labels_r = torch.zeros(score_r.size(0)).long().to(self.args.gpu)
        # loss_r = self.ce_loss(score_r, labels_r)
        return loss_e

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.args.cl_temp)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    def get_loss_gcl(self, ent_emb, ent_emb_out, rel_emb, rel_emb_out):
        #for ent
        l1 = self.semi_loss(ent_emb, ent_emb_out)
        l2 = self.semi_loss(ent_emb_out, ent_emb)
        ret = (l1 + l2) * 0.5
        ret_e = ret.mean()

        l1_r = self.semi_loss(rel_emb, rel_emb_out)
        l2_r = self.semi_loss(rel_emb_out, rel_emb)
        ret_r = (l1_r + l2_r) * 0.5
        ret_r = ret_r.mean()

        "Full"
        return ret_e+ret_r


    def split_emb(self, emb, split_list):
        split_list = [np.sum(split_list[0: i], dtype=np.int) for i in range(len(split_list) + 1)]
        emb_split = [emb[split_list[i]: split_list[i + 1]] for i in range(len(split_list) - 1)]
        return emb_split

    def train_one_step(self):
        batch = next(self.train_subgraph_iter)
        batch_loss = 0

        batch_pattern_g = dgl.batch([d[1] for d in batch]).to(self.args.gpu)

        # for what? batch graph. with the id
        # num_metarel = 0
        for idx, d in enumerate(batch):
            d[0].edata['b_rel'] = d[0].edata['rel'] + torch.sum(batch_pattern_g.batch_num_nodes()[:idx]).cpu()
            # d[1].edata['b_rel'] = d[1].edata['rel'] + num_metarel
            # num_metarel += torch.unique(d[1].edata['rel']).shape[0]

        batch_sup_g = dgl.batch([d[0] for d in batch]).to(self.args.gpu) # dglbatch will re-idx the all-subgraph to avoid cross

        # ent_emb, ent_emb_out, rel_emb, rel_emb_out, time_emb, time_emb_out, metarel_emb, metarel_emb_out
        batch_ent_emb, batch_ent_emb_out, batch_rel_emb, batch_rel_emb_out, time_emb, time_emb_out, metarel_emb, metarel_emb_out = self.model(batch_sup_g, batch_pattern_g)

        # batch_vgae_loss = self.model.vgae_loss_f(batch_ent_emb, batch_ent_emb_out, batch_rel_emb, batch_rel_emb_out)/len(batch)

        batch_ent_emb = self.split_emb(batch_ent_emb, batch_sup_g.batch_num_nodes().tolist())
        batch_rel_emb = self.split_emb(batch_rel_emb, batch_pattern_g.batch_num_nodes().tolist())
        batch_ent_emb_out = self.split_emb(batch_ent_emb_out, batch_sup_g.batch_num_nodes().tolist())
        batch_rel_emb_out = self.split_emb(batch_rel_emb_out, batch_pattern_g.batch_num_nodes().tolist())

        total_time_emd = time_emb+time_emb_out
        total_metarel_emb = metarel_emb+metarel_emb_out

        # batch_ent_emb = self.split_emb(batch_ent_emb, batch_sup_g.batch_num_nodes().tolist())
        # batch_rel_emb = self.split_emb(batch_rel_emb, batch_pattern_g.batch_num_nodes().tolist())

        for batch_i, data in enumerate(batch):
            que_tri, que_neg_tail_ent, que_neg_head_ent = [d.to(self.args.gpu) for d in data[2:]]
            ent_emb = batch_ent_emb[batch_i]
            rel_emb = batch_rel_emb[batch_i]
            ent_emb_out = batch_ent_emb_out[batch_i]
            rel_emb_out = batch_rel_emb_out[batch_i]
            #Full
            loss_task = self.get_loss_task(que_tri, que_neg_tail_ent, que_neg_head_ent, ent_emb+ent_emb_out, rel_emb+rel_emb_out, total_time_emd)
            loss_vgae = self.model.vgae_loss_f(ent_emb, ent_emb_out, rel_emb, rel_emb_out)

            loss_secl_e = self.get_loss_secl_e(que_tri, que_neg_tail_ent, que_neg_head_ent, ent_emb+ent_emb_out, rel_emb+rel_emb_out, total_time_emd)

            loss_gcl = self.get_loss_gcl(ent_emb, ent_emb_out, rel_emb, rel_emb_out)

            #Full
            batch_loss += (self.args.alpha*loss_task+self.args.beta*(loss_gcl+loss_secl_e+loss_vgae))

        batch_loss /= len(batch)

        return batch_loss

    def get_eval_emb(self, eval_data):
        ent_emb, ent_emb_out, rel_emb, rel_emb_out, time_emb, time_emb_out, metarel_emb, metarel_emb_out = self.model(eval_data.g, eval_data.pattern_g)

        return ent_emb, ent_emb_out, rel_emb, rel_emb_out, time_emb, time_emb_out, metarel_emb, metarel_emb_out
