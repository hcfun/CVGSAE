import dgl
import torch.nn as nn
import torch
import dgl.function as fn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class Ext_entLayer(nn.Module):
    def __init__(self, args, act=None):
        super(Ext_entLayer, self).__init__()
        self.args = args
        self.act = act

        # define in/out/loop transform layer
        self.W_O = nn.Linear(args.time_dim + args.rel_dim + args.ent_dim, args.ent_dim)
        self.W_I = nn.Linear(args.time_dim + args.rel_dim + args.ent_dim, args.ent_dim)
        self.W_S = nn.Linear(args.ent_dim, args.ent_dim)

        self.W_T = nn.Linear(args.time_dim, args.time_dim)
        self.drop = nn.Dropout(args.gcn_drop)
    def msg_func(self, edges):
        comp_h = torch.cat((edges.data['h'], edges.src['h'], edges.data['time_h']), dim=-1)

        non_inv_idx = (edges.data['inv'] == 0)
        inv_idx = (edges.data['inv'] == 1)

        msg = torch.zeros_like(edges.src['h'])
        msg[non_inv_idx] = self.W_I(comp_h[non_inv_idx])
        msg[inv_idx] = self.W_O(comp_h[inv_idx])

        return {'msg': msg}

    def apply_node_func(self, nodes):
        h_new = self.W_S(nodes.data['h']) + self.drop(nodes.data['h_agg'])

        if self.act is not None:
            h_new = self.act(h_new)

        return {'h': h_new}

    def time_update(self, time_emb):
        time_edge_new = self.W_T(time_emb)

        if self.act is not None:
            time_edge_new = self.act(time_edge_new)

        return time_edge_new



    def forward(self, g, ent_emb, rel_emb, time_emb):
        with g.local_scope():
            g.edata['h'] = rel_emb[g.edata['b_rel']]
            g.edata['time_h'] = torch.index_select(time_emb, dim=0, index=g.edata['time'])
            g.ndata['h'] = ent_emb

            g.update_all(self.msg_func, fn.mean('msg', 'h_agg'), self.apply_node_func)

            time_emb = self.time_update(time_emb)
            ent_emb = g.ndata['h']

            # rel_emb = self.rel_update(g, ent_emb, rel_emb, time_emb)

        return ent_emb, time_emb


class Ext_relLayer(nn.Module):
    def __init__(self, args, act=None):
        super(Ext_relLayer, self).__init__()
        self.args = args
        self.act = act

        # define in/out/loop transform layer
        self.W_O = nn.Linear(args.rel_dim*2, args.rel_dim)
        self.W_I = nn.Linear(args.rel_dim*2, args.rel_dim)
        self.W_S = nn.Linear(args.rel_dim, args.rel_dim)

        self.W_M = nn.Linear(args.rel_dim, args.rel_dim)
        self.drop = nn.Dropout(args.gcn_drop)
    def msg_func(self, edges):
        comp_h = torch.cat((edges.data['h'], edges.src['h']), dim=-1)

        non_inv_idx = (edges.data['inv'] == 0)
        inv_idx = (edges.data['inv'] == 1)

        msg = torch.zeros_like(edges.src['h'])
        msg[non_inv_idx] = self.W_I(comp_h[non_inv_idx])
        msg[inv_idx] = self.W_O(comp_h[inv_idx])

        return {'msg': msg}

    def apply_node_func(self, nodes):
        h_new = self.W_S(nodes.data['h']) + self.drop(nodes.data['h_agg'])

        if self.act is not None:
            h_new = self.act(h_new)

        return {'h': h_new}

    def metaedge_update(self, metarel_emb):
        h_edge_new = self.W_M(metarel_emb)

        if self.act is not None:
            h_edge_new = self.act(h_edge_new)

        return h_edge_new



    def forward(self, g, rel_emb, metarel_emb):
        with g.local_scope():
            g.edata['h'] = metarel_emb[g.edata['rel']]
            g.ndata['h'] = rel_emb

            g.update_all(self.msg_func, fn.mean('msg', 'h_agg'), self.apply_node_func)

            metarel_emb = self.metaedge_update(metarel_emb)

            rel_emb = g.ndata['h']

        return rel_emb, metarel_emb


class Ext_GNN(nn.Module):
    # knowledge extrapolation with GNN
    def __init__(self, args):
        super(Ext_GNN, self).__init__()
        self.args = args
        self.layers = nn.ModuleList()

        for idx in range(args.num_layers):
            if idx == args.num_layers - 1:
                self.layers.append(Ext_entLayer(args, act=None))
                self.layers.append(Ext_relLayer(args, act=None))
            else:
                # layers before the last one need act
                self.layers.append(Ext_entLayer(args, act=F.relu))
                self.layers.append(Ext_relLayer(args, act=F.relu))

    def forward(self, g, pattern_g, **param):
        rel_emb = param['rel_feat']
        ent_emb = param['ent_feat']
        time_emb = param['time_emb']
        metarel_emb = param['metarel_emb']
        for i in range(len(self.layers)//2):
            ent_emb, time_emb = self.layers[2*i](g, ent_emb, rel_emb, time_emb)
            rel_emb, metarel_emb = self.layers[2*i+1](pattern_g, rel_emb, metarel_emb)

        return ent_emb, rel_emb, time_emb, metarel_emb

class VGAE(nn.Module):
    def __init__(self, args):
        super(VGAE, self).__init__()
        self.args = args
        self.enc_layers = Ext_GNN(args)
        self.dec_layers = Ext_GNN(args)

        self.e_mean_layer = Ext_entLayer(args, act=None)
        self.e_log_std_layer = Ext_entLayer(args, act=None)
        self.r_mean_layer = Ext_relLayer(args, act=None)
        self.r_log_std_layer = Ext_relLayer(args, act=None)



    def encoder(self, g, pattern_g, ent_feat, rel_feat, time_emb, metarel_emb):
        ent_emb, rel_emb, time_emb, metarel_emb = self.enc_layers(g, pattern_g, ent_feat=ent_feat, rel_feat=rel_feat, time_emb=time_emb, metarel_emb=metarel_emb)

        self.mean_e = self.e_mean_layer(g, ent_emb, rel_emb, time_emb)[0]
        self.log_std_e = self.e_log_std_layer(g, ent_emb, rel_emb, time_emb)[0]
        self.mean_r = self.r_mean_layer(pattern_g, rel_emb, metarel_emb)[0]
        self.log_std_r = self.r_log_std_layer(pattern_g, rel_emb, metarel_emb)[0]

        gaussian_noise_e = torch.randn(ent_feat.size(0), self.args.ent_dim).to(self.args.gpu)
        sampled_z_e = self.mean_e + gaussian_noise_e * torch.exp(self.log_std_e).to(self.args.gpu)
        gaussian_noise_r = torch.randn(rel_feat.size(0), self.args.rel_dim).to(self.args.gpu)
        sampled_z_r = self.mean_r + gaussian_noise_r * torch.exp(self.log_std_r).to(self.args.gpu)

        return ent_emb, rel_emb, time_emb, metarel_emb, sampled_z_e, sampled_z_r
    def decoder(self, g, pattern_g, sampled_z_e, sampled_z_r, time_emb, metarel_emb):
        ent_emb, rel_emb, time_emb, metarel_emb = self.dec_layers(g, pattern_g, ent_feat=sampled_z_e, rel_feat=sampled_z_r, time_emb=time_emb, metarel_emb=metarel_emb)
        return ent_emb, rel_emb, time_emb, metarel_emb

    def forward(self, g, pattern_g, **param):
        rel_emb = param['rel_feat']
        ent_emb = param['ent_feat']
        time_emb = param['time_emb']
        metarel_emb = param['metarel_emb']

        ent_emb, rel_emb, time_emb, metarel_emb, sampled_z_e, sampled_z_r = self.encoder(g, pattern_g, ent_emb, rel_emb, time_emb, metarel_emb)
        ent_emb_out, rel_emb_out, time_emb_out, metarel_emb_out = self.decoder(g, pattern_g, sampled_z_e, sampled_z_r, time_emb, metarel_emb)  # 当前ent emb是重构表示ent_emb_out
        # g.ndata['h'] = x
        #g.ndata['g'] = x_output

        return ent_emb, ent_emb_out, rel_emb, rel_emb_out, time_emb, time_emb_out, metarel_emb, metarel_emb_out

    def vgae_loss_f(self, e, out_e, r, out_r):

        #reconstruct loss
        recon_loss = F.mse_loss(out_e, e) + F.mse_loss(out_r, r)
        kld_loss_e = -(0.5 * (1 + 2*self.log_std_e - self.mean_e**2 - torch.exp(self.log_std_e)**2).sum(1).mean())
        kld_loss_r = -(0.5 * (1 + 2 * self.log_std_r - self.mean_r ** 2 - torch.exp(self.log_std_r) ** 2).sum(1).mean())
        return recon_loss + kld_loss_e + kld_loss_r







