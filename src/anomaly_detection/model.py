#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import gc


class MF(nn.Module):

    def __init__(self, num_users, num_items, embedding_size):

        super(MF, self).__init__()

        self.users = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.init_params()
    
    def init_params(self):

        stdv = 1. / math.sqrt(self.users.size(1))
        self.users.data.uniform_(-stdv, stdv)
        self.items.data.uniform_(-stdv, stdv)
    
    def forward(self, user, item_p, item_n):

        user = self.users[user]
        item_p = self.items[item_p]
        item_n = self.items[item_n]

        p_score = torch.sum(user * item_p, 1)
        n_score = torch.sum(user * item_n, 1)

        return p_score, n_score
    
    def test_forward(self, user, item):

        user = self.users[user]
        item = self.items[item]
        score = torch.sum(user * item, 1)

        return score

class BasePUP(nn.Module):

    def __init__(self, dropout, alpha, split_dim, gc):

        super(BasePUP, self).__init__()

        self.dropout = dropout
        self.alpha = alpha
        self.split_dim = split_dim
        self.gc = gc

    
    def forward(self, feature, adj, user, item_p, item_n, cat_p, cat_n, price_p, price_n):
        #feature和adj都是num_nodes*num*nodes的矩阵，feature为对角线上为1的单矩阵，adj为对称的邻接矩阵
        x = self.encode(feature, adj)
        pred_p, pred_n = self.decode(x, user, item_p, item_n, cat_p, cat_n, price_p, price_n)

        return pred_p, pred_n
    
    def encode(self, feature, adj, training=True):

        x = self.gc(feature, adj)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=training)

        return x
    
    def test_encode(self, feature, adj):

        return self.encode(feature, adj, False)
    
    def decode(self, x, user, item_p, item_n, cat_p, cat_n, price_p, price_n):

        pred_p = self.decode_core(x, user, item_p, cat_p, price_p)
        pred_n = self.decode_core(x, user, item_n, cat_n, price_n)

        return pred_p, pred_n
    
    def decode_core(self, x, user, item, cat, price):

        user_embedding = x[user]
        item_embedding = x[item]
        cat_embedding = x[cat]
        price_embedding = x[price]

        (user_global, user_category) = torch.split(user_embedding, self.split_dim, 1)
        (item_global, _) = torch.split(item_embedding, self.split_dim, 1)
        (_, cat_category) = torch.split(cat_embedding, self.split_dim, 1)
        (price_global, price_category) = torch.split(price_embedding, self.split_dim, 1)

        pred_global = self.fm([user_global, item_global, price_global])
        pred_category = self.fm([user_category, cat_category, price_category])
        scores = pred_global + self.alpha * pred_category

        return scores
    
    def test_decode(self, x, user, item, cat, price):

        return self.decode_core(x, user, item, cat, price)
    
    def fm(self, features):

        sum_feature = sum(features)
        sum_sqr_feature = sum([f**2 for f in features])
        fm = torch.sum(0.5 * (sum_feature ** 2 - sum_sqr_feature), 1)

        return fm

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):

        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.spmm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphAttentionLayer(nn.Module):

    def __init__(self, feature_size):
        super(GraphAttentionLayer, self).__init__()
        self.feature_size = feature_size

        #self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        #nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*feature_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


    def forward(self, x, adj):
        '''
        :param x: nodes_num * embedding_size
        :param adj: nodes_num * nodes_num
        :return: nodes_num * embedding_size
        '''
        #h = torch.mm(input, self.W) # shape [N, embedding_size]
        #N = x.size()[0]
        N = adj.size()[0]
        '''
        a_input = torch.cat([x.repeat(1, N).view(N * N, -1), x.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features) # shape[N, N, 2*embedding_size]
        #e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N,N,1] -> [N,N]
        e = torch.matmul(a_input, self.a).squeeze(2)  # [N,N,1] -> [N,N]
        '''
        '''
        e = torch.norm(x, dim=1)
        print(adj.size())
        print(e.size())
        zero_vec = -9e15*torch.ones_like(e)
        adj = F.softmax(adj, dim=1)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        '''
        attention = adj
        #此处的attention为权重值
        h_prime = torch.spmm(attention, x)  # [N,N], [N, embedding_size] --> [N, embedding_size]

        return h_prime

class flow_GraphAttention(nn.Module):
    def __init__(self, feature_size, dropout):
        super(flow_GraphAttention, self).__init__()
        self.dropout = dropout
        self.feature_size = feature_size
        self.aggregator = Flow_Aggregator(feature_size)
        self.linear1 = nn.Linear(2 * self.feature_size, self.feature_size)  #

    def forward(self, item_id, features, history_list, new_features = None, training = True):

        tmp_history = []
        for item in item_id:
            tmp_history.append(history_list[int(item)])

        if training:
            self_feats = features[item_id]
        else:
            self_feats = new_features

        neigh_feats = self.aggregator.forward(features, tmp_history, self_feats)  # user-item network

        # self-connection could be considered.

        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))
        combined = F.dropout(combined, self.dropout, training=training)

        return combined

class Flow_Aggregator(nn.Module):
    """
    flow aggregator: for aggregating embeddings of neighbors (flow aggreagator).
    """
    def __init__(self, feature_size):
        super(Flow_Aggregator, self).__init__()
        self.att = Attention(feature_size)

    def forward(self, flow_latent, history_list, self_flow_latent):

        for i in range(len(history_list)):
            history_tmp = history_list[i]
            num_neigh = len(history_tmp)

            flow_latent_tmp = self_flow_latent[i]

            flow_latent_neigh = flow_latent[history_tmp]

            att_w = self.att(flow_latent_neigh, flow_latent_tmp, num_neigh)
            att_history = torch.mm(flow_latent_neigh.t(), att_w)
            att_history = att_history.t()

            self_flow_latent[i] = att_history
            gc.collect()
            torch.cuda.empty_cache()

        return self_flow_latent

class character_GraphConvolution(nn.Module):

    def __init__(self, feature_size):

        super(character_GraphConvolution, self).__init__()
        self.feature_size = feature_size

    def forward(self, input, flow_char_adj):

        output = torch.spmm(flow_char_adj, input)

        return output

class IEAD(nn.Module):

    def __init__(self, feature_size, embedding_size, dropout):

        super(IEAD, self).__init__()
        self.dropout = dropout
        self.weight_emb = Parameter(torch.FloatTensor(feature_size, embedding_size))
        self.bias_emb = Parameter(torch.FloatTensor(embedding_size))
        self.weight_character = Parameter(torch.FloatTensor(2*embedding_size, embedding_size))
        self.character_gc = character_GraphConvolution(embedding_size)
        self.flow_gc = flow_GraphAttention(embedding_size, self.dropout)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight_emb.size(1))
        self.weight_emb.data.uniform_(-stdv, stdv)
        self.bias_emb.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_character.size(1))
        self.weight_character.data.uniform_(-stdv, stdv)

    
    def forward(self, feature, flow_adj, flow_char_adj, item_id, category_r, category_n, PA_level):

        flow_emb, character_emb = self.embedding_encode(feature, flow_char_adj)
        character_latent_r = self.character_modeling(character_emb, category_r, PA_level)
        character_latent_n = self.character_modeling(character_emb, category_n, PA_level)
        flow_latent = self.flow_modeling(item_id, flow_emb, character_emb, flow_adj, flow_char_adj)
        pred_r = self.decode_score(flow_latent, character_latent_r)
        pred_n = self.decode_score(flow_latent, character_latent_n)

        return pred_r, pred_n

    def embedding_encode(self, features, flow_char_adj):

        '''
        :param features: n*feature_size
        :param flow_adj: n*n的邻接矩阵
        :param flow_char_adj: m(category numbers+PA level number)*n(nodes numbers)的邻接矩阵
        :return:
        '''
        flow_emb = torch.matmul(features, self.weight_emb)
        flow_emb += self.bias_emb

        character_emb = self.character_gc(flow_emb, flow_char_adj)
        return flow_emb, character_emb

    def test_encode(self, feature, flow_char_adj):

        flow_emb, character_emb = self.embedding_encode(feature, flow_char_adj)

        return flow_emb, character_emb

    def character_modeling(self, character_emb, category, PA):
        category_embedding = character_emb[category]
        PA_embedding = character_emb[PA]
        character_emb = torch.cat([category_embedding, PA_embedding], dim=1)
        character_latent = torch.mm(character_emb, self.weight_character)
        character_latent = torch.sigmoid(character_latent)

        return character_latent

    def flow_modeling(self, item_id, flow_emb, character_emb, flow_adj, flow_char_adj):

        #取item_id的adj
        '''
        support = sp.coo_matrix((np.ones(len(item_id)), (np.array(range(len(item_id))), item_id)),
                            shape=(len(item_id), flow_adj.size()[1]), dtype=np.float32)

        support = torch.sparse_coo_tensor(
            torch.tensor([np.array(range(len(item_id))), item_id]),
            torch.ones(len(item_id)),
            size = (len(item_id), flow_adj.size()[1]))
        support = support.to_dense()
        flow_adj =
        flow_adj = torch.mm(support.t(), flow_adj)
        '''
        flow_latent = self.attribute_agg(flow_emb, character_emb, flow_char_adj)
        flow_latent = self.flow_gc(item_id, flow_latent, flow_adj, training=True)

        return flow_latent

    def attribute_agg(self, flow_emb, character_emb, flow_char_adj):

        flow_char_adj = flow_char_adj.to_dense()
        category_emb = character_emb[:2]   #category_num * emb_size
        flow_cate_adj = (flow_char_adj[:2]).T   #node_size * category_num
        flow_emb_agg = (flow_emb+ torch.spmm(flow_cate_adj, category_emb))/2  #node_size * emb_size

        return flow_emb_agg
    
    def decode(self, x, node, category_p, categroy_n, PA):

        pred_p = self.decode_core(x, node, category_p, PA)
        pred_n = self.decode_core(x, node, categroy_n, PA)

        return pred_p, pred_n
    
    def decode_score(self, flow_latent, character_latent):
        lht = torch.mul(character_latent, flow_latent)
        scores = torch.sum(lht, dim=1)

        return scores
    
    def test_decode(self, flow_emb, character_emb, items, cats, PA_level, flow_adj):
        character_latent = self.character_modeling(character_emb, cats, PA_level)
        flow_latent = self.flow_modeling(items, flow_emb, character_emb, flow_adj, flow_char_adj)
        pred = self.decode_score(flow_latent, character_latent)

        return pred

class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, 1)
        self.att1 = nn.Linear(self.embed_dim, 1)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)

        self.softmax = nn.Softmax(0)

    def forward(self, node1, u_rep, num_neighs):
        gc.collect()
        torch.cuda.empty_cache()
        '''
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)
        '''
        x = self.att1(node1)
        att = F.softmax(x, dim=0)
        return att