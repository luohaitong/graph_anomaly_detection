#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import math
import time

import numpy
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


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
        self.a = nn.Parameter(torch.zeros(size=(feature_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


    def forward(self, x, adj):
        '''
        :param x: nodes_num * embedding_size
        :param adj: nodes_num * nodes_num
        :return: nodes_num * embedding_size
        '''
        #h = torch.mm(input, self.W) # shape [N, embedding_size]
        #N = x.size()[0]
        #N = adj.size()[0]

        #邻接矩阵按行归一化
        #e = torch.norm(x, dim=1)

        e = torch.matmul(x, self.a).squeeze(1)
        col = adj._indices()[1]
        values = e[col]

        adj_new = torch.sparse.FloatTensor(adj._indices(), values, adj.shape)

        attention = torch.sparse.softmax(adj_new, dim=1)

        #此处的attention为权重值
        h_prime = torch.sparse.mm(attention, x)  # [batch_size,N], [N, embedding_size] --> [batch_size, embedding_size]

        '''
        batch_size = adj.size()[0]
        item_num = adj.size()[1]
        x1 = x.repeat(1, batch_size).view(batch_size * item_num, -1)
        x2 = x.repeat(batch_size, 1)
        print(x1.shape)
        print(x2.shape)
        a_input = torch.cat([x1, x2], dim=1).view(batch_size, -1, 2 * self.feature_size) # shape[N, N, 2*embedding_size]
        #e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N,N,1] -> [N,N]
        e = torch.matmul(a_input, self.a).squeeze(2)  # [N,N,1] -> [N,N]
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        #此处的attention为权重值
        h_prime = torch.sparse.mm(attention, x)  # [batch_size,N], [N, embedding_size] --> [batch_size, embedding_size]
        '''

        return h_prime

    def normalize(self, mx):
        """Row-normalize sparse matrix"""

        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)

        '''
        print(mx.shape)
        print("processing")
        mx = F.softmax(mx, dim=1)
        print(mx.shape)
        '''
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)

class flow_GraphAttention(nn.Module):
    def __init__(self, feature_size, dropout):
        super(flow_GraphAttention, self).__init__()
        self.dropout = dropout
        self.attentions = GraphAttentionLayer(feature_size)
        self.weight = Parameter(torch.FloatTensor(2*feature_size, feature_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, item_id, x, flow_adj, training, test_x = None):

        if len(item_id) < flow_adj.shape[0]:
            new_row = []
            new_col = []
            for i in range(len(item_id)):
                col_tmp = flow_adj[item_id[i]]._indices()[0]
                new_col.extend(col_tmp)
                new_row.extend(torch.full(col_tmp.shape, i))
            new_col = torch.tensor(new_col)
            new_row = torch.tensor(new_row)
            indices = torch.vstack((new_row, new_col))
            values = torch.ones_like(new_row)
            shape = torch.Size([len(item_id), flow_adj.shape[1]])
            flow_adj_tmp = torch.sparse.FloatTensor(indices, values, shape).to(flow_adj.device)



        x_agg = self.attentions(x, flow_adj_tmp)
        if training:
            x = torch.cat([x[item_id], x_agg], dim=1)
        else:
            x = torch.cat([test_x, x_agg], dim=1)
        x = torch.mm(x, self.weight)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=training)

        return x

class character_GraphConvolution(nn.Module):

    def __init__(self, feature_size):

        super(character_GraphConvolution, self).__init__()
        self.feature_size = feature_size

    def forward(self, input, flow_char_adj):

        output = torch.spmm(flow_char_adj, input)

        return output

class IEAD(nn.Module):

    def __init__(self, feature_size, embedding_size, num_cats, dropout):

        super(IEAD, self).__init__()
        self.dropout = dropout
        self.num_cats = num_cats
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

    
    def forward(self, feature, flow_adj, flow_char_adj, item_id, category, PA_level):

        item_id_a = item_id[:,0]
        item_id_n = item_id[:,1]
        category_a = category[:,0]
        category_n = category[:,1]
        PA_level_a = PA_level[:,0]
        PA_level_n = PA_level[:,1]

        flow_emb, character_emb = self.embedding_encode(feature, flow_char_adj)
        flow_latent_a = self.flow_modeling(item_id_a, flow_emb, character_emb, flow_adj, flow_char_adj)
        character_latent_aa = self.character_modeling(character_emb, category_a, PA_level_a)
        #character_latent_n = self.character_modeling(character_emb, category_n, PA_level)
        #flow_latent = flow_latent[item_id]
        character_latent_na = self.character_modeling(character_emb, category_n, PA_level_a)
        pred_a_p = self.decode_score(flow_latent_a, character_latent_aa)
        pred_a_n = self.decode_score(flow_latent_a, character_latent_na)
        #pred_n = self.decode_score(flow_latent, character_latent_n)

        flow_latent_n = self.flow_modeling(item_id_n, flow_emb, character_emb, flow_adj, flow_char_adj)
        character_latent_nn = self.character_modeling(character_emb, category_n, PA_level_n)
        character_latent_an = self.character_modeling(character_emb, category_a, PA_level_n)
        pred_n_p = self.decode_score(flow_latent_n, character_latent_nn)
        pred_n_n = self.decode_score(flow_latent_n, character_latent_an)

        return pred_a_p, pred_a_n, pred_n_p, pred_n_n

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

    def character_modeling(self, character_emb, category, PA):

        category_embedding = character_emb[category]

        PA_embedding = character_emb[PA]

        character_emb = torch.cat([category_embedding, PA_embedding], dim=1)

        character_latent = torch.mm(character_emb, self.weight_character)

        character_latent = torch.sigmoid(character_latent)

        return character_latent

    def flow_modeling(self, item_id, flow_emb, character_emb, flow_adj, flow_char_adj, training = True, test_flow_emb = None):

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
        if training:
            flow_cat_adj = flow_char_adj.to_dense()
            flow_cat_adj = flow_cat_adj[:self.num_cats].T  # node_size * category_num
            flow_latent = self.attribute_agg(flow_emb, character_emb, flow_cat_adj)
            flow_latent = self.flow_gc(item_id, flow_latent, flow_adj, training=True)
        else:
            test_flow_latent = self.attribute_agg(test_flow_emb, character_emb, flow_char_adj)
            flow_latent = self.flow_gc(item_id, flow_emb, flow_adj, training=False, test_x = test_flow_latent)

        return flow_latent

    def attribute_agg(self, flow_emb, character_emb, flow_cat_adj):

        category_emb = character_emb[:self.num_cats]   #category_num * emb_size
        tem = torch.mm(flow_cat_adj, category_emb)
        flow_emb_agg = (flow_emb + tem)*0.5  #node_size * emb_size

        return flow_emb_agg
    
    def decode(self, x, node, category_p, categroy_n, PA):

        pred_p = self.decode_core(x, node, category_p, PA)
        pred_n = self.decode_core(x, node, categroy_n, PA)

        return pred_p, pred_n
    
    def decode_score(self, flow_latent, character_latent):

        #scores = torch.sum(torch.mul(character_latent, flow_latent), dim=1)
        scores = torch.cosine_similarity(character_latent, flow_latent)

        return scores

    def test_encode(self, feature, flow_char_adj):

        flow_emb, character_emb = self.embedding_encode(feature, flow_char_adj)
        flow_cat_adj = flow_char_adj.to_dense()
        flow_cat_adj = flow_cat_adj[:self.num_cats].T  # node_size * category_num
        flow_latent = self.attribute_agg(flow_emb, character_emb, flow_cat_adj)

        return flow_latent, character_emb

    def test_decode(self, flow_emb, character_emb, item_id, PA_level, features, test_flow_adj):


        test_flow_emb = torch.matmul(features, self.weight_emb)
        test_flow_emb += self.bias_emb

        cats_n = torch.zeros_like(PA_level)
        #flow_cat_adj_n = torch.cat([torch.ones_like(PA_level, dtype=torch.float32).view(-1, 1),
                                    #torch.zeros_like(PA_level, dtype=torch.float32).view(-1, 1)], dim=1)
        flow_cat_adj_n = torch.zeros([PA_level.shape[0], self.num_cats], dtype=torch.float32)
        flow_cat_adj_n[:,0] = 1
        flow_cat_adj_n = flow_cat_adj_n.to(PA_level.device)
        character_latent_n = self.character_modeling(character_emb, cats_n, PA_level)
        flow_latent_n = self.flow_modeling(item_id, flow_emb, character_emb, test_flow_adj, flow_cat_adj_n,
                                         training=False, test_flow_emb = test_flow_emb)
        pred_n = self.decode_score(flow_latent_n, character_latent_n)

        for i in range(1, self.num_cats):

            cats_a = torch.full_like(PA_level, i)
            flow_cat_adj_a = torch.zeros([PA_level.shape[0], self.num_cats], dtype=torch.float32)
            flow_cat_adj_a[:,i] = 1
            flow_cat_adj_a = flow_cat_adj_a.to(PA_level.device)
            #flow_cat_adj_a = torch.cat([torch.zeros_like(PA_level, dtype=torch.float32).view(-1, 1), torch.ones_like(PA_level, dtype=torch.float32).view(-1, 1)], dim=1)
            character_latent_a = self.character_modeling(character_emb, cats_a, PA_level)
            flow_latent_a = self.flow_modeling(item_id, flow_emb, character_emb, test_flow_adj, flow_cat_adj_a,
                                             training=False, test_flow_emb = test_flow_emb)
            pred_a_tmp = self.decode_score(flow_latent_a, character_latent_a)
            if i==1:
                pred_a_all = pred_a_tmp.reshape(-1, 1)
            else:
                pred_a_all = torch.cat([pred_a_tmp.reshape(-1, 1), pred_a_all], dim=1)

        pred_a = torch.max(pred_a_all, dim=1)[0]

        return pred_a, pred_n

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