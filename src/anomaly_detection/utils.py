#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import os
import datetime
import setproctitle
from absl import logging

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from visdom import Visdom

import config.const as const_util
import data
import recommender
import pandas as pd


class ContextManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name
        self.output = flags_obj.output
        self.workspace = flags_obj.workspace
    
    def set_recommender(self, flags_obj, workspace, cm):

        if flags_obj.model == 'MF':
            return recommender.MFRecommender(flags_obj, workspace, cm)      
        elif flags_obj.model == 'PUP':
            return recommender.PUPRecommender(flags_obj, workspace, cm)       
        elif flags_obj.model == 'PUP-C':
            return recommender.PUPMinusCRecommender(flags_obj, workspace, cm)
        elif flags_obj.model == 'PUP-P':
            return recommender.PUPMinusPRecommender(flags_obj, workspace, cm)
        elif flags_obj.model == 'PUP-CP':
            return recommender.PUPMinusCPRecommender(flags_obj, workspace, cm)
        elif flags_obj.model =='PUPRANK':
            return recommender.PUPRankRecommender(flags_obj, workspace, cm)
        elif flags_obj.model =='IEAD':
            return recommender.IEADDetector(flags_obj, workspace, cm)
    
    def set_device(self, flags_obj):

        if not flags_obj.use_gpu:

            return torch.device('cpu')
        
        else:

            return torch.device('cuda:{}'.format(flags_obj.gpu_id))
    
    def set_default_ui(self):

        self.set_workspace()
        self.set_process_name()
        self.set_logging()
    
    def set_test_ui(self):

        self.set_process_name()
        self.set_test_logging()
    
    def set_workspace(self):

        date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        dir_name = self.name + '_' + date_time
        self.workspace = os.path.join(self.output, dir_name)
        os.mkdir(self.workspace)
    
    def set_process_name(self):

        setproctitle.setproctitle(self.name + '@luohaitong')
    
    def set_logging(self):

        self.log_path = os.path.join(self.workspace, 'log')
        if not os.path.exists(self.log_path):

            os.mkdir(self.log_path)

        logging.flush()
        logging.get_absl_handler().use_absl_log_file(self.name + '.log', self.log_path)
    
    def set_test_logging(self):

        self.log_path = os.path.join(self.workspace, 'test_log')
        if not os.path.exists(self.log_path):

            os.mkdir(self.log_path)

        logging.flush()
        logging.get_absl_handler().use_absl_log_file(self.name + '.log', self.log_path)


class VizManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name
        self.port = flags_obj.port
        self.set_visdom()
    
    def set_visdom(self):

        self.viz = Visdom(port=self.port, env=self.name)
    
    def show_basic_info(self, flags_obj):

        basic = self.viz.text('Basic Information:')
        self.viz.text('Name: {}'.format(flags_obj.name), win=basic, append=True)
        self.viz.text('Model: {}'.format(flags_obj.model), win=basic, append=True)
        self.viz.text('Dataset: {}'.format(flags_obj.dataset), win=basic, append=True)
        self.viz.text('Embedding Size: {}'.format(flags_obj.embedding_size), win=basic, append=True)
        self.viz.text('Initial lr: {}'.format(flags_obj.lr), win=basic, append=True)
        self.viz.text('Batch Size: {}'.format(flags_obj.batch_size), win=basic, append=True)

        self.basic = basic
    
    def show_test_info(self, flags_obj):

        test = self.viz.text('Test Information:')
        self.viz.text('Test Mode: {}'.format(flags_obj.mode), win=test, append=True)
        self.viz.text('Workspace: {}'.format(flags_obj.workspace), win=test, append=True)

        self.test = test
    
    def update_line(self, title, epoch, loss):

        if epoch == 0:

            setattr(self, title, self.viz.line([loss], [epoch], opts=dict(title=title)))
        
        else:

            self.viz.line([loss], [epoch], win=getattr(self, title), update='append')
    
    def show_result(self, result):

        self.viz.text('-----Results-----', win=self.test, append=True)

        for i, k in enumerate(result['topk']):
            
            self.viz.text('topk: {}'.format(k), win=self.test, append=True)
            self.viz.text('Recall: {}'.format(result['recall'][i]), win=self.test, append=True)
            self.viz.text('NDCG: {}'.format(result['ndcg'][i]), win=self.test, append=True)
        
        self.viz.text('-----------------', win=self.test, append=True)


class ResourceManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name
        self.datafile_prefix = flags_obj.testfile_prefix
        self.all_items = np.arange(flags_obj.num_items, dtype=np.int32)
        self.load_items(flags_obj)
        self.load_cats(flags_obj)
        self.load_PA(flags_obj)
        self.load_features(flags_obj)
        self.num_workers = flags_obj.num_workers
        self.topk = flags_obj.topk

    def load_items(self, flags_obj):

        self.item_id = torch.load(os.path.join(flags_obj.testfile_prefix, const_util.item_id))

    def load_features(self, flags_obj):

        self.features = torch.load(os.path.join(flags_obj.testfile_prefix, const_util.features))

    def load_cats(self, flags_obj):

        self.cats = torch.load(os.path.join(flags_obj.testfile_prefix, const_util.cats))
    
    def load_PA(self, flags_obj):

        self.PA_level = torch.load(os.path.join(flags_obj.testfile_prefix, const_util.PA_level))

    def get_test_dataloader(self):

        return DataLoader(data.TestDataset(self), batch_size=1024, shuffle=False, num_workers=self.num_workers, drop_last=False)


class BaseGraphManager(object):

    def __init__(self, flags_obj, training = True):

        self.num_items_col = flags_obj.num_items
        self.num_cats = flags_obj.num_cats
        self.num_PA = flags_obj.num_PA
        self.training = training
        if self.training:
            self.num_items_row = flags_obj.num_items
        else:
            self.num_items_row = flags_obj.test_num_items
    
    def transfer_data(self, device):

        self.flow_adj = self.flow_adj.to(device)
        if self.training:
            self.feature = self.feature.to(device)
            self.flow_char_adj = self.flow_char_adj.to(device)
    
    def generate_id_feature(self):

        i = torch.cat((torch.arange(self.num_nodes, dtype=torch.int64), torch.arange(self.num_nodes, dtype=torch.int64)), 0)
        i = i.reshape(2, -1)
        v = torch.ones(self.num_nodes)
        #生成的矩阵为对角矩阵
        self.feature = torch.sparse.FloatTensor(i, v, torch.Size([self.num_nodes, self.num_nodes]))

    def generate_feature_and_adj(self, flags_obj):

        if self.training:
            edge_index, PA_index, cat_index = self.load_index(flags_obj)
            features = self.load_features(flags_obj)
            self.feature = features

            flow_row, flow_col = self.generate_flow_coo_row_col(edge_index)
            self.flow_adj = self.generate_adj_from_coo_row_col(flow_row, flow_col, is_flow_adj=True)
            flow_char_row, flow_char_col = self.generate_char_coo_row_col(flags_obj, PA_index, cat_index)

            self.flow_char_adj = self.generate_adj_from_coo_row_col(flow_char_row, flow_char_col, is_flow_adj=False)

        else:
            edge_index = self.load_index(flags_obj)
            flow_row, flow_col = self.generate_flow_coo_row_col(edge_index)
            self.flow_adj = self.generate_adj_from_coo_row_col(flow_row, flow_col, is_flow_adj=True)


    def load_features(self, flags_obj):

        features = torch.load(os.path.join(flags_obj.trainfile_prefix, const_util.features))

        return features

    def load_index(self, flags_obj):

        if self.training:
            with open(flags_obj.datafile_prefix + 'train_' +const_util.cat_index, 'r') as f:
                cat_index = json.loads(f.read())
            with open(flags_obj.datafile_prefix + 'train_' + const_util.PA_index, 'r') as f:
                PA_index = json.loads(f.read())
            with open(flags_obj.datafile_prefix + 'train_' + const_util.edge_index, 'r') as f:
                edge_index = json.loads(f.read())['edge_index']

            return edge_index, PA_index, cat_index
        else:
            with open(flags_obj.datafile_prefix + 'test_' + const_util.edge_index, 'r') as f:
                edge_index = json.loads(f.read())['edge_index']

            return edge_index



    def generate_flow_coo_row_col(self, edge_index):

        # 前count个数，row为user、col为对应的评分item;后2*num_items个数，每两个一组，第一个row为item、col为类别id；第二个row为item、col为价格id
        flow_row = np.zeros(len(edge_index), dtype=np.int32)
        flow_col = np.zeros(len(edge_index), dtype=np.int32)

        cursor = 0
        for pair in edge_index:
            flow_row[cursor] = pair[0]
            flow_col[cursor] = pair[1]
            cursor += 1

        return flow_row, flow_col

    def generate_char_coo_row_col(self, flags_obj, PA_index, cat_index):

        #前count个数，row为user、col为对应的评分item;后2*num_items个数，每两个一组，第一个row为item、col为类别id；第二个row为item、col为价格id
        flow_char_row = np.zeros(2 * flags_obj.num_items, dtype=np.int32)
        flow_char_col = np.zeros(2 * flags_obj.num_items, dtype=np.int32)

        cursor = 0
        for item_id in cat_index:
            c1 = cat_index[item_id]
            c2 = PA_index[item_id] + flags_obj.num_cats
            flow_char_col[cursor: cursor + 2 ] = item_id
            flow_char_row[cursor] = c1
            flow_char_row[cursor + 1] = c2
            cursor += 2

        return flow_char_row, flow_char_col
    
    def gennerate_adj_from_coo_row_col(self, row, col, is_flow_adj):

        pass


class GraphManager(BaseGraphManager):

    def __init__(self, flags_obj, training = True):

        super(GraphManager, self).__init__(flags_obj, training)
    
    def generate_adj_from_coo_row_col(self, row, col, is_flow_adj):

        if is_flow_adj:
            adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(self.num_items_row, self.num_items_col), dtype=np.float32)
            #将矩阵转为对称矩阵
            if self.training:
                adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            #adj = self.normalize(adj + sp.eye(adj.shape[0]))
            adj = self.normalize(adj)
        else:
            adj = sp.coo_matrix((np.ones(len(row)), (row, col)),
                                shape=(self.num_cats + self.num_PA, self.num_items_col), dtype=np.float32)
            # adj = self.normalize(adj + sp.eye(adj.shape[0]))
            adj = self.normalize(adj)
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        return adj
    
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)

        return torch.sparse.FloatTensor(indices, values, shape)
    