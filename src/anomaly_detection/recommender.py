#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

import data
import model
import utils
import config.const as const_util

import os

class Anomaly_Detector(object):

    def __init__(self, flags_obj, workspace, cm):
        self.cm = cm
        self.model_name = flags_obj.model
        self.num_items = flags_obj.num_items
        self.num_cats = flags_obj.num_cats
        self.num_PA = flags_obj.num_PA
        self.feature_size = flags_obj.feature_size
        self.embedding_size = flags_obj.embedding_size
        self.datafile_prefix = flags_obj.datafile_prefix
        self.lr = flags_obj.lr
        self.set_device(flags_obj)
        self.set_model(flags_obj)
        self.workspace = workspace

    def set_device(self, flags_obj):
        self.device = self.cm.set_device(flags_obj)

    def set_model(self, flags_obj):
        pass

    def transfer_model(self):
        self.model = self.model.to(self.device)

    def save_ckpt(self, epoch):
        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')
        torch.save(self.model.state_dict(), model_path)

    def load_ckpt(self):
        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        model_path = os.path.join(ckpt_path, const_util.model)
        self.model.load_state_dict(torch.load(model_path))

    def get_dataloader_generator(self, datafile_prefix):
        pass

    def get_optimizer(self, lr, weight_decay):
        pass

    def inference(self, sample):
        pass

    def test_inference(self, sample):
        pass

    def prepare_test(self):
        pass


class BaseIEADDetector(Anomaly_Detector):

    def __init__(self, flags_obj, workspace, cm):
        super(BaseIEADDetector, self).__init__(flags_obj, workspace, cm)
        self.weight_decay = flags_obj.weight_decay
        self.set_gm(flags_obj)
        self.generate_transfer_feature_adj(flags_obj)

    def set_gm(self, flags_obj):
        pass

    def generate_transfer_feature_adj(self, flags_obj):
        #self.gm.generate_id_feature()
        self.gm.generate_feature_and_adj(flags_obj)
        self.gm.transfer_data(self.device)

    def set_model(self, flags_obj):
        self.set_pup_hyper_params(flags_obj)
        self.set_pup_model()

    def set_pup_model(self):
        pass

    def set_pup_hyper_params(self, flags_obj):
        #self.set_feature_size(flags_obj)
        self.dropout = flags_obj.dropout
        self.alpha = flags_obj.alpha

    def set_feature_size(self, flags_obj):
        self.feature_size = flags_obj.num_users + flags_obj.num_items + flags_obj.num_cats + flags_obj.num_prices

    def get_dataloader_generator(self):
        return data.FactorizationDataLoaderGenerator(self.datafile_prefix)

    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.5, 0.99))

    def inference(self, sample):

        item_id, cat, PA_level = sample
        item_id = item_id.to(self.device)
        cat_r = cat.to(self.device)
        PA_level = PA_level.to(self.device) + self.num_cats
        a_p_score, a_n_score, n_p_score, n_n_score = self.model(self.gm.feature, self.gm.flow_adj, self.gm.flow_char_adj, item_id, cat_r, PA_level)

        return a_p_score, a_n_score, n_p_score, n_n_score

    def prepare_test(self):
        self.flow_emb, self.character_emb = self.model.test_encode(self.gm.feature, self.gm.flow_char_adj)

    def test_inference(self, sample, test_flow_adj):
        items, cats, PA_level, features = sample
        items = torch.squeeze(items.to(self.device))
        cats = torch.squeeze(cats.to(self.device))
        PA_level = torch.squeeze(PA_level.to(self.device)) + self.num_cats
        features = features.to(self.device)

        scores_a, scores_n = self.model.test_decode(self.flow_emb, self.character_emb, items, PA_level, features, test_flow_adj)
        #pred = torch.sigmoid(scores_a -scores_n)
        score = torch.cat([scores_n.reshape(-1,1), scores_a.reshape(-1,1)], dim=1)
        pred = torch.squeeze(score.argmax(dim=1))
        scores = scores_a - scores_n



        return scores_a, scores_n, cats, scores


class IEADDetector(BaseIEADDetector):

    def __init__(self, flags_obj, workspace, cm):
        super(IEADDetector, self).__init__(flags_obj, workspace, cm)

    def set_gm(self, flags_obj):
        self.gm = utils.GraphManager(flags_obj)

    def set_pup_model(self):
        self.model = model.IEAD(self.feature_size, self.embedding_size, self.num_cats, self.dropout)

    def get_dataloader_generator(self):
        return data.PriceLevelPairwiseDataLoaderGenerator(self.datafile_prefix)