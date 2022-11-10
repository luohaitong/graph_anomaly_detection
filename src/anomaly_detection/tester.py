#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import os
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

import utils
import recommender
import metrics
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,precision_score


class Tester(object):

    def __init__(self, flags_obj, trained_recommender, cm):

        self.cm = cm
        self.name = flags_obj.name
        self.model = flags_obj.model
        self.dataset = flags_obj.dataset
        self.test_num_items = flags_obj.test_num_items
        self.set_device(flags_obj)
        self.workspace = cm.workspace
        self.set_recommender(flags_obj, trained_recommender, cm.workspace)
        self.set_rm(flags_obj)
        self.set_dataloader()
        self.set_gm(flags_obj)
        self.generate_transfer_feature_adj(flags_obj)

    def set_device(self, flags_obj):
        self.device = self.cm.set_device(flags_obj)

    def set_gm(self, flags_obj):
        self.gm = utils.GraphManager(flags_obj, training=False)

    def generate_transfer_feature_adj(self, flags_obj):
        self.gm.generate_feature_and_adj(flags_obj)
        self.gm.transfer_data(self.device)

    def set_recommender(self, flags_obj, trained_recommender, workspace):

        pass
    
    def set_rm(self, flags_obj):

        self.rm = utils.ResourceManager(flags_obj)
    
    def set_dataloader(self):
 
        self.dataloader = self.rm.get_test_dataloader()
    
    def test(self):

        #hit_recall = np.zeros(len(self.rm.topk), dtype=np.float64)
        #hit_ndcg = np.zeros(len(self.rm.topk), dtype=np.float64)
        pred = []
        label = []
        score2 = []
        score_a_list = []
        score_n_list = []

        with torch.no_grad():

            self.recommender.prepare_test()

            for _, sample in enumerate(tqdm(self.dataloader)):
                scores_a, scores_n, labels, scores_2 = self.recommender.test_inference(sample, self.gm.flow_adj)

                #scores = scores.to(torch.device('cpu'))
                labels = labels.to(torch.device('cpu'))
                scores_2 = scores_2.to(torch.device('cpu'))

                #pred.extend(list(scores))
                label.extend(list(labels))
                score2.extend(list(scores_2))
                score_a_list.extend(list(scores_a))
                score_n_list.extend(list(scores_n))
                #hit_recall_u, hit_ndcg_u = metrics.calc_hit_recall_ndcg(scores_2, labels, self.rm.topk, True)

                #hit_recall = hit_recall + hit_recall_u
                #hit_ndcg = hit_ndcg + hit_ndcg_u

        '''
        print("sum label:", sum(label))
        print("sum pred", sum(pred))
        print("acc is:",accuracy_score(label, pred))
        print("recall is:", recall_score(label, pred))
        print("f1 is:", f1_score(label, pred))
        print("precision is:",precision_score(label, pred))
        '''
        print("auc is:", roc_auc_score(label, score2))
        label = np.array(label)
        score_a_list = np.array(score_a_list)
        score_n_list = np.array(score_n_list)
        score2 = np.array(score2)
        score_aa_list = score_a_list[label == 1]
        score_an_list = score_a_list[label == 0]
        score_na_list = score_n_list[label == 1]
        score_nn_list = score_n_list[label == 0]
        score_a_list = score2[label == 1]
        score_n_list = score2[label == 0]



        print("score_a_list:", sum(score_a_list) / len(score_a_list))
        print("score_n_list:", sum(score_n_list) / len(score_n_list))
        print("score_aa_list:",sum(score_aa_list) / len(score_aa_list))
        print("score_an_list:", sum(score_an_list) / len(score_an_list))
        print("score_na_list:", sum(score_na_list) / len(score_na_list))
        print("score_nn_list:", sum(score_nn_list) / len(score_nn_list))
        for k in self.rm.topk:
            topk_score = np.sort(score2)[-k]
            print("topk_score:", topk_score)
            score_tem = np.array(score2)
            a_index = score_tem >= topk_score
            n_index = score_tem < topk_score
            score_tem[a_index] = 1
            score_tem[n_index] = 0
            print("k is:", k)
            print("acc is:",accuracy_score(label, score_tem))
            print("recall is:", recall_score(label, score_tem))
            print("f1 is:", f1_score(label, score_tem))
            print("precision is:",precision_score(label, score_tem))
        '''
        dict = {}
        dict['score_a_list'] = score_a_list.tolist()
        dict['score_n_list'] = score_n_list.tolist()
        with open('test_score.json', 'w') as f:
            f.write(json.dumps(dict))
        '''
        #recall = hit_recall / self.rm.num_users
        #ndcg = hit_ndcg / self.rm.num_users

        #self.report(recall, ndcg)
    
    def report(self, recall, ndcg):

        metrics_path = os.path.join(self.workspace, 'metrics')
        if not os.path.exists(metrics_path):
            os.mkdir(metrics_path)
        
        result_path = os.path.join(metrics_path, 'basic.json')

        result = {
            'topk': self.rm.topk,
            'recall': recall.tolist(),
            'ndcg': ndcg.tolist(),
        }

        with open(result_path, 'w') as f:

            f.write(json.dumps(result))
        
        #self.vm.show_result(result)


class InstantTester(Tester):

    def __init__(self, flags_obj, trained_recommender, cm):

        super(InstantTester, self).__init__(flags_obj, trained_recommender, cm)
        self.workspace = trained_recommender.workspace
    
    def set_recommender(self, flags_obj, trained_recommender, workspace):

        self.recommender = trained_recommender


class PostTester(Tester):

    def __init__(self, flags_obj, trained_recommender, cm, vm):

        super(PostTester, self).__init__(flags_obj, trained_recommender, cm, vm)
        self.prepare_user_study()
    
    def set_recommender(self, flags_obj, trained_recommender, workspace):

        self.recommender = self.cm.set_recommender(flags_obj, workspace, self.cm)
        
        self.recommender.transfer_model()
        self.recommender.load_ckpt()

    def prepare_user_study(self):

        self.user_study = []

    def get_low_high_cate_price(self, uid):

        items = self.rm.positive[str(uid)]
        cates = self.rm.cats[items]
        prices = self.rm.prices[items]

        df = pd.DataFrame({'cate': cates, 'price': prices})
        df = df.groupby('cate').mean().reset_index()

        df_min = df[df.price == df.price.min()].reset_index(drop=True)
        lo_cate = df_min['cate'][0]
        lo_price = df_min['price'][0]

        df_max = df[df.price == df.price.max()].reset_index(drop=True)
        hi_cate = df_max['cate'][0]
        hi_price = df_max['price'][0]

        return lo_cate, lo_price, hi_cate, hi_price

    def get_recommend_low_high_price(self, top_items, lo_cate, hi_cate):

        lo_items = top_items[self.rm.cats[top_items] == lo_cate]
        if len(lo_items) < 1:
            return False, None, None
        recommend_lo_price = self.rm.prices[lo_items].mean()

        hi_items = top_items[self.rm.cats[top_items] == hi_cate]
        if len(hi_items) < 1:
            return False, None, None
        recommend_hi_price = self.rm.prices[hi_items].mean()

        return True, recommend_lo_price, recommend_hi_price

    def update_user_study(self, sample, scores):

        users, items, _, _, _ = sample
        uid = users[0][0].item()
        lo_cate, lo_price, hi_cate, hi_price = self.get_low_high_cate_price(uid)

        _, top_indices = torch.topk(scores, 1000)
        top_items = items[0][top_indices].numpy()

        recommended, recommend_lo_price, recommend_hi_price = self.get_recommend_low_high_price(top_items, lo_cate, hi_cate)

        if recommended:
            self.user_study.append([uid, lo_cate, lo_price, recommend_lo_price, hi_cate, hi_price, recommend_hi_price])

    def report(self, recall, ndcg):

        metrics_path = os.path.join(self.workspace, 'metrics')
        if not os.path.exists(metrics_path):
            os.mkdir(metrics_path)
        
        result_path = os.path.join(metrics_path, 'user_study.npy')
        result = np.array(self.user_study)

        np.save(result_path, result)

        result_path = os.path.join(metrics_path, 'basic.json')

        result = {
            'topk': self.rm.topk,
            'recall': recall.tolist(),
            'ndcg': ndcg.tolist(),
        }

        with open(result_path, 'w') as f:

            f.write(json.dumps(result))
        
        self.vm.show_result(result)

    def test(self):

        hit_recall = np.zeros(len(self.rm.topk), dtype=np.float64)
        hit_ndcg = np.zeros(len(self.rm.topk), dtype=np.float64)

        with torch.no_grad():

            self.recommender.prepare_test()

            for _, sample in enumerate(tqdm(self.dataloader)):
                
                scores, num_positive = self.recommender.test_inference(sample)
                scores = scores.to(torch.device('cpu'))

                self.update_user_study(sample, scores)

                hit_recall_u, hit_ndcg_u = metrics.calc_hit_recall_ndcg(scores, num_positive.item(), self.rm.topk, True)

                hit_recall = hit_recall + hit_recall_u
                hit_ndcg = hit_ndcg + hit_ndcg_u

        recall = hit_recall / self.rm.num_users
        ndcg = hit_ndcg / self.rm.num_users

        self.report(recall, ndcg)
    