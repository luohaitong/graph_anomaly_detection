#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import logging

from tqdm import tqdm

import recommender

import numpy as np
import torch
import torch.optim as optim
from tester import InstantTester, PostTester

class Trainer(object):

    def __init__(self, flags_obj, cm, val):

        self.cm = cm
        self.val = val
        #self.vm = vm
        self.flags_obj = flags_obj
        self.name = flags_obj.name
        self.model = flags_obj.model
        self.dataset = flags_obj.dataset
        self.epochs = flags_obj.epochs
        self.lr = flags_obj.lr
        self.weight_decay = flags_obj.weight_decay
        self.dropout = flags_obj.dropout
        self.batch_size = flags_obj.batch_size
        self.lr_decay_epochs = flags_obj.lr_decay_epochs
        self.num_workers = flags_obj.num_workers
        self.datafile_prefix = flags_obj.datafile_prefix
        self.output = flags_obj.output
        self.set_recommender(flags_obj, cm.workspace)
        self.recommender.transfer_model()
    
    def set_recommender(self, flags_obj, workspace):

        self.recommender = self.cm.set_recommender(flags_obj, workspace, self.cm)
    
    def train(self):

        self.set_dataloader_generator()
        self.set_optimizer()
        self.set_scheduler()

        for epoch in range(self.epochs):

            self.train_one_epoch(epoch)
            if self.val:
                self.set_tester()
                self.tester.test()
            #self.recommender.save_ckpt(epoch)
            #self.scheduler.step()
    
    def set_dataloader_generator(self):

        self.generator = self.recommender.get_dataloader_generator()
    
    def set_optimizer(self):

        self.optimizer = self.recommender.get_optimizer()

    def set_tester(self):
        #self.cm.set_test_logging()
        #vm.show_test_info(flags_obj)
        self.tester = InstantTester(self.flags_obj, self.recommender, self.cm)
    
    def set_scheduler(self):

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_decay_epochs, gamma=0.1)
    
    def train_one_epoch(self, epoch):

        data_loader = self.generator.generate(self.batch_size, self.num_workers)

        running_loss = 0.0
        total_loss = 0.0
        total_cross_loss = 0.0
        total_triplet_loss = 0.0
        num_batch = len(data_loader)
        distances = np.zeros(num_batch)
        a_p_score_mean = 0
        a_n_score_mean = 0
        n_p_score_mean = 0
        n_n_score_mean = 0
        logging.info('learning rate : {}'.format(self.optimizer.param_groups[0]['lr']))

        for batch_count, sample in enumerate(tqdm(data_loader, ncols=50)):
            self.optimizer.zero_grad()
            a_p_score, a_n_score, n_p_score, n_n_score = self.recommender.inference(sample)
            #distances[batch_count] = (p_score - n_score).mean().item()
            pos_score = a_p_score + n_p_score
            neg_score = a_n_score + n_n_score

            #loss = self.bpr_loss(a_p_score, a_n_score, n_p_score, n_n_score)
            cross_loss = self.cross_entropy_loss(pos_score, neg_score)
            triplet_loss = self.triplet_loss(pos_score, neg_score)
            loss = cross_loss + triplet_loss
            loss.backward()
            self.optimizer.step()

            total_cross_loss += cross_loss.item()
            total_triplet_loss += triplet_loss.item()
            running_loss += loss.item()
            total_loss += loss.item()
            a_p_score_mean += sum(a_p_score)
            a_n_score_mean += sum(a_n_score)
            n_p_score_mean += sum(n_p_score)
            n_n_score_mean += sum(n_n_score)

            if batch_count % (num_batch // 5) == num_batch // 5 - 1:
                logging.info('epoch {}: running loss = {}'.format(epoch, running_loss / (num_batch // 5)))
                running_loss = 0.0

        print("a_p_score_mean:", a_p_score_mean/857899)
        print("a_n_score_mean:", a_n_score_mean/857899)
        print("n_p_score_mean:", n_p_score_mean/857899)
        print("n_n_score_mean:", n_n_score_mean/857899)
        print('epoch {}: average cross loss = {} ,average triplet loss = {}'.format(epoch, total_cross_loss / num_batch,
                                                                                total_triplet_loss / num_batch))
        print('epoch {}: average loss = {}'.format(epoch, total_loss/num_batch))
        logging.info('epoch {}: total loss = {}'.format(epoch, total_loss/num_batch))
        #self.vm.update_line('loss', epoch, total_loss)
        #self.vm.update_line('distance', epoch, distances.mean())


    def cross_entropy_loss(self, pos_score, neg_score):
        out = torch.cat([torch.unsqueeze(pos_score, -1), torch.unsqueeze(neg_score, -1)], 1)

        prob = torch.softmax(out, dim=1)
        prob_p = prob[:,0]

        return  -torch.mean(torch.log(prob_p))

    def bpr_loss(self, a_p_score, a_n_score, n_p_score, n_n_score):

        return -torch.mean(torch.log(torch.sigmoid(a_p_score- n_n_score + n_p_score- a_n_score)))

    def triplet_loss(self, pos_score, neg_score):

        return -torch.mean(pos_score - neg_score)