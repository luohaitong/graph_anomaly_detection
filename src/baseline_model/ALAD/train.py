import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as vutils

import numpy as np
from barbar import Bar

from model_DGraphFin import Generator, Encoder, Discriminatorxz, Discriminatorzz, Discriminatorxx
from utils.utils import weights_init_normal
from sklearn.metrics import roc_auc_score

class ALADTrainer:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
        self.build_models()


    def train(self):
        """Training the ALAD"""
        
        if self.args.pretrained:
            self.load_weights()

        optimizer_ge = optim.Adam(list(self.G.parameters()) +
                                  list(self.E.parameters()), lr=self.args.lr, betas=(0.5, 0.999))
        params_ = list(self.Dxz.parameters()) \
                + list(self.Dzz.parameters()) \
                + list(self.Dxx.parameters())
        optimizer_d = optim.Adam(params_, lr=self.args.lr, betas=(0.5, 0.999))

        fixed_z = Variable(torch.randn((16, self.args.latent_dim, 1, 1)),
                           requires_grad=False).to(self.device)
        criterion = nn.BCELoss()
        max_val_auc = 0
        for epoch in range(self.args.num_epochs+1):
            ge_losses = 0
            d_losses = 0
            for x, _ in Bar(self.train_loader):
                #Defining labels
                y_true = Variable(torch.ones((x.size(0), 1)).to(self.device))
                y_fake = Variable(torch.zeros((x.size(0), 1)).to(self.device))

                #Cleaning gradients.
                optimizer_d.zero_grad()
                optimizer_ge.zero_grad()

                #Generator:
                z_real = Variable(torch.randn((x.size(0), self.args.latent_dim)).to(self.device),
                                  requires_grad=False)
                x_gen = self.G(z_real)

                #Encoder:
                x_real = x.float().to(self.device)
                z_gen = self.E(x_real)

                #Discriminatorxz
                out_truexz, _ = self.Dxz(x_real, z_gen)
                out_fakexz, _ = self.Dxz(x_gen, z_real)

                #Discriminatorzz
                out_truezz, _ = self.Dzz(z_real, z_real)
                out_fakezz, _ = self.Dzz(z_real, self.E(self.G(z_real)))

                #Discriminatorxx
                out_truexx, _ = self.Dxx(x_real, x_real)
                out_fakexx, _ = self.Dxx(x_real, self.G(self.E(x_real)))

                #Losses
                loss_dxz = criterion(out_truexz, y_true) + criterion(out_fakexz, y_fake)
                loss_dzz = criterion(out_truezz, y_true) + criterion(out_fakezz, y_fake)
                loss_dxx = criterion(out_truexx, y_true) + criterion(out_fakexx, y_fake)
                loss_d = loss_dxz + loss_dzz + loss_dxx

                loss_gexz = criterion(out_fakexz, y_true) + criterion(out_truexz, y_fake)
                loss_gezz = criterion(out_fakezz, y_true) + criterion(out_truezz, y_fake)
                loss_gexx = criterion(out_fakexx, y_true) + criterion(out_truexx, y_fake)
                cycle_consistency = loss_gezz + loss_gexx
                loss_ge = loss_gexz + cycle_consistency

                #Computing gradients and backpropagate.
                loss_d.backward(retain_graph=True)
                optimizer_d.step()

                loss_ge.backward()
                optimizer_ge.step()
                
                ge_losses += loss_ge.item()
                d_losses += loss_d.item()

            print("Training... Epoch: {}, Discrimiantor Loss: {:.3f}, Generator Loss: {:.3f}".format(
                epoch, d_losses/len(self.train_loader), ge_losses/len(self.train_loader)
            ))

            #validation
            if epoch % 3 == 0:
                label_list = []
                score_list = []
                with torch.no_grad():
                    for x, labels in Bar(self.test_loader):

                        x_real = x.float().to(self.device)
                        z_gen = self.E(x_real)
                        x_rec = self.G(z_gen)

                        _, score_n = self.Dxx(x_real, x_real)
                        _, score_a = self.Dxx(x_real, x_rec)

                        score = score_a - score_n
                        #score = torch.flatten(score, dims=1)
                        score = torch.norm(score, 1, dim=1, keepdim=False)
                        score = torch.squeeze(score)

                        labels = labels.to(torch.device('cpu'))
                        score = score.to(torch.device('cpu'))

                        # pred.extend(list(scores))
                        label_list.extend(list(labels))
                        score_list.extend(list(score))
                val_auc = roc_auc_score(label_list, score_list)
                if val_auc > max_val_auc:
                    self.save_weights(val_auc, epoch)
                    max_val_auc = val_auc
                print("Validation... Epoch: {}, AUC: {:.3f}".format(epoch, roc_auc_score(label_list, score_list)))

        #self.save_weights()

    def build_models(self):           
        self.G = Generator(self.args.latent_dim).to(self.device)
        self.E = Encoder(self.args.latent_dim, self.args.spec_norm).to(self.device)
        self.Dxz = Discriminatorxz(self.args.latent_dim, self.args.spec_norm).to(self.device)
        self.Dxx = Discriminatorxx(self.args.spec_norm).to(self.device)
        self.Dzz = Discriminatorzz(self.args.latent_dim, self.args.spec_norm).to(self.device)
        self.G.apply(weights_init_normal)
        self.E.apply(weights_init_normal)
        self.Dxz.apply(weights_init_normal)
        self.Dxx.apply(weights_init_normal)
        self.Dzz.apply(weights_init_normal)
                
    def save_weights(self, auc, epoch):
        """Save weights."""
        state_dict_Dxz = self.Dxz.state_dict()
        state_dict_Dxx = self.Dxx.state_dict()
        state_dict_Dzz = self.Dzz.state_dict()
        state_dict_E = self.E.state_dict()
        state_dict_G = self.G.state_dict()
        torch.save({'Generator': state_dict_G,
                    'Encoder': state_dict_E,
                    'Discriminatorxz': state_dict_Dxz, 
                    'Discriminatorxx': state_dict_Dxx,
                    'Discriminatorzz': state_dict_Dzz}, 'weights/model_parameters_epoch{}_auc{}.pth'.format(epoch, auc))

    def load_weights(self):
        """Load weights."""
        state_dict = torch.load('weights/model_parameters.pth')

        self.Dxz.load_state_dict(state_dict['Discriminatorxz'])
        self.Dxx.load_state_dict(state_dict['Discriminatorxx'])
        self.Dzz.load_state_dict(state_dict['Discriminatorzz'])
        self.G.load_state_dict(state_dict['Generator'])
        self.E.load_state_dict(state_dict['Encoder'])

        

