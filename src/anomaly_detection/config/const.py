#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

train_data = 'train_data/'

train_positive = 'train_positive.json'
val_positive = 'val_positive.json'
test_positive = 'test_positive.json'
positive = 'positive.json'

cats = 'cats.npy'
prices = 'prices.npy'
item_cat = 'item_cat.json'
item_index = 'item_index.json'
cat_index = 'cat_index.json'
item_lux = 'item_lux.json'
item_lux_rank = 'item_lux_rank.json'
PA_index = 'PA_index.json'
train_edge_index = 'train_edge_index.npy'

item_id = 'item_id.pth'
features = 'features.pth'
cats_r = 'cats_r.pth'
cats_n = 'cats_n.pth'
PA_level = 'PA_level.pth'

ckpt = 'ckpt/'
model = 'epoch_199.pth'

prices_absolute = 'prices_absolute.npy'
prices_rank = 'prices_rank.npy'
