#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import app
from absl import flags
from absl import logging

import utils
from trainer import Trainer
from tester import InstantTester, PostTester


FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'IEAD', 'Experiment name.')
flags.DEFINE_enum('model', 'IEAD', ['MF', 'PUP', 'PUPRANK', 'PUP-C', 'PUP-P', 'PUP-CP', 'IEAD'], 'Model name.')
flags.DEFINE_enum('mode', 'train', ['train', 'overall_test', 'CIR', 'UCIR', 'ug'], 'Train or test.')
flags.DEFINE_bool('use_gpu', True, 'Use GPU or not.')
flags.DEFINE_integer('gpu_id', '7', 'GPU ID.')
flags.DEFINE_enum('dataset', 'DGraphFin', ['CICIDS2017', 'DGraphFin'], 'Dataset.')
flags.DEFINE_integer('num_items', 418733, 'Number of items.')
flags.DEFINE_integer('num_cats', 13, 'Number of categories.')
flags.DEFINE_integer('num_PA', 7, 'Number of PA levels.')
flags.DEFINE_integer('test_num_items', 89728, 'Number of test items.')
flags.DEFINE_string('price_format', 'absolute', 'Price format.')
flags.DEFINE_integer('feature_size', 3, 'feature size for original features.')
flags.DEFINE_integer('embedding_size', 10, 'Embedding size for embedding based models.')
flags.DEFINE_float('alpha', 0.5, 'Weight of category branch (1.0 for global branch).')
flags.DEFINE_multi_integer('split_dim', [56, 8], 'Embedding dimension distribution for global branch and category branch.')
flags.DEFINE_integer('epochs', 10, 'Max epochs for training.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_float('weight_decay', 5e-8, 'Weight decay.')
flags.DEFINE_float('dropout', 0.2, 'Dropout ratio.')
flags.DEFINE_integer('batch_size', 1024, 'Batch Size.')
flags.DEFINE_multi_integer('lr_decay_epochs', [40, 80], 'Epochs at which to perform learning rate decay.')
flags.DEFINE_multi_integer('topk', [100, 500, 2326, 5000], 'Topk for testing recommendation performance.')
flags.DEFINE_integer('num_workers', 8, 'Number of processes for training and testing.')
flags.DEFINE_string('datafile_prefix', '../data/CICIDS2017/multi_label/', 'Path prefix to data files.')
flags.DEFINE_string('trainfile_prefix', '../data/CICIDS2017/multi_label/train_data/', 'Path prefix to data files.')
flags.DEFINE_string('testfile_prefix', '../data/CICIDS2017/multi_label/test_data/', 'Path prefix to data files.')
flags.DEFINE_string('output', './IEAD/', 'Directory to save model/log/metrics.')
flags.DEFINE_integer('port', 33333, 'Port to show visualization results.')
flags.DEFINE_string('workspace', '/IEAD/yelp-PUP-test-basePUP-2_2019-07-13-15-58-12/', 'Path to workspace.')


def main(argv):

    flags_obj = flags.FLAGS
    cm = utils.ContextManager(flags_obj)
    #vm = utils.VizManager(flags_obj)
    
    if flags_obj.mode == 'train':
    
        cm.set_default_ui()
        #vm.show_basic_info(flags_obj)
        trainer = Trainer(flags_obj, cm, True)
        trainer.train()

        #cm.set_test_logging()
        #vm.show_test_info(flags_obj)
        #tester = InstantTester(flags_obj, trainer.recommender, trainer.cm)
        #tester.test()

    elif flags_obj.mode == 'overall_test':

        cm.set_test_ui()
        #vm.show_test_info(flags_obj)
        tester = PostTester(flags.FLAGS, None, cm)
        tester.test()


if __name__ == "__main__":
    
    app.run(main)
