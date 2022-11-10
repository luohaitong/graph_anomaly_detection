from warnings import simplefilter

import numpy as np

simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import csv
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from meta_aad.env import make_eval_env
from meta_aad.ppo2 import PPO2, evaluate_policy
from meta_aad.utils import generate_csv_writer
import time

from tensorflow.python.client import device_lib
import os
import json
from  sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,f1_score
import matplotlib.pyplot as plt
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
print(device_lib.list_local_devices())

def plot_confusion_matrix(cm, savename,classes,title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    #plt.show()

def argsparser():
    parser = argparse.ArgumentParser("Active Anomaly Detection")
    parser.add_argument('--test', help='Testing datasets', default='CIRA-CIC-DoHBrw-2020')
    parser.add_argument('--budget', help='Budget in testing', type=int, default=50)
    parser.add_argument('--log', help='the directory to save the evaluation results', default='results')
    parser.add_argument('--load', help='the model directory', default='log/model_DoH_close.zip')
    parser.add_argument('--eval_interval', help='the interval of recording results in evaluation', type=int, default=10)

    return parser
    
def evaluate(args):

    if not os.path.exists(args.log):
        os.makedirs(args.log)

    test_datasets = args.test.split(',')

    # Make the testing environments
    eval_envs = {}
    for d in test_datasets:
        path = os.path.join('../dataset_vmeng/', d+'/binary/close_world/data_test.csv')
        output_path = os.path.join(args.log, d+'.csv') 
        csv_file, csv_writer = generate_csv_writer(output_path)
        eval_envs[d] = {'env': make_eval_env(datapath=path, budget=args.budget),
                        'csv_writer': csv_writer,
                        'csv_file': csv_file,
                        'mean_reward': 0,
                       }

    # Load model
    model = PPO2.load(args.load)

    for d in eval_envs:
        print('Dataset {}'.format(d))
        print('-----------------------')
        reward, _, results = evaluate_policy(model, eval_envs[d]['env'], n_eval_episodes=1, eval_interval=500, deterministic=False, use_batch=True)
        print('Reward: {}'.format(reward))

        #计算acc,f1,recall,pre
        if 0:
            a_num=50
            num = 125
            y_test = np.concatenate((np.ones((a_num,)),np.zeros((num-a_num,))),axis=0)
            test_predictions = np.concatenate((np.ones(int(reward),),np.zeros(a_num-int(reward)),np.ones(a_num-int(reward)),np.zeros(num-a_num-a_num+int(reward))),axis=0)
            dict_result = {}
            dict_result['accuracy'] = accuracy_score(y_test, test_predictions)
            dict_result['f1-micro'] = f1_score(y_test, test_predictions, average='micro')
            dict_result['f1-macro'] = f1_score(y_test, test_predictions, average='macro')
            dict_result['precision-micro'] = precision_score(y_test, test_predictions, average='micro')
            dict_result['precision-macro'] = precision_score(y_test, test_predictions, average='macro')
            dict_result['recall-micro'] = recall_score(y_test, test_predictions, average='micro')
            dict_result['recall-macro'] = recall_score(y_test, test_predictions, average='macro')

            outputpath = '../dataset_vmeng/'+str(test_datasets[0]) +'/binary/close_world/'
            print(outputpath)
            with open(outputpath+'result_MetaAAD.json', 'a') as outfile:
                json.dump(dict_result, outfile, ensure_ascii=False)
                outfile.write('\n')

            matrix = confusion_matrix(y_test, test_predictions)
            matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
            plot_confusion_matrix(matrix, outputpath + 'confusion_matrix_MetaAAD', classes=[0,1])

        eval_envs[d]['csv_writer'].writerow(results)
        eval_envs[d]['csv_file'].flush()
        eval_envs[d]['csv_file'].close()
        print('Results written!')
        print('')

if __name__ == "__main__":
    time1 =time.time()
    parser = argsparser()
    args = parser.parse_args()
    evaluate(args)
    time2 =time.time()
    print("time:",time2-time1)
