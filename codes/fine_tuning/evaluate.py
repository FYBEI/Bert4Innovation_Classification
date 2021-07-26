import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

file = 'tmp/innovation2_output_layer-1_rand/results_ep4.txt'
f = open(file, 'r', encoding='utf-8')
y = [int(i) for i in f.readlines()][0:650]
y = np.array(y)

file = '../../data/innovation2/test.csv'
data = pd.read_csv(file, header=None)
print(data)
target = data.values[:, 0]
target = np.array(target).astype(np.int32)


def caculate_sklearn():
    a = 0
    b = 0
    c = 0
    for i in range(len(y)):
        print(target[i], y[i], target[i]==y[i])
        if target[i] == 1 and y[i] == 0:
            a += 1
        elif target[i] == 0 and y[i] == 1:
            b += 1
        else:
            c += 1

    print('Pos error', a)
    print('Nav error', b)
    print('Right', c)

    # print(classification_report(target, y, digits=4))
    acc_score = accuracy_score(y_true=target, y_pred=y)
    print('accuracy_score:', acc_score)
    pre_score = precision_score(y_true=target, y_pred=y)
    print('precision_score:', pre_score)
    rc_score = recall_score(y_true=target, y_pred=y)
    print('recall_score', rc_score)
    f_score = f1_score(y_true=target, y_pred=y)
    print('f1_score:', f_score)
