import pandas as pd
import os
import random


def get_corpus():
    path = 'data/innovation_pretrain_data_1.0.csv'

    data = pd.read_csv(path)

    files = data['files']
    sents = data['sents']

    with open("codes/further_pre_training/corpus/Innovation2_corpus.txt", "w", encoding="utf-8") as f:
        for i in range(len(data)):
            f.write(files[i]+'\n')

            ss = str(sents[i]).split(' 。 ')
            for i, s in enumerate(ss):
                if len(s) > 0:
                    if i != len(ss)-1:
                        f.write(s+' 。\n')
                    else:
                        f.write(s + '\n')

            f.write('\n')


def get_data_set():
    path = 'data/innovation_fine_tune_data_0.3.csv'

    data = pd.read_csv(path)

    files = [file for file in data['files']]
    sents = [sent for sent in data['sents']]
    target = [str(target) for target in data['target']]


    train = {
        "target": target[0:3500],
        "files": files[0:3500],
        "sents": sents[0:3500]
    }

    l = len(data)
    test = {
        "target": target[3500:l],
        "files": files[3500:l],
        "sents": sents[3500:l]
    }

    train_dataframe = pd.DataFrame(train)
    test_dataframe = pd.DataFrame(test)

    train_dataframe.to_csv('data/innovation_unbalance/train.csv', header=False, index=False)
    test_dataframe.to_csv('data/innovation_unbalance/test.csv', header=False, index=False)


def caculate_max():
    path = 'data/innovation_fine_tune_data_0.2.csv'

    data = pd.read_csv(path)

    files = [file for file in data['files']]
    sents = [sent for sent in data['sents']]

    max_l = 0
    all_l = 0
    max_file_l = 0
    all_file_l = 0
    max_sent_l = 0
    all_sent_l = 0
    for i in range(len(files)):
        file = files[i].split(' ')
        sent = sents[i].split(' ')

        l = len(file) + len(sent)

        all_l += l
        all_file_l += len(file)
        all_sent_l += len(sent)
        if max_l < l:
            max_l = l
        if max_file_l < len(file):
            max_file_l = len(file)
        if max_sent_l < len(file):
            max_sent_l = len(sent)

    ave_l = all_l/len(files)
    print(ave_l)
    print(max_l)
    ave_file_l = all_file_l/len(files)
    print(ave_file_l)
    print(max_file_l)
    ave_sent_l = all_sent_l/len(files)
    print(ave_sent_l)
    print(max_sent_l)


def get_rand_data_set():
    path = 'data/innovation_fine_tune_data_0.3.csv'

    data = pd.read_csv(path)

    files = [file for file in data['files']]
    sents = [sent for sent in data['sents']]
    target = [str(target) for target in data['target']]

    innovation = list()
    common = list()

    for i in range(len(target)):
        if target[i] == '0':
            common.append(i)
        elif target[i] == '1':
            innovation.append(i)

    start = random.randint(0, len(common) - len(innovation))
    common = common[start: start + len(innovation)]

    train_files = []
    train_sents = []
    train_target = []
    test_files = []
    test_sents = []
    test_target = []
    index = len(innovation)- int(len(innovation)/7)
    for i in range(len(innovation)):
        if i < index:
            train_files.append(files[innovation[i]])
            train_sents.append(sents[innovation[i]])
            train_target.append('1')
            train_files.append(files[common[i]])
            train_sents.append(sents[common[i]])
            train_target.append('0')
        else:
            test_files.append(files[innovation[i]])
            test_sents.append(sents[innovation[i]])
            test_target.append('1')
            test_files.append(files[common[i]])
            test_sents.append(sents[common[i]])
            test_target.append('0')

    train = {
        'target': train_target,
        'files': train_files,
        'sents': train_sents
    }
    test = {
        'target': test_target,
        'files': test_files,
        'sents': test_sents
    }

    train_dataframe = pd.DataFrame(train)
    test_dataframe = pd.DataFrame(test)

    train_dataframe.to_csv('data/innovation2.1/train.csv', header=False, index=False)
    test_dataframe.to_csv('data/innovation2.1/test.csv', header=False, index=False)


get_corpus()
# get_data_set()
# caculate_max()
# get_rand_data_set()