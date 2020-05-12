import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import numpy as np
import multiprocessing as mp
import nltk
from toolz import curry, compose, concat
import threading
import subprocess as sp
from collections import Counter, deque
from glob import glob
import pandas as pd
import sys


nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def make_n_grams(seq, n):
    ngrams = (tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
    return ngrams


def _n_gram_match(summ, ref, n):
    summ_grams = Counter(make_n_grams(summ, n))
    ref_grams = Counter(make_n_grams(ref, n))
    grams = min(summ_grams, ref_grams, key=len)
    # print(summ_grams)
    # print(ref_grams)
    # print(grams)
    count = sum(min(summ_grams[g], ref_grams[g]) for g in grams)
    return count


@curry
def compute_rouge_n(output, reference, n=1, mode='f'):
    assert mode in list('fpr')
    match = _n_gram_match(referece, output, n)
    if match == 0:
        score = 0.0
    else:
        precision = match / len(output)
        recall = match / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'r':
            score = recall
        elif mode == 'p':
            score = precision
        else:
            score = f_score
    return score


def _lcs_dp(a, b):
    """compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b) + 1)]
          for _ in range(0, len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp


def _lcs_len(a, b):
    dp = _lcs_dp(a, b)
    return dp[-1][-1]


@curry
def compute_rouge_l(output, reference, mode='f'):
    assert mode in list('fpr')
    lcs = _lcs_len(output, reference)
    if lcs == 0:
        score = 0.0
    else:
        precision = lcs / len(output)
        recall = lcs / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def _lcs(a, b):
    dp = _lcs_dp(a, b)
    i = len(a)
    j = len(b)
    lcs = deque()
    while (i > 0 and j > 0):
        if a[i - 1] == b[j - 1]:
            lcs.appendleft(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    assert len(lcs) == dp[-1][-1]
    return lcs


def compute_rouge_l_summ(summs, refs, mode='f'):
    assert mode in list('fpr')
    tot_hit = 0
    ref_cnt = Counter(concat(refs))
    summ_cnt = Counter(concat(summs))
    for ref in refs:
        for summ in summs:
            lcs = _lcs(summ, ref)
            for gram in lcs:
                if ref_cnt[gram] > 0 and summ_cnt[gram] > 0:
                    tot_hit += 1
                ref_cnt[gram] -= 1
                summ_cnt[gram] = -1
    if tot_hit == 0:
        score = 0.0
    else:
        precision = tot_hit / sum(len(s) for s in summs)
        recall = tot_hit / sum(len(s) for s in refs)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def _split_words(texts):
    return map(lambda t: t.split(), texts)


def get_extract_labels(art_sents, abs_sents):
    extracted = []
    scores = []
    indices = list(range(len(art_sents)))
    for abst in abs_sents:
        rouges = list(map(compute_rouge_l(reference=abst, mode='r'), art_sents))
        ext = max(indices, key=lambda i: rouges[i])
        indices.remove(ext)
        extracted.append(ext)
        scores.append(rouges[ext])
        if not indices:
            break
    return extracted, scores


DATA_DIR = r"E:\finsummary\Data"


@curry
def process(split, k):
    start = time()
    k = k.split('.')[0]
    data_dir = join(DATA_DIR, split)
    ext_sentences = []
    # print("reached_1")
    try:
        with open(join(data_dir, 'annual_reports/{}.txt'.format(k)),encoding='utf8') as f:
            ext_data = f.read()
    except:
        return
    # print("reached_2")
    for sent in tokenizer.tokenize(ext_data):
        sent = sent.replace('\n', ' ')
        ext_sentences.append(sent)
    token = compose(list, _split_words)
    ext_sents = token(ext_sentences)
    files = os.listdir(join(data_dir, 'gold_summaries/'))
    results = []
    score_list = []
    for i in range(len(files)):
        if files[i].startswith(k):
            res = {}
            print(files[i])

            abs_sentences = []
            try:
                with open(join(data_dir, 'gold_summaries/{}'.format(files[i])),encoding='utf8') as f:
                    abs_data = f.read()
            except:
                continue
            # print("reached_3")
            for sent in tokenizer.tokenize(abs_data):
                sent = sent.replace('\n', ' ')
                abs_sentences.append(sent)
            abs_sents = token(abs_sentences)
            # print(abs_sentences)
            print('Sentences seperated')
            # print(ext_sentences)
            if ext_sentences and abs_sentences:
                extracted, scores = get_extract_labels(ext_sents, abs_sents)
                print("sentence extraction done")
                # print(extracted)
                score = compute_rouge_l_summ(abs_sents, ext_sents)
                exted_sents = []
                for j in extracted:
                    exted_sents.append(ext_sents[j])
                # print(score)
            else:
                extracted, scores = [], []
            res = {'filename': files[i], 'extracted_labels': extracted, 'scores_l': scores,
                   'extracted_sentences': exted_sents, 'summary_score': score}
            results.append(res)
            score_list.append(score)

    max_i = np.argmax(score_list)

    with open(join(extr_dir_path, "/{}.json".format(k)), 'w') as f:
        json.dump(results[max_i], f)
    print((time() - start) / 60)


def label_mp(split, files):

    print('Start processing {} split..'.format(split))
    data_dir = join(DATA_DIR, split)
    # n_data=len(os.listdir(join(data_dir,'annual_reports')))
    # data_f=os.listdir(join(data_dir,'annual_reports'))
    files_list = list(files)
    n_data = len(files_list)
    print(n_data)
    return files_list,n_data


filename = "one_half_408.csv"
num = filename.split('.')[0].split('_')[-1]
extr_dir_path = os.path.join(DATA_DIR, "training/extraction_" + num)

if __name__ == '__main__':
    # start 4 worker processes
    start = time()
    # filename = "one_half_1088.csv"
    df = pd.read_csv(os.path.join(DATA_DIR, 'training/' + filename))
    # num = filename.split('.')[0].split('_')[-1]

    if (not (os.path.isdir(extr_dir_path))):
        os.mkdir(os.path.join(DATA_DIR, "training/extraction_" + num))
    # sys.exit()
    files_list,n_data=label_mp('training',df['f_name'])
    split='training'
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(split), files_list, chunksize=4))
    print('finished in {}'.format(timedelta(seconds=time() - start)))

