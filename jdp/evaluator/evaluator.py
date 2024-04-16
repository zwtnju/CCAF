# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import pickle
import sys
import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def read_answers(filename):
    answers = []
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        for label in data[1]:
            answers.append(label)
    return answers


def read_predictions(filename):
    predictions = []
    with open(filename, 'r') as f:
        for line in f:
            line = int(line.strip().split('\t')[1])
            predictions.append(line)
    return predictions


def calculate_scores(answers, predictions):
    scores = {'Acc': round(accuracy_score(answers, predictions) * 100, 2),
              'AUC': round(roc_auc_score(answers, predictions) * 100, 2)}
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate AUC for six project data.')
    parser.add_argument('--answers', '-a', help="filename of the labels, in txt format.", required=True)
    parser.add_argument('--predictions', '-p', help="filename of the predictions, in txt format.", required=True)

    args = parser.parse_args()
    answers = read_answers(args.answers)
    predictions = read_predictions(args.predictions)
    scores = calculate_scores(answers, predictions)
    print(scores)


if __name__ == '__main__':
    main()
