# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def read_answers(filename):
    answers = []
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        for i in range(len(data)):
            answers.append(data[i]['label'])
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
              'Pre': round(precision_score(answers, predictions) * 100, 2),
              'Rec': round(recall_score(answers, predictions) * 100, 2),
              'F1': round(f1_score(answers, predictions) * 100, 2),
              'AUC': round(roc_auc_score(answers, predictions) * 100, 2), }
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate metrics for apca task data.')
    parser.add_argument('--answers', '-a', help="filename of the labels, in txt format.", required=True)
    parser.add_argument('--predictions', '-p', help="filename of the leaderboard predictions, in txt format.", required=True)

    args = parser.parse_args()
    answers = read_answers(args.answers)
    predictions = read_predictions(args.predictions)
    scores = calculate_scores(answers, predictions)
    print(scores)


if __name__ == '__main__':
    main()
