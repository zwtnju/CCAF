# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import json
import os
from meteor.meteor import Meteor
from rouge.rouge import Rouge


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate metrics for FIRA data.')
    parser.add_argument('--references', '-ref', help="filename of the labels, in json format.", required=True)
    parser.add_argument('--predictions', '-pre', help="filename of the leaderboard predictions, in txt format.",
                        required=True)

    args = parser.parse_args()

    with open(args.predictions, 'r') as r:
        hypothesis = json.load(r)
        res = {k: [v.strip().lower()] for k, v in enumerate(hypothesis)}
    with open(args.references, 'r') as r:
        references = json.load(r)
        tgt = {k: [v.strip().lower()] for k, v in enumerate(references)}

    meteor, _ = Meteor().compute_score(tgt, res)
    rouge, _ = Rouge().compute_score(tgt, res)
    b_bleu = os.popen(f'python B-Norm_BLEU.py {args.references} < {args.predictions}')

    scores = {'B-Norm': round(float(b_bleu.read().replace('\n', '')), 2),
              'Meteor': round(meteor * 100, 2),
              'Rouge': round(rouge * 100, 2),
              }
    print(scores)


if __name__ == '__main__':
    main()
