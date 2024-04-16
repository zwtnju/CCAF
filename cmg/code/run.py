# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library model for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import re
import time

import adapters
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from accelerate import Accelerator
from adapters import Fuse
from meteor.meteor import Meteor
from rouge.rouge import Rouge

from utils import ADAPTER_TYPE, get_adapter, show_gpu, get_model_size
from model import Seq2Seq
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (AdamW, get_linear_schedule_with_warmup, RobertaConfig,
                          RobertaModel, RobertaTokenizer, T5Config, T5ForConditionalGeneration)

MODEL_CLASSES = {
    'codebert': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
}

logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename, 'r') as f1:
        diffs = json.load(f1)
    with open(trg_filename, 'r') as f2:
        msgs = json.load(f2)

    for diff, msg in zip(diffs, msgs):
        code = diff['diff']
        code = re.sub(r'diff --git \S* \S*\n', '', code)
        code = re.sub(r'index \S* \S*\n', '', code)
        code = re.sub(r'@@ \S* \S* @@ ', '', code)
        code = ' '.join(code.strip().split()).lower()
        msg = ' '.join(msg.strip().split()).lower()
        examples.append(
            Example(
                idx=idx,
                source=code.strip(),
                target=msg.strip(),
            )
        )
        idx += 1
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


def load_and_cache_examples(args, tokenizer, is_eval=False, is_test=False):
    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path)
    cache_fn = os.path.join(args.cache_path, "test" if is_test else ("eval" if is_eval else "train"))
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        dataset = torch.load(cache_fn)
    else:
        file_path = args.test_filename if is_test else args.dev_filename if is_eval else args.train_filename
        examples = read_examples(file_path)
        features = convert_examples_to_features(examples, tokenizer, args,
                                                stage='test' if is_test else 'eval' if is_eval else 'train')
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)
        dataset = TensorDataset(all_source_ids, all_target_ids, all_source_mask, all_target_mask)
        torch.save(dataset, cache_fn)
    return dataset


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_data, model, tokenizer, accelerator, adapter_fusion=None):
    train_sampler = SequentialSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size // args.gradient_accumulation_steps)

    num_train_optimization_steps = args.train_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    else:
        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
    # Start training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Batch size = %d", args.train_batch_size * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Num epoch = %d", num_train_optimization_steps * args.train_batch_size * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1) // len(train_data))

    model.train()
    nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 10
    bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    train_dataloader = cycle(train_dataloader)
    eval_flag = True
    is_show_train_gpu = True
    for step in bar:
        batch = next(train_dataloader)
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids, source_mask, target_mask = batch
        if args.model_type == 'codebert':
            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                               target_ids=target_ids, target_mask=target_mask)
        else:
            outputs = model(input_ids=source_ids, attention_mask=source_mask,
                            labels=target_ids, decoder_attention_mask=target_mask)
            loss = outputs.loss

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        tr_loss += loss.item()
        train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
        if (global_step + 1) % 1000 == 0:
            logger.info("  step {} loss {}".format(global_step + 1, train_loss))
        nb_tr_examples += source_ids.size(0)
        nb_tr_steps += 1
        # loss.backward()
        accelerator.backward(loss)
        if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            eval_flag = True

        if args.local_rank in [-1, 0] and args.evaluate_during_training and (
                (global_step + 1) % args.eval_steps == 0) and eval_flag:
            # Eval model with dev data
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            eval_flag = False

            if args.local_rank == -1:

                # Calculate bleu
                eval_data = load_and_cache_examples(args, tokenizer, is_eval=True)
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                pred_ids = []
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids, target_ids, source_mask, target_mask = batch
                    with torch.no_grad():
                        if args.model_type == 'codebert':
                            preds = model(source_ids=source_ids, source_mask=source_mask)

                            top_preds = [pred[0].cpu().numpy() for pred in preds]

                        else:
                            if hasattr(model, 'module'):
                                unwrapped_model = accelerator.unwrap_model(model)
                                unwrapped_model.to(args.device)
                                preds = unwrapped_model.generate(source_ids,
                                                                 attention_mask=source_mask,
                                                                 use_cache=True,
                                                                 num_beams=args.beam_size,
                                                                 max_length=args.max_target_length)
                            else:
                                preds = model.generate(source_ids,
                                                       attention_mask=source_mask,
                                                       use_cache=True,
                                                       num_beams=args.beam_size,
                                                       max_length=args.max_target_length)
                            top_preds = list(preds.cpu().numpy())
                        pred_ids.extend(top_preds)
                pred = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in
                        pred_ids]
                model.train()
                with open(os.path.join(args.output_dir, "eval_msg.json"), 'w') as f:
                    json.dump(pred, f, indent=4)

                reference_path = args.dev_filename.split(",")[1]
                prediction_path = os.path.join(args.output_dir, "eval_msg.json")

                b_bleu = os.popen(f'python ../evaluator/B-Norm_BLEU.py {reference_path} < {prediction_path}')
                dev_bleu = round(float(b_bleu.read().replace('\n', '')), 2)

                with open(prediction_path, 'r') as r:
                    hypothesis = json.load(r)
                    res = {k: [v.strip().lower()] for k, v in enumerate(hypothesis)}
                with open(reference_path, 'r') as r:
                    references = json.load(r)
                    tgt = {k: [v.strip().lower()] for k, v in enumerate(references)}
                meteor, _ = Meteor().compute_score(tgt, res)
                rouge, _ = Rouge().compute_score(tgt, res)

                logger.info("  %s = %s " % ("B-norm bleu", str(dev_bleu)))
                logger.info("  %s = %s " % ("Meteor", str(round(meteor * 100, 2))))
                logger.info("  %s = %s " % ("Rouge", str(round(rouge * 100, 2))))
                logger.info("  " + "*" * 20)

                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                    if args.do_adapter:
                        if args.model_type == "codebert":
                            model_to_save.encoder.save_adapter(output_dir, args.adapter_name)
                        else:
                            model_to_save.save_adapter(output_dir, args.adapter_name)
                        logger.info("Saving model adapter to %s", output_dir)

                    if args.do_adapterfusion:
                        adapter_fusion_path = os.path.join(output_dir, 'apca_adapter_fusion')
                        if args.model_type == "codebert":
                            model_to_save.encoder.save_adapter_fusion(adapter_fusion_path, adapter_fusion)
                        else:
                            model_to_save.save_adapter_fusion(adapter_fusion_path, adapter_fusion)
                        logger.info("Saving adapter fusion to %s", adapter_fusion_path)

                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.local_rank == 0:
                eval_data = load_and_cache_examples(args, tokenizer, is_eval=True)
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_dataloader))
                logger.info("  Batch size = %d", args.eval_batch_size * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))

                model.eval()
                pred_ids = []
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids, target_ids, source_mask, target_mask = batch
                    with torch.no_grad():
                        if args.model_type == 'codebert':
                            preds = model(source_ids=source_ids, source_mask=source_mask)

                            top_preds = [pred[0].cpu().numpy() for pred in preds]

                        else:
                            if hasattr(model, 'module'):
                                unwrapped_model = accelerator.unwrap_model(model)
                                unwrapped_model.to(args.device)
                                preds = unwrapped_model.generate(source_ids,
                                                                 attention_mask=source_mask,
                                                                 use_cache=True,
                                                                 num_beams=args.beam_size,
                                                                 max_length=args.max_target_length)
                            else:
                                preds = model.generate(source_ids,
                                                       attention_mask=source_mask,
                                                       use_cache=True,
                                                       num_beams=args.beam_size,
                                                       max_length=args.max_target_length)
                            top_preds = list(preds.cpu().numpy())
                        pred_ids.extend(top_preds)
                pred = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in
                        pred_ids]
                model.train()
                with open(os.path.join(args.output_dir, "eval_msg.json"), 'w') as f:
                    json.dump(pred, f, indent=4)

                reference_path = args.dev_filename.split(",")[1]
                prediction_path = os.path.join(args.output_dir, "eval_msg.json")

                b_bleu = os.popen(f'python ../evaluator/B-Norm_BLEU.py {reference_path} < {prediction_path}')
                dev_bleu = round(float(b_bleu.read().replace('\n', '')), 2)

                with open(prediction_path, 'r') as r:
                    hypothesis = json.load(r)
                    res = {k: [v.strip().lower()] for k, v in enumerate(hypothesis)}
                with open(reference_path, 'r') as r:
                    references = json.load(r)
                    tgt = {k: [v.strip().lower()] for k, v in enumerate(references)}
                meteor, _ = Meteor().compute_score(tgt, res)
                rouge, _ = Rouge().compute_score(tgt, res)

                logger.info("  %s = %s " % ("B-norm bleu", str(dev_bleu)))
                logger.info("  %s = %s " % ("Meteor", str(round(meteor * 100, 2))))
                logger.info("  %s = %s " % ("Rouge", str(round(rouge * 100, 2))))
                logger.info("  " + "*" * 20)

                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    model_to_save = accelerator.unwrap_model(model)

                    if args.do_adapter:
                        if args.model_type == "codebert":
                            model_to_save.encoder.save_adapter(output_dir, args.adapter_name)
                        else:
                            model_to_save.save_adapter(output_dir, args.adapter_name)
                        logger.info("Saving model adapter to %s", output_dir)

                    if args.do_adapterfusion:
                        adapter_fusion_path = os.path.join(output_dir, 'apca_adapter_fusion')
                        if args.model_type == "codebert":
                            model_to_save.encoder.save_adapter_fusion(adapter_fusion_path, adapter_fusion)
                        else:
                            model_to_save.save_adapter_fusion(adapter_fusion_path, adapter_fusion)
                        logger.info("Saving adapter fusion to %s", adapter_fusion_path)

                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Saving model checkpoint to %s", output_dir)

        if is_show_train_gpu and step > int(args.train_steps / 3):
            show_gpu()
            is_show_train_gpu = False


def test(args, model, tokenizer, accelerator):
    logger.info("Test file: {}".format(args.test_filename))
    # Calculate bleu
    eval_data = load_and_cache_examples(args, tokenizer, is_test=True)
    logger.info('dataset size : {}'.format(len(eval_data)))
    eval_sampler = SequentialSampler(eval_data)  # if args.local_rank == -1 else DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    pred_ids = []

    logger.info("\n***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1))

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids, source_mask, target_mask = batch
        with torch.no_grad():
            if args.model_type == 'codebert':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]

            else:
                if hasattr(model, 'module'):
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.to(args.device)
                    preds = unwrapped_model.generate(source_ids,
                                                     attention_mask=source_mask,
                                                     use_cache=True,
                                                     num_beams=args.beam_size,
                                                     max_length=args.max_target_length)
                else:
                    preds = model.generate(source_ids,
                                           attention_mask=source_mask,
                                           use_cache=True,
                                           num_beams=args.beam_size,
                                           max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)
    pred = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in pred_ids]

    with open(os.path.join(args.output_dir, "predict_msg.json"), 'w') as f:
        json.dump(pred, f, indent=4)

    reference_path = args.test_filename.split(",")[1]
    prediction_path = os.path.join(args.output_dir, "predict_msg.json")

    b_bleu = os.popen(f'python ../evaluator/B-Norm_BLEU.py {reference_path} < {prediction_path}')
    dev_bleu = round(float(b_bleu.read().replace('\n', '')), 2)

    with open(prediction_path, 'r') as r:
        hypothesis = json.load(r)
        res = {k: [v.strip().lower()] for k, v in enumerate(hypothesis)}
    with open(reference_path, 'r') as r:
        references = json.load(r)
        tgt = {k: [v.strip().lower()] for k, v in enumerate(references)}
    meteor, _ = Meteor().compute_score(tgt, res)
    rouge, _ = Rouge().compute_score(tgt, res)

    logger.info("  %s = %s " % ("B-norm bleu", str(dev_bleu)))
    logger.info("  %s = %s " % ("Meteor", str(round(meteor * 100, 2))))
    logger.info("  %s = %s " % ("Rouge", str(round(rouge * 100, 2))))
    logger.info("  " + "*" * 20)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--tokenizer_name", default="",
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filenames (source and target files).")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. (source and target files).")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. (source and target files).")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1),
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--do_adapter", action='store_true', help="Whether to use adapter in model.")
    parser.add_argument('--adapter_name', type=str, default='cmg_adapter', help="Adapter name for each layer.")
    parser.add_argument("--adapter_type", type=str, default="pfeiffer",
                        choices=ADAPTER_TYPE, help="Adapter type to use.")
    parser.add_argument("--adapter_file", type=str, default=None,
                        help="Optional directory to store the pre-trained adapter.")

    parser.add_argument("--cache_path", type=str, default="",
                        help="Directory to store the cached data.")

    parser.add_argument("--do_adapterfusion", action='store_true', help="Whether to use adapter fusion in model.")
    parser.add_argument("--project", type=str, default=None)

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = torch.cuda.device_count()
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    accelerator = Accelerator()
    # device = accelerator.device
    args.device = device
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Barrier to make sure only the first process in distributed training download model & vocab

    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    adapter_fusion = None
    # build model
    if args.model_type == "codebert":
        encoder = model_class.from_pretrained(args.model_name_or_path, config=config)

        # add adapter
        if args.do_adapter:
            adapters.init(encoder)
            adapter_config = get_adapter(args.adapter_type)
            if args.adapter_file:
                encoder.load_adapter(args.adapter_file)
            else:
                # task adapter - only add if not existing
                if args.adapter_name not in encoder.adapters_config:
                    # add a new adapter
                    encoder.add_adapter(args.adapter_name, config=adapter_config)
            # Enable adapter training
            encoder.train_adapter(args.adapter_name)

            logger.info('Used Adapter: {}'.format(args.adapter_type))

        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                        beam_size=args.beam_size, max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    else:
        model = model_class.from_pretrained(args.model_name_or_path, config=config)

        # add adapter
        if args.do_adapter:
            adapters.init(model)
            adapter_config = get_adapter(args.adapter_type)
            if args.adapter_file:
                model.load_adapter(args.adapter_file)
            else:
                # task adapter - only add if not existing
                if args.adapter_name not in model.adapters_config:
                    # add a new adapter
                    model.add_adapter(args.adapter_name, config=adapter_config)
            # Enable adapter training
            model.train_adapter(args.adapter_name)
            logger.info('Used Adapter: {}'.format(args.adapter_type))

        # add adapter fusion
        if args.do_adapterfusion:
            adapters.init(model)
            task_adapters = ['all_codet5_pfeiffer', 'cv_5_codet5_pfeiffer']  # 'fira_codet5_pfeiffer'
            for task_adapter in task_adapters:
                model.load_adapter(task_adapter)
            adapter_fusion = Fuse('all_codet5_pfeiffer', 'cv_5_codet5_pfeiffer')
            model.add_adapter_fusion(adapter_fusion)
            model.train_adapter_fusion(adapter_fusion)

            logger.info('Used adapters: {}'.format(adapter_fusion))

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training
        # download model & vocab
    logger.info("Training/evaluation parameters %s", args)
    num_param = get_model_size(model)
    num_total_param = get_model_size(model, required=False)
    logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

    if args.do_train:

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training
            # process the data, and the test will use the cache

        train_data = load_and_cache_examples(args, tokenizer)

        if args.local_rank == 0:
            torch.distributed.barrier()

        train_time = time.time()
        train(args, train_data, model, tokenizer, accelerator, adapter_fusion)
        logger.info(f'training time:{time.time() - train_time:.3f}s')

    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))

        logger.info("reload model from {}".format(output_dir))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test_time = time.time()
        test(args, model, tokenizer, accelerator)
        logger.info(f'testing time:{time.time() - test_time:.3f}s')


if __name__ == "__main__":
    main()
