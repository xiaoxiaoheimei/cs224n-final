"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD, BertSQuAD

from pytorch_pretrained_bert import modeling
from joint_context_models import JointContextQA, compute_loss, compute_loss_KL_v2


def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get model
    log.info('Building model...')
    char_vectors = util.torch_from_json("./data/char_bert_emb.json")
    char_emb_dim = 16
    bert_hidden_size = 768
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=bert_hidden_size,
                                  num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = JointContextQA(config, char_emb_dim, char_vectors, drop_prob=0.2, word_emb_dim=bert_hidden_size,
                            cat_reduce_factor=1, lstm_dim_bm=64, lstm_dim_pred=64, device=torch.device("cuda"))
    model = nn.DataParallel(model, gpu_ids)
    log.info('Loading checkpoint from {}...'.format(args.load_path))
    model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)['{}_bert_record_file'.format(args.split)]
    dataset = BertSQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    # Evaluate
    log.info('Evaluating on {} split...'.format(args.split))
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)['{}_bert_eval_file'.format(args.split)]
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)
            print(cw_idxs.size())

            # Forward
            ans_logits, log_p_start, log_p_end = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            loss = compute_loss_KL_v2(ans_logits, log_p_start, log_p_end, y1, y2, cw_idxs)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p_start.exp(), log_p_end.exp()
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = util.bert_convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      ans_logits,
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                for k, v in results.items())
        log.info('{} {}'.format(args.split.title(), results_str))

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info('Writing submission file to {}...'.format(sub_path))
    with open(sub_path, 'w') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    main(get_test_args())
