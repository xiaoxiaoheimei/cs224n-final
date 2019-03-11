import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD

#####################################
## File: joint_context_layers.py   ##
#####################################

class AnswerablePredictor(nn.Module):
      '''
      This layer predict whether the Qestion is answerable
      '''
      
      def __init__(self, input_dim, lstm_hdim, drop_prob=0):
          '''
          Args:
            @param: input_dim (int): input dimension
            @param: lstm_hdim (int): the hidden dimension of the LSTM of location predictor
          '''
          super(AnswerablePredictor, self).__init__()
          self.input_dim = input_dim
          self.lstm_hdim = lstm_hdim
          self.drop_prob = drop_prob
          self.att_weight = nn.Parameter(torch.zeros(input_dim, 1)) #weight to compute attention with respect to each input vector towards answer prediction
          self.h_proj = nn.Linear(input_dim, 2*lstm_hdim) #the project layer to obtain initialization of next bi-LSTM hidden state
          self.c_proj = nn.Linear(input_dim, 2*lstm_hdim) #the project layer to obtain initialization of next bi-LSTM memory state
          self.logits_proj = nn.Linear(input_dim, 2) #the project layer to obtain (no answer, answerable) logits

          for weight in (self.att_weight, self.h_proj.weight, self.c_proj.weight, self.logits_proj.weight):
            nn.init.xavier_uniform_(weight)


      def forward(self, M, G, ctx_mask):
          '''
          Args:
            @param: M (tensor): output of the BiDAF model layer (batch_size, ctx_len, m_dim)
            @param: G (tensor): output of the context condition on question (batch_size, ctx_len, g_dim)
            @param: ctx_mask (tensor): mask of the valid word (batch_size, ctx_len)
          Return:
            logits (tensor): logits of (0,1), 0 for no answer, 1 for answerable.
            (h0, c0) (tensor): initialization state of the start-loc LSTM  
          '''
          ctx = torch.cat((G, M), 2) #(batch_size, ctx_len, self.input_dim)
          att = torch.matmul(ctx, self.att_weight).transpose(1,2) #(batch_size, 1, ctx_len)
          att_prob = util.masked_softmax(att, ctx_mask.unsqueeze(1)) #(batch_size, 1, ctx_len)
          ctx_summary = torch.bmm(att_prob, ctx).squeeze(1) #(batch_size, self.input_dim)
          h0 = self.h_proj(ctx_summary) #(batch_size, 2*self.lstm_hdim)
          h0 = h0.view(-1, 2, self.lstm_hdim).transpose(0,1) #(2, batch_size, self.lstm_hdim)
          c0 = self.c_proj(ctx_summary) #(batch_size, 2*self.lstm_hdim)
          c0 = c0.view(-1, 2, self.lstm_hdim).transpose(0,1) #(2, batch_size, self.lstm_hdim)

          logits = self.logits_proj(ctx_summary) #(batch_size, self.lstm_hdim)
          h0 = F.dropout(h0, self.drop_prob, self.training).contiguous()
          c0 = F.dropout(c0, self.drop_prob, self.training).contiguous()
          
          return logits, (h0, c0)

class StartLocationPredictor(nn.Module):
      '''
      This layer compute the conditional distribution P(start_loc | Answerable).
      '''

      def __init__(self, M_dim, G_dim, lstm_hdim, drop_prob=0):
          '''
          Args:
            @param: input_dim (int): input dimension
            @param: lstm_hdim (int): the hidden dimension of the LSTM
          '''
          super(StartLocationPredictor, self).__init__()
          self.M_dim = M_dim
          self.G_dim = G_dim
          self.lstm_hdim = lstm_hdim
          self.cat_dim = G_dim + 2*lstm_hdim
          self.rnn = RNNEncoder(M_dim, lstm_hdim, 1, drop_prob)
          self.loc_proj = nn.Linear(self.cat_dim, 1)
          self.h_proj = nn.Linear(self.cat_dim, 2*lstm_hdim)
          self.c_proj = nn.Linear(self.cat_dim, 2*lstm_hdim)
          self.drop_prob=drop_prob

          for weight in (self.loc_proj.weight, self.h_proj.weight, self.c_proj.weight):
            nn.init.xavier_uniform_(weight)


      def forward(self, M, G, ctx_mask, enc_init):
          '''
          Args:
            @param: M (tensor): output of the BiDAF model layer (batch_size, ctx_len, M_dim)
            @param: G (tensor): output of the context condition on question (batch_size, ctx_len, G_dim)
            @param: ctx_mask (tensor): mask of the valid word (batch_size, ctx_len)
            @param: enc_init (tensor): initiation state of the interal LSTM (2, batch_size, self.lstm_hdim)
          Rets:
            p_start (tensor): probability distribution of the start location (batch_size, ctx_len)
            (h0, c0) (tensor): initialization state of the end-loc LSTM
          '''
          lengths = ctx_mask.sum(-1) #(batch_size)
          M1 = self.rnn(M, lengths, enc_init) #M2: (batch_size, ctx_len, 2*self.lstm_hdim)
          ctx = torch.cat((G, M1), 2) #(batch_size, ctx_len, self.cat_dim)
          logits = self.loc_proj(ctx).squeeze(-1) #(batch_size, ctx_len)
          p_start = util.masked_softmax(logits, ctx_mask, log_softmax=True) #(batch_size, ctx_len)
          
          #use probability as attention explaining the start prediction
          att = p_start.unsqueeze(1) #(batch_size, 1, ctx_len)
          #use attention to obtain a summary of the context in respect to start prediction
          ctx_summary = torch.bmm(att, ctx).squeeze(1) #(batch_size, self.cat_dim)

          h0 = self.h_proj(ctx_summary) #(batch_size, 2*self.lstm_hdim)
          h0 = h0.view(-1, 2, self.lstm_hdim).transpose(0,1) #(2, batch_size, self.lstm_hdim)
          c0 = self.c_proj(ctx_summary) #(batch_size, 2*self.lstm_hdim)
          c0 = c0.view(-1, 2, self.lstm_hdim).transpose(0,1) #(2, batch_size, self.lstm_hdim)

          h0 = F.dropout(h0, self.drop_prob, self.training).contiguous()
          c0 = F.dropout(c0, self.drop_prob, self.training).contiguous()
         
          return p_start, (h0, c0)

class EndLocationPredictor(nn.Module):
      '''
      This layer compute the conditional distribution P(end_loc | Answerable, start_loc)
      '''
      
      def __init__(self, M_dim, G_dim, lstm_hdim, drop_prob=0.):
          '''
          Args:
            @param: input_dim (int): input dimension
            @param: lstm_hdim (int): the hidden dimension of the LSTM
          '''
          super(EndLocationPredictor, self).__init__()
          self.M_dim = M_dim
          self.G_dim = G_dim
          self.lstm_hdim = lstm_hdim
          self.drop_prob = drop_prob
          self.cat_dim = G_dim + 2*lstm_hdim
          self.rnn = RNNEncoder(M_dim, lstm_hdim, 1, drop_prob) 
          self.h_proj = nn.Linear(2*lstm_hdim, lstm_hdim)
          self.c_proj = nn.Linear(2*lstm_hdim, lstm_hdim)
          self.loc_proj = nn.Linear(self.cat_dim, 1)
          
          for weight in (self.h_proj.weight, self.c_proj.weight):
              nn.init.xavier_uniform_(weight)


      def forward(self, M, G, ctx_mask, enc_init_A, enc_init_start):
          '''
          Args:
            @param: M (tensor): output of the BiDAF model layer (batch_size, ctx_len, M_dim)
            @param: G (tensor): output of the context condition on question (batch_size, ctx_len, G_dim)
            @param: ctx_mask (tensor): mask of the valid word (batch_size, ctx_len)
            @param: enc_init_A (tensor): initialtion state computed by Answerable predictor (2, batch_size, self.lstm_hdim)
            @param: enc_init_start (tensor): initialtion state computed by start position predictor (2, batch_size, self.lstm_hdim)
          Rets:
            p_end (tensor): probability distribution of the end location (batch_size, ctx_len)
          '''
          lengths = ctx_mask.sum(-1) #(batch_size)
          (hA, cA) = enc_init_A
          (hs, cs) = enc_init_start
          h0 = self.h_proj(torch.cat((hA, hs), 2)) #(2, batch_size, self.lstm_hdim)
          c0 = self.h_proj(torch.cat((cA, cs), 2)) #(2, batch_size, self.lstm_hdim)
          M2 = self.rnn(M, lengths, (h0, c0)) #(batch_size, ctx_len, 2*lstm_hdim)
          ctx = torch.cat((G, M2), 2) #(batch_size, ctx_len, self.cat_dim)
          logits = self.loc_proj(ctx).squeeze(-1) #(batch_size, ctx_len)
          p_end = util.masked_softmax(logits, ctx_mask, log_softmax=True)#(batch_size, ctx_len)

          return p_end
          

#####################################
## File: models.py                 ##
#####################################

class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out



#####################################
## File: train.py                  ##
#####################################

def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info('Using random seed {}...'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    # Get model
    log.info('Building model...')
    model = BiDAF(word_vectors=word_vectors,
                  hidden_size=args.hidden_size,
                  drop_prob=args.drop_prob)
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info('Loading checkpoint from {}...'.format(args.load_path))
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                log_p1, log_p2 = model(cw_idxs, qw_idxs)
                y1, y2 = y1.to(device), y2.to(device)
                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info('Evaluating at step {}...'.format(step))
                    ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                            for k, v in results.items())
                    log.info('Dev {}'.format(results_str))

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar('dev/{}'.format(k), v, step)
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=args.dev_eval_file,
                                   step=step,
                                   split='dev',
                                   num_visuals=args.num_visuals)


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_p1, log_p2 = model(cw_idxs, qw_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
