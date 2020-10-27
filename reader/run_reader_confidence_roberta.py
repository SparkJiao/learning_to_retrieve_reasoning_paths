from __future__ import absolute_import, division, print_function

import argparse
import collections
import logging
import os
import random
import sys
from io import open
import oss2
import io

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from modeling_reader import RobertaForQuestionAnsweringConfidence
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

from rc_utils import convert_examples_to_features_yes_no_roberta, read_squad_examples, write_predictions_yes_no_no_empty_answer_roberta
from oss_utils import torch_save_to_oss

from tensorboardX import SummaryWriter

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "switch_logits"])


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--train_file", default=None, type=str,
                        help="SQuAD-format json file for training.")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD-format json file for evaluation.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', default='O1', type=str)
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--server_ip', type=str, default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--no_masking',
                        action='store_true',
                        help='If true, we do not mask the span loss for no-answer examples.')
    parser.add_argument('--skip_negatives',
                        action='store_true',
                        help='If true, we skip negative examples during training; this is mainly for ablation.')
    # For Natural Questions
    parser.add_argument('--max_answer_len',
                        type=int,
                        default=1000000,
                        help="maximum length of answer tokens (might be set to 5 for Natural Questions!)")

    # balance the two losses.
    parser.add_argument('--lambda_scale',
                        type=float, default=1.0,
                        help="If you would like to change the two losses, please change the lambda scale.")

    # Save checkpoints more
    parser.add_argument('--save_gran',
                        type=str, default="10,2",
                        help='"10,5" means saving a checkpoint every 1/10 of the total updates, but start saving from the 5th attempt')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--cached_features', type=str, default=None)
    args = parser.parse_args()
    print(args)

    # torch.distributed.init_process_group(backend='nccl')
    # print(f"local rank: {args.local_rank}")
    # print(f"global rank: {torch.distributed.get_rank()}")
    # print(f"world size: {torch.distributed.get_world_size()}")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    # global_rank = torch.distributed.get_rank()
    # world_size = torch.distributed.get_world_size()
    # if world_size > 1:
    #     args.local_rank = global_rank

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Prepare model and tokenizer
    print(f"AutoTokenizer do lower case: {args.do_lower_case}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    model = RobertaForQuestionAnsweringConfidence.from_pretrained(args.bert_model,
                                                                  num_labels=4,
                                                                  no_masking=args.no_masking,
                                                                  lambda_scale=args.lambda_scale)

    model.to(device)

    # prepare training data
    train_examples = None
    train_features = None
    num_train_optimization_steps = None
    if args.do_train:
        cached_train_features_file = args.train_file + '_{0}_{1}_{2}_{3}'.format(
            list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
        try:
            if args.cached_features is not None:
                train_features = torch.load(args.cached_features)
                # with open(args.cached_features, 'rb') as reader:
                #     train_features = pickle.load(reader)
            else:
                with open(cached_train_features_file, "rb") as reader:
                    train_features = pickle.load(reader)
        except:
            train_examples = read_squad_examples(input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative,
                                                 max_answer_len=args.max_answer_len, skip_negatives=args.skip_negatives)

            train_features = convert_examples_to_features_yes_no_roberta(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                if args.cache_dir is not None:
                    cached_train_features_file = cached_train_features_file.split('/')[-1]
                    cached_train_features_file = os.path.join(args.cache_dir, cached_train_features_file)
                    torch_save_to_oss(train_features, cached_train_features_file)
                    logger.info(
                        "  Saving train features into cached file %s", cached_train_features_file)
                else:
                    logger.info(
                        "  Saving train features into cached file %s", cached_train_features_file)
                    with open(cached_train_features_file, "wb") as writer:
                        pickle.dump(train_features, writer)
        
        num_train_optimization_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(num_train_optimization_steps * args.warmup_proportion), num_train_optimization_steps)

    if args.fp16:
        from apex import amp

        if args.loss_scale == 0:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.fp16_opt_level)
        else:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.fp16_opt_level, loss_scale=args.loss_scale)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP

            model = DDP(model, delay_allreduce=True)
        except ImportError:

            from torch.distributed.parallel import DistributedDataParallel as DDP

            model = DDP(model, find_unused_parameters=True)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank in [-1, 0] and args.do_train:
        summary_dir = os.path.join(args.output_dir, 'tensorboard')
        os.makedirs(summary_dir, exist_ok=True)
        summary_writer = SummaryWriter(summary_dir)

    global_step = 0
    if args.do_train:

        logger.info("***** Running training *****")
        if train_examples:
            logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor(
            [f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor(
            [f.end_position for f in train_features], dtype=torch.long)
        all_switches = torch.tensor(
            [f.switch for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions, all_switches)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size, 
            pin_memory=True, num_workers=4)

        if args.save_gran is not None:
            save_chunk, save_start = args.save_gran.split(',')
            save_chunk = num_train_optimization_steps // int(save_chunk)
            save_start = int(save_start)

        model.train()
        for _epc in trange(int(args.num_train_epochs), desc="Epoch"):
            if args.local_rank != -1:
                train_dataloader.sampler.set_epoch(_epc)
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
                if n_gpu == 1:
                    # multi-gpu does scattering it-self
                    batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, start_positions, end_positions, switches = batch
                loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                             start_positions=start_positions, end_positions=end_positions, switch_list=switches)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0]:
                        summary_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        summary_writer.add_scalar("train/loss", loss.item(), global_step)

                    if global_step % 5 == 0:
                        logger.info(f"training loss: {loss.item()}")

                    if args.save_gran is not None and args.local_rank in [-1, 0]:
                        if (global_step % save_chunk == 0) and (global_step // save_chunk >= save_start):
                            logger.info('Saving a checkpoint....')
                            output_dir_per_epoch = os.path.join(
                                args.output_dir, str(global_step) + 'steps')
                            os.makedirs(output_dir_per_epoch)

                            # Save a trained model, configuration and tokenizer
                            model_to_save = model.module if hasattr(
                                model, 'module') else model  # Only save the model it-self

                            # If we save using the predefined names, we can load using
                            # `from_pretrained`
                            # output_model_file = os.path.join(
                            #     output_dir_per_epoch, WEIGHTS_NAME)
                            # output_config_file = os.path.join(
                            #     output_dir_per_epoch, CONFIG_NAME)

                            # torch.save(model_to_save.state_dict(),
                            #            output_model_file)
                            # model_to_save.config.to_json_file(
                            #     output_config_file)
                            # tokenizer.save_vocabulary(output_dir_per_epoch)

                            model_to_save.save_pretrained(output_dir_per_epoch)
                            tokenizer.save_pretrained(output_dir_per_epoch)

                            logger.info('Done')

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using
        # `from_pretrained`
        # output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        # torch.save(model_to_save.state_dict(), output_model_file)
        # model_to_save.config.to_json_file(output_config_file)
        # tokenizer.save_vocabulary(args.output_dir)

        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = RobertaForQuestionAnsweringConfidence.from_pretrained(
            args.output_dir,  num_labels=4, no_masking=args.no_masking)
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)

    if args.do_train is False and args.do_predict is True:
        model = RobertaForQuestionAnsweringConfidence.from_pretrained(
            args.output_dir,  num_labels=4, no_masking=args.no_masking)
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        
        print("====================== Loading here.... ============================")

    elif args.do_train is True and args.do_predict is True:
        model = RobertaForQuestionAnsweringConfidence.from_pretrained(
            args.output_dir,  num_labels=4, no_masking=args.no_masking)
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)

    else:
        model = RobertaForQuestionAnsweringConfidence.from_pretrained(
            args.bert_model,  num_labels=4, no_masking=args.no_masking, lambda_scale=args.lambda_scale)

    model.to(device)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_squad_examples(
            input_file=args.predict_file, is_training=False, version_2_with_negative=args.version_2_with_negative,
            max_answer_len=args.max_answer_length, skip_negatives=args.skip_negatives)
        eval_features = convert_examples_to_features_yes_no_roberta(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(
            all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits, batch_switch_logits = model(
                    input_ids, segment_ids, input_mask)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                switch_logits = batch_switch_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits,
                                             switch_logits=switch_logits))
        output_prediction_file = os.path.join(
            args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(
            args.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(
            args.output_dir, "null_odds.json")
        write_predictions_yes_no_no_empty_answer_roberta(eval_examples, eval_features, all_results,
                                                         args.n_best_size, args.max_answer_length,
                                                         args.do_lower_case, output_prediction_file,
                                                         output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                                                         args.version_2_with_negative, args.null_score_diff_threshold,
                                                         tokenizer, args.no_masking)


if __name__ == "__main__":
    main()