from __future__ import absolute_import, division, print_function

import argparse
import collections
import logging
import os
import random
import sys

import numpy as np
import torch
from rc_utils import convert_examples_to_features_yes_no, read_squad_examples, write_predictions_yes_no_no_empty_answer
from torch import distributed as dist
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from oss_utils import torch_save_to_oss, load_buffer_from_oss

logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "switch_logits"])

oss_features_cache_dir = 'reader_bert_feature_cache_dir/'


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Optimizer parameters
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--adam_betas", default="(0.9, 0.999)", type=str)
    parser.add_argument("--no_bias_correction", default=False, action='store_true')

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
    parser.add_argument("--do_label", action='store_true')
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

    parser.add_argument('--model_version', default='v1', type=str)

    # Save checkpoints more
    parser.add_argument('--save_gran',
                        type=str, default="10,3",
                        help='"10,5" means saving a checkpoint every 1/10 of the total updates,'
                             'but start saving from the 5th attempt')
    parser.add_argument('--oss_cache_dir', default=None, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--dist', default=False, action='store_true')

    args = parser.parse_args()
    print(args)

    if args.dist:
        dist.init_process_group(backend='nccl')
        print(f"local rank: {args.local_rank}")
        print(f"global rank: {dist.get_rank()}")
        print(f"world size: {dist.get_world_size()}")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of
        # synchronizing nodes/GPUs
        dist.init_process_group(backend='nccl')

    if args.dist:
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size > 1:
            args.local_rank = global_rank

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

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

    if args.model_version == 'v1':
        from modeling_reader import IterBertForQuestionAnsweringConfidence
    elif args.model_version == 'v2':
        from modeling_reader import IterBertForQuestionAnsweringConfidenceV2 as IterBertForQuestionAnsweringConfidence
    elif args.model_version == 'v3':
        from modeling_reader import IterBertForQuestionAnsweringConfidenceV3 as IterBertForQuestionAnsweringConfidence
    elif args.model_version == 'v4':
        from modeling_reader import IterBertForQuestionAnsweringConfidenceV4 as IterBertForQuestionAnsweringConfidence
    else:
        raise RuntimeError(f"No compatible model version: {args.model_version}")

    # Prepare model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    model = IterBertForQuestionAnsweringConfidence.from_pretrained(args.bert_model,
                                                                   num_labels=4,
                                                                   no_masking=args.no_masking,
                                                                   lambda_scale=args.lambda_scale)

    model.to(device)

    train_examples = None
    train_features = None
    num_train_optimization_steps = None
    if args.do_train:
        cached_train_features_file = args.train_file + '_{0}_{1}_{2}_{3}_{4}'.format(
            model.base_model_prefix, str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length),
            tokenizer.do_lower_case)
        cached_train_features_file_name = cached_train_features_file.split('/')[-1]
        _oss_feature_save_path = os.path.join(oss_features_cache_dir, cached_train_features_file_name)

        try:
            if args.cache_dir is not None and os.path.exists(os.path.join(args.cache_dir,
                                                                          cached_train_features_file_name)):
                logger.info(f"Loading pre-processed features from {os.path.join(args.cache_dir, cached_train_features_file_name)}")
                train_features = torch.load(os.path.join(args.cache_dir, cached_train_features_file_name))
            else:
                logger.info(f"Loading pre-processed features from oss: {_oss_feature_save_path}")
                train_features = torch.load(load_buffer_from_oss(_oss_feature_save_path))
        except:
            train_examples = read_squad_examples(
                input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative,
                max_answer_len=args.max_answer_len, skip_negatives=args.skip_negatives)
            train_features = convert_examples_to_features_yes_no(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)
            if args.local_rank in [-1, 0]:
                torch_save_to_oss(train_features, _oss_feature_save_path)
                logger.info(f"Saving train features into oss: {_oss_feature_save_path}")

        num_train_optimization_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    if args.do_label:
        logger.info("finished.")
        return

    if args.do_train:
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_optimization_steps
        if args.local_rank != -1:
            t_total = t_total // dist.get_world_size()

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=eval(args.adam_betas),
                        eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(t_total * args.warmup_proportion), num_train_optimization_steps)

        if args.fp16:
            from apex import amp

            if args.fp16_opt_level == 'O1':
                amp.register_half_function(torch, "einsum")

            if args.loss_scale == 0:
                model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
            else:
                model, optimizer = amp.initialize(model, optimizer,
                                                opt_level=args.fp16_opt_level, loss_scale=args.loss_scale)
        if args.local_rank != -1:
            if args.fp16_opt_level == 'O2':
                try:
                    import apex
                    model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
                except ImportError:
                    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0

        logger.info("***** Running training *****")
        if train_examples:
            logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (dist.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        all_switches = torch.tensor([f.switch for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions, all_switches)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      pin_memory=True, num_workers=4)

        if args.save_gran is not None:
            save_chunk, save_start = args.save_gran.split(',')
            save_chunk = t_total // int(save_chunk)
            save_start = int(save_start)

        model.train()
        tr_loss = 0
        for _epc in trange(int(args.num_train_epochs), desc="Epoch"):
            if args.local_rank != -1:
                train_dataloader.sampler.set_epoch(_epc)
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration",
                                              disable=args.local_rank not in [-1, 0])):
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

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 50 == 0:
                        logger.info(f"Training loss: {tr_loss / global_step}\t"
                                    f"Learning rate: {scheduler.get_lr()[0]}\t"
                                    f"Global step: {global_step}")

                    if args.save_gran is not None and args.local_rank in [-1, 0]:
                        if (global_step % save_chunk == 0) and (global_step // save_chunk >= save_start):
                            logger.info('Saving a checkpoint....')
                            output_dir_per_epoch = os.path.join(
                                args.output_dir, str(global_step) + 'steps')
                            os.makedirs(output_dir_per_epoch)

                            # Save a trained model, configuration and tokenizer
                            model_to_save = model.module if hasattr(
                                model, 'module') else model  # Only save the model it-self

                            if args.oss_cache_dir is not None:
                                _oss_model_save_path = os.path.join(args.oss_cache_dir, f"{global_step}steps")
                                torch_save_to_oss(model_to_save.state_dict(),
                                                  _oss_model_save_path + "/pytorch_model.bin")
                            model_to_save.save_pretrained(output_dir_per_epoch)
                            tokenizer.save_pretrained(output_dir_per_epoch)
                            logger.info('Done')

    if args.do_train and (args.local_rank == -1 or dist.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self

        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch_save_to_oss(model_to_save.state_dict(), os.path.join(args.oss_cache_dir, "pytorch_model.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        # model = IterBertForQuestionAnsweringConfidence.from_pretrained(
        #     args.output_dir, num_labels=4, no_masking=args.no_masking)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    if args.do_train is False and args.do_predict is True:
        model = IterBertForQuestionAnsweringConfidence.from_pretrained(
            args.output_dir, num_labels=4, no_masking=args.no_masking)
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
    elif args.do_train is True and args.do_predict is True:
        model = IterBertForQuestionAnsweringConfidence.from_pretrained(
            args.output_dir, num_labels=4, no_masking=args.no_masking)
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = IterBertForQuestionAnsweringConfidence.from_pretrained(
            args.bert_model, num_labels=4, no_masking=args.no_masking, lambda_scale=args.lambda_scale)

    model.to(device)

    if args.do_predict and (args.local_rank == -1 or dist.get_rank() == 0):
        eval_examples = read_squad_examples(
            input_file=args.predict_file, is_training=False, version_2_with_negative=args.version_2_with_negative,
            max_answer_len=args.max_answer_length, skip_negatives=args.skip_negatives)
        eval_features = convert_examples_to_features_yes_no(
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
        for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating",
                                                                        disable=args.local_rank not in [-1, 0]):
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
        write_predictions_yes_no_no_empty_answer(eval_examples, eval_features, all_results,
                                                 args.n_best_size, args.max_answer_length,
                                                 args.do_lower_case, output_prediction_file,
                                                 output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                                                 args.version_2_with_negative, args.null_score_diff_threshold,
                                                 args.no_masking)


if __name__ == "__main__":
    main()
