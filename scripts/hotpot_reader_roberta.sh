hotpot_train_file_path='../learning_to_retrieve_reasoning_paths/models/hotpot_reader_data/hotpot_reader_train_data.json'
hotpot_dev_file_path='../learning_to_retrieve_reasoning_paths/models/hotpot_reader_data/hotpot_dev_squad_v2.0_format.json'
bert_dir='../../pretrained_model/roberta-base'

output_dir='experiments/hotpot_open/reader_roberta_base/hotpot_roberta_reader3'

python reader/run_reader_confidence_roberta.py \
--bert_model $bert_dir \
--output_dir $output_dir \
--train_file $hotpot_train_file_path \
--predict_file $hotpot_dev_file_path \
--learning_rate 3e-5 \
--max_seq_length 512 \
--doc_stride 160 --max_query_length -1 \
--do_predict \
--max_answer_len 45 \
--version_2_with_negative \
--warmup_proportion 0.06 \
--verbose_logging

output_dir='experiments/hotpot_open/reader_roberta_base/hotpot_roberta_reader3/4266steps'

python reader/run_reader_confidence_roberta.py \
--bert_model $bert_dir \
--output_dir $output_dir \
--train_file $hotpot_train_file_path \
--predict_file $hotpot_dev_file_path \
--learning_rate 3e-5 \
--max_seq_length 512 \
--doc_stride 160 --max_query_length -1 \
--do_predict \
--version_2_with_negative \
--warmup_proportion 0.06

output_dir='experiments/hotpot_open/reader_roberta_base/hotpot_roberta_reader3/6399steps'

python reader/run_reader_confidence_roberta.py \
--bert_model $bert_dir \
--output_dir $output_dir \
--train_file $hotpot_train_file_path \
--predict_file $hotpot_dev_file_path \
--learning_rate 3e-5 \
--max_seq_length 512 \
--doc_stride 160 --max_query_length -1 \
--do_predict \
--version_2_with_negative \
--warmup_proportion 0.06

output_dir='experiments/hotpot_open/reader_roberta_base/hotpot_roberta_reader3/8532steps'

python reader/run_reader_confidence_roberta.py \
--bert_model $bert_dir \
--output_dir $output_dir \
--train_file $hotpot_train_file_path \
--predict_file $hotpot_dev_file_path \
--learning_rate 3e-5 \
--max_seq_length 512 \
--doc_stride 160 --max_query_length -1 \
--do_predict \
--version_2_with_negative \
--warmup_proportion 0.06

output_dir='experiments/hotpot_open/reader_roberta_base/hotpot_roberta_reader3/10665steps'

python reader/run_reader_confidence_roberta.py \
--bert_model $bert_dir \
--output_dir $output_dir \
--train_file $hotpot_train_file_path \
--predict_file $hotpot_dev_file_path \
--learning_rate 3e-5 \
--max_seq_length 512 \
--doc_stride 160 --max_query_length -1 \
--do_predict \
--version_2_with_negative \
--warmup_proportion 0.06
--verbose_logging

output_dir='experiments/hotpot_open/reader_roberta_base/hotpot_roberta_reader3/12798steps'

python reader/run_reader_confidence_roberta.py \
--bert_model $bert_dir \
--output_dir $output_dir \
--train_file $hotpot_train_file_path \
--predict_file $hotpot_dev_file_path \
--learning_rate 3e-5 \
--max_seq_length 512 \
--doc_stride 160 --max_query_length -1 \
--do_predict \
--version_2_with_negative \
--warmup_proportion 0.06

output_dir='experiments/hotpot_open/reader_roberta_base/hotpot_roberta_reader3/14931steps'

python reader/run_reader_confidence_roberta.py \
--bert_model $bert_dir \
--output_dir $output_dir \
--train_file $hotpot_train_file_path \
--predict_file $hotpot_dev_file_path \
--learning_rate 3e-5 \
--max_seq_length 512 \
--doc_stride 160 --max_query_length -1 \
--do_predict \
--version_2_with_negative \
--warmup_proportion 0.06

output_dir='experiments/hotpot_open/reader_roberta_base/hotpot_roberta_reader3/17064steps'

python reader/run_reader_confidence_roberta.py \
--bert_model $bert_dir \
--output_dir $output_dir \
--train_file $hotpot_train_file_path \
--predict_file $hotpot_dev_file_path \
--learning_rate 3e-5 \
--max_seq_length 512 \
--doc_stride 160 --max_query_length -1 \
--do_predict \
--version_2_with_negative \
--warmup_proportion 0.06

output_dir='experiments/hotpot_open/reader_roberta_base/hotpot_roberta_reader3/19197steps'

python reader/run_reader_confidence_roberta.py \
--bert_model $bert_dir \
--output_dir $output_dir \
--train_file $hotpot_train_file_path \
--predict_file $hotpot_dev_file_path \
--learning_rate 3e-5 \
--max_seq_length 512 \
--doc_stride 160 --max_query_length -1 \
--do_predict \
--version_2_with_negative \
--warmup_proportion 0.06

output_dir='experiments/hotpot_open/reader_roberta_base/hotpot_roberta_reader3/21330steps'

python reader/run_reader_confidence_roberta.py \
--bert_model $bert_dir \
--output_dir $output_dir \
--train_file $hotpot_train_file_path \
--predict_file $hotpot_dev_file_path \
--learning_rate 3e-5 \
--max_seq_length 512 \
--doc_stride 160 --max_query_length -1 \
--do_predict \
--version_2_with_negative \
--warmup_proportion 0.06
