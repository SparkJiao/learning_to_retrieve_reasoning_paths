hotpot_train_file_path='../learning_to_retrieve_reasoning_paths/models/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db'
cached_features='../learning_to_retrieve_reasoning_paths/models/hotpotqa_new_selector_train_data_db_2017_10_12_fix/torch_cached_features'
output_dir='experiments/hotpot_open/graph_retriever/test/'
bert_dir='../../pretrained_model/bert-base-uncased'

python graph_retriever/run_graph_retriever.py \
--task hotpot_open \
--bert_model $bert_dir \
--train_file_path $hotpot_train_file_path \
--cached_features $cached_features \
--output_dir $output_dir \
--max_para_num 50 \
--tfidf_limit 40 \
--neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 4 \
--learning_rate 3e-5 --num_train_epochs 3 \
--use_redundant \
--max_select_num 4