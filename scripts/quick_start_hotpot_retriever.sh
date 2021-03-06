# download trained models
# mkdir models
# cd models
# gdown https://drive.google.com/uc?id=1ra37xtEXSROG_f90XxR4kgElGJWUHQyM
# unzip hotpot_models.zip
# rm hotpot_models.zip
# cd ..

# download eval data
# mkdir data
# cd data
# mkdir hotpot
# cd hotpot
# gdown https://drive.google.com/uc?id=1m_7ZJtWQsZ8qDqtItDTWYlsEHDeVHbPt
# gdown https://drive.google.com/uc?id=1D-Uj4DPMZWkSouzw5Gg5YhkGiBHSqCJp
# wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
# cd ../..

reader_path='models/hotpot_models/reader'
# reader_path='experiments/hotpot_open/reader_roberta_base/hotpot_roberta_reader3'
retriever_path='models/hotpot_models/graph_retriever_path/pytorch_model.bin'
# retriever_path='models/hotpot_models/graph_retriever_bert_iter_1/pytorch_model_1.bin'
saved_tfidf_retrieval_outputs_path='evaluation_results/baseline/tfidf_results/tfidf_results.json'
# saved_selector_outputs_path='evaluation_results/iter_bert_base_reader/graph_retriever_results/retriever_results.json'


retriever_path='models/hotpot_models/graph_retriever_bert_iter_2/pytorch_model_40000.bin'

# # iter retriever all data
# python eval_main.py \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_2/graph_retriever_iter_results/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_2/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_2/sequence_sentence_selector_results/sequence_sentence_selector_results.json


# retriever_path='models/hotpot_models/graph_retriever_bert_iter_2/pytorch_model_85000.bin'

# # iter retriever all data
# python eval_main.py \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_2/graph_retriever_iter_results/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_2/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_2/sequence_sentence_selector_results/sequence_sentence_selector_results.json



# retriever_path='models/hotpot_models/graph_retriever_bert_iter_2/pytorch_model_85000.bin'

# # iter retriever all data
# python eval_main.py \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_2/graph_retriever_iter_results/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_2/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_2/sequence_sentence_selector_results/sequence_sentence_selector_results.json


# retriever_path='models/hotpot_models/graph_retriever_bert_iter_1/pytorch_model_40000.bin'

# # iter retriever all data
# python eval_main.py \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_1/graph_retriever_iter_results/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_1/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_1/sequence_sentence_selector_results/sequence_sentence_selector_results.json


retriever_path='models/hotpot_models/graph_retriever_bert_iter_4/pytorch_model_85000.bin'

# # iter retriever all data
# python eval_main.py \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_4/graph_retriever_iter_results/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_4/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_4/sequence_sentence_selector_results/sequence_sentence_selector_results.json

# retriever_path='models/hotpot_models/graph_retriever_bert_iter_v2_0/pytorch_model_85000.bin' # better performance
# # retriever_path='oss:graph_retriever_bert_iter_v2_0/pytorch_model_60000.bin'

# # iter retriever all data
# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v2' \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_v2_0/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_v2_0/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_v2_0/sequence_sentence_selector_results.json

retriever_path='oss:graph_retriever_bert_iter_5/pytorch_model_85000.bin'

# # iter retriever all data
# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v1' \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_v2_0/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_v2_0/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_v2_0/sequence_sentence_selector_results.json


retriever_path='oss:graph_retriever_bert_iter_7/pytorch_model_15000.bin'

# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v1' \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path 


retriever_path='oss:graph_retriever_bert_iter_7/pytorch_model_25000.bin'

# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v1' \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path 

retriever_path='oss:graph_retriever_bert_iter_7/pytorch_model_45000.bin'

# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v1' \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_7_pt45k/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_7_pt45k/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_7_pt45k/sequence_sentence_selector_results.json

retriever_path='oss:graph_retriever_bert_iter_v3_0/pytorch_model_45000.bin'

# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v3' \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_v3_0_pt45k/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_v3_0_pt45k/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_v3_0_pt45k/sequence_sentence_selector_results.json

retriever_path='oss:graph_retriever_bert_iter_7/pytorch_model_85000.bin'

# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v1' --disable_rnn_layer_norm \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_7_pt85k/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_7_pt85k/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_7_pt85k/sequence_sentence_selector_results.json

retriever_path='oss:graph_retriever_bert_iter_v3_0/pytorch_model_85000.bin'

# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v3' --disable_rnn_layer_norm \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_v3_0_pt85k/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_v3_0_pt85k/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_v3_0_pt85k/sequence_sentence_selector_results.json

# retriever_path='oss:graph_retriever_bert_iter_v3_1/pytorch_model_85000.bin'

# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v3' --disable_rnn_layer_norm \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_v3_1_pt85k/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_v3_1_pt85k/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_v3_1_pt85k/sequence_sentence_selector_results.json


retriever_path='oss:graph_retriever_bert_iter_v3_1/pytorch_model_70000.bin'

# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v3' --disable_rnn_layer_norm \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_v3_1_pt70k/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_v3_1_pt70k/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_v3_1_pt70k/sequence_sentence_selector_results.json

retriever_path='oss:graph_retriever_bert_iter_v3_2/pytorch_model_85000.bin'

# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v3' \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_v3_2_pt85k/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_v3_2_pt85k/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_v3_2_pt85k/sequence_sentence_selector_results.json

retriever_path='oss:graph_retriever_bert_iter_v3_2/pytorch_model_85000.bin'
# Remove --sampled and re-test
python eval_main.py \
--reader_version 'bert' --graph_retriever_version 'iter_v3' \
--eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
--eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
--graph_retriever_path $retriever_path \
--reader_path $reader_path \
--sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
--tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
--db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
--bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
--tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
--beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
--saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
--saved_selector_outputs_path evaluation_results/iter_bert_base_retriever_v3_2_pt85k/retriever_results.json \
--reader_results_save_path evaluation_results/iter_bert_base_retriever_v3_2_pt85k/reader_results.json \
--sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_v3_2_pt85k/sequence_sentence_selector_results.json

# retriever_path='models/hotpot_models/graph_retriever_bert_iter_v3_2/pytorch_model_3.bin'

# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v3' \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_v3_2/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_v3_2/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_v3_2/sequence_sentence_selector_results.json


# retriever_path='oss:graph_retriever_bert_iter_v3_2/pytorch_model_80000.bin'

# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v3' \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_v3_2_pt80k/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_v3_2_pt80k/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_v3_2_pt80k/sequence_sentence_selector_results.json

retriever_path='oss:graph_retriever_bert_iter_v3_3/pytorch_model_85000.bin'

# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'iter_v3' \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_v3_3_pt85k/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_v3_3_pt85k/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_v3_3_pt85k/sequence_sentence_selector_results.json

retriever_path='oss:graph_retriever_roberta_baseline1/pytorch_model_85000.bin'

# python eval_main.py \
# --reader_version 'bert' --graph_retriever_version 'roberta' \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --bert_model_graph_retriever /home/admin/workspace/roberta-base \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/iter_bert_base_retriever_v3_3_pt85k/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_retriever_v3_3_pt85k/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_retriever_v3_3_pt85k/sequence_sentence_selector_results.json


# retriever_path='models/hotpot_models/graph_retriever_bert_1/pytorch_model_85000.bin'

# # iter retriever all data
# python eval_main.py \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --selector_results_save_path evaluation_results/bert_base_retriever_1/graph_retriever_iter_results/retriever_results.json \
# --reader_results_save_path evaluation_results/bert_base_retriever_1/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/bert_base_retriever_1/sequence_sentence_selector_results/sequence_sentence_selector_results.json
