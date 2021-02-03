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
saved_selector_outputs_path='evaluation_results/iter_bert_base_reader/graph_retriever_results/retriever_results.json'

# run evaluation scripts
# python eval_main.py \
# --eval_file_path data/hotpot/hotpot_fullwiki_first_100.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled


# run evaluation scripts / first 100 data
# python eval_main.py \
# --eval_file_path data/hotpot/hotpot_fullwiki_first_100.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --selector_results_save_path evaluation_results/baseline/graph_retriever_results/retriever_results_first100.json

# run evaluation scripts / all data
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
# --selector_results_save_path evaluation_results/baseline/graph_retriever_results/retriever_results.json \
# --tfidf_results_save_path evaluation_results/baseline/tfidf_results/tfidf_results.json \
# --reader_results_save_path evaluation_results/baseline/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/baseline/sequence_sentence_selector_results/sequence_sentence_selector_results.json

# python eval_main.py \
# --eval_file_path data/hotpot/hotpot_fullwiki_first_100.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --selector_results_save_path evaluation_results/iter_bert_base/graph_retriever_iter_results/retriever_results_first100.json

# iter retriever all data

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
# --selector_results_save_path evaluation_results/iter_bert_base/sequence_sentence_selector_results/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base/sequence_sentence_selector_results/sequence_sentence_selector_results.json


# reader_path='models/hotpot_models/reader_bert_iter/44645steps'

# # run evaluation scripts / bert-base-iter-reader
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
# --selector_results_save_path evaluation_results/iter_bert_base_reader/graph_retriever_results/retriever_results.json \
# --reader_results_save_path evaluation_results/iter_bert_base_reader/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_reader/sequence_sentence_selector_results/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_bert_base_baseline_1'

# run evaluation scripts / bert-base-iter-reader
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/iter_bert_base_reader/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_reader/sequence_sentence_selector_results/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_bert_base_baseline_2'

# # run evaluation scripts / bert-base-baseline
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/bert_base_reader_baseline_2/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/bert_base_reader_baseline_2/sequence_sentence_selector_results/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_bert_iter_2/45720steps'

# run evaluation scripts / bert-iter-base
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/iter_bert_base_reader_2/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_reader_2/sequence_sentence_selector_results/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_bert_base_baseline_3'

# # run evaluation scripts / bert-base-baseline
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/bert_base_reader_baseline_3/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/bert_base_reader_baseline_3/sequence_sentence_selector_results/sequence_sentence_selector_results.json

reader_path='models/hotpot_models/reader_bert_base_baseline_4'

# # run evaluation scripts / bert-base-baseline
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/bert_base_reader_baseline_4/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/bert_base_reader_baseline_4/sequence_sentence_selector_results/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_bert_base_baseline_5'

# # run evaluation scripts / bert-base-baseline
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/bert_base_reader_baseline_5/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/bert_base_reader_baseline_5/sequence_sentence_selector_results/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_bert_base_baseline_6'
reader_path='models/hotpot_models/reader_bert_base_baseline_6/68589steps'

# # run evaluation scripts / bert-base-baseline
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/bert_base_reader_baseline_6/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/bert_base_reader_baseline_6/sequence_sentence_selector_results/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_bert_base_baseline_7'

# # run evaluation scripts / bert-base-baseline
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/bert_base_reader_baseline_7/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/bert_base_reader_baseline_7/sequence_sentence_selector_results/sequence_sentence_selector_results.json

# reader_path='models/hotpot_models/reader_bert_base_baseline_8'

# # run evaluation scripts / bert-base-baseline
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/bert_base_reader_baseline_7/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/bert_base_reader_baseline_7/sequence_sentence_selector_results/sequence_sentence_selector_results.json

reader_path='models/hotpot_models/reader_bert_base_baseline_9'

# # run evaluation scripts / bert-base-baseline
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/bert_base_reader_baseline_9/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/bert_base_reader_baseline_9/sequence_sentence_selector_results/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_bert_base_baseline_10/42645steps'

# # run evaluation scripts / bert-base-baseline / max_seq_length = 512
# python eval_main.py \
# --eval_file_path data/hotpot/hotpot_fullwiki_data.jsonl \
# --eval_file_path_sp data/hotpot/hotpot_dev_distractor_v1.json \
# --graph_retriever_path $retriever_path \
# --reader_path $reader_path \
# --sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
# --tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
# --db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
# --bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case --max_seq_length 512 --max_query_length 128 \
# --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
# --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled \
# --saved_tfidf_retrieval_outputs_path $saved_tfidf_retrieval_outputs_path \
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/bert_base_reader_baseline_10/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/bert_base_reader_baseline_10/sequence_sentence_selector_results/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_bert_base_baseline_12/48768steps'

# # run evaluation scripts / bert-base-baseline / max_seq_length = 512
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/bert_base_reader_baseline_12/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/bert_base_reader_baseline_12/sequence_sentence_selector_results/sequence_sentence_selector_results.json

# reader_path='models/hotpot_models/reader_bert_base_baseline_13/45720steps'

# # run evaluation scripts / bert-base-baseline / max_seq_length = 512
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/bert_base_reader_baseline_13/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/bert_base_reader_baseline_13/sequence_sentence_selector_results/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_bert_base_baseline_14'

# # run evaluation scripts / bert-base-baseline
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/bert_base_reader_baseline_14/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/bert_base_reader_baseline_14/sequence_sentence_selector_results/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_bert_iter_6'

# # run evaluation scripts / bert-base-iter
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/iter_bert_base_reader_6/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_reader_6/sequence_sentence_selector_results/sequence_sentence_selector_results.json


# reader_path='models/hotpot_models/reader_bert_iter_7'

# # run evaluation scripts / bert-base-iter
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/iter_bert_base_reader_7/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_reader_7/sequence_sentence_selector_results/sequence_sentence_selector_results.json

# reader_path='models/hotpot_models/reader_bert_iter_8'

# # run evaluation scripts / bert-base-iter
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/iter_bert_base_reader_8/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_reader_8/sequence_sentence_selector_results/sequence_sentence_selector_results.json


# reader_path='models/hotpot_models/reader_bert_iter_v2_0/41148steps'

# # run evaluation scripts / bert-base-iter
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/iter_bert_base_reader_v2_0/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_reader_v2_0/sequence_sentence_selector_results/sequence_sentence_selector_results.json

# reader_path='models/hotpot_models/reader_bert_iter_v3_0'

# # run evaluation scripts / bert-base-iter
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/iter_bert_base_reader_v3_0/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_reader_v3_0/sequence_sentence_selector_results/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_bert_iter_v3_1'

# # run evaluation scripts / bert-base-iter
# python eval_main.py --reader_version iter_v3 \
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/iter_bert_base_reader_v3_1/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_reader_v3_1/sequence_sentence_selector_results/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_roberta_baseline_1'

# python eval_main.py --reader_version roberta \
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/roberta_reader_baseline_1/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/roberta_reader_baseline_1/sequence_sentence_selector_results.json

reader_path='models/hotpot_models/reader_roberta_baseline_5'

# python eval_main.py --reader_version roberta \
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/roberta_reader_baseline_5/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/roberta_reader_baseline_5/sequence_sentence_selector_results.json


reader_path='models/hotpot_models/reader_roberta_baseline_5'

# python eval_main.py --reader_version roberta_iter \
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/roberta_reader_baseline_5/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/roberta_reader_baseline_5/sequence_sentence_selector_results.json

# # Test
# reader_path='models/hotpot_models/reader'

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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/baseline/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/baseline/sequence_sentence_selector_results/sequence_sentence_selector_results.json


# reader_path='models/hotpot_models/reader_bert_iter_5'

# # run evaluation scripts / bert-base-baseline
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
# --saved_selector_outputs_path $saved_selector_outputs_path \
# --reader_results_save_path evaluation_results/iter_bert_base_reader_5/reader_results/reader_results.json \
# --sequence_sentence_selector_save_path evaluation_results/iter_bert_base_reader_5/sequence_sentence_selector_results/sequence_sentence_selector_results.json



# ======================================== Predict

reader_path='models/hotpot_models/reader_roberta_iter_3'

python eval_main.py --reader_version roberta_iter \
--eval_file_path data/hotpot/hotpot_test_fullwiki_v1.jsonl \
--graph_retriever_path $retriever_path \
--reader_path $reader_path \
--sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
--tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
--db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
--bert_model_sequential_sentence_selector /home/admin/workspace/bert-large-uncased --do_lower_case \
--tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
--beam_sequential_sentence_selector 8 --max_para_num 2000 --predict \
--tfidf_results_save_path evaluation_results/iter_rob_base_reader_v3_predict/tfidf_results.json \
--selector_results_save_path evaluation_results/iter_rob_base_reader_v3_predict/graph_retriever_results.json \
--reader_results_save_path evaluation_results/roberta_reader_baseline_5/reader_results.json \
--sequence_sentence_selector_save_path evaluation_results/roberta_reader_baseline_5/sequence_sentence_selector_results.json