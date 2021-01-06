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



retriever_path='models/hotpot_models/graph_retriever_bert_1/pytorch_model_85000.bin'

# iter retriever all data
python eval_main.py \
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
--selector_results_save_path evaluation_results/bert_base_retriever_1/graph_retriever_iter_results/retriever_results.json \
--reader_results_save_path evaluation_results/bert_base_retriever_1/reader_results/reader_results.json \
--sequence_sentence_selector_save_path evaluation_results/bert_base_retriever_1/sequence_sentence_selector_results/sequence_sentence_selector_results.json
