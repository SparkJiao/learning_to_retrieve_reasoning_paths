<!-- iter_retriever_4 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path oss:graph_retriever_bert_iter_4/pytorch_model_85000.bin --reader_path /data/volume1/baseline_reader_large --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --bert_model_graph_retriever /data/volume3/bert-base-uncased --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --selector_results_save_path /data/output1/retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/transformers" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_retriever_4_eval" -Dcluster="{\"worker\":{\"cpu\":100, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;


<!-- iter_retriever_v2_0 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path oss:graph_retriever_bert_iter_v2_0/pytorch_model_55000.bin --reader_path /data/volume1/baseline_reader_large --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --bert_model_graph_retriever /data/volume3/bert-base-uncased --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --selector_results_save_path /data/output1/retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/transformers" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_retri_v2_0_5_5K_eval" -Dcluster="{\"worker\":{\"cpu\":100, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;


<!-- iter_retriever_v3_2 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path oss:graph_retriever_bert_iter_v3_2/pytorch_model_70000.bin --reader_path /data/volume1/baseline_reader_large --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --bert_model_graph_retriever /data/volume3/bert-base-uncased --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --selector_results_save_path /data/output1/retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version bert --graph_retriever_version iter_v3 "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/transformers" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_retri_v3_2_70K_eval" -Dcluster="{\"worker\":{\"cpu\":400, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;


<!-- iter_retriever_v3_2 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path oss:graph_retriever_bert_iter_v3_2/pytorch_model_75000.bin --reader_path /data/volume1/baseline_reader_large --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --bert_model_graph_retriever /data/volume3/bert-base-uncased --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --selector_results_save_path /data/output1/retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version bert --graph_retriever_version iter_v3 "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/transformers" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_retri_v3_2_75K_eval" -Dcluster="{\"worker\":{\"cpu\":400, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;

<!-- iter_retriever_v3_2 // remove --sampled -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path oss:graph_retriever_bert_iter_v3_2/pytorch_model_85000.bin --reader_path /data/volume1/baseline_reader_large --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --bert_model_graph_retriever /data/volume3/bert-base-uncased --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --selector_results_save_path /data/output1/retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version bert --graph_retriever_version iter_v3 "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/transformers" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_retri_v3_2_85K_eval" -Dcluster="{\"worker\":{\"cpu\":600, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;


<!-- iter_retriever_v3_3 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path /data/volume4/pytorch_model_3.bin --reader_path /data/volume1/baseline_reader_large --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --bert_model_graph_retriever /data/volume3/bert-base-uncased --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --selector_results_save_path /data/output1/retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version bert --graph_retriever_version iter_v3 "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_v3_3" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_retri_v3_3_3_eval" -Dcluster="{\"worker\":{\"cpu\":800, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;



<!-- roberta retriever iter v1 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path oss:graph_retriever_roberta_baseline1/pytorch_model_85000.bin --reader_path /data/volume1/baseline_reader_large --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --bert_model_graph_retriever /data/volume3/roberta-base --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --selector_results_save_path /data/output1/retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version bert --graph_retriever_version roberta_iter "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_rob_iter_1" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_iter_retri1_85k_eval" -Dcluster="{\"worker\":{\"cpu\":800, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;
