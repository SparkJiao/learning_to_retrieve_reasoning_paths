

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path /data/volume1/baseline_graph_retriever/pytorch_model.bin --reader_path /data/volume3/ --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --saved_selector_outputs_path /data/volume1/saved_baseline_retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_v4_0" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_v4_0_eval" -Dcluster="{\"worker\":{\"cpu\":100, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path /data/volume1/baseline_graph_retriever/pytorch_model.bin --reader_path /data/volume3/ --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --saved_selector_outputs_path /data/volume1/saved_baseline_retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_v4_1" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_v4_1_eval" -Dcluster="{\"worker\":{\"cpu\":100, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;

<!-- hotpot_iter_bb_reader_7_wpt20k -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path /data/volume1/baseline_graph_retriever/pytorch_model.bin --reader_path /data/volume3/ --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --saved_selector_outputs_path /data/volume1/saved_baseline_retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_7_wpt20k" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_7_wpt_eval" -Dcluster="{\"worker\":{\"cpu\":100, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;


<!-- hotpot_iter_bb_reader_7_wpt40k -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path /data/volume1/baseline_graph_retriever/pytorch_model.bin --reader_path /data/volume3/ --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --saved_selector_outputs_path /data/volume1/saved_baseline_retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version iter"
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_7_wpt40k" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader7_wpt40k_eval" -Dcluster="{\"worker\":{\"cpu\":100, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;


<!-- hotpot_roberta_reader_baseline_2 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path /data/volume1/baseline_graph_retriever/pytorch_model.bin --reader_path /data/volume3/ --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --saved_selector_outputs_path /data/volume1/saved_baseline_retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version roberta "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl_2" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl2_eval" -Dcluster="{\"worker\":{\"cpu\":300, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;


<!-- hotpot_roberta_reader_baseline_3 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path /data/volume1/baseline_graph_retriever/pytorch_model.bin --reader_path /data/volume3/ --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --saved_selector_outputs_path /data/volume1/saved_baseline_retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version roberta "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl_3" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl3_eval" -Dcluster="{\"worker\":{\"cpu\":300, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;


<!-- hotpot_roberta_reader_baseline_4 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path /data/volume1/baseline_graph_retriever/pytorch_model.bin --reader_path /data/volume3/ --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --saved_selector_outputs_path /data/volume1/saved_baseline_retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version roberta "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl_4" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl4_eval" -Dcluster="{\"worker\":{\"cpu\":300, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;



<!-- hotpot_roberta_reader iter 2 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path /data/volume1/baseline_graph_retriever/pytorch_model.bin --reader_path /data/volume3/ --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --saved_selector_outputs_path /data/volume1/saved_baseline_retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version roberta_iter "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_iter_2" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_it2_eval" -Dcluster="{\"worker\":{\"cpu\":300, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;




<!-- hotpot_roberta_reader_baseline_6 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path /data/volume1/baseline_graph_retriever/pytorch_model.bin --reader_path /data/volume3/ --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --saved_selector_outputs_path /data/volume1/saved_baseline_retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version roberta "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl_6" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl6_eval" -Dcluster="{\"worker\":{\"cpu\":300, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;


<!-- hotpot_roberta_reader iter 3 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path /data/volume1/baseline_graph_retriever/pytorch_model.bin --reader_path /data/volume3/ --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --sampled --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --saved_selector_outputs_path /data/volume1/saved_baseline_retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version roberta_iter "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_iter_3" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_it3_eval" -Dcluster="{\"worker\":{\"cpu\":300, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;


<!-- hotpot_roberta_reader iter 3 w/o pt -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path /data/volume1/baseline_graph_retriever/pytorch_model.bin --reader_path /data/volume3/ --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --saved_selector_outputs_path /data/volume1/saved_baseline_retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version roberta_iter "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_iter_3_wopt" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_it3_wopt_eval" -Dcluster="{\"worker\":{\"cpu\":300, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;


<!-- hotpot_roberta_reader  w. mlm_baseline pre-training -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="eval_main.py" 
-DuserDefinedParameters="--eval_file_path /data/volume1/hotpot_fullwiki_data.jsonl --eval_file_path_sp /data/volume1/hotpot_dev_distractor_v1.json --graph_retriever_path /data/volume1/baseline_graph_retriever/pytorch_model.bin --reader_path /data/volume3/ --sequential_sentence_selector_path /data/volume1/sequential_sentence_selector/pytorch_model.bin --tfidf_path /data/volume1/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --db_path /data/volume1/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db --bert_model_sequential_sentence_selector /data/volume2/ --do_lower_case --tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 --beam_sequential_sentence_selector 8 --max_para_num 2000 --sp_eval --saved_tfidf_retrieval_outputs_path /data/volume1/saved_tfidf_results.json --saved_selector_outputs_path /data/volume1/saved_baseline_retriever_results.json --reader_results_save_path /data/output1/reader_results.json --sequence_sentence_selector_save_path /data/output1/sequence_sentence_selector_results.json --reader_version bert "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/hotpot_eval_cache,odps://crm_nlp_dev/volumes/fangxi/bert_large_uncased,odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_mlm_14" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_mlm_14_eval" -Dcluster="{\"worker\":{\"cpu\":300, \"memory\":30000, \"gpu\":100}}" 
-DworkerCount=1;
