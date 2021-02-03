


## baseline
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_roberta.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/roberta-base --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 2e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_roberta_baseline1/ --max_seq_length 380 --model_version roberta " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_roberta_1" -Dcluster="{\"worker\":{\"cpu\":600, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;

<!-- roberta + s/r pre-training -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_roberta.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/roberta-base --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 2e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_roberta_baseline1/ --max_seq_length 380 --model_version v3 --oss_pretrain roberta_iter_sr_mlm_s_2/pytorch_model_80000.bin " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_rob_iter_1" -Dcluster="{\"worker\":{\"cpu\":800, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;


# pre-process features
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_roberta.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/roberta-base --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 2e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_roberta_baseline1/ --max_seq_length 400 --model_version roberta --do_label " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_rob_bl_1" -Dcluster="{\"worker\":{\"cpu\":400, \"memory\":200000, \"gpu\":0}}" 
-DworkerCount=1;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_roberta.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/roberta-base --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 2e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_roberta_baseline1/ --model_version roberta --do_label " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_rob_bl_1" -Dcluster="{\"worker\":{\"cpu\":400, \"memory\":200000, \"gpu\":0}}" 
-DworkerCount=1;
