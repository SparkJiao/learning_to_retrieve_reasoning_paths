
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert_iter_sr_mlm_1 --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4  --fp16 --fp16_opt_level O1 --oss_cache_dir graph_retriever_bert_iter_1/" 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/bert_iter_sr_mlm_1,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/hotpot_retriever_cache_train" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_1" -Dcluster="{\"worker\":{\"cpu\":400, \"memory\":100000, \"gpu\":400}}" 
-DworkerCount=1;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert_iter_sr_mlm_1 --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4  --fp16 --fp16_opt_level O1 --oss_cache_dir graph_retriever_bert_iter_2/ --cache_dir /data/volume3/" 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/bert_iter_sr_mlm_1,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_1" -Dcluster="{\"worker\":{\"cpu\":200, \"memory\":160000, \"gpu\":400}}" 
-DworkerCount=1;


# distributed training
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert_iter_sr_mlm_1 --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 16 --train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4  --fp16 --fp16_opt_level O2 --oss_cache_dir graph_retriever_bert_iter_2/ --cache_dir /data/volume3/ --dist --resume 25000 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/bert_iter_sr_mlm_1,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_2" -Dcluster="{\"worker\":{\"cpu\":300, \"memory\":120000, \"gpu\":100}}" 
-DworkerCount=4;

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert_iter_sr_mlm_1 --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 3 --use_redundant --max_select_num 4  --fp16 --fp16_opt_level O1 --oss_cache_dir graph_retriever_bert_iter_1/ --cache_dir /data/volume3/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/bert_iter_sr_mlm_1,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_1" -Dcluster="{\"worker\":{\"cpu\":500, \"memory\":100000, \"gpu\":100}}" 
-DworkerCount=8;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert_iter_sr_mlm_1 --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 5 --use_redundant --max_select_num 4  --fp16 --fp16_opt_level O1 --oss_cache_dir graph_retriever_bert_iter_3/ --cache_dir /data/volume3/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/bert_iter_sr_mlm_1,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_3" -Dcluster="{\"worker\":{\"cpu\":500, \"memory\":100000, \"gpu\":100}}" 
-DworkerCount=8;

# Re-run
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert_iter_sr_mlm_1 --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_4/ --cache_dir /data/volume3/ --dist" 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/bert_iter_sr_mlm_1,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_4" -Dcluster="{\"worker\":{\"cpu\":400, \"memory\":120000, \"gpu\":100}}" 
-DworkerCount=4;

<!-- 40k model -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_5/ --oss_pretrain bert_iter_sr_mlm_2r/pytorch_model_40000.bin "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_5" -Dcluster="{\"worker\":{\"cpu\":400, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;

<!-- 40k model with less epoch -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 2 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_6/ --oss_pretrain bert_iter_sr_mlm_2r/pytorch_model_40000.bin "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_6" -Dcluster="{\"worker\":{\"cpu\":800, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;

<!-- 20k v2 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert_iter_sr_mlm_1 --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_v2_0/ --cache_dir /data/volume3/ --model_version v2 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/bert_iter_sr_mlm_1,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_v2_0" -Dcluster="{\"worker\":{\"cpu\":400, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;


<!-- 20k model // Fix target setting // Disable layer norm -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_7/ --oss_pretrain bert_iter_sr_mlm_1/pytorch_model_20000.bin --disable_rnn_layer_norm "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_7" -Dcluster="{\"worker\":{\"cpu\":800, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;

<!-- 20k model // Fix target setting // Disable layer norm // dist-->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_7d/ --oss_pretrain bert_iter_sr_mlm_1/pytorch_model_20000.bin --disable_rnn_layer_norm --dist "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_7d" -Dcluster="{\"worker\":{\"cpu\":600, \"memory\":120000, \"gpu\":100}}" 
-DworkerCount=4;
<!-- distributed training caused non-convergence. -->


<!-- 20k model // Fix target setting // version v3 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_v3_0/ --oss_pretrain bert_iter_sr_mlm_1/pytorch_model_20000.bin --model_version v3 --disable_rnn_layer_norm "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_v3_0" -Dcluster="{\"worker\":{\"cpu\":600, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;

<!-- 40k model // Fix target setting // version v3 -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_v3_1/ --oss_pretrain bert_iter_sr_mlm_2r/pytorch_model_40000.bin --model_version v3 --disable_rnn_layer_norm "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_v3_1" -Dcluster="{\"worker\":{\"cpu\":600, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;

<!-- 20k model // Fix target setting // version v3_2 // remove ``--disable_rnn_layer_norm`` option -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_v3_2/ --oss_pretrain bert_iter_sr_mlm_1/pytorch_model_20000.bin --model_version v3 "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_v3_2" -Dcluster="{\"worker\":{\"cpu\":600, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;


<!-- 40k model // Fix target setting // version v3_3 // remove ``--disable_rnn_layer_norm`` option -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_v3_3/ --oss_pretrain bert_iter_sr_mlm_2r/pytorch_model_40000.bin --model_version v3 "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_v3_3" -Dcluster="{\"worker\":{\"cpu\":600, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;

<!-- 20k model // Fix target setting // version v3_4 // remove ``--disable_rnn_layer_norm`` option // smaller learning rate -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 2e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_v3_4/ --oss_pretrain bert_iter_sr_mlm_1/pytorch_model_20000.bin --model_version v3 "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_v3_4" -Dcluster="{\"worker\":{\"cpu\":600, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;


<!-- 20k model // Fix target setting // version v3_4 // remove ``--disable_rnn_layer_norm`` option // larger learning rate -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 4e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_v3_5/ --oss_pretrain bert_iter_sr_mlm_1/pytorch_model_20000.bin --model_version v3 "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_v3_5" -Dcluster="{\"worker\":{\"cpu\":600, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;

<!-- 20k model // Fix target setting // version v3 // dist -->
<!-- pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_v3_0d/ --oss_pretrain bert_iter_sr_mlm_1/pytorch_model_20000.bin --model_version v3 --disable_rnn_layer_norm --dist "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_v3_0d" -Dcluster="{\"worker\":{\"cpu\":500, \"memory\":120000, \"gpu\":100}}" 
-DworkerCount=4; -->


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_iter_v3_0/ --oss_pretrain bert_iter_sr_mlm_1/pytorch_model_20000.bin --model_version v3 --disable_rnn_layer_norm "
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_v3_0" -Dcluster="{\"worker\":{\"cpu\":600, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;


## baseline
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_baseline_1/ --cache_dir /data/volume3/ --model_version bert " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_1" -Dcluster="{\"worker\":{\"cpu\":600, \"memory\":100000, \"gpu\":400}}" 
-DworkerCount=1;

# Single node + no precision
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_bert.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4 --oss_cache_dir graph_retriever_bert_3/ --cache_dir /data/volume3/ " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_387_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_3" -Dcluster="{\"worker\":{\"cpu\":500, \"memory\":120000, \"gpu\":400}}" 
-DworkerCount=1;


## baseline + max_seq_length == 512
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_bert.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert-base-uncased --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 3e-5 --num_train_epochs 3 --use_redundant --max_select_num 4  --fp16 --fp16_opt_level O2 --oss_cache_dir graph_retriever_bert_2/ --max_seq_length 512 --cache_dir /data/volume3/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models,odps://crm_nlp_dev/volumes/fangxi/retri_bert_cache_512_50" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_2" -Dcluster="{\"worker\":{\"cpu\":300, \"memory\":140000, \"gpu\":100}}" 
-DworkerCount=4;

# pre-process features
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="graph_retriever/run_graph_retriever_iter.py" 
-DuserDefinedParameters="--task hotpot_open --bert_model /data/volume1/bert_iter_sr_mlm_1 --train_file_path /data/volume2/hotpotqa_new_selector_train_data_db_2017_10_12_fix/db --output_dir /data/output1/ --max_para_num 50 --tfidf_limit 40 --neg_chunk 8 --train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 3 --use_redundant --max_select_num 4  --fp16 --fp16_opt_level O1 --oss_cache_dir graph_retriever_bert_iter_tmp/ --max_seq_length 512 --do_label " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/bert_iter_sr_mlm_1,odps://crm_nlp_dev/volumes/fangxi/retrieve_datasets_and_models" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_retri_bert_iter_tmp" -Dcluster="{\"worker\":{\"cpu\":400, \"memory\":200000, \"gpu\":0}}" 
-DworkerCount=1;
