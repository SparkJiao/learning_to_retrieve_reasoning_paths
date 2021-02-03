

# roberta-baseline
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_roberta.py" -DuserDefinedParameters="--bert_model /data/volume1/roberta-base --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 386 --num_train_epochs 3.0 --learning_rate 3e-5 --do_train --version_2_with_negative --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir roberta_reader_baseline1/ --save_gran 20,5 --model_version roberta " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl_1" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_roberta.py" -DuserDefinedParameters="--bert_model /data/volume1/roberta-base --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 386 --num_train_epochs 3.0 --learning_rate 3e-5 --do_train --version_2_with_negative --train_batch_size 48 --gradient_accumulation_steps 3 --oss_cache_dir roberta_reader_baseline2/ --save_gran 20,5 --model_version roberta " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl_2" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_roberta.py" -DuserDefinedParameters="--bert_model /data/volume1/roberta-base --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 386 --num_train_epochs 2.0 --learning_rate 2.5e-5 --do_train --version_2_with_negative --train_batch_size 48 --gradient_accumulation_steps 3 --oss_cache_dir roberta_reader_baseline3/ --save_gran 20,5 --model_version roberta " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl_3" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;

<!-- Fix span tagging bug. -->

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_roberta.py" -DuserDefinedParameters="--bert_model /data/volume1/roberta-base --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 386 --num_train_epochs 3.0 --learning_rate 3e-5 --do_train --version_2_with_negative --train_batch_size 48 --gradient_accumulation_steps 3 --oss_cache_dir roberta_reader_baseline4/ --save_gran 20,5 --model_version roberta " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl_4" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_roberta.py" -DuserDefinedParameters="--bert_model /data/volume1/roberta-base --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 386 --num_train_epochs 2.0 --learning_rate 2.5e-5 --do_train --version_2_with_negative --train_batch_size 48 --gradient_accumulation_steps 3 --oss_cache_dir roberta_reader_baseline5/ --save_gran 20,5 --model_version roberta " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl_5" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_roberta.py" -DuserDefinedParameters="--bert_model /data/volume1/roberta-base --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 386 --num_train_epochs 4.0 --learning_rate 3e-5 --do_train --version_2_with_negative --train_batch_size 48 --gradient_accumulation_steps 3 --oss_cache_dir roberta_reader_baseline6/ --save_gran 20,5 --model_version roberta " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_bl_6" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;


<!-- roberta + s/r pre-training -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_roberta.py" -DuserDefinedParameters="--bert_model /data/volume1/roberta-base --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 386 --num_train_epochs 3.0 --learning_rate 3e-5 --do_train --version_2_with_negative --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir roberta_reader_baseline1/ --save_gran 20,5 --model_version v1 --oss_pretrain roberta_iter_sr_mlm_s_2/pytorch_model_80000.bin" 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_iter_1" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_roberta.py" -DuserDefinedParameters="--bert_model /data/volume1/roberta-base --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 386 --num_train_epochs 3.0 --learning_rate 3e-5 --do_train --version_2_with_negative --train_batch_size 48 --gradient_accumulation_steps 3 --oss_cache_dir roberta_reader_iter_2/ --save_gran 20,5 --model_version v1 --oss_pretrain roberta_iter_sr_mlm_s_2/pytorch_model_80000.bin" 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_iter_2" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_roberta.py" -DuserDefinedParameters="--bert_model /data/volume1/roberta-base --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 386 --num_train_epochs 4.0 --learning_rate 3e-5 --do_train --version_2_with_negative --train_batch_size 48 --gradient_accumulation_steps 3 --oss_cache_dir roberta_reader_iter_3/ --save_gran 20,5 --model_version v1 --oss_pretrain roberta_iter_sr_mlm_s_2/pytorch_model_80000.bin" 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_iter_3" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;

<!-- roberta-r w/o pre-training -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_roberta.py" -DuserDefinedParameters="--bert_model /data/volume1/roberta-base --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 386 --num_train_epochs 4.0 --learning_rate 3e-5 --do_train --version_2_with_negative --train_batch_size 48 --gradient_accumulation_steps 3 --oss_cache_dir roberta_reader_iter_3_wopt/ --save_gran 20,5 --model_version v1 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_rob_reader_iter_3_wopt" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;


# pre-process features

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 512 --max_query_length 128 --num_train_epochs 2 --learning_rate 4e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 16 --gradient_accumulation_steps 1 --oss_cache_dir bert_reader_baseline_tmp/ --do_label --save_gran 10,2 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_tmp" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":80000, \"gpu\":0}}" -DworkerCount=1;
