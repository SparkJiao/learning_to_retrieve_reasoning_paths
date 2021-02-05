pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --do_train --version_2_with_negative --fp16 --fp16_opt_level O2 --oss_cache_dir iter_bert_reader_1/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader1" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=4;

# do lower case
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 16 --oss_cache_dir iter_bert_reader_2/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader2" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 2 --learning_rate 3e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir iter_bert_reader_3/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader3" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=4;

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3 --learning_rate 3e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir iter_bert_reader_4/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader4" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=4;

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 2 --learning_rate 4e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir iter_bert_reader_5/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader5" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=4;


# No mixed precision + single GPU

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3.0 --learning_rate 5e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir iter_bert_reader_6/ --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_6" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;


# Fix duplicate dropout
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3.0 --learning_rate 5e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir iter_bert_reader_7/ --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_7" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;

# Fix loading pre-trained model
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3.0 --learning_rate 5e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir iter_bert_reader_7_wpt20k/ --save_gran 20,5 --oss_pretrain bert_iter_sr_mlm_1/pytorch_model_20000.bin " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_7_wpt20k" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;


# Fix loading pre-trained model of 40k pre-training
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3.0 --learning_rate 5e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir iter_bert_reader_7_wpt40k/ --save_gran 20,5 --oss_pretrain bert_iter_sr_mlm_2r/pytorch_model_40000.bin " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_7_wpt40k" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;


# low learning rate
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3.0 --learning_rate 4e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir iter_bert_reader_8/ --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_8" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;


# V2 version
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter_v2.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3.0 --learning_rate 5e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir iter_bert_reader_v2_0/ --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_v2_0" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;

# V3 version
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter_v3.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3.0 --learning_rate 5e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir iter_bert_reader_v3_0/ --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_v3_0" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;

# low learning_rate 
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter_v3.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3.0 --learning_rate 4e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir iter_bert_reader_v3_1/ --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_v3_1" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;

# V4
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3.0 --learning_rate 5e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir iter_bert_reader_v4_0/ --model_version v4 --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_v4_0" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;

# Larger learning rate
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_iter.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3.0 --learning_rate 6e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir iter_bert_reader_v4_1/ --model_version v4 --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_iter_bb_reader_v4_1" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;

==========================================

# bert-baseline
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --do_train --version_2_with_negative --fp16 --fp16_opt_level O1 --train_batch_size 16 --oss_cache_dir bert_reader_baseline_1/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_1" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;

# bert-baseline do lower case
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 16 --oss_cache_dir bert_reader_baseline_2/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_2" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;

<!-- bert-baseline w mlm-baseline pre-train -->
pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert_mlm_baseline_2 --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3.0 --learning_rate 5e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir bert_reader_mlm_baseline/ --dist --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/bert_mlm_baseline2,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_mlm_14" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 2 --learning_rate 3e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir bert_reader_baseline_3/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_3" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=4;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3 --learning_rate 3e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 16 --gradient_accumulation_steps 1 --oss_cache_dir bert_reader_baseline_4/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_4" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3 --learning_rate 4e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 16 --gradient_accumulation_steps 1 --oss_cache_dir bert_reader_baseline_5/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_5" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 5 --learning_rate 3e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 16 --gradient_accumulation_steps 1 --oss_cache_dir bert_reader_baseline_6/ --dist " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_6" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 4 --learning_rate 3e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 16 --gradient_accumulation_steps 1 --oss_cache_dir bert_reader_baseline_7/ --dist --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_7" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 4 --learning_rate 3e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 16 --gradient_accumulation_steps 1 --oss_cache_dir bert_reader_baseline_7/ --dist --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_7" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 2 --learning_rate 5e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 16 --gradient_accumulation_steps 1 --oss_cache_dir bert_reader_baseline_8/ --dist --save_gran 10,2 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_8" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 2 --learning_rate 4e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 16 --gradient_accumulation_steps 1 --oss_cache_dir bert_reader_baseline_9/ --dist --save_gran 10,2 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_9" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 512 --max_query_length 128 --num_train_epochs 4 --learning_rate 3e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 16 --gradient_accumulation_steps 2 --oss_cache_dir bert_reader_baseline_10/ --dist --save_gran 20,4 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_10" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 512 --max_query_length 128 --num_train_epochs 4 --learning_rate 2e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 16 --gradient_accumulation_steps 2 --oss_cache_dir bert_reader_baseline_11/ --dist --save_gran 20,4 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_11" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;

# no mixed precision

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 4 --learning_rate 3e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 16 --gradient_accumulation_steps 1 --oss_cache_dir bert_reader_baseline_12/ --dist --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_12" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3.0 --learning_rate 6e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 16 --gradient_accumulation_steps 1 --oss_cache_dir bert_reader_baseline_13/ --dist --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_13" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":40000, \"gpu\":100}}" -DworkerCount=2;


pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 384 --num_train_epochs 3.0 --learning_rate 5e-5 --do_train --version_2_with_negative --do_lower_case --train_batch_size 32 --gradient_accumulation_steps 2 --oss_cache_dir bert_reader_baseline_14/ --dist --save_gran 20,5 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_14" 
-Dcluster="{\"worker\":{\"cpu\":400, \"memory\":40000, \"gpu\":100}}" -DworkerCount=1;

# pre-process features

pai -name pytorch151 
-Dscript="file:///home/admin/workspace/project/learning_to_retrieve.tar.gz" 
-DentryFile="reader/run_reader_confidence_bert.py" -DuserDefinedParameters="--bert_model /data/volume1/bert-base-uncased --train_file /data/volume2/hotpot_reader_train_data.json --predict_file /data/volume2/hotpot_dev_squad_v2.0_format.json --output_dir /data/output1/ --max_seq_length 512 --max_query_length 128 --num_train_epochs 2 --learning_rate 4e-5 --do_train --version_2_with_negative --do_lower_case --fp16 --fp16_opt_level O1 --train_batch_size 16 --gradient_accumulation_steps 1 --oss_cache_dir bert_reader_baseline_tmp/ --do_label --save_gran 10,2 " 
-Dvolumes="odps://crm_nlp_dev/volumes/fangxi/transformers,odps://crm_nlp_dev/volumes/fangxi/hotpot_reader_data" 
-Doutputs="odps://crm_nlp_dev/volumes/fangxi/hotpot_bb_reader_bl_tmp" 
-Dcluster="{\"worker\":{\"cpu\":300, \"memory\":80000, \"gpu\":0}}" -DworkerCount=1;
