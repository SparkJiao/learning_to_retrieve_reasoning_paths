hotpot_dev_file_path='data/hotpot/hotpot_dev_squad_v2.0_format.json'

output_dir='models/hotpot_models/reader_bert_base_baseline_6/22863steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_6/30484steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_6/38105steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_6/45726steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_6/53347steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_6/60968steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_6/68589steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_6/76210steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_6'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json
