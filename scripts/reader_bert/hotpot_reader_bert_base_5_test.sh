hotpot_dev_file_path='data/hotpot/hotpot_dev_squad_v2.0_format.json'

output_dir='models/hotpot_models/reader_bert_base_baseline_5/13716steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_5/18288steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_5/22860steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_5/27432steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_5/32004steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_5/36576steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_5/41148steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_5/45720steps'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json

output_dir='models/hotpot_models/reader_bert_base_baseline_5'

# python reader/run_reader_confidence_bert.py \
# --bert_model $output_dir \
# --output_dir $output_dir \
# --predict_file $hotpot_dev_file_path \
# --max_seq_length 384 \
# --do_predict \
# --do_lower_case \
# --version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json
