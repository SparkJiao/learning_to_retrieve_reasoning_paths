hotpot_dev_file_path='data/hotpot/hotpot_dev_squad_v2.0_format.json'

output_dir='models/hotpot_models/reader_roberta_baseline_1'

python reader/run_reader_confidence_roberta.py \
--model_version roberta \
--bert_model $output_dir \
--output_dir $output_dir \
--predict_file $hotpot_dev_file_path \
--max_seq_length 384 \
--do_predict \
--version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json



output_dir='models/hotpot_models/reader_roberta_baseline_1/27636steps'

python reader/run_reader_confidence_roberta.py \
--model_version roberta \
--bert_model $output_dir \
--output_dir $output_dir \
--predict_file $hotpot_dev_file_path \
--max_seq_length 384 \
--do_predict \
--version_2_with_negative 

python squad2_eval.py $hotpot_dev_file_path $output_dir/predictions.json
