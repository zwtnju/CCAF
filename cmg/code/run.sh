model=codet5
data=fira
python -m accelerate.commands.launch run.py \
  --output_dir ./saved_models/${model}/${data} \
  --model_type ${model} \
  --model_name_or_path ../model/${model} \
  --do_train \
  --do_test \
  --train_filename ../data/${data}/train_diff.json,../data/${data}/train_msg.json \
  --dev_filename ../data/${data}/test_diff.json,../data/${data}/test_msg.json \
  --test_filename ../data/${data}/test_diff.json,../data/${data}/test_msg.json \
  --cache_path ./cache/${model}/${data} \
  --max_source_length 512 \
  --max_target_length 512 \
  --beam_size 10 \
  --train_batch_size 32 \
  --eval_batch_size 16 \
  --learning_rate 5e-5 \
  --evaluate_during_training \
  --warmup_steps 100 \
  --train_steps 5000 \
  --eval_steps 2500  2>&1 | tee ${data}_${model}.log

cd ../evaluator
python evaluator.py -ref ../data/${data}/test_msg.json -pre ../code/saved_models/${model}/${data}/predict_msg.json | tee ../code/${data}_${model}.result
cd ../code