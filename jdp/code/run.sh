model=codebert
for project in gerrit go jdt openstack platform qt
do
python run.py \
    --output_dir ./saved_models/${model}/${project}/ \
    --model_type ${model} \
    --model_name_or_path ../model/${model} \
    --do_train \
    --do_test \
    --local_rank -1 \
    --train_data_file ../data/${project}/train.pkl \
    --test_data_file ../data/${project}/test.pkl \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --warmup_steps 100 \
    --seed 123456 2>&1 | tee ${project}_${model}.log

python ../evaluator/evaluator.py -a ../data/${project}/test.pkl -p saved_models/${model}/${project}/predictions.txt | tee ${project}_${model}.result
done
