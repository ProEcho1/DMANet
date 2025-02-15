export CUDA_VISIBLE_DEVICES=1

model_name=DMANet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 512 \
  --batch_size 64 \
  --learning_rate 0.0002 \
  --global_topk 3\
  --auxi_lambda 0.5 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 512 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --global_topk 3\
  --auxi_lambda 0.5 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ETT-small/\
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 512 \
  --batch_size 32 \
  --learning_rate 0.0002 \
  --global_topk 3 \
  --auxi_lambda 1.0 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/ETT-small/\
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 512 \
  --batch_size 16 \
  --learning_rate 0.0002 \
  --global_topk 3\
  --auxi_lambda 1.0 \
  --des 'Exp' \
  --itr 1