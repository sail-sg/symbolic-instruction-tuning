accelerate launch --config_file train_config.yaml train_model.py \
  --model_name_or_path google/flan-t5-large \
  --dataset_name sail/symbolic-instruction-tuning \
  --eval_func get_denotation_accuracy \
  --input_column input \
  --output_column output \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 6 \
  --gradient_accumulation_steps 32 \
  --learning_rate 3e-5 \
  --preprocessing_num_workers 16 \
  --generation_max_length 128 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --max_steps 20000 \
  --logging_strategy steps \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --warmup_steps 1000 \
  --max_seq_length 2048 \
  --max_answer_length 128 \
  --val_max_answer_length 128 \
  --output_dir checkpoints/tapex_zero_large \
  --run_name tapex_zero_large