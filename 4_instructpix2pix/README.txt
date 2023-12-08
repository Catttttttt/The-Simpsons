export MODEL_ID="timbrooks/instruct-pix2pix"
export DATASET_ID="aldenn13l/182-fine-tune"
export OUTPUT_DIR="geo-finetuned"

accelerate launch --mixed_precision="fp16" finetune_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_ID \
  --use_ema \
  --resolution=256 --random_flip \
  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=1000 \
  --checkpointing_steps=300 --checkpoints_total_limit=1 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_image_url="https://datasets-server.huggingface.co/assets/aldenn13l/182-fine-tune/--/1014744dd1c828c7d7a4837b8b32a176b1daec13/--/default/train/76/original_image/image.jpg" \
  --validation_prompt="Remove the power lines on the top of the bridge." \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --report_to=tensorboard \
  --push_to_hub



  ^ for training the model (can change) train steps and checkpointing_steps
  
  https://huggingface.co/datasets/aldenn13l/182-fine-tune - dataset
  https://huggingface.co/aldenn13l/geo-finetuned - model