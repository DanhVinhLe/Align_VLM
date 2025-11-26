WORK_DIR=$(cd "$(dirname "$0")";pwd)
export PYTHONPATH=${WORK_DIR}

LANGUAGE_MODEL=mtgv/MobileLLaMA-1.4B-Chat  # or 2.7B
VISION_MODEL=openai/clip-vit-large-patch14-336
ARCH=mobilevlm_v2_1.7b
TASK=finetune.lora
OUTPUT_DIR=${WORK_DIR}/outputs/${ARCH}_$(date +"%Y%m%d_%H%M%S")
DATA_PATH=${WORK_DIR}/data
DISTILL=1
OUTPUT_DIR_PT=mtgv/MobileVLM_V2-1.7B
OUTPUT_DIR_FT=${OUTPUT_DIR}/mobilevlm_v2-2.finetune-lora
mkdir -p ${OUTPUT_DIR_FT}
echo ">>> Start Fine-tuning with LoRA ..."

deepspeed mobilevlm/train/train_mem.py \
    --distill ${DISTILL} \
    --deepspeed scripts/deepspeed/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --learning_rate 2e-4 \
    --model_name_or_path mtgv/MobileVLM_V2-1.7B \
    --version v1 \
    --data_path ${DATA_PATH}/finetune_data/MobileVLM_V2_FT_Mix2M.json \
    --image_folder ${DATA_PATH}/finetune_data \
    --vision_tower ${VISION_MODEL} \
    --vision_tower_type clip \
    --mm_projector_type ldpnetv2 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR_FT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee -a ${OUTPUT_DIR_FT}/log.txt &&
python3 scripts/mergelora.py ${OUTPUT_DIR_PT} ${OUTPUT_DIR}/mobilevlm_v2-2.finetune-lora ${OUTPUT_DIR}/mobilevlm_v2-2.finetune \
    2>&1 | tee -a ${OUTPUT_DIR_FT}/log.txt &&
echo "Done."
