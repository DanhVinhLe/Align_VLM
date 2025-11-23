LANGUAGE_MODEL=mtgv/MobileLLaMA-1.4B-Chat  # or 2.7B
VISION_MODEL=openai/clip-vit-large-patch14-336
bash run.sh mobilevlm_v2_1.7b finetune.lora ${LANGUAGE_MODEL} ${VISION_MODEL}