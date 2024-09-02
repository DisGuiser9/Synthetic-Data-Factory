#python main.py\
#    --top_k 40\
#    --top_p 0.3\
#    --numbers 2\
#    --langcode 'zh'\
#    --type 'dpo'\
#    --mode 'literary'\
#    --literary '博客'\
# stepback single augment
# 0,1,2,3,4,5,6,7,8,9
# /data/share9/zwh/XPT/ms-swift/output/qwen1half-32b-chat/v3-20240819-131540/checkpoint-83-merged
#

#
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=2,3
export NUMBER=50
echo "chosen-model-path: $1 rejected-model-path: $2 chosen-nums-gpu: $3 rejected-nums-gpu: $4 gpus: $CUDA_VISIBLE_DEVICES numbers-per-book: $NUMBER"
python main.py \
    --chosen-model $1 \
    --rejected-model $2 \
    --chosen-nums-gpu $3 \
    --rejected-nums-gpu $4 \
    --mode stepback \
    --numbers $NUMBER \
    --max-new-tokens 2048
