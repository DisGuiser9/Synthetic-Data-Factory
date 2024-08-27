#python main.py\
#    --top_k 40\
#    --top_p 0.3\
#    --numbers 2\
#    --langcode 'zh'\
#    --type 'dpo'\
#    --mode 'literary'\
#    --literary '博客'\
# stepback single

export CUDA_VISIBLE_DEVICES=8,6
echo "model-path: $1 nums-gpu: $2 gpus: $CUDA_VISIBLE_DEVICES"
python main.py \
    --model $1 \
    --nums-gpu $2 \
    --mode stepback \
    --numbers 4
