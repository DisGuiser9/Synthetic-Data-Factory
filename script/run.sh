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
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
export NUMBER=50
echo "model-path: $1 nums-gpu: $2 gpus: $CUDA_VISIBLE_DEVICES numbers-per-book: $NUMBER"
python main.py \
    --model $1 \
    --nums-gpu $2 \
    --mode augment \
    --numbers $NUMBER \
    --max-new-tokens 2048
