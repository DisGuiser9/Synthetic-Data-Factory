# from utils import *
import os

from synthetic_method import *
import argparse
from tqdm import tqdm
# from evaluation import *

def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for the Synthetic Dataset')

    parser.add_argument('--vllm', type=bool,
                        default=True,
                        help='vllm or ollama launching of model')

    parser.add_argument('--chosen-nums-gpu', type=int, default=1)
    parser.add_argument('--rejected-nums-gpu', type=int, default=1)

    parser.add_argument('--chosen-gpu-rate', type=float, default=0.9)
    parser.add_argument('--rejected-gpu-rate', type=float, default=0.9)

    parser.add_argument('--chosen-model', type=str,
                        default='qwen2:72b-instruct-fp16',
                        help='The model to be used')
    parser.add_argument('--rejected-model', type=str,
                        default='qwen2:72b-instruct-fp16',
                        help='The model to be used')

    parser.add_argument('--top_k', type=int, 
                        default=40,
                        help='Reduces the probability of generating nonsense from Ollama model')
    
    parser.add_argument('--top_p', type=float, 
                        default=0.35,
                        help='Work with top_k')

    parser.add_argument('--file_name', type=str, 
                        default='all',
                        help='The file name to be chosen, and will be concated with the original root path')
    
    parser.add_argument('--numbers', type=int, 
                        default=6,
                        help='The number of augmented queries, ranging from [1, 20]')

    parser.add_argument('--langcode', type=str,
                        default='zh',
                        help='The language code of the dataset')

    parser.add_argument('--type', type=str,
                        default='dpo',
                        help='The type of the dataset, could be dpo/sft')

    parser.add_argument('--mode', type=str,
                        default='augment',
                        help='The number of turns in questions')

    parser.add_argument('--literary', type=str,
                        default='博客',
                        help='Literary style of the synthetic answer, only works when the mode choice is literary')

    parser.add_argument('--output_directory', type=str,
                        default='./output_data/',
                        help='The output directory')

    parser.add_argument('--max-new-tokens', type=int,
                        default=512,
                        help='The maximum number of new tokens'
    )

    args = parser.parse_args()
    return args

def post_processing(seed_prompts, rag_content, llm_content, mode, data_type, output_directory, dialogue):
    # score_dataframe = evaluation(seed_prompts, rag_content, llm_content)
    score_dataframe = 'no eval'
    if data_type == 'dpo':
        processed_dataset = post_processing_for_dpo(seed_prompts, rag_content, llm_content, mode, dialogue)
    else:
        processed_dataset = post_processing_for_sft(seed_prompts, rag_content, mode, dialogue)
    dataset = dump_into_json(processed_dataset, output_directory)
    return score_dataframe, dataset

def main():
    args = parse_args()
    directory = '/data/share9/XPT/dataset/dataset_8/book/data'
    running = 'terminal'


    # Get the model in the model list
    available_model_list = local_ollama_models(args.vllm)
    model = get_ollama_model(args.chosen_model)
    if model not in available_model_list:
        if not args.vllm:
            raise ValueError('The model you choose is not in the available model list')
        else:
            print('loading model by vllm')
    else:
        print(f"获取模型{model}中...")

    # Check the number of augmented queries
    numbers = args.numbers
    if args.numbers < 1:
        raise ValueError('The number of augmented queries should be greater than 0')
    # elif args.numbers > 20:
    #     numbers = 20

    top_k = args.top_k
    top_p = args.top_p
    data_type = args.type
    mode = args.mode
    literary = args.literary
    output_directory = args.output_directory

    print(f"开始{mode}问题类生成……")
    # Get the files in the directory
    files_list = get_files_in_directory(directory, return_paths=True)
    files_choices = [os.path.basename(files) for files in files_list]
    """
        TODO
        langchain VLLM setup of Sampling configs e.g. repetition_penalty etc.
        the kwargs collector dont make effect
    """
    model = VLLM(
        model=args.chosen_model,
        max_new_tokens=args.max_new_tokens,
        top_k=10,
        top_p=0.8,
        temperature=0.7,
        tensor_parallel_size=args.chosen_nums_gpu,
        repetition_penalty=1.05,
        restsdasdada=1,
        dadasaf='no effect kwargs collector',
        vllm_kwargs={
            'gpu_memory_utilization': args.chosen_gpu_rate,
        },
    )
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    rejected_model = VLLM(
        model=args.rejected_model,
        max_new_tokens=args.max_new_tokens,
        top_k=10,
        top_p=0.8,
        temperature=0.7,
        tensor_parallel_size=args.rejected_nums_gpu,
        repetition_penalty=1.05,
        restsdasdada=1,
        dadasaf=' no effect kwargs collector',
        vllm_kwargs={
            'gpu_memory_utilization': args.rejected_gpu_rate,
        },
        # vllm_kwargs={
        #     'repetition_penalty': 1.05
        # }
    )

    # /share149/huggingface/models--Qwen--Qwen2-0.5B/snapshots/ff3a49fac17555b8dfc4db6709f480cc8f16a9fe
    # /data/share8/huggingface/models--Qwen--Qwen1.5-14B-Chat/snapshots/17e11c306ed235e970c9bb8e5f7233527140cdcf
    # /data/share9/huggingface/models--Qwen--Qwen1.5-32B-Chat/snapshots/0997b012af6ddd5465d40465a8415535b2f06cfc
    # /share148/huggingface/models--Qwen--Qwen1.5-32B/snapshots/cefef80dc06a65f89d1d71d0adbc56d335ca2490
    for file_name in files_choices:
        if args.file_name == file_name:
            file_path = os.path.join(directory, file_name)

        elif args.file_name == "all":
            file_path = os.path.join(directory, file_name)
            print(f"获取文件《{file_name}》中...")

            if mode == 'single':
                seed_prompts = seed_prompt_generation(model, file_path, numbers)
                rag_content = retrieve_answer(model, seed_prompts, file_path, top_k, top_p, running, mode)
                llm_content = llm_answer(rejected_model, seed_prompts, file_path, top_k, top_p, running, mode)
                post_processing(seed_prompts, rag_content, llm_content, mode, data_type, output_directory, dialogue=None)

            elif mode == 'stepback':
                seed_prompts = seed_prompt_generation(model, file_path, numbers, mode='single')
                # 如果下面的mode赋值为single，那就是两个根据文本提取的随机问题；如果保持，就是stepback式问题
                stepback_prompts = seed_prompt_generation(model, file_path, numbers, mode, prev_question=seed_prompts)
                intermidiate_answers = retrieve_answer(model, stepback_prompts, file_path, top_k, top_p, running, mode='single', second_questions=seed_prompts)
                dialogue = conversation_concat(seed_prompts, intermidiate_answers, numbers, running, mode, stepback_prompts)
                # dialogue在RAG中，和文档一起作为参考资料，抽取第二句问题输入才能获得最好的RAG效果
                rag_content = retrieve_answer(model, dialogue, file_path, top_k, top_p, running, mode, second_questions=seed_prompts)
                llm_content = llm_answer(rejected_model, dialogue, file_path, top_k, top_p, running, mode)
                post_processing(seed_prompts, rag_content, llm_content, mode, data_type, output_directory, dialogue)
            
            elif mode == 'literary' or mode == 'augment':

                # first turn prompt to generate query-1
                seed_prompts = seed_prompt_generation(model, file_path, numbers, mode='single')
                # first turn query to generate answer-1
                intermidiate_answers = retrieve_answer(model, seed_prompts, file_path, top_k, top_p, running, mode='single', second_questions=seed_prompts)

                # second turn prompt to generate query-2
                # dialogue在RAG中，和文档一起作为参考资料，抽取第二句问题输入才能获得最好的RAG效果
                second_prompts = multi_prompts_generation(model, file_path, numbers, seed_prompts, intermidiate_answers, running, mode, literary)
                dialogue = conversation_concat(seed_prompts, intermidiate_answers, numbers, running, mode, second_prompts)

                # second turn query to generate answer-2 that compose chosen and reject answer for dpo data answer
                rag_content = retrieve_answer(model, dialogue, file_path, top_k, top_p, running, mode, literary, second_questions=second_prompts)
                llm_content = llm_answer(rejected_model, dialogue, file_path, top_k, top_p, running, mode, literary)
                post_processing(seed_prompts, rag_content, llm_content, mode, data_type, output_directory, dialogue)
        
        elif args.file_name != file_name:
            continue
        else:
            raise ValueError('The file name you choose is not in the directory')
if __name__ == '__main__':
    main()

    


    


    

    
    