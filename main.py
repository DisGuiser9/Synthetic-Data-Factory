from utils import *
import argparse
from tqdm import tqdm
from evaluation import *

def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for the Synthetic Dataset')
    
    parser.add_argument('--model', type=str, 
                        default='qwen2:72b-instruct-fp16',
                        help='The model to be used')
    
    parser.add_argument('--top_k', type=int, 
                        default=40,
                        help='Reduces the probability of generating nonsense from Ollama model')
    
    parser.add_argument('--top_p', type=float, 
                        default=0.35,
                        help='Work with top_k')

    parser.add_argument('--file_name', type=str, 
                        default='报关实务 282.jsonl',
                        help='The file name to be chosen, and will be concated with the original root path')
    
    parser.add_argument('--numbers', type=int, 
                        default=2,
                        help='The number of augmented queries, ranging from [1, 20]')
    
    parser.add_argument('--langcode', type=str, 
                        default='zh',
                        help='The language code of the dataset')
    
    parser.add_argument('--type', type=str,
                        default='dpo',
                        help='The type of the dataset, could be dpo/sft')
    
    parser.add_argument('--mode', type=str,
                        default='literary',
                        help='The number of turns in questions')
    
    parser.add_argument('--literary', type=str,
                        default='博客',
                        help='Literary style of the synthetic answer, only works when the mode choice is literary')
    
    args = parser.parse_args()
    return args
    
def post_processing(seed_prompts, rag_content, llm_content, dataset_type='dpo'):
        score_dataframe = evaluation(seed_prompts, rag_content, llm_content)
        if dataset_type == 'dpo':
            processed_dataset = post_processing_for_dpo(seed_prompts, rag_content, llm_content)
        dataset = dump_into_json(processed_dataset)
        return score_dataframe, dataset

def main():
    args = parse_args()
    directory = '/data/share9/XPT/dataset/dataset_8/book/data'
    running = 'terminal'
    # model = "qwen2:72b-instruct-fp16"
    # Get the files in the directory
    files_list = get_files_in_directory(directory, return_paths=True)
    files_choices = [os.path.basename(files) for files in files_list]
    file_name = args.file_name
    if file_name not in files_choices:
        raise ValueError('The file you choose is not in the directory')
    else:
        file_path = os.path.join(directory, file_name)
        print(f"获取文件{file_path}中...")

    # Get the model in the model list
    available_model_list = local_ollama_models()
    model = get_ollama_model(args.model)
    if model not in available_model_list:
        raise ValueError('The model you choose is not in the available model list')
    else:
        print(f"获取模型{model}中...")

    # Check the number of augmented queries
    numbers = args.numbers
    if args.numbers < 1:
        raise ValueError('The number of augmented queries should be greater than 0')
    elif args.numbers > 20:
        numbers = 20

    top_k = args.top_k
    top_p = args.top_p
    data_type = args.type
    mode = args.mode
    literary = args.literary

    if mode == 'single':
        seed_prompts = seed_prompt_generation(model, file_path, numbers)
        rag_content = retrieve_answer(model, seed_prompts, file_path, top_k, top_p, running, mode)
        llm_content = llm_answer(model, seed_prompts, file_path, top_k, top_p, running, mode)
        post_processing(seed_prompts, rag_content, llm_content, data_type)

    elif mode == 'stepback':
        seed_prompts = seed_prompt_generation(model, file_path, numbers, mode='single')
        stepback_prompts = seed_prompt_generation(model, file_path, numbers, seed_prompts, mode)
        intermidiate_answers = retrieve_answer(model, stepback_prompts, file_path, top_k, top_p, running, mode)
        dialogue = conversation_concat(seed_prompts, intermidiate_answers, numbers, running, mode, stepback_prompts)
        rag_content = retrieve_answer(model, dialogue, file_path, top_k, top_p, running, mode)
        llm_content = llm_answer(model, dialogue, file_path, top_k, top_p, running, mode)
        post_processing(dialogue, rag_content, llm_content, data_type)
    
    elif mode == 'literary' or mode == 'augment':
        seed_prompts = seed_prompt_generation(model, file_path, numbers, mode='single')
        intermidiate_answers = retrieve_answer(model, seed_prompts, file_path, top_k, top_p, running, mode)
        second_prompts = multi_prompts_generation(model, file_path, numbers, seed_prompts, intermidiate_answers, running, mode, None, literary)
        dialogue = conversation_concat(seed_prompts, intermidiate_answers, numbers, running, mode, second_prompts)
        print(dialogue)
        rag_content = retrieve_answer(model, dialogue, file_path, top_k, top_p, running, mode)        
        llm_content = llm_answer(model, dialogue, file_path, top_k, top_p, running, mode)
        print(rag_content)
        post_processing(second_prompts, rag_content, llm_content, data_type)

if __name__ == '__main__':
    main()

    


    


    

    
    