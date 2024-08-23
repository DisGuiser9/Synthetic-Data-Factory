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
    
    args = parser.parse_args()
    return args
    
def post_processing(seed_prompts, rag_content, llm_content, mode, data_type, output_directory, dialogue):
    score_dataframe = evaluation(seed_prompts, rag_content, llm_content)
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
    output_directory = args.output_directory

    print(f"开始{mode}问题类生成……")
    # Get the files in the directory
    files_list = get_files_in_directory(directory, return_paths=True)
    files_choices = [os.path.basename(files) for files in files_list]

    for file_name in files_choices:
        if args.file_name == file_name:
            file_path = os.path.join(directory, file_name)

        elif args.file_name == "all":
            file_path = os.path.join(directory, file_name)
            print(f"获取文件《{file_name}》中...")
        
            if mode == 'single':
                seed_prompts = seed_prompt_generation(model, file_path, numbers)
                rag_content = retrieve_answer(model, seed_prompts, file_path, top_k, top_p, running, mode)
                llm_content = llm_answer(model, seed_prompts, file_path, top_k, top_p, running, mode)
                post_processing(seed_prompts, rag_content, llm_content, mode, data_type, output_directory, dialogue=None)

            elif mode == 'stepback':
                seed_prompts = seed_prompt_generation(model, file_path, numbers, mode='single')
                # 如果下面的mode赋值为single，那就是两个根据文本提取的随机问题；如果保持，就是stepback式问题
                stepback_prompts = seed_prompt_generation(model, file_path, numbers, mode, prev_question=seed_prompts)
                intermidiate_answers = retrieve_answer(model, stepback_prompts, file_path, top_k, top_p, running, mode='single', second_questions=seed_prompts)
                dialogue = conversation_concat(seed_prompts, intermidiate_answers, numbers, running, mode, stepback_prompts)
                # dialogue在RAG中，和文档一起作为参考资料，抽取第二句问题输入才能获得最好的RAG效果
                rag_content = retrieve_answer(model, dialogue, file_path, top_k, top_p, running, mode, second_questions=seed_prompts)
                llm_content = llm_answer(model, dialogue, file_path, top_k, top_p, running, mode)
                post_processing(seed_prompts, rag_content, llm_content, mode, data_type, output_directory, dialogue)
            
            elif mode == 'literary' or mode == 'augment':
                seed_prompts = seed_prompt_generation(model, file_path, numbers, mode='single')
                intermidiate_answers = retrieve_answer(model, seed_prompts, file_path, top_k, top_p, running, mode='single', second_questions=seed_prompts)
                second_prompts = multi_prompts_generation(model, file_path, numbers, seed_prompts, intermidiate_answers, running, mode, literary)
                dialogue = conversation_concat(seed_prompts, intermidiate_answers, numbers, running, mode, second_prompts)
                # dialogue在RAG中，和文档一起作为参考资料，抽取第二句问题输入才能获得最好的RAG效果
                rag_content = retrieve_answer(model, dialogue, file_path, top_k, top_p, running, mode, literary, second_questions=second_prompts)        
                llm_content = llm_answer(model, dialogue, file_path, top_k, top_p, running, mode, literary)
                post_processing(seed_prompts, rag_content, llm_content, mode, data_type, output_directory, dialogue)
        
        elif args.file_name != file_name:
            continue
        else:
            raise ValueError('The file name you choose is not in the directory')
if __name__ == '__main__':
    main()

    


    


    

    
    