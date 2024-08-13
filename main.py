from utils import *
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for the Synthetic Dataset')
    
    parser.add_argument('--ollama_model', type=str, 
                        default='qwen2:72b-instruct-fp16',
                        help='The model to be used')
    
    parser.add_argument('--top_k', type=int, 
                        default=40,
                        help='Reduces the probability of generating nonsense from Ollama model')
    
    parser.add_argument('--top_p', type=float, 
                        default=0.7,
                        help='Work with top_k')
    
    parser.add_argument('--numbers', type=int, 
                        default=5,
                        help='The number of augmented queries')
    
    parser.add_argument('--langcode', type=str, 
                        default='zh',
                        help='The language code of the dataset')
    
    parser.add_argument('--turns', type=str,
                        default='single_turn',
                        help='The number of turns in questions')
    
    args = parser.parse_args()
    return args
    
    
    
    