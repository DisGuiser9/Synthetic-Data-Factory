from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
import math
import jieba
import numpy as np
import pandas as pd
import requests
import json
import ast
import logging
from datetime import datetime

from collections import Counter
from openai import OpenAI
import language_tool_python
from rouge_score import rouge_scorer


# path = '/share149/huggingface/models--meta-llama--Meta-Llama-3-70B/snapshots/b4d08b7db49d488da3ac49adf25a6b9ac01ae338'
tool = language_tool_python.LanguageTool('zh-CN')
path = '/share148/huggingface/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6'
model = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

class AnswerEvaluator:
    def __init__(self):

        self.device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")

        self.gpt_preference1 = []
        self.gpt_preference2 = []

        self.grammar_scores1 = []
        self.grammar_scores2 = []
        
        self.ttr_scores1 = []
        self.ttr_scores2 = []
        
        self.keyword_matches1 = []
        self.keyword_matches2 = []
        
        self.lengths_distributions1 = []
        self.lengths_distributions2 = []
        
        self.freq_similarities1 = []
        self.freq_similarities2 = []
        
        self.perplexities1 = []
        self.perplexities2 = []

        self.distinct1_scores1 = []
        self.distinct1_scores2 = []
        self.distinct2_scores1 = []
        self.distinct2_scores2 = []

    def distinct_n_score(self, text, n):
        words = text.split()
        if len(words) < n:
            return 0.0
        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        unique_ngrams = set(ngrams)
        return len(unique_ngrams) / len(ngrams)
    
    def round_results(self, results, decimal_places=2):
        def round_value(value):
            if isinstance(value, list):
                return [round(v, decimal_places) if isinstance(v, (int, float)) else v for v in value]
            return round(value, decimal_places)

        rounded_results = {}
        for key, value in results.items():
            rounded_results[key] = [round_value(sublist) for sublist in value]
        return rounded_results


    def gpt_preference(self, question, rag_answer, llm_answer):
        def call(query: str):
            client = OpenAI(
                            api_key = "sk-h5ujoj44PqVvQF5P3oH7oMwEGhpvZOqCiH7n2Fma3Od2Fxyz",
                            base_url = "https://api.moonshot.cn/v1",
                            )
            completion = client.chat.completions.create(
                        model = "moonshot-v1-8k",
                        messages = [
                            {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
                            {"role": "user", "content": query}
                        ],
                        temperature = 0.3,
                            )
            content = completion.choices[0].message.content
            return content
        prompt = f"""请比较以下两个回答，选择更符合人类偏好且更好地回答问题的回答，并将其判为1，另一个判为0，最后将结果返回为一个数组。评判标准如下：
                    1.前后逻辑是否一致,是否围绕问题{question}展开回答；
                    2.对于问题的回答是否有理有据、摆事实讲道理。
                    如果回答符合以上标准，应加分；如果回答存在胡言乱语的幻觉现象、前后逻辑看似正确但泛泛而谈、没有实际依据、说了很多但没有回答问题，应扣分。
                    请注意：其中一个回答是经过检索增强生成的，一个回答是没有经过检索增强，直接生成的，所以可能会在某个回答中有一些专有名词等情况，这并不是幻觉，应该加分而并非扣分；
                    请根据这些标准进行打分。如果你认为第一个回答更符合人类偏好且更好地回答了问题，输出[1,0]，反之输出[0,1]。不要输出你的判断和任何的分析或其他内容，只需要返回一个数组。
                    """
        query = f"{prompt} {rag_answer} {llm_answer}"
        return call(query)
        

    # 计算语法错误数量
    def grammar_errors(self, text):
        matches = tool.check(text)
        return len(matches)
        
    # 计算词汇丰富度（TTR）
    def type_token_ratio(self, text):
        tokens = jieba.lcut(text)
        return len(set(tokens)) / len(tokens)

    # 计算关键词匹配
    def keyword_matching(self, question, answer):
        question_tokens = set(jieba.lcut(question))
        answer_tokens = set(jieba.lcut(answer))
        common_tokens = question_tokens & answer_tokens
        return len(common_tokens) / len(question_tokens)

    # 计算句子长度分布
    def sentence_length_distribution(self, text):
        sentences = text.split('。')
        lengths = [len(jieba.lcut(sentence)) for sentence in sentences if sentence]
        return lengths

    # 计算词频分布
    def word_frequency(self, text):
        tokens = jieba.lcut(text)
        return Counter(tokens)

    #困惑度
    def calculate_perplexity(self, text):
        # 对文本进行编码
        encodings = tokenizer(text, return_tensors='pt').to(self.device)
        input_ids = encodings.input_ids
        
        # 禁用梯度计算
        with torch.no_grad():
            # 计算模型输出
            model.to(self.device)
            outputs = model(input_ids, labels=input_ids)
            
            # 获取交叉熵损失
            loss = outputs.loss
            perplexity = math.exp(loss.item())
        
        return perplexity

    def freq_similarity(self, freq1, freq2):
        # 计算两个词频分布的余弦相似度
        common_keys = set(freq1.keys()).union(set(freq2.keys()))
        vec1 = np.array([freq1.get(key, 0) for key in common_keys])
        vec2 = np.array([freq2.get(key, 0) for key in common_keys])
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def evaluate(self, question, rag_answer, llm_answer):

        for i in range(len(question)):
            # if len(question) <= 5 or len(rag_answer[i]) <= 5 or len(llm_answer[i]) <= 5:
            #     continue

            # self.grammar_scores1.append(self.grammar_errors(rag_answer[i]))
            # self.grammar_scores2.append(self.grammar_errors(llm_answer[i]))

            # self.ttr_scores1.append(self.type_token_ratio(rag_answer[i]))
            # self.ttr_scores2.append(self.type_token_ratio(llm_answer[i]))

            # self.keyword_matches1.append(self.keyword_matching(question[i], rag_answer[i]))
            # self.keyword_matches2.append(self.keyword_matching(question[i], llm_answer[i]))

            # self.lengths_distributions1.append(self.sentence_length_distribution(rag_answer[i]))
            # self.lengths_distributions2.append(self.sentence_length_distribution(llm_answer[i]))

            # freq_q = self.word_frequency(question[i])
            # freq_a1 = self.word_frequency(rag_answer[i])
            # freq_a2 = self.word_frequency(llm_answer[i])
            # self.freq_similarities1.append(self.freq_similarity(freq_q, freq_a1))
            # self.freq_similarities2.append(self.freq_similarity(freq_q, freq_a2))

            # self.distinct1_scores1.append(self.distinct_n_score(rag_answer[i], 1))
            # self.distinct1_scores2.append(self.distinct_n_score(llm_answer[i], 1))
            # self.distinct2_scores1.append(self.distinct_n_score(rag_answer[i], 2))
            # self.distinct2_scores2.append(self.distinct_n_score(llm_answer[i], 2))

            self.perplexities1.append(self.calculate_perplexity(rag_answer[i]))
            self.perplexities2.append(self.calculate_perplexity(llm_answer[i]))

            preference_list = ast.literal_eval(self.gpt_preference(question[i],rag_answer[i], llm_answer[i]))
            self.gpt_preference1.append(preference_list[0])
            self.gpt_preference2.append(preference_list[1])

    def calculate(self):
        results = {
                # "grammar_scores": [self.grammar_scores1, self.grammar_scores2],
                # "ttr_scores 词汇丰富度": [self.ttr_scores1, self.ttr_scores2],
                # "keyword_matches 关键词匹配": [self.keyword_matches1, self.keyword_matches2],
                # "lengths_distributions": [self.lengths_distributions1, self.lengths_distributions2],
                # "freq_similarities 词频分布余弦相似度": [self.freq_similarities1, self.freq_similarities2],
                "perplexities 困惑度": [self.perplexities1, self.perplexities2],
                # "distinct1_scores": [self.distinct1_scores1, self.distinct1_scores2],
                # "distinct2_scores": [self.distinct2_scores1, self.distinct2_scores2]
                "LLM选择": [self.gpt_preference1, self.gpt_preference2]
                }
        rounded_results = self.round_results(results)
        values = [value for value in rounded_results.values()]
        rag_scores = []
        llm_scores = []

        for pair in values:
            # 提取rag_score
            rag_score = pair[0]
            rag_scores.append(rag_score)

            # 提取llm_score
            llm_score = pair[1]
            llm_scores.append(llm_score)

        output  = pd.DataFrame({
        "Metric": rounded_results.keys(),
        "RAG_Score": rag_scores,
        "LLM_Socre": llm_scores
        })
        print(output)
        return output

def decorate(df):
    # 定义 CSS 样式
    css_style = """
        .custom-table {
            font-size: 14px;
            border-collapse: collapse;
        }
        .custom-table td, .custom-table th {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .custom-table tr:nth-child(even){background-color: #f2f2f2;}
        .custom-table tr:hover {background-color: #ddd;}
        .custom-table th {
            padding-top: 12px;
            padding-bottom: 12px;
            background-color: #4CAF50;
            color: white;
        }
    """
    # 应用样式
    styled_df = df.style.set_table_styles([
        {'selector': 'table', 'props': [('border-collapse', 'collapse')]}
    ]).set_properties(**{'font-size': '14px'}).set_table_attributes('class="custom-table"')
    return styled_df

def setup_logging():
    log_file_path = os.path.join('/home/mth/RAG-Align', 'evaluation_log.txt')
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def evaluation(questions, rag_answers, llm_answers):
    # setup_logging()
    now = datetime.now()
    current_date = now.date()
    formatted_date = current_date.strftime("%Y-%m-%d")

    if questions is None:
        results = pd.DataFrame(
            {
            "Matrix": ["keyword_matches", "freq_similarities", "Perplexity"],
            'RAG_Score': [0, 0, 0],
            'LLM_Score': [0, 0, 0]
            }
            )
    evaluator = AnswerEvaluator()
    if type(questions) is str:
        questions = [question.replace('\n', '').strip() for question in questions.split('<eos>') if len(question) >=5]
        n = len(questions)
        rag_answers = [answer.replace('\n', '').strip() for answer in rag_answers.split('<eos>') if len(answer) >=5]
        llm_answers = [answer.replace('\n', '').strip() for answer in llm_answers.split('<eos>') if len(answer) >=5]
        rag_answers = rag_answers[:n]
        llm_answers = llm_answers[:n]
        print(f"评测完成，评测数据长度{len(questions), len(rag_answers), len(llm_answers)}")

    evaluator.evaluate(questions, rag_answers, llm_answers)
    results = evaluator.calculate()
    results.to_csv(f'/home/mth/RAG-Align/statistics_data/evaluation_results_{formatted_date}.csv',mode='a', index=False)
    return results

# # 示例文本
# questions = ['如何根据国际市场需求命名商品？', '在外贸中，商品名称需要遵循哪些规定或标准？']
# answers1 = ["1. 根据国际市场需求命名商品时,应考虑目标市场的文化、语言习惯和消费心理。使用通俗易懂且具有吸引力的名称，如以产地（青岛啤酒）、主要用途（护手霜）或原材料（棉布）等命名，能增强消费者识别度和购买意愿。", "2. 在外贸中，商品名称的规定需明确具体，适应商品特点，翻译准确，与国际通用名称一致，避免含糊不清。遵循《国际贸易术语解释通则》等相关国际标准，确保名称的规范性和一致性，便于全球贸易伙伴理解和接受。"]
# answers2 = ["1. 根据国际市场需求命名商品时，应考虑目标市场的语言习惯、文化背景和产品特性。使用易于理解且具有吸引力的词汇，避免直译导致的文化误解。", "2. 外贸商品名称需遵守国际贸易规则，如不侵犯他人知识产权，避免使用误导性或通用术语。同时，符合进口国的商品标识法律法规，确保合规性和市场准入。"]

# evaluator = AnswerEvaluator()

# # 进行评估
# evaluator.evaluate(questions, answers1, answers2)

# # 获取结果
# results = evaluator.calaculate()
# print(results)

