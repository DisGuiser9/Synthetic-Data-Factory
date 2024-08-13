import os 
import json
import chromadb
from pathlib import Path
from typing import List, Dict, Iterator
import matplotlib.pyplot as plt
import re
import umap
import numpy as np
import requests
from datetime import datetime
from tqdm import tqdm
from IPython.display import display, HTML

from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.vectorstores import Chroma
from langchain_community.document_transformers import LongContextReorder
from langchain_community.retrievers import BM25Retriever, ElasticSearchBM25Retriever
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.output_parsers import StrOutputParser

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


class MyParser(BaseBlobParser):
    """A simple parser that creates a document from each line."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Parse a blob into a document line by line."""
        line_number = 0
        with blob.as_bytes_io() as f:
            for line in f:
                line_number += 1
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": blob.source},
                )

def load_file_to_text(file_path) -> List:
    """
    提取纯文本
    """
    file_path = Path(file_path)
    format_name = os.path.splitext(file_path)[1]
    
    if format_name == '.json' or format_name == '.jsonl' or format_name == '.txt':
        loader = Blob.from_path(file_path)
        parser = MyParser()
        output = parser.lazy_parse(loader)
        try:
            while True:
                documents_text = json.loads(next(output).page_content).get('text')
        except StopIteration:
            pass    
    elif format_name == '.pdf':
        loader = PyPDFLoader(file_path=file_path)
        documents_text = [p.extract_text().strip() for p in loader.pages]
            
        # 去除空行
        documents_text = [text for text in documents_text if text]
    else:
        raise ValueError("Unsupported file format: {file_path.suffix}")
    
    return documents_text

def get_files_in_directory(directory, return_paths=False):
    """
    获取指定目录下的所有JSONL文件。
    如果return_paths为True，则返回文件路径；否则，返回文件名。
    """
    files = [file for file in os.listdir(directory) if file.endswith('.jsonl') and os.path.isfile(os.path.join(directory, file))]
    return [os.path.join(directory, file) if return_paths else file for file in files]

def local_ollama_models():
    url = "http://127.0.0.1:11434/api/tags"  
    models_list = []
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json()
            for items in models.items():
                for item in items[1]:
                    model_name = item.get("name")
                    models_list.append(model_name)
        else:
            print(f"请求失败，状态码：{response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"请求错误：{e}")
    return models_list

def get_ollama_model(model):
    return model

def get_collection_name(selected_file):
    """根据所选文件路径，返回文件名"""
    model = Ollama(model="qwen2:7b")
    template = """
                你是一个优秀的翻译，翻译{name}，你只需要给出一个文件名字的翻译，保证翻译准确，不需要任何额外解释，中英一一对应，{query}。
                """
    prompt = PromptTemplate.from_template(template=template)
    collection_name = os.path.splitext(selected_file)[0]
    chain = prompt | model
    collection_name = chain.invoke({"query": "Give the English translation of the name?", "name": collection_name})
    collection_name = collection_name.replace('"',"")
    cleaned_name = re.sub(r"[^a-zA-Z0-9._-]", "_", collection_name)
    
    return f"{cleaned_name[:60]}"  # 只取前60个字符

def get_file_name(file_path, directory='/data/share9/XPT/dataset/dataset_8/book/data'):
    if type(file_path) == list:
        file_path = ''
    file_name = os.path.join(directory,file_path)
    return file_name

def load_chroma(filename, collection_name, embedding_function, langcode='zh'):

    # Read the text from the file
    texts = load_file_to_text(filename)

    # Chunk the texts based on the language code
    chunks = _chunk_texts(texts, langcode)

    # Create a new ChromaDB client
    chroma_client = chromadb.Client()

    # Create a new collection with the specified name and embedding function
    chroma_collection = chroma_client.create_collection(name=collection_name,
                                                        embedding_function=embedding_function)

    # Generate IDs for the chunks
    ids = [str(i) for i in range(len(chunks))]

    # Add the chunks to the collection
    chroma_collection.add(ids=ids, documents=chunks)

    return chroma_collection

def word_wrap(string, n_chars=72):
    """
    在指定字符数后的下一个空格处换行字符串.
    Wrap the string at the next space after a specified number of characters

    Args:
        string: The input string
        n_chars: The maximum number of characters before wrapping

    Returns:
        The wrapped string
    """
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0])+1:], n_chars)


def get_embeddings_function(langcode: str):
    if langcode == 'zh':
        embedding_function = FastEmbedEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        # embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    elif langcode == 'en':
        embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    else:
        print("Unsupported language code:", langcode)
    return embedding_function


def _chunk_texts(texts, langcode="zh"):
    assert langcode in ['zh', 'en'], "langcode must be 'zh' or 'en'"

    # 1.用\n\n拼接所有文本；2.字符级分割，分割成多个块(chunk)；3. 对每一个块分割成token
    if langcode == 'zh':
        # 中文句子分割
        def chinese_sentence_segmentation(text):
            # 使用正则表达式匹配中文句子的分割符号，将中文分句
            sentences = re.split(r'[，。！？]', text)
            return [s.strip() for s in sentences if s.strip()]
        
        character_split_texts = []
        for text in texts:
            # 每个段落逐个分割句子
            character_split_texts += chinese_sentence_segmentation(text)
    else:
        # 英文句子分割
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""], # 英文
            # separators=["\n\n", "\n", "。", "，", ""], # 中文
            chunk_size=1000,
            chunk_overlap=0
        )
        character_split_texts = character_splitter.split_text('\n\n'.join(texts))

    # Split each character-based chunk into tokens
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, 
                                                           model_name='/home/mth/RAG-Align/all-mpnet-base-v2', 
                                                           tokens_per_chunk=256)

    token_split_texts = []
    if langcode == "zh":
        # 中文分句后不再进一步分割，保留完整语义信息，方便嵌入查询
        token_split_texts = character_split_texts
    else:
        # 英文分句后继续分割token（实际上这一步分割得很少）
        for text in character_split_texts:
            token_split_texts += token_splitter.split_text(text)

    return token_split_texts


def augment_multiple_query(model, query, numbers):
# 扩展query，扩大数据集涉及区域
    extended_queries = []
    template = """
    为模型提供指令，告诉它扮演一个优秀的外贸研究助理。
    指令包括基于提供的查询生成最多{numbers}个相关的额外查询问题。
    提示模型生成的问题应该简短，不包含复合句，并涵盖主题的不同方面。
    要确保问题是完整的，并且与原始查询相关联。
    输出格式为每行一个问题，不要对问题编号，输出中文。
    Question: {query}
    """
    prompt = PromptTemplate.from_template(template)
    llm = Ollama(model=model)
    chain = prompt | llm | StrOutputParser()
    extended_query = chain.invoke({"query": query, "numbers": numbers}) 
    # extended_queries = extended_query.strip().split("\n")
    for line in extended_query.split('\n'):
        question = line.strip()
        if question:
            extended_queries.append(question)
    return extended_queries


def prompt_generation(model, documents, numbers):
    with open(documents, "r", encoding='utf-8') as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    for data in data:
        text_string = ""
        text_string += data["text"]
    file_length = len(text_string)
    full_file_name = os.path.split(documents)[1]
    file_name, _ = os.path.splitext(full_file_name)

    _chunk_size = file_length // (numbers*10)
    _chunk_overlap = _chunk_size // 10

    llm = Ollama(model=model)
    embedding_function = get_embeddings_function("zh")
    generated_prompts = []
    template = """
                请扮演一位优秀的外贸研究助理。我将提供与外贸领域相关的文本，请你按照以下步骤操作：
                1.你必须参考我给你的检索内容{context}，这些内容已按照与文本主旨相关性进行了降序排列，越靠前的主题越相关。
                2.根据排序文本概括这段文本内容，并根据此生成{number}个相关问题。这些问题的答案必须存在于文本中，并且与原始文本紧密相关。
                3.确保每个问题都是完整的句子，且两个问题不相关；只输出问题本身，不需要提供答案、分析或总结。
                4.生成的问题应该简短明了，避免使用复合句，禁止输出无意义或意义不明的问题。
                注意：禁止输出无关问题的内容!输出中文问题即可，问题长度必须在10-30字。
                特别注意：禁止对问题进行编号!！
                例如，一个合格的问题可能是：“命名商品的方法主要有哪些？” ，请仅返回问题列表。
                """
    PROMPT = PromptTemplate.from_template(template=template)
    searching_blob = Blob.from_path(documents)
    parser = MyParser()
    output = parser.lazy_parse(searching_blob)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max(_chunk_size,3000),chunk_overlap=max(_chunk_overlap,300), add_start_index=True,
                                                    length_function=len,is_separator_regex=False)
    output = text_splitter.split_documents(output)
    
    vector_db = Chroma.from_documents(output, embedding=embedding_function)   #创建向量数据库
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": max(numbers,10), "lambda_mult": 0.3})

    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    docs = compression_retriever.invoke(f"关于{file_name}，你可以告诉我什么？请返回更多样全面的信息。")
    
    reordering = LongContextReorder()
    chain = create_stuff_documents_chain(llm, PROMPT) | StrOutputParser()
    reordered_docs = reordering.transform_documents(docs)
    result = chain.invoke({"context": reordered_docs, "number": numbers, "query": "请按要求生成提问，不要给问题标序号！"}) 
    for line in result.split('\n'):
        question = line.strip() + '<eos>'
        if len(question) >= 10:
            generated_prompts.append(question)
    print(generated_prompts[0])
    return generated_prompts


def retrieve_result(model, extended_queries, documents, _top_k, _top_p):
    llm = Ollama(model=model, top_k=_top_k, top_p=_top_p)
    embedding_function = get_embeddings_function("zh")
    template = """
                首先，请严格按照提供的检索内容和要求输出文本，这些内容已按照与查询的相关性进行了降序排列。请按以下步骤和要求完成任务：
                1. 分析：请分析哪些检索内容与我提出的问题最为相关。
                2. 回答：严格按照{context}，先结合你的分析结果组织语言，再详细且专业地回答{query}。
                请在回答时必须遵循以下原则：
                1. 回答必须在0-300字之间，保持客观，符合人类偏好逻辑和语气，只需要输出与问题有关的回答，不要输出任何额外的内容或无关的举例！
                2. 回答时禁止输出“根据题目描述”、“根据书本”等类似文本，回答时保持原来的交流状态和逻辑，直接输出回答，而不是按检索后逻辑回答问题！！
                2. 禁止回答中包含额外的内容或无关的举例，禁止输出一切你的判断和总结！如果你有不知道的请严格参考所给内容，禁止胡言乱语重复输出！
                3. 禁止输出令人困惑或具歧义的文本，尽可能少地依赖你的先验知识，但必须保持语言能力，保持输出文本的质量，必要时可以罗列排序文本！
                4. 禁止输出任何对回答的评价，禁止总结回答内容，禁止总结回答是否遵循提示词要求或上下文，禁止用总起句介绍回答是否遵循或提示词上下文！！              
               """
    RETRIEVAL_PROMPT = PromptTemplate.from_template(template=template)
    extended_queries = [query.strip() for query in extended_queries.split('<eos>')]
    full_file_name = os.path.split(documents)[1]
    file_name, _ = os.path.splitext(full_file_name)

    # 加载整个文档
    searching_blob = Blob.from_path(documents)
    parser = MyParser()
    output = parser.lazy_parse(searching_blob)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200, add_start_index=True,
                                                   length_function=len,is_separator_regex=False)
    output = text_splitter.split_documents(output)

    # Sparse and Dense retrieval
    vector_db = Chroma.from_documents(output, embedding=embedding_function)   #创建向量数据库
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.25})
    bm25_retriever = BM25Retriever.from_documents(output)
    bm25_retriever.k = 5

    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, compression_retriever], weights=[0.5, 0.5])
    
    chain = create_stuff_documents_chain(llm, RETRIEVAL_PROMPT) | StrOutputParser()

    results, context = "", []
    for query in extended_queries:
        if len(query) <= 5:
            continue
        query = query.replace('[','').replace(']','').replace("'",'').replace(",",'').replace('"','')
        query = query.strip()
        docs = ensemble_retriever.invoke(f"关于{file_name}，请提供所有与提问{query}相关的详细信息和回答。")
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(docs)
        result = chain.invoke({"context": reordered_docs, "query": query}) 
        results += result + '<eos>' + '\n\n'
        context.append(reordered_docs) 
    return results, context


def chat_with_llm(model, extended_queries, _top_k, _top_p):
    llm = Ollama(model=model, top_k=_top_k, top_p=_top_p)
    results = ""
    template = """
                你是一个优秀的外贸研究助理，你现在需要回答我所给的问题，回答需要详细且专业；
                必须在0-300字，语气保持客观，符合人类偏好逻辑和语气，以中文回答，不要给回答标号。
                {query}"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    extended_queries = [query.strip() for query in extended_queries.split('<eos>')]

    for query in extended_queries:
        if len(query) <= 5:
            continue
        query = query.replace('[','').replace(']','').replace("'",'').replace(",",'').replace('"','')
        query = query.strip()
        result = chain.invoke({"query": query}) 
        results += result + '<eos>' + '\n\n'
    return results


def project_umap_embeddings(embeddings, umap_transform):
    """
    用 UMAP 将高维的embeddings矩阵投影到2维，方便可视化。
    Project embeddings to a lower dimensional space using UMAP transformation.
    
    Args:
        embeddings (array-like): The original embeddings to be transformed.
        umap_transform (umap.UMAP): The UMAP transformation model.
    
    Returns:
        array-like: The UMAP-transformed embeddings.
    """
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings


def embeddings_plot(original_query, projected_dataset_embeddings, projected_retrieved_embeddings, projected_original_query_embedding, projected_augmented_query_embedding):
    plot = plt.figure()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置全局字体为微软雅黑，显示中文
    plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray')
    plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')
    plt.scatter(projected_original_query_embedding[:, 0], projected_original_query_embedding[:, 1], s=150, marker='X', color='r')
    plt.scatter(projected_augmented_query_embedding[:, 0], projected_augmented_query_embedding[:, 1], s=150, marker='X', color='orange')

    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'{original_query}')
    plt.axis('off')
    return plot

def post_processing_for_dpo(questions, rag_result, llm_result):
    questions = [question.strip().replace('\n', '') for question in questions.split('<eos>') if len(question) > 5]
    rag_answers = [answer.strip().replace('\n', '') for answer in rag_result.split('<eos>') if len(answer) > 5]
    llm_answers = [answer.strip().replace('\n', '') for answer in llm_result.split('<eos>') if len(answer) > 5]

    print(len(questions), len(rag_answers), len(llm_answers))
    if len(questions) != len(rag_answers) or len(questions) != len(llm_answers):
        raise ValueError("The number of questions does not match the number of answers.")

    # data = []
    json_result = ""
    for i in range(len(questions)):
        entry = {
            "question": [
                {
                    "from": "human",
                    "value": questions[i].strip().replace('[','').replace(']','').replace("'",'').replace(",",'').replace('"','').replace(" ",'')
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": rag_answers[i].strip().replace('<eos>','').replace('[','').replace(']','').replace("'",'').replace(",",'').replace('"','').replace(" ",'')
            },
            "rejected": {
                "from": "gpt",
                "value": llm_answers[i].strip().replace('<eos>','').replace('[','').replace(']','').replace("'",'').replace(",",'').replace('"','').replace(" ",'')
            }
        }
        json_result += json.dumps(entry, ensure_ascii=False)+ '\n'
        
    # data.append(json_result)    
    return json_result

def dump_into_json(json_result):
    now = datetime.now()
    current_date = now.date()
    formatted_date = current_date.strftime("%Y-%m-%d")
    with open(f'/home/mth/RAG-Align/output_data/dpo-data_{formatted_date}.jsonl', 'a', encoding='utf-8') as file:
        for line in [json_result]:
            file.write(line) 

def visualize_embeddings(original_query, queries, results, text, collection_name, file_path):
    # character_splitter = RecursiveCharacterTextSplitter(#separators=["\n\n", "\n", ". ", " ", ""],英文文档的分割符示例
    #                                                     separators=["\n\n", "\n", "。", "，", "、", "；", ""],
    #                                                     chunk_size=1000,chunk_overlap=0)
    # character_split_texts = character_splitter.split_text('\n\n'.join(documents_text))
    # token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0,tokens_per_chunk=128)
    # token_split_texts = []
    # # for text in preprocessed_texts:
    # for text in character_split_texts:
    #     token_split_texts += token_splitter.split_text(text)

    embedding_function = get_embeddings_function("zh")
    chroma_client = chromadb.Client()
    collection_list_file_path = "/home/mth/RAG-Align/collection_list.jsonl"
    documents_text = load_file_to_text(text)
    chunks = _chunk_texts(documents_text, langcode="zh")

    if os.path.exists(collection_list_file_path):
        with open(collection_list_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = {}
    # 检查 filename 是否已经在文件中
    if file_path not in data:
        # 如果不存在，则根据 collection_name 创建集合，并写入文件
        chroma_collection = chroma_client.create_collection(name=collection_name,
                                                            embedding_function=embedding_function)
        data[file_path] = collection_name
        with open(collection_list_file_path, "a", encoding="utf-8") as file:
            file.write(json.dump(data, file, ensure_ascii=False) + "\n")
    else:
        chroma_collection = chroma_client.get_collection(name=collection_name,
                                                         embedding_function=embedding_function)
    ids = [str(i) for i in range(len(chunks))]
    chroma_collection.add(ids=ids, documents=chunks)
    
    collection_embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
    original_query_embedding = embedding_function.embed_query(original_query)
    queries_embeddings = embedding_function.embed_documents(queries)
    results_embeddings = embedding_function.embed_documents(results)

    umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(collection_embeddings)
    projected_dataset_embeddings = project_umap_embeddings(collection_embeddings, umap_transform)
    projected_original_query_embedding = project_umap_embeddings(original_query_embedding, umap_transform)
    projected_augmented_query_embedding = project_umap_embeddings(queries_embeddings, umap_transform)
    projected_retrieved_embeddings = project_umap_embeddings(results_embeddings, umap_transform)
    plot = embeddings_plot(projected_dataset_embeddings,projected_original_query_embedding, 
                           projected_augmented_query_embedding, projected_retrieved_embeddings)
    return plot

def decorate(x):
    # 复制输入的数据框
    df = x.copy()

    # 定义一个函数来为特定的列添加样式
    df[['Matrix']] = 'color: orange'
    # df[['Matrix']] = 'width: 100px'

    # 显示或返回样式化后的 DataFrame
    return df

