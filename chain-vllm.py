from langchain_community.llms.vllm import VLLM
from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate
def fun():
    # llm = VLLM(
    #     model="/data/share9/huggingface/models--Qwen--Qwen1.5-32B-Chat/snapshots/0997b012af6ddd5465d40465a8415535b2f06cfc",
    #     tensor_parallel_size=2,
    #     # trust_remote_code=True,  # mandatory for hf models
    # )

    # model = VLLM(
    #     model="/data/share9/huggingface/models--Qwen--Qwen1.5-32B-Chat/snapshots/0997b012af6ddd5465d40465a8415535b2f06cfc",
    #     max_new_tokens=512,
    #     top_k=10,
    #     top_p=0.8,
    #     temperature=0.7,
    #     tensor_parallel_size=2,
    #     # repetition_penalty=1.05,
    #     debug=True
    # )
    """
    "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    """
    raw_prompt = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>{user}\n{content}<|im_end|>\n<|im_start|>assistant\n"""
    prompt = PromptTemplate.from_template(raw_prompt)




    full_template = """{introduction}
    
    {example}
    
    {start}"""
    full_prompt = PromptTemplate.from_template(full_template)
    pre_prompt = PromptTemplate.from_template('pre-ctx: {subcontent}')

    mid_prompt = PromptTemplate.from_template('mid-ctx: {midcontent}')

    end_prompt = PromptTemplate.from_template('end-ctx: {endcontent}')

    input_prompts = [
        ("introduction", pre_prompt),
        ("example", mid_prompt),
        ("start", end_prompt),
    ]
    print(full_prompt)
    pip_prompt = PipelinePromptTemplate(
        final_prompt=full_prompt, pipeline_prompts=input_prompts
    )


    print(pip_prompt.partial())
    start = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    end = '<|im_end|>\n<|im_start|>assistant\n'
    spt = PromptTemplate.from_template(start)
    ept = PromptTemplate.from_template(end)
    wrapper_prompt = PipelinePromptTemplate(
        final_prompt=PromptTemplate.from_template("""{start}{inner}{end}"""),
        pipeline_prompts=[
            ("start", spt),
            ("inner", pip_prompt),
            ("end", ept)
        ]
    )
    a = wrapper_prompt.invoke({
        'subcontent':'sss',
        'midcontent':'mmm',
        'endcontent':'eee',
    })
    print(a)

    pass
    def preprompt():
        pre_prompt = PromptTemplate.from_template('sub-ctx: {subcontent}')
        # content = pre_prompt.invoke(
        #     {
        #         'subcontent': text
        #     }
        # )
        res = {
            'user': 'user',
            'raw_content': f'{pre_prompt.pretty_repr()}'
        }
        return res
    # content = preprompt()
    # msg = prompt.invoke({
    #     "user": "user",
    #     "content": content,
    # })
    # msg = prompt.invoke(content)
    from transformers import AutoTokenizer
    model="/data/share9/huggingface/models--Qwen--Qwen1.5-32B-Chat/snapshots/0997b012af6ddd5465d40465a8415535b2f06cfc"
    def tokenizer_add_chat_template(prompt, model):
        pure_prompt = prompt.pretty_repr()
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model)
        msg = {
            'role': 'user',
            'content': pure_prompt,
        }
        prompt = tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
        # print(prompt)
        # prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{pure_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return PromptTemplate.from_template(prompt)
    def add_chat_template(prompt):
        pure_prompt = prompt.pretty_repr()
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{pure_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return PromptTemplate.from_template(prompt)

    pt = add_chat_template(pip_prompt)

    ptt = tokenizer_add_chat_template(pip_prompt, model)




    print(pt)
    print(ptt)
    a = ''



    while a!='quit':
        a = input()
        raw_msg = {
            'user': 'user',
            'content': f'{a}',
        }
        msg = prompt.invoke(raw_msg)
        res = model.invoke(msg)
        print(res)

def fun():
    # from langchain_chroma import Chroma
    # from langchain_huggingface import HuggingFaceEmbeddings

    # Get embeddings.
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    from utils import *

    texts = [
        "Basquetball is a great sport.",
        "Fly me to the moon is one of my favourite songs.",
        "The Celtics are my favourite team.",
        "This is a document about the Boston Celtics",
        "I simply love going to the movies",
        "The Boston Celtics won the game by 20 points",
        "This is just a random text.",
        "Elden Ring is one of the best games in the last 15 years.",
        "L. Kornet is one of the best Celtics players.",
        "Larry Bird was an iconic NBA player.",
    ]
    text = ''.join(texts)
    embedding_function = get_embeddings_function("zh")
    generated_prompts = []
    print("********开始单次提问*********")

    # searching_blob = Blob.from_path(documents)
    # parser = MyParser()
    # output = parser.lazy_parse(searching_blob)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=30,
                                                   chunk_overlap=10, add_start_index=True,
                                                   length_function=len, is_separator_regex=False)
    output = text_splitter.split_documents(text)

    vector_db = Chroma.from_documents(output, embedding=embedding_function)
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 10})

    compressor = FlashrankRerank(top_n=10)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    docs = compression_retriever.invoke(
        f"关于外贸，你可以告诉我什么相关知识？请返回更全面的有关信息，并过滤所有无关的、有特定场景或任何人名的信息。")
    reordering = LongContextReorder()

    PROMPT_TEMPLATE, _, _ = prompt_choices(mode='single')

    # query = "What can you tell me about the Celtics?"

    chain = create_stuff_documents_chain(llm, prompt)
    response = chain.invoke({"context": reordered_docs, "query": query})
    print(response)
