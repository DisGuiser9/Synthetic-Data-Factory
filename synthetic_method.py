from utils import *
def seed_prompt_generation(model, documents, numbers, mode: Optional[str] = "single",
                           prev_question: Optional[str] = None,
                           dialogues: Optional[str] = None, literary: Optional[str] = "博客"):
    with open(documents, "r", encoding='utf-8') as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    for data in data:
        text_string = ""
        text_string += data["text"]
    file_length = len(text_string)
    assert file_length > 99, 'file_length too short'

    full_file_name = os.path.split(documents)[1]
    file_name, _ = os.path.splitext(full_file_name)

    _chunk_size = file_length // (numbers * 10)
    _chunk_size = max(_chunk_size, 100)
    _chunk_overlap = _chunk_size // 10
    ref_doc_len = 30000 // _chunk_size


    # llm = Ollama(model=model)
    # llm = VLLM(
    #     model=model,
    #     max_new_tokens=512,
    #     top_k=10,
    #     top_p=0.8,
    #     temperature=0.7,
    # )
    llm = model
    embedding_function = get_embeddings_function("zh")
    generated_prompts = []
    print("********开始单次提问*********")

    searching_blob = Blob.from_path(documents)
    parser = MyParser()
    output = parser.lazy_parse(searching_blob)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=min(_chunk_size, 3000),
                                                   chunk_overlap=min(_chunk_overlap, 300), add_start_index=True,
                                                   length_function=len, is_separator_regex=False, )
    output = text_splitter.split_documents(output)

    vector_db = Chroma.from_documents(output, embedding=embedding_function)
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": max(numbers, 20), "fetch_k": 50})

    compressor = FlashrankRerank(top_n=ref_doc_len)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    docs = compression_retriever.invoke(
        f"关于{file_name}，你可以告诉我什么相关知识？请返回更全面的有关信息，并过滤所有无关的、有特定场景或任何人名的信息。")
    reordering = LongContextReorder()

    PROMPT_TEMPLATE, _, _ = prompt_choices(mode=mode)

    chain = create_stuff_documents_chain(llm, PROMPT_TEMPLATE) | StrOutputParser()
    reordered_docs = reordering.transform_documents(docs)

    input_variables_dict = {
        "single": {"file_name": file_name, "context": reordered_docs, "number": numbers},
        "stepback": {"file_name": file_name, "context": reordered_docs},
        "augment": {"file_name": file_name, "context": reordered_docs, "prompt": dialogues},
        "literary": {"file_name": file_name, "context": reordered_docs, "prompt": dialogues, "literary": literary}
    }
    input_variables = input_variables_dict.get(mode)
    input_variables["query"] = "请按要求生成提问"  #invoke function must have key "query"

    if mode == "stepback":
        for question in prev_question:
            input_variables["question"] = question
            result = chain.invoke(input_variables)
            generated_prompts.append(result)
    else:
        # print('PROMPT_TEMPLATE: ', PROMPT_TEMPLATE)
        # print('input_variables:', input_variables)
        result = chain.invoke(input_variables)
        print('synthetic query: \n', result)

        for line in result.split('\n'):
            question = line.strip() + '<eos>'
            if len(question) >= 10:
                generated_prompts.append(question)

    print(f"问题数量：{len(generated_prompts)}")
    # 确保长度是与numbers一样，让程序能运行，后期再删除多余数据
    while len(generated_prompts) < numbers:
        generated_prompts.append(f"回答一个关于{file_name}的问题")

    return generated_prompts[:numbers]

def retrieve_answer(model, extended_queries, documents, _top_k, _top_p, running='demo', mode: Optional[str] = "single",
                    literary: Optional[str] = "博客", second_questions: Optional[str] = ''):
    # llm = Ollama(model=model, top_k=_top_k, top_p=_top_p)
    # llm = VLLM(
    #     model=model,
    #     max_new_tokens=512,
    #     top_k=_top_k,
    #     top_p=_top_p,
    #     temperature=0.7,
    # )
    llm = model
    embedding_function = get_embeddings_function("zh")

    full_file_name = os.path.split(documents)[1]
    file_name, _ = os.path.splitext(full_file_name)

    print("********开始检索*********")
    searching_blob = Blob.from_path(documents)
    parser = MyParser()
    output = parser.lazy_parse(searching_blob)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, add_start_index=True,
                                                   length_function=len, is_separator_regex=False)
    output = text_splitter.split_documents(output)

    # Sparse and Dense retrieval
    vector_db = Chroma.from_documents(output, embedding=embedding_function)
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 30})
    bm25_retriever = BM25Retriever.from_documents(output)
    bm25_retriever.k = 5

    compressor = FlashrankRerank(top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, compression_retriever], weights=[0.5, 0.5])

    _, RETRIEVAL_PROMPT, _ = prompt_choices(mode)
    chain = create_stuff_documents_chain(llm, RETRIEVAL_PROMPT) | StrOutputParser()

    if isinstance(extended_queries, str):
        str_results = ""
        extended_queries = [string_processing(query) for query in extended_queries.split('<eos>') if
                            len(query.strip()) > 5]
    elif isinstance(extended_queries, list):
        if isinstance(extended_queries[0], list):
            extended_queries = extended_queries
        else:
            extended_queries = [query.replace('<eos>', '') for query in extended_queries if len(query.strip()) > 5]
        list_results = []

    n = len(extended_queries)

    for index in tqdm(range(n)):
        if mode == "single":
            docs = ensemble_retriever.invoke(
                f"关于{file_name}，请提供所有与{extended_queries[index]}相关的详细信息和回答。")
        else:
            docs = ensemble_retriever.invoke(
                f"关于{file_name}，请提供所有与{second_questions[index]}相关的详细信息和回答。")
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(docs)

        # input_variables_dict = {
        #     "single": {"file_name": file_name, "context": reordered_docs, "prompt": extended_queries[index]},
        #     "stepback": {"file_name": file_name, "context": reordered_docs, "prompt": extended_queries[index],
        #                  "question": second_questions[index]},
        #     "augment": {"file_name": file_name, "context": reordered_docs, "prompt": extended_queries[index],
        #                 "question": second_questions[index]},
        #     "literary": {"file_name": file_name, "context": reordered_docs, "literary": literary,
        #                  "prompt": extended_queries[index], "question": second_questions[index]}
        # }
        if mode == "single":
            input_variables = {"file_name": file_name, "context": reordered_docs, "prompt": extended_queries[index]}
        elif mode == "stepback":
            input_variables = {"file_name": file_name, "context": reordered_docs, "prompt": extended_queries[index],
                               "question": second_questions[index]}
        elif mode == "augment":
            input_variables = {
                "file_name": file_name, "context": reordered_docs, "prompt": extended_queries[index],
                "question": second_questions[index]}
        elif mode == "literary":
            input_variables = {"file_name": file_name, "context": reordered_docs, "literary": literary,
                               "prompt": extended_queries[index], "question": second_questions[index]}
        else:
            raise ValueError(f"mode {mode} not supported")

        # invoke function should have key words "query"
        input_variables["query"] = "请按要求回答问题"
        result = chain.invoke(input_variables)
        if running == "terminal":
            list_results.append(result)
        else:
            str_results += result + '<eos>' + '\n\n'

    return_results = list_results if running == "terminal" else str_results
    return return_results

def llm_answer(model, extended_queries, documents, _top_k, _top_p, running="demo", mode: Optional[str] = "single",
               literary: Optional[str] = "博客"):
    # llm = Ollama(model=model, top_k=_top_k, top_p=_top_p)
    # llm = VLLM(
    #     model=model,
    #     max_new_tokens=512,
    #     top_k=_top_k,
    #     top_p=_top_p,
    #     temperature=0.7,
    # )
    llm = model

    full_file_name = os.path.split(documents)[1]
    file_name, _ = os.path.splitext(full_file_name)

    str_results, list_results = "", []
    mode_dict = {"single": llm_template.single_turn_llm_template,
                 "stepback": llm_template.stepback_llm_template,
                 "augment": llm_template.augmented_prompt_llm_template,
                 "literary": llm_template.literary_prompt_llm_template}

    print("********开始LLM回答*********")
    if mode not in mode_dict:
        raise ValueError(f"Invalid mode: {mode}, Valid Key is {list(mode_dict.keys())}")

    _, _, LLM_PROMPT = prompt_choices(mode)
    chain = LLM_PROMPT | llm | StrOutputParser()
    if type(extended_queries) is str:
        extended_queries = [string_processing(query) for query in extended_queries.split('<eos>') if
                            len(query.strip()) > 5]

    for query in tqdm(extended_queries):
        input_variables_dict = {
            "single": {"file_name": file_name, "prompt": query},
            "stepback": {"file_name": file_name, "prompt": query},
            "augment": {"file_name": file_name, "prompt": query},
            "literary": {"file_name": file_name, "literary": literary, "prompt": query}
        }
        input_variables = input_variables_dict.get(mode)
        # invoke function should have key words "query"
        input_variables["query"] = "请按要求回答问题"
        result = chain.invoke(input_variables)
        if running == "terminal":
            list_results.append(result)
        else:
            str_results += result + '<eos>' + '\n\n'

    return_results = list_results if running == "terminal" else str_results
    return return_results

def conversation_concat(seed_questions, rag_inter_answers, numbers, running='demo', mode='single',
                        second_questions=None):
    if type(seed_questions) is str:
        seed_questions = [string_processing(inter_question) for inter_question in seed_questions.split('<eos>') if
                          len(inter_question) > 5]
        rag_answers = [string_processing(answer) for answer in rag_inter_answers.split('<eos>') if len(answer) > 5]
        if second_questions is not None:
            second_questions = [string_processing(question) for question in second_questions.split('<eos>') if
                                len(question) > 5]
    else:
        seed_questions = [question.replace("<eos>", "") for question in seed_questions]
        rag_answers = [answer.replace("<eos>", "") for answer in rag_inter_answers]
        if second_questions is not None:
            second_questions = [second_question.replace("<eos>", "") for second_question in second_questions]
            print(len(seed_questions), len(rag_answers), len(second_questions))

    json_result = ""
    list_result = []
    for i in range(numbers):
        if mode == 'single':
            entry = [
                {
                    "from": "human",
                    "value": seed_questions[i]
                },
                {
                    "from": "gpt",
                    "value": rag_answers[i]
                }
            ]
        elif mode == 'stepback':  #对话全部完成后，平凑在一起
            entry = [
                {
                    "from": "human",
                    "value": second_questions[i]  #后退式提问对应于第二个生成的问题
                },
                {
                    "from": "gpt",
                    "value": rag_answers[i]
                },
                {
                    "from": "human",
                    "value": seed_questions[i]
                }
            ]

        else:
            entry = [
                {
                    "from": "human",
                    "value": seed_questions[i]
                },
                {
                    "from": "gpt",
                    "value": rag_answers[i]
                },
                {
                    "from": "human",
                    "value": second_questions[i]  #增进提问对应于第二个问题的生成
                }
            ]

        if running == "terminal":
            list_result.append(entry)
        else:
            json_result += json.dumps(entry, ensure_ascii=False) + '<eos>'

    return_result = list_result if running == "terminal" else json_result
    return return_result

def multi_prompts_generation(model, documents, numbers, seed_questions, rag_inter_answers, running='demo',
                             mode: Optional[str] = "single", literary: Optional[str] = None):
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

    _chunk_size = file_length // (numbers * 10)
    _chunk_overlap = _chunk_size // 10

    # llm = Ollama(model=model)
    # llm = VLLM(
    #     model=model,
    #     max_new_tokens=512,
    #     top_k=10,
    #     top_p=0.8,
    #     temperature=0.7,
    # )
    llm = model
    embedding_function = get_embeddings_function("zh")

    print("********开始多轮提问*********")
    searching_blob = Blob.from_path(documents)
    parser = MyParser()
    output = parser.lazy_parse(searching_blob)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max(_chunk_size, 3000),
                                                   chunk_overlap=max(_chunk_overlap, 300), add_start_index=True,
                                                   length_function=len, is_separator_regex=False)
    output = text_splitter.split_documents(output)

    vector_db = Chroma.from_documents(output, embedding=embedding_function)
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": max(numbers, 20), "fetch_k": 50})

    compressor = FlashrankRerank(top_n=10)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    dialogues = conversation_concat(seed_questions, rag_inter_answers, numbers, running)  #single模式下先整合前两句对话
    if type(dialogues) is str:
        str_results = ''
        dialogues = [string_processing(dialogue) for dialogue in dialogues.split('<eos>')]

    list_results = []
    for i in tqdm(range(len(dialogues))):
        docs = compression_retriever.invoke(
            f"你认为{dialogues[i]}与本文哪些核心内容有关？请返回更全面的有关信息，并过滤所有无关的、有特定场景的信息。")
        reordering = LongContextReorder()

        PROMPT_TEMPLATE, _, _ = prompt_choices(mode=mode)

        chain = create_stuff_documents_chain(llm, PROMPT_TEMPLATE) | StrOutputParser()
        reordered_docs = reordering.transform_documents(docs)

        input_variables_dict = {
            "single": {"file_name": file_name, "context": reordered_docs, "number": numbers},
            "augment": {"file_name": file_name, "context": reordered_docs, "prompt": dialogues[i]},
            "literary": {"file_name": file_name, "context": reordered_docs, "prompt": dialogues[i],
                         "literary": literary}
        }
        input_variables = input_variables_dict.get(mode)
        input_variables["query"] = "请按要求生成提问"
        result = chain.invoke(input_variables)

        if running == "terminal":
            list_results.append(result)
        else:
            str_results += result + '<eos>' + '\n\n'
    return_results = list_results if running == "terminal" else str_results

    return return_results