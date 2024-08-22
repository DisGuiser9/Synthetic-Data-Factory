from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate

def single_turn_rag_template():
    """
    correspondence to single turn prompt
    input_variables: prompt, context
    """
    full_template = """
            {system}

            {instruction}
            
            {notice}
            """
    full_prompt = PromptTemplate.from_template(full_template)

    system_template = """
            请你扮演一位优秀的外贸研究助理，现在需要你根据要求生成文本。
            """
    system_prompt = PromptTemplate.from_template(system_template)

    instruction_template = """
                首先，请严格按照提供的检索内容和要求输出文本，提供的内容已按照与{prompt}的相关性进行了降序排列。
                请按以下步骤和要求完成任务：
                1. 分析：检索内容中可能最只是存在少部分有关语句，请分析哪些检索内容对回答问题最有帮助，能够成为回答的关键依据。
                2. 回答：严格按照先前的对话、分析结果，以及所给的参考资料{context}，再详细且专业地回答{prompt}。
                      """
    instruction_prompt = PromptTemplate.from_template(instruction_template)

    notice_template = """
                请在回答时必须遵循以下原则：
                1. 回答必须在0-300字之间，保持客观，符合人类偏好逻辑和语气，只需要输出与问题有关的回答，不要输出任何额外的内容或无关的举例！
                2. 回答时禁止输出“根据题目描述”、“根据书本”等类似文本，回答时保持原来的交流状态和逻辑，直接输出回答，而不是按检索后逻辑回答问题！！
                2. 禁止回答中包含额外的内容或无关的举例，禁止输出一切你的判断和总结！如果你有不知道的请严格参考所给内容，禁止胡言乱语重复输出！
                3. 禁止输出令人困惑或具歧义的文本，尽可能少地依赖你的先验知识，但必须保持语言能力，保持输出文本的质量，必要时可以罗列排序文本！
                4. 禁止输出任何对回答的评价，禁止总结回答内容，禁止总结回答是否遵循提示词要求或上下文，禁止用总起句介绍回答是否遵循或提示词上下文！！              
               """
    notice_prompt = PromptTemplate.from_template(notice_template)

    input_prompts = [
        ("system", system_prompt),
        ("instruction", instruction_prompt),
        ("notice",notice_prompt)
    ]
    SINGLE_TURN_LLM_PROMPT = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
    return SINGLE_TURN_LLM_PROMPT
    

def stepback_rag_template():
    """
    correspondence to stepback prompt
    input_variables: file_name, prompt, context, question
    """
    full_template = """
            {system}

            {instruction}

            {notice}
            """
    full_prompt = PromptTemplate.from_template(full_template)

    system_template = """
            请你扮演一位优秀的外贸研究助理，现在需要你根据要求生成对应的文本。
            """
    system_prompt = PromptTemplate.from_template(system_template)

    instruction_template = """
            以下是一份对话{prompt}，由“提问、回答、提问”组合而成，主题是{file_name}；
            请按以下步骤和要求完成任务：
            1. 分析：第一个问题与第二个问题是高度相关的，第一个问题的回答也能作为回答第二个问题的参考；
                    检索内容中可能最只是存在少部分有关语句，分析得到与对话最有关的文本片段
            2. 整合：结合按照先前的对话、分析结果，以及所给的参考资料{context}
            3. 回答：严格按照分析和整合结果以及要求，详细且专业地回答第二个问题{question}。
            """
    instruction_prompt = PromptTemplate.from_template(instruction_template)

    notice_template = """
            请在回答时必须遵循以下原则：
            1. 回答必须在0-300字之间，中文，保持客观，符合人类偏好逻辑和语气，只需要输出与问题有关的回答，不要输出任何额外的内容或无关的举例！
            2. 回答时禁止输出“根据题目描述”、“根据书本、对话”等类似文本，回答时保持原来的交流状态和逻辑，直接输出回答，而不是按检索后逻辑回答问题！！
            3. 禁止输出令人困惑或具歧义的文本，尽可能少地依赖你的先验知识，但必须保持语言能力，保持输出文本的质量，必要时可以罗列排序文本！
            4. 禁止输出任何对回答的评价，禁止总结回答内容，禁止总结回答是否遵循提示词要求或上下文，禁止用总起句介绍回答是否遵循或提示词上下文！！
                    """
    notice_prompt = PromptTemplate.from_template(notice_template)

    input_prompts = [
        ("system", system_prompt),
        ("instruction", instruction_prompt),
        ("notice", notice_prompt)
    ]
    STEPBACK_RAG_PROMPT = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
    return STEPBACK_RAG_PROMPT

def augmented_prompt_rag_template():
    """
    correspondence to augment prompt
    input_variables: file_name, context, prompt, question
    """
    full_template = """
            {system}

            {instruction}

            {notice}
            """
    full_prompt = PromptTemplate.from_template(full_template)

    system_template = """
            请你扮演一位优秀的外贸研究助理，现在需要你根据要求生成对应的文本。
            """
    system_prompt = PromptTemplate.from_template(system_template)

    instruction_template = """
            以下是一份对话{prompt}，由“提问、回答、提问”组合而成，主题是{file_name}；
            请按以下步骤和要求完成任务：
            1. 分析：第一个问题与第二个问题是高度相关的，第一个问题的回答也能作为回答第二个问题的参考；
                    检索内容中可能最只是存在少部分有关语句，分析得到与对话最有关的文本片段
            2. 整合：结合按照先前的对话、分析结果，以及所给的参考资料{context}，结合你的分析结果组织语言，再。
            3. 回答：严格按照分析和整合结果以及要求，详细且专业地回答第二个问题{question}。
            """
    instruction_prompt = PromptTemplate.from_template(instruction_template)

    notice_template = """
            请在回答时必须遵循以下原则：
            1. 回答必须在0-300字之间，中文，保持客观，符合人类偏好逻辑和语气，只需要输出与问题有关的回答，不要输出任何额外的内容或无关的举例！
            2. 回答时禁止输出“根据题目描述”、“根据书本、对话、提示词”等类似文本，回答时保持原来的交流状态和逻辑，直接输出回答，而不是按检索后逻辑回答问题！！
            3. 禁止输出令人困惑或具歧义的文本，尽可能少地依赖你的先验知识，但必须保持语言能力，保持输出文本的质量，必要时可以罗列排序文本！
            4. 禁止输出任何对回答的评价，禁止总结回答内容，禁止总结回答是否遵循提示词要求或上下文，禁止用总起句介绍回答是否遵循或提示词上下文！！
                    """
    notice_prompt = PromptTemplate.from_template(notice_template)

    input_prompts = [
        ("system", system_prompt),
        ("instruction", instruction_prompt),
        ("notice", notice_prompt)
    ]
    AUGMENT_RAG_PROMPT = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
    return AUGMENT_RAG_PROMPT


def literary_prompt_rag_template():
    """
    correspondence to augment prompt
    input_variables: file_name, prompt, literary, context
    """
    full_template = """
            {system}
            
            {instruction}
            
            {notice}
            """
    full_prompt = PromptTemplate.from_template(full_template)

    system_template = """
            请你扮演一位优秀的外贸研究助理，现在需要你根据要求生成对应的文本。
            """
    system_prompt = PromptTemplate.from_template(system_template)

    instruction_template = """
            以下是一段与外贸相关的文本段落{prompt}，其中包含了所需要生成的文体和主题是{file_name}的外贸相关文本；
            请按以下步骤和要求完成任务：
            1. 检索：请先参考{context}，在相关文本段落中，检索生成文体所需的文本片段；
            2. 生成：根据检索的片段，遵循问题{question}生成与文本段落相关，且符合{literary}文体的文本；
            """
    instruction_prompt = PromptTemplate.from_template(instruction_template)

    notice_template = """    
            请在回答时必须遵循以下原则：
            1. 回答字数与所要求生成的文体的平均长度相同，中文回答，保持客观，符合人类偏好逻辑和语气，生成文本的语气、格式、逻辑等必须符合对应文体！！
            2. 回答时禁止输出“根据题目描述”、“根据书本、对话、提示词”等类似文本，回答时保持原来的交流状态和逻辑，直接输出回答，而不是按检索后逻辑回答问题！！
            3. 禁止输出令人困惑或具歧义的文本，尽可能少地依赖你的先验知识，但必须保持语言能力，保持输出文本的质量，必要时可以罗列排序文本！
            4. 禁止输出任何对回答的评价，禁止总结回答内容，禁止总结回答是否遵循提示词要求或上下文，禁止用总起句介绍回答是否遵循或提示词上下文！！禁止输出无意义的符号！！！
                    """
    notice_prompt = PromptTemplate.from_template(notice_template)

    input_prompts = [
        ("system", system_prompt),
        ("instruction", instruction_prompt),
        ("notice", notice_prompt)
    ]
    LITERARY_RAG_PROMPT = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
    return LITERARY_RAG_PROMPT