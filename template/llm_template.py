from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate

def single_turn_llm_template():
    """
    correspondence to single turn prompt
    input_variables: file_name, query
    """
    full_template = """
            {system}

            {notice}
                      
            """
    full_prompt = PromptTemplate.from_template(full_template)

    system_template = """
            请你扮演一位优秀的外贸研究助理，现在需要你回答我所给的{file_name}的相关问题{query}。
            """
    system_prompt = PromptTemplate.from_template(system_template)

    notice_template = """
            回答必须在300字左右，语气保持客观，符合人类偏好逻辑和语气，
            必须以中文回答！禁止给回答标号！禁止输出自己的评价、总结等无关语句！
            """
    notice_prompt = PromptTemplate.from_template(notice_template)

    input_prompts = [
        ("system", system_prompt),
        ("notice", notice_prompt)
    ]
    SINGLE_TURN_LLM_PROMPT = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
    return SINGLE_TURN_LLM_PROMPT

def stepback_llm_template():
    """
    correspondence to stepback prompt
    input_variables: query, file_name
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
            以下是一份对话{query}，由“提问、回答、提问”组合而成，主题为{file_name}；
            第一个问题与第二个问题是高度相关的，第一个问题的回答也能作为回答第二个问题的参考；
            因此，在你回答第二个问题时，请你理解并结合这份对话的上下文，参考后再做回答。
            """
    instruction_prompt = PromptTemplate.from_template(instruction_template)

    notice_template = """
            必须先参考上下文，再回答第二个问题，
            回答必须在300字左右，语气保持客观，符合人类偏好逻辑和语气，
            必须以中文回答！禁止给回答标号！禁止不看上下文胡乱回答！
                    """
    notice_prompt = PromptTemplate.from_template(notice_template)

    input_prompts = [
        ("system", system_prompt),
        ("instruction", instruction_prompt),
        ("notice", notice_prompt)
    ]
    STEPBACK_LLM_PROMPT = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
    return STEPBACK_LLM_PROMPT

def augmented_prompt_llm_template():
    """
    correspondence to augment prompt
    input_variables: file_name, query
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
            以下是一份对话{query}，由“提问、回答、提问”组合而成，主题为{file_name}；
            第一个问题与第二个问题是高度相关的，第一个问题的回答也能作为回答第二个问题的参考；
            因此，在你回答第二个问题时，请你理解并结合这份对话的上下文，参考后再做回答。
            """
    instruction_prompt = PromptTemplate.from_template(instruction_template)

    notice_template = """
            必须先参考上下文，再回答第二个问题，
            回答必须在300字左右，语气保持客观，符合人类偏好逻辑和语气，
            必须以中文回答！禁止给回答标号！禁止不看上下文胡乱回答！
                    """
    notice_prompt = PromptTemplate.from_template(notice_template)

    input_prompts = [
        ("system", system_prompt),
        ("instruction", instruction_prompt),
        ("notice", notice_prompt)
    ]
    AUGMENT_LLM_PROMPT = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
    return AUGMENT_LLM_PROMPT

def literary_prompt_llm_template():
    """
    correspondence to augment prompt
    input_variables: query, file_name, literary
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
            以下是一段与外贸有关的文本{query}，主题是{file_name}；
            你需要根据文本内容，生成符合{literary}文体的文本
            """
    instruction_prompt = PromptTemplate.from_template(instruction_template)

    notice_template = """
            字数与所要求生成的文体的平均长度相同，
            必须以中文回答！禁止给回答标号！禁止生成无关符号！
            禁止输出令人困惑或具歧义的文本！
            生成文本的语气、格式、逻辑等必须符合对应文体！
                    """
    notice_prompt = PromptTemplate.from_template(notice_template)

    input_prompts = [
        ("system", system_prompt),
        ("instruction", instruction_prompt),
        ("notice", notice_prompt)
    ]
    LITERARY_LLM_PROMPT = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
    return LITERARY_LLM_PROMPT