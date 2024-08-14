from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

def single_turn_prompt_template():
    """
    input_variables in chain: file_name, context, number
    """
    full_template = """
            {system}

            {instruction}

            {notice}
            
            {example}            
            """
    full_prompt = PromptTemplate.from_template(full_template)

    system_template = "请扮演一位优秀的外贸研究助理生成与{file_name}内容相关的问题。"
    system_prompt = PromptTemplate.from_template(system_template)

    instruction_template = """
                            我将提供与外贸领域相关的文本，请你按照以下步骤操作：                            
                            1.你必须参考我给你的，与文本主旨最相关的内容{context}，这些内容已按照与文本主旨相关性进行了降序排列，越靠前的主题越相关。
                            2.根据排序文本概括这段文本内容，并根据此生成{number}个相关问题。这些问题的答案必须存在于文本中，并且与原始文本紧密相关。
                            3.确保每个问题都是完整的句子，且两个问题不相关；只输出问题本身，不需要提供答案、分析或总结。禁止输出标号！！！
                            4.生成的问题应该简短明了，避免使用复合句，禁止输出无意义或意义不明的问题。
                            """
    instruction_prompt = PromptTemplate.from_template(instruction_template)

    notice_template = """
                禁止输出与检索到文本无关的问题!
                输出中文问题即可，问题长度必须在10-30字！
                禁止对问题进行编号！！
                禁止输出含义不明的问题！
                仅返回问题列表！避免生成类似反面例子的问题！
                """
    notice_prompt = PromptTemplate.from_template(notice_template)

    example_template = """
                以下是几个优秀的例子：
                1. 液体类货物的包装有何特殊要求？
                2. 国际货物买卖引起的国际贸易结算为什么需要研究票据行为规律？
                3. 在会计规范体系中，合法性、合理性和实践性分别对应什么？
                以下是不合格的例子，存在意义不明、有歧义的问题：
                1. 目的港是哪里？（问题不完整存在歧义）
                """
    example_prompt = PromptTemplate.from_template(example_template)

    input_prompts = [
        ("system", system_prompt),
        ("instruction", instruction_prompt),
        ("notice", notice_prompt),
        ("example", example_prompt),
    ]

    SINGLE_TURN_TEMPLATE = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

    return SINGLE_TURN_TEMPLATE


def stepback_prompt_template():
    """
    input_variables in chain: file_name, context, question, 
    """
    full_template = """
            {system}

            {step_back_prompt}

            {notice}            
            """
    full_prompt = PromptTemplate.from_template(full_template)

    system_template = "请扮演一位优秀的外贸研究助理解决关于生成与{file_name}内容相关的一个问题。"
    system_prompt = PromptTemplate.from_template(system_template)

    step_back_template = """
                            你的任务是后退一步，同时参考文档{context}，将问题转述为一个与文档内容相关的，且更通用的后退式问题；
                            以下是几个例子：
                            原始问题：如果温度增加 2 倍，体积增加 8 倍，理想气体的压强 P 会发生什么变化？
                            后退式问题：与气体、压强和温度相关的物理原理是什么？
                            原始问题：如何利用大语言模型成功发表高质量 SCI 论文？
                            后退式问题：如何有效利用大语言模型进行学术研究？
                            原始问题：{question}
                            后退式问题：
                        """
    step_back_prompt = PromptTemplate.from_template(step_back_template)

    notice_template = """
                必须只输出这个后退式问题!输出中文！禁止与原问题重复！
                问题长度必须在10-30字！必须简练具体且没有歧义！
                必须是基于问题，且参考文档，生成相关的后退式问题，禁止发散随意生成！
                可以参考给定的例子生成后退式问题！！
                禁止输出含义不明的问题！禁止对问题标号！！
                """
    notice_prompt = PromptTemplate.from_template(notice_template)

    input_prompts = [
        ("system", system_prompt),
        ("instruction", step_back_prompt),
        ("notice", notice_prompt)
    ]

    STEP_BACK_PROMPT_TEMPLATE = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
    return STEP_BACK_PROMPT_TEMPLATE


def augment_dialogue_prompt_template():
    """
    input_variables in chain: file_name, dialogue, context
    """
    full_template = """
            {system}

            {instruction}
            
            {notice}
            """
    
    system_template = "请扮演一位优秀的外贸研究助理，我将给你一个主题为{file_name}外贸相关的对话，请你基于提供的对话扩展1个额外查询问题。"
    system_prompt = PromptTemplate.from_template(system_template)

    instruction_template = """
                            这是一段对话：{dialogue}，请你按照以下步骤操作：
                            1. 理解以上对话内容；
                            2. 参考{context}，生成一个与对话相关的问题
                            """
    instruction_prompt = PromptTemplate.from_template(instruction_template)

    notice_template = """
                必须只输出这个问题!输出中文！禁止与原问题重复！
                问题长度必须在10-30字！必须简练具体且没有歧义，与上文不冲突！
                必须是基于对话生成的相关扩展问题，必须保证对话延续性，禁止发散随意生成！
                """
    notice_prompt = PromptTemplate.from_template(notice_template)

    input_prompts = [
        ("system", system_prompt),
        ("instruction", instruction_prompt),
        ("notice", notice_prompt)
    ]
    AUGMENT_DIALOGUE_TEMPLATE = PipelinePromptTemplate(final_prompt=full_template, pipeline_prompts=input_prompts)
    return AUGMENT_DIALOGUE_TEMPLATE


def literary_prompt_template():
    """
    input_variables in chain: file_name, literary, context;
    literary example: 博客、报纸、教案、 文章
    """
    final_template = """
            {system}
            
            {instruction}

            {notice}
            """
    full_prompt = PromptTemplate.from_template(final_template)

    system_template = """
                    请扮演一位优秀的外贸研究助理，我将给你一个主题为{file_name}外贸相关文本，
                    你的任务是生成一段用于指导AI创作符合{literary}文体，与主题相关的目标提示词。
                    """
    system_prompt = PromptTemplate.from_template(system_template)

    instruction_template = """
                            这是一段与文本主旨最相关的内容{context}，请模型总结以上文本的内容：
                            先整合成一份元提示词：
                            再生成最后的目标提示词：
                            """
    instruction_prompt = PromptTemplate.from_template(instruction_template)

    notice_template = """
                必须只输出这个提示词!输出中文！提示词长度必须在包含所给文本，必须具体且没有歧义！
                输出提示词的长度应在200字左右，包括相关内容总结和文体要求！
                只需要输出最后的目标提示词！
                以下是一个例子：
                元提示词：这段文章的主旨是“现行有关国际结算的规则与惯例有哪些？”，需要一份文体形式为报告的文本
                目标提示词：请你写一段报告，主题为“现行有关国际结算的规则与惯例有哪些？”
                """
    notice_prompt = PromptTemplate.from_template(notice_template)
    input_prompts = [
        ("system", system_prompt),
        ("instruction", instruction_prompt),
        ("notice", notice_prompt)
    ]
    LITERARY_PROMPT_TEMPLATE = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
    return LITERARY_PROMPT_TEMPLATE

