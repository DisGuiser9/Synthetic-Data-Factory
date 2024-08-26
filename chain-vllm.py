from langchain_community.llms.vllm import VLLM
from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate
#
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
def add_chat_xtemplate(prompt):
    pure_prompt = prompt.pretty_repr()
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{pure_prompt}<|im_end|>\n<|im_start|>assistant\n"
    return PromptTemplate.from_template(prompt)
pt = add_chat_template(pip_prompt)

ptt = tokenizer_add_chat_template(pip_prompt, model)




print(pt)
print(ptt)
a = ''
# while a!='quit':
#     a = input()
#     raw_msg = {
#         'user': 'user',
#         'content': f'{a}',
#     }
#     msg = prompt.invoke(raw_msg)
#     res = model.invoke(msg)
#     print(res)
