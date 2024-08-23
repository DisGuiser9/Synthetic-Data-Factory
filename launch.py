import gradio as gr
import pandas as pd
from typing import Dict, Optional

from utils import *
from evaluation import *

directory = '/data/share9/XPT/dataset/dataset_8/book/data'
# root_directory = '/data/share9/XPT/dataset'
# directory = get_directory(root_directory)
files_list = get_files_in_directory(directory, return_paths=True)
available_model_list = local_ollama_models()
files_choices = [os.path.basename(files) for files in files_list]
question_choices = ["augment", "literary"]

with gr.Blocks(title="数据合成") as demo:
    gr.Markdown("""# Synthetic Data Factory""")
    with gr.Row():
        directory = gr.Textbox(label="Directory", value=directory, interactive=False, scale=4)
        available_models = gr.Dropdown(choices=available_model_list, value="qwen2:72b-instruct-fp16", label="Ollama Model", scale=2)
        data = gr.Dropdown(choices=files_choices, label="文件列表", scale=4)
        # collection_name = gr.Textbox(label="Collection Name", interactive=True, scale=4)
        file_path = gr.Textbox(label="文件地址", interactive=False, scale=7, show_copy_button=True)

    with gr.Row():
        with gr.Column(scale=4):
            generated_seed_queries = gr.Textbox(label="种子合成提示词",lines=12, interactive=False, show_copy_button=True)
        with gr.Column(scale=6):
            previous_rag_answer = gr.Textbox(label="RAG生成中间回答", lines=12, max_lines=14, interactive=False)
        with gr.Column(scale=1):
            seed_question_numbers = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="种子提示词数量", scale=1)
            button_generate_q = gr.Button(value="生成种子提示词")
            top_k = gr.Slider(minimum=20, maximum=80, value=40, step=1, label="Top-K（越大越多样）")
            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.05, label="Top-P")
            button_intermediate_a = gr.Button(value="中间回答", variant='primary')

    with gr.Row():
        # with gr.Column(scale=5):
        #     first_queries = gr.Textbox(label="一轮对话提示词",lines=10, interactive=False, show_copy_button=True)
            # button_synthe = gr.Button(value="生成一轮对话提示词")
        with gr.Column(scale=7):
            second_queries = gr.Textbox(label="二轮对话提示词",lines=12, interactive=False, show_copy_button=True)
            # multi_question_numbers = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="多轮提示词数量", scale=1)
        with gr.Column(scale=2):
            question_mode = gr.Radio(choices=question_choices, value='augment', label="Question的不同形式")
            literary = gr.Textbox(label="请输入想生成的文体(仅literary模式有效)", value='报纸' ,info='报纸、博客、教材、论文等', interactive=True, scale=1)
            button_gen_multi_q = gr.Button(value="生成多轮合成提示词")
            button_final_answer = gr.Button(value="生成回答", variant='primary')

    with gr.Row():
        with gr.Column(scale=3):
            rag_content = gr.Textbox(label="RAG回答", lines=10, max_lines=14, interactive=False)
        with gr.Column(scale=3):
            llm_content = gr.Textbox(label="LLM回答", lines=10, max_lines=14, interactive=False)
        #     retrieved_docs = gr.Textbox(label="检索到的文档", lines=23, max_lines=30, interactive=False)

    with gr.Row():
        # with gr.Column(scale=4):
        #     rag_plot = gr.Plot(label="Umap Embeddings for RAG")
        #     llm_plot = gr.Plot(label="Umap Embeddings for LLM")
        with gr.Column(scale=6):
            score_dataframe = gr.DataFrame(label="Score Statistics", 
                                           type="array",
                                           row_count=5,
                                           col_count=3,
                                           show_label=True,
                                           column_widths=[30,40,40]
                                           )
    with gr.Row():
        # button_gen_figs = gr.Button("生成Umap图")
        button_score = gr.Button("生成统计数据")
    with gr.Row():
        output_file_preview = gr.Textbox(label="输出文件预览", lines=10, max_lines=14, interactive=False)
    with gr.Row():
        with gr.Column(scale=2):
            button_clear = gr.ClearButton(value="清空记录", scale=1, components=[generated_seed_queries, rag_content, llm_content,
                                                                                output_file_preview, score_dataframe])
        with gr.Column(scale=6):
            button_preview = gr.Button("预览", scale=1, variant='secondary')
            button_save = gr.Button("保存文件", scale=1, variant='primary')


    # data.change(fn=get_collection_name,inputs=data, outputs=collection_name)
    data.change(fn=get_file_name,inputs=[data, directory], outputs=file_path)
    available_models.change(fn=get_ollama_model,inputs=[available_models], outputs=available_models)
    
    button_generate_q.click(fn=seed_prompt_generation,inputs=[available_models,file_path,seed_question_numbers],
                            outputs=generated_seed_queries)
    button_intermediate_a.click(fn=retrieve_answer,inputs=[available_models,generated_seed_queries, file_path, top_k, top_p],
                                outputs=[previous_rag_answer])

    question_mode.change(inputs=[question_mode], outputs=[question_mode])
    # button_synthe.click(fn=conversation_concat,inputs=[generated_seed_queries, previous_rag_answer, seed_question_numbers],outputs=[first_queries])
    button_gen_multi_q.click(fn=multi_prompts_generation,inputs=[available_models,file_path,seed_question_numbers,generated_seed_queries, previous_rag_answer,question_mode,literary],
                             outputs=[second_queries])
    button_final_answer.click(fn=retrieve_answer,inputs=[available_models,generated_seed_queries, file_path, top_k, top_p, question_mode, literary],
                              outputs=[rag_content])
    button_final_answer.click(fn=llm_answer,inputs=[available_models,generated_seed_queries, file_path, top_k, top_p, question_mode, literary],
                              outputs=llm_content)

    # button_gen_figs.click(fn=visualize_embeddings,inputs=[input_query, extended_queries, retrieved_docs, file_path, collection_name, file_path], outputs=rag_plot)
    # button_gen_figs.click(fn=visualize_embeddings,inputs=[input_query, extended_queries, llm_content, file_path, collection_name, file_path], outputs=llm_plot)
    # score_dataframe.change(fn=decorate, inputs=score_dataframe, outputs=score_dataframe)

    
    button_score.click(fn=evaluation,inputs=[generated_seed_queries, rag_content, llm_content],outputs=score_dataframe)

    button_preview.click(fn=post_processing_for_dpo,inputs=[generated_seed_queries, rag_content, llm_content],outputs=output_file_preview)
    button_save.click(fn=dump_into_json,inputs=[output_file_preview, output_directory])


if __name__ == "__main__":
    demo.launch()
