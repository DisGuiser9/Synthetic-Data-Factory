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
question_choices = ["single", "stepback", "augment", "literary"]

with gr.Blocks(title="数据合成") as demo:
    gr.Markdown("""# Synthetic Data Factory""")
    with gr.Row():
        directory = gr.Textbox(label="Directory", value=directory, interactive=False, scale=4)
        available_models = gr.Dropdown(choices=available_model_list, value="qwen2:72b-instruct-fp16", label="Ollama Model", scale=2)
        data = gr.Dropdown(choices=files_choices, label="文件列表", scale=4)
        # collection_name = gr.Textbox(label="Collection Name", interactive=True, scale=4)
        file_path = gr.Textbox(label="文件地址", interactive=False, scale=7, show_copy_button=True)
    with gr.Row():
        with gr.Column(scale=10):
            generated_queries = gr.Textbox(label="扩展提示词",lines=7, interactive=False, show_copy_button=True)
        with gr.Column(scale=3):
            question_numbers = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="提示词数量", scale=1)
            question_mode = gr.Dropdown(choices=question_choices, label="Question的不同形式")
            button_generate_q = gr.Button(value="生成扩展提示词", scale=1)
        with gr.Column(scale=1):
            top_k = gr.Slider(minimum=20, maximum=80, value=40, step=1, label="Top-K", info='越大越多样，以下同样')
            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.05, label="Top-P")
            button_answer = gr.Button(value="生成回答", scale=1, variant='primary')
    # with gr.Row():
        # extended_queries = gr.Textbox(label="扩展提示词", lines=3, interactive=False, show_copy_button=True, scale=15)
        
    with gr.Row():
        with gr.Column(scale=6):
            rag_content = gr.Textbox(label="RAG回答", lines=10, max_lines=14, interactive=False)
            llm_content = gr.Textbox(label="LLM回答", lines=10, max_lines=14, interactive=False)
        with gr.Column(scale=3):
            retrieved_docs = gr.Textbox(label="检索到的文档", lines=23, max_lines=30, interactive=False)
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
            button_clear = gr.ClearButton(value="清空记录", scale=1, components=[generated_queries, rag_content, llm_content,
                                                                                retrieved_docs, output_file_preview, score_dataframe])
        with gr.Column(scale=6):
            button_preview = gr.Button("预览", scale=1, variant='secondary')
            button_save = gr.Button("保存文件", scale=1, variant='primary')


    # data.change(fn=get_collection_name,inputs=data, outputs=collection_name)
    data.change(fn=get_file_name,inputs=[data, directory], outputs=file_path)
    available_models.change(fn=get_ollama_model,inputs=[available_models], outputs=available_models)
    button_generate_q.click(fn=prompt_generation,inputs=[available_models,file_path,question_numbers,question_mode],outputs=generated_queries)
    
    question_mode.change(inputs=[question_mode], outputs=[question_mode])
    button_answer.click(fn=retrieve_result,inputs=[available_models,generated_queries, file_path, top_k, top_p, question_mode],outputs=[rag_content,retrieved_docs])
    button_answer.click(fn=chat_with_llm,inputs=[available_models,generated_queries, file_path, top_k, top_p, question_mode],outputs=llm_content)

    # button_gen_figs.click(fn=visualize_embeddings,inputs=[input_query, extended_queries, retrieved_docs, file_path, collection_name, file_path], outputs=rag_plot)
    # button_gen_figs.click(fn=visualize_embeddings,inputs=[input_query, extended_queries, llm_content, file_path, collection_name, file_path], outputs=llm_plot)
    # score_dataframe.change(fn=decorate, inputs=score_dataframe, outputs=score_dataframe)
    button_score.click(fn=evaluation,inputs=[generated_queries, rag_content, llm_content],outputs=score_dataframe)

    button_preview.click(fn=post_processing_for_dpo,inputs=[generated_queries, rag_content, llm_content],outputs=output_file_preview)
    button_save.click(fn=dump_into_json,inputs=[output_file_preview])


if __name__ == "__main__":
    demo.launch()
