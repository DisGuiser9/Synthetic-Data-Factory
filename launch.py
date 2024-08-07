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

# dataframe = pd.DataFrame(
#     {
#     "Matrix": ["Perplexity", "Distinct", "freq_similarities", "keyword_matches","LLM-Judger"],
#     'RAG_Score': [0, 0, 0, 0, 0],
#     'LLM_Score': [0, 0, 0, 0, 0]
#     }
# )
# styled_df = dataframe.style.apply(decorate, axis = None)

with gr.Blocks(title="数据合成") as demo:
    gr.Markdown("""# Synthetic Data Factory""")
    with gr.Row():
        directory = gr.Textbox(label="Directory", value=directory, interactive=False, scale=4)
        available_models = gr.Dropdown(choices=available_model_list, label="Ollama Model", scale=2)
        data = gr.Dropdown(choices=files_choices, label="File", scale=4)
        # collection_name = gr.Textbox(label="Collection Name", interactive=True, scale=4)
        file_path = gr.Textbox(label="File Path", interactive=False, scale=7, show_copy_button=True)
    with gr.Row():
        with gr.Column(scale=6):
            generated_queries = gr.Textbox(label="扩展提示词",lines=6, interactive=False, show_copy_button=True)
        with gr.Column(scale=2):
            numbers = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="提示词数量")
            button_generate_q = gr.Button("生成扩展提示词")
            button_answer = gr.Button("生成回答", scale=1, variant='primary')
    # with gr.Row():
        # extended_queries = gr.Textbox(label="扩展提示词", lines=3, interactive=False, show_copy_button=True, scale=15)
        
    with gr.Row():
        with gr.Column(scale=6):
            rag_content = gr.Textbox(label="RAG回答", lines=10, max_lines=14, interactive=False)
            llm_content = gr.Textbox(label="LLM回答", lines=10, max_lines=14, interactive=False)
        with gr.Column(scale=4):
            retrieved_docs = gr.Textbox(label="检索到的文档", lines=25, max_lines=30, interactive=False)
    with gr.Row():
        # with gr.Column(scale=4):
        #     rag_plot = gr.Plot(label="Umap Embeddings for RAG")
        #     llm_plot = gr.Plot(label="Umap Embeddings for LLM")
        with gr.Column(scale=6):
            score_dataframe = gr.DataFrame(label="Score Statistics", 
                                           type="array",
                                           row_count=5,
                                           col_count=(3, "fixed"),
                                           show_label=True,
                                           column_widths=[30,40,40]
                                           )
    with gr.Row():
        # button_gen_figs = gr.Button("生成Umap图")
        button_score = gr.Button("生成统计数据")
    with gr.Row():
        output_file_preview = gr.Textbox(label="输出文件预览", lines=10, interactive=False)
    with gr.Row():
        button_preview = gr.Button("预览", scale=1, variant='secondary')
        button_save = gr.Button("保存文件", scale=1, variant='primary')
    
    # data.change(fn=get_collection_name,inputs=data, outputs=collection_name)
    data.change(fn=get_file_name,inputs=[data, directory], outputs=file_path)
    available_models.change(fn=get_ollama_model,inputs=[available_models], outputs=available_models)
    button_generate_q.click(fn=prompt_generation,inputs=[available_models,file_path,numbers],outputs=generated_queries)
    
    button_answer.click(fn=retrieve_result,inputs=[available_models,generated_queries, file_path],outputs=[rag_content,retrieved_docs])
    button_answer.click(fn=chat_with_llm,inputs=[available_models,generated_queries],outputs=llm_content)

    # button_gen_figs.click(fn=visualize_embeddings,inputs=[input_query, extended_queries, retrieved_docs, file_path, collection_name, file_path], outputs=rag_plot)
    # button_gen_figs.click(fn=visualize_embeddings,inputs=[input_query, extended_queries, llm_content, file_path, collection_name, file_path], outputs=llm_plot)
    # score_dataframe.change(fn=decorate, inputs=score_dataframe, outputs=score_dataframe)
    button_score.click(fn=evaluation,inputs=[generated_queries, rag_content, llm_content],outputs=score_dataframe)

    button_preview.click(fn=post_processing,inputs=[generated_queries, rag_content, llm_content],outputs=output_file_preview)
    button_save.click(fn=dump_into_json,inputs=[output_file_preview])


if __name__ == "__main__":
    demo.launch()
