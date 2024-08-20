# Synthetic Data Factory
## Synthesize LLM SFT & Alignment Dataset with RAG System

### About this
Files:
```
├── dataset            # put original data in
├── output_data        # synthetic data named by date
├── statistics_data    # the statistics of the pair data
```

Usage:
**Example**
```
python main.py\
    --model_name "qwen2:72b-instruct-fp16"\
    --top_k 40\
    --top_p 0.3\
    --file_name '报关实务 282.jsonl'\
    --numbers 5\
    --langcode 'zh'\
    --type 'dpo'\
    --mode 'literary'\
    --literary '博客'\
```

#### Ollama
Model: Qwen2-72B-Instruct-fp16
```
ollama pull qwen2:72b-instruct-fp16
```

#### Embedding Model
Ollama Embedding: nomic-embed-text
```
ollama pull nomic-embed-text
```

FastEmbedEmbeddings: BAAI/bge-small-zh-v1.5
```
git clone https://huggingface.co/BAAI/bge-small-zh-v1.5
```

#### Langchain

- PromptTemplate
    - PipelinePromptTemplate

- Retriver
    - FlashrankRerank
    - ContextualCompressionRetriever
    - LongContextReorder

#### Trulenth


### TODO
- Trulenth Evaluation with OpenAI
- Gradio Adaption(TextBox outputs str format)
- Template Improvement
- First Query Templates Enrichment
- Retrieval Augmentation Improvement
