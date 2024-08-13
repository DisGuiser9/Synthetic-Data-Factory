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
    --model_name="qwen2:72b-instruct-fp16"\
    --top_k=50\
    --top_p=0.35\
    --numbers=10\
    --langcode='zh'
```

#### Ollama
Model: Qwen2-72B-Instruct-fp16

#### Langchain

- PromptTemplate
    - PipelinePromptTemplate

- Retriver
    - FlashrankRerank
    - ContextualCompressionRetriever
    - LongContextReorder

#### Trulenth
