## 1. Setup Environment Variables

Add a ```.env``` file at project root level with the following variables

```bash
HF_TOKEN=<your_hf_token>     
HF_MODEL_API_GATEWAY=https://<your_model>.us-east-1.aws.endpoints.huggingface.cloud      
HF_EMBEDDING_API_GATEWAY=https://<your_embedding>.us-east-1.aws.endpoints.huggingface.cloud    
LANGCHAIN_TRACING_V2=true    
LANGCHAIN_API_KEY=<your-api-key>     
LANGCHAIN_PROJECT=<your-project>     
```

## 2. Create environment

```bash
make environment
```

## 3. Install dependencies
```
make install
```

## 4. Run app locally

```
make run
```

