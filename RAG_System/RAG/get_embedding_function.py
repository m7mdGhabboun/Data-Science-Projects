from langchain_huggingface import HuggingFaceEmbeddings

# Quick configuration toggle
RUNNING_ON_LAPTOP = True

if RUNNING_ON_LAPTOP:
    #LLM_MODEL = "qwen2.5:3b"
    EMBED_DEVICE = "cpu"
else:
    #LLM_MODEL = "qwen2.5:7b"
    EMBED_DEVICE = "cuda"


def get_embedding_function(model_id: str = "BAAI/bge-m3"):
    
    print(f"Loading embedding model: {model_id}...")
    
    local_execution_kwargs = {'device': EMBED_DEVICE}
    encoding_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs=local_execution_kwargs,
        encode_kwargs=encoding_kwargs
    )
    
    return embeddings

# Example usage:
# arabic_embedder = get_embedding_function()
# test_vector = arabic_embedder.embed_query("المادة الثانية من القانون")
# print(f"Vector length: {len(test_vector)}")