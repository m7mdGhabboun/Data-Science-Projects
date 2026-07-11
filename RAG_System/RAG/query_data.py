import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Context:
{context}

---

Question: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str, chat_history: list = None):

    # 1. Provide a default empty list if no history is passed
    if chat_history is None:
        chat_history = []

    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=3)

    # Format the context and question
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = PromptTemplate.from_template(template=PROMPT_TEMPLATE, template_format='f-string')
    user_prompt = prompt_template.format(context=context_text, question=query_text)
    
    print(f"Sending prompt to DeepSeek:\n{user_prompt}")

    # 2. Instantiate ChatDeepSeek natively
    model = ChatDeepSeek(
        api_key="API_KEY",
        model="deepseek-chat", 
        max_tokens=1024,
        temperature=0
    )

    # 3. Use the modern messages structure
    # This clearly separates the system's instructions from the user's retrieved data
    messages = [
        ("system", "You are an expert Arabic legal assistant. Answer the user's question strictly based on the provided context. If the answer is not in the context, say so. Answer in Arabic."),
        ("human", user_prompt)
    ]

    # 3. Inject the past conversation
    for msg in chat_history:
        # Streamlit uses 'user'/'assistant', but LangChain prefers 'human'/'ai'
        role = "human" if msg["role"] == "user" else "ai"
        
        # Skip the default greeting to save tokens if you want, or just append everything
        if msg["content"] != "مرحباً! كيف يمكنني مساعدتك في قانون الضمان الاجتماعي لعام 2014؟":
            messages.append((role, msg["content"]))

    # 4. Append the CURRENT question wrapped in its legal context at the very end
    messages.append(("human", user_prompt))

    # Invoke the model and extract the content
    response = model.invoke(messages)
    response_text = response.content

    # Output the results
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"\nResponse: {response_text}\nSources: {sources}"
    print(formatted_response)
    
    return response_text

def classify_query(query_text: str) -> str:
    """
    Step 2 in diagram: Query Classifier (LLM)
    Uses a strict prompt to force the LLM to output only a routing keyword.
    """
    classifier_model = ChatDeepSeek(
        api_key="API_KEY", 
        model="deepseek-chat", 
        temperature=0, # MUST be 0 so the model doesn't get creative with its output
        max_tokens=10  # We only need one word, save bandwidth
    )

    messages = [
        ("system", """You are a query classifier.
Classify the user query into one of the following:
1. GENERAL (not related to the domain knowledge or legal documents)
2. DOMAIN_SPECIFIC (related to the Social Security Corporation / Social Security Law for the year 2014 / requires specific internal knowledge)

Respond with only one word: GENERAL or DOMAIN_SPECIFIC"""),
        ("human", query_text)
    ]

    response = classifier_model.invoke(messages)
    
    # Clean the string just in case the LLM adds a period or extra space
    classification = response.content.strip().upper()
    
    # Fallback safety check
    if "GENERAL" in classification:
        return "GENERAL"
    return "DOMAIN_SPECIFIC"

def general_chat(query_text: str, chat_history: list) -> str:
    """
    Step 3A in diagram: General LLM Path
    Bypasses the vector database completely.
    """
    model = ChatDeepSeek(
        api_key="API_KEY", 
        model="deepseek-chat", 
        temperature=0.7 # A bit higher for natural conversational flow
    )
    
    messages = [
        ("system", "You are a helpful Arabic assistant. Answer the user's general questions naturally.")
    ]
    
    # Inject history so it remembers the chat
    for msg in chat_history:
        role = "human" if msg["role"] == "user" else "ai"
        messages.append((role, msg["content"]))
        
    messages.append(("human", query_text))
    
    response = model.invoke(messages)
    return response.content

def process_user_input(query_text: str, chat_history: list = None) -> str:
    """
    The Master Router. This is the function Streamlit will call.
    """
    if chat_history is None:
        chat_history = []
        
    print("Classifying query...")
    route = classify_query(query_text)
    print(f"Route selected: {route}")
    
    if route == "DOMAIN_SPECIFIC":
        # Execute Step 3B: RAG Retrieval Path
        return query_rag(query_text, chat_history)
    else:
        # Execute Step 3A: General LLM Path
        return general_chat(query_text, chat_history)

if __name__ == "__main__":
    main()
