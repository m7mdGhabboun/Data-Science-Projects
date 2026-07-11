import os
import shutil
import stat
import argparse
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

import sys
# Force the standard output to handle Arabic (UTF-8) characters safely
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


DATA_PATH = "data"
CHROMA_PATH = "chroma"

def populate_database():

    # Check if the database should be cleared (using the --clear flag).
    
    print("✨ Clearing Database")
    clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = clean_merge_and_split_articles(documents)
    add_to_chroma(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# def split_documents_tokens(documents: list[Document]):
#     text_splitter = TokenTextSplitter(
#         #encoding_name = "cl100k_base",
#         chunk_size = 800,
#         chunk_overlap = 80,
#         length_function = len,
#     )
#     chunks = text_splitter.split_documents(documents)
#     print(f"Number of splitted chunks = {len(chunks)}")
#     return chunks


def clean_merge_and_split_articles(documents: list[Document]) -> list[Document]:
    """
    1. Cleans headers/footers from PDF pages.
    2. Merges all pages into a single string to fix cross-page articles.
    3. Splits the massive string by the RTL article marker.
    """
    
    # STEP 1 & 2: Clean and Merge
    merged_text = ""
    
    # This regex targets the exact noise you found. 
    # It looks for the date/Qistas string, the URL, and the page numbers (e.g., 28/37)
    # \s* handles unpredictable spaces or newlines between the elements.
    footer_pattern = r'2021/\s*5/\s*19ﻗﺳطﺎس\s*https://qistas\.com\S+\s+\d+/\d+'
    
    for doc in documents:
        # Erase the footer/header noise from the current page
        cleaned_page = re.sub(footer_pattern, '', doc.page_content)
        
        # Append to our master string with a newline to act as a space between pages
        merged_text += cleaned_page + "\n"

    # STEP 3: Split the Merged Text
    # We use the same RTL pattern we built earlier
    rtl_pattern = r'\n(?=\s*\(\s*\d+\s*(?:المادة|اﻟﻤﺎدة))'
    raw_sections = re.split(rtl_pattern, merged_text)
    
    # STEP 4: Package into Documents with Article Metadata
    article_chunks = []
    
    for section in raw_sections:
        cleaned_text = section.strip()
        
        if not cleaned_text:
            continue
            
        # Extract the specific article number to use as our reliable metadata
        # This looks for the number inside the parenthesis right next to "المادة"
        num_match = re.search(r'\(\s*(\d+)\s*(?:المادة|اﻟﻤﺎدة)', cleaned_text)
        article_number = num_match.group(1) if num_match else "Preamble/Intro"
        
        # Create the clean LangChain Document
        article_chunks.append(
            Document(
                page_content=cleaned_text, 
                metadata={"article": article_number, "source": "Qistas_Legal_Doc"}
            )
        )
        
    print(f"Extraction complete. Created {len(article_chunks)} clean article chunks.")
    return article_chunks

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    """
    Creates a unique ID for each chunk based on its source and article number.
    Example output: 'Qistas_Legal_Doc:Article_12'
    """
    for chunk in chunks:
        source = chunk.metadata.get("source")
        article = chunk.metadata.get("article")
        
        # Create a clean, perfectly unique ID
        chunk_id = f"{source}:Article_{article}"
        
        # Add it to the chunk metadata
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        # Define a helper function to handle read-only files
        def remove_readonly(func, path, _):
            # Change the file permission to writeable
            os.chmod(path, stat.S_IWRITE)
            # Try the operation again
            func(path)
            
        print(f"Deleting Chroma directory at {CHROMA_PATH}...")
        
        # Pass the helper function to the onerror argument
        shutil.rmtree(CHROMA_PATH, onerror=remove_readonly)
        
        print("Database cleared successfully.")


if __name__ == "__main__":
    populate_database()