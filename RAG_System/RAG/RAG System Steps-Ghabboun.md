**RAG System Steps:**



**langchain -> LLM Library**

**chromadb -> Vector Storage**

**pypdf -> Loading PDFs**

**pytest -> Unit Testing**



**1)Save the required data as documents.**



**2)Load the documents into python using:
DirectoryLoader module from langchain.**



**3)It is better to split the original data into chunks before saving them as document, so that each chunk is more specific and relevant, using:**

**RecursiveCharacterTextSplitter, specify chunk size and chunk overlap in characters.**



**4)Use ChromaDB database to save the chunks as Vector Embeddings**



**5)To create an embedding from a chunk, use:**

**OpenAIEmbeddings().embed\_query(chunk)**



**6)To find the distance between two vectors you need to first load an "evaluator" (a utility function, that will help you find the distances between embeddings):**

**load\_evaluator("pairwise\_embedding\_distance")**

**then use this evaluator to calculate the distance:**

**evaluator.evaluate\_string\_pairs(str1, str2)**



**7)Prepare the DB, then start searching for the most relevant chunks, using:**

**.similarity\_search\_with\_relevance\_scores(query\_text, k)**



**8)Create the prompt, containing the context (relevant chunks), and the actual question to answer, then send it to an LLM to format the final answer.**

