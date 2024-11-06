from typing import List
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Define a rating score model
class RatingScore(BaseModel):
    relevance_score = Field(..., description="The relevance score of a document to a query.")

# Reranking function
def rerank_documents(query, docs, top_n = 3):
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""On a scale of 1-10, rate the relevance of the following document to the query. 
        Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {doc}
        Relevance Score:"""
    )
    
    # Use the GPT-2 model (adjust as needed)
    llm = ChatOpenAI(temperature=0, model_name="gpt-2", max_tokens=4000)
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)
    
    scored_docs = []
    for doc in docs:
        input_data = {"query": query, "doc": doc.page_content}
        score = llm_chain.invoke(input_data).relevance_score
        try:
            score = float(score)
        except ValueError:
            score = 0  # Default score if parsing fails
        scored_docs.append((doc, score))
    
    # Sort documents based on scores and return top_n
    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]

# Retrieve passages with reranking
def retrieve_passages(query, num_docs, vectorstore):
    # Step 1: Initial retrieval
    initial_docs = vectorstore.similarity_search(query, k=num_docs * 5)  # Fetch more docs initially
    
    # Step 2: Rerank the documents based on relevance using LLM
    reranked_docs = rerank_documents(query, initial_docs, top_n=num_docs)
    
    return reranked_docs


# Example usage after applying the above code to the PDF document or the data we have.
query = "" # THE INPUT QUERY

# Need to check about the type of documents we have
vectorstore = encode_pdf("data.pdf") # Vectorstore of PDF documents
retrieved_docs = retrieve_passages(query, num_docs=3, vectorstore=vectorstore)

# Print top 3 reranked documents and first 200 characters of each
for i, doc in enumerate(retrieved_docs):
    print(f"\nDocument {i+1}:")
    print(doc.page_content[:200] + "...")
