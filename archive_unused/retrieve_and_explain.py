import os
import sys
from dotenv import load_dotenv

# Load environment variables from a .env file for the OpenAI API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Set up the system path to import helper functions and evaluation tools
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper_functions import *
from evaluation.evaluate_rag import *

# Import necessary modules for the ExplainableRetriever class
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import ChatOpenAI
from langchain.prompts import PromptTemplate

# Define the ExplainableRetriever class
class ExplainableRetriever:
    """
    The ExplainableRetriever class performs retrieval and provides explanations
    for the relevance of each document. It combines vector-based search with
    language model-generated explanations for enhanced transparency.
    """
    
    def __init__(self, texts):
        """
        Initializes the ExplainableRetriever with a list of documents (texts).
        
        Args:
            texts (list): List of strings, each representing a document.
        """
        # Initialize the embedding model to convert text into vector representations
        self.embeddings = OpenAIEmbeddings()
        
        # Create a FAISS vector store from the text embeddings for efficient similarity search
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        
        # Initialize the language model (LLM) for generating natural language explanations
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
        
        # Set up a base retriever from the vector store, returning top 5 similar documents
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Define a prompt for generating explanations with the LLM
        explain_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Analyze the relationship between the following query and the retrieved context.
            Explain why this context is relevant to the query and how it might help answer the query.
            
            Query: {query}
            
            Context: {context}
            
            Explanation:
            """
        )
        # Combine the explanation prompt with the LLM to create an explanation chain
        self.explain_chain = explain_prompt | self.llm

    def retrieve_and_explain(self, query):
        """
        Retrieves documents relevant to the query and generates an explanation
        for each retrieved document.
        
        Args:
            query (str): The search query.
        
        Returns:
            List[dict]: List of dictionaries, each containing the document content
                        and its explanation of relevance.
        """
        # Retrieve relevant documents using the base retriever
        docs = self.retriever.get_relevant_documents(query)
        
        explained_results = []
        
        # For each retrieved document, generate an explanation of its relevance to the query
        for doc in docs:
            input_data = {"query": query, "context": doc.page_content}
            explanation = self.explain_chain.invoke(input_data).content
            
            # Append the document content and explanation to the results
            explained_results.append({
                "content": doc.page_content,
                "explanation": explanation
            })
        
        return explained_results

# Example usage
texts = [
    "The sky is blue because of the way sunlight interacts with the atmosphere.",
    "Photosynthesis is the process by which plants use sunlight to produce energy.",
    "Global warming is caused by the increase of greenhouse gases in Earth's atmosphere."
]

# Initialize the ExplainableRetriever with the example documents
explainable_retriever = ExplainableRetriever(texts)

# Define a query for testing
query = "Why is the sky blue?"

# Retrieve and explain the results for the query
results = explainable_retriever.retrieve_and_explain(query)

# Display each retrieved document with its generated explanation
for i, result in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"Content: {result['content']}")
    print(f"Explanation: {result['explanation']}")
    print()
