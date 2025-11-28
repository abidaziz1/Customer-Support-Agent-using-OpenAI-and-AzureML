import faiss
import numpy as np

from typing import Tuple, List

import pandas as pd

from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI

API_KEY = 'YOUR KEY HERE'

def create_embeddings(df: pd.DataFrame, column_name: str, model: str) -> np.ndarray:
    """
    Purpose: Generates vector embeddings for text data in a DataFrame column.
Steps:
Loads the OpenAI embedding model.
Applies the embedding model to each entry in the specified column.
Stores the resulting vectors in a new column.
Stacks all vectors into a NumPy array and returns it.    """
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY, model=model)
    #encode the text data in the specified column using the sentence transformer model
    df[f"{column_name}_vector"] = df[column_name].apply(lambda x: embeddings.embed_query(x))
    #stack the encoded vectors into a NumPy array
    vectors = np.stack(df[f"{column_name}_vector"].values)
    
    return vectors

def create_index(vectors: np.ndarray, index_file_path: str) -> faiss.Index:
    """
    Purpose: Creates a FAISS index from the embeddings and saves it.
Steps:
Determines the vector dimension.
Initializes a FAISS index using L2 distance.
Adds vectors to the index.
Saves the index to disk.
Returns the index object.   """
    #get the dimension of the vectors
    dimension = vectors.shape[1]
    #create a FAISS index with L2 distance metric (cosine similarity)
    index = faiss.IndexFlatL2(dimension)
    #add the vectors to the index
    index.add(vectors)
    #save the index to a file
    faiss.write_index(index, index_file_path)
    print("FAISS index is created and vectors are added to the index.")

    return index

def semantic_similarity(query: str, index: faiss.Index, model: str, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Purpose: Finds the top-k most similar vectors to a query.
Steps:
Embeds the query using the specified model.
Searches the FAISS index for the k most similar vectors.
Returns distances and indices of the matches.    """
    model = OpenAIEmbeddings(openai_api_key=API_KEY, model=model)
    #embed the query
    query_vector = model.embed_query(query)
    query_vector = np.array([query_vector]).astype('float32')
    #search the FAISS index
    D, I = index.search(query_vector, k)
    
    return D, I

def call_llm(query: str, responses: List[str]) -> str:
    """
    Purpose: Uses OpenAI’s chat model to generate a response based on the query and example responses.
Steps:
Initializes the OpenAI client.
Prepares a system prompt and user message, including instructions for urgency, categorization, and response generation.
Calls the chat completion API.
Returns the generated response.    """
    #assuming your KEY is saved in your environment variable as described in the Readme
    client = OpenAI(api_key=API_KEY)

    messages = [
        {"role": "system", "content": "You are a helpful assistant and help answer customer's query and request. Answer on the basis of provided context only."},
        {"role": "user", "content": f'''On the basis of the input customer query determine or suggest the following things about the input query: {query}:
                                          1. Urgency of the query based on the input query on a scale of 1-5 where 1 is least urgent and 5 is most urgent. Only tell me the number.
                                          2. Categorize the input query into sales, product, operations etc. Only tell me the category.
                                          3. Generate 1 best humble response to the input query which is similar to examples in the python list: {responses} from the internal database and is helpful to the customer.
                                          If the input query form customer is not clear then ask a follow up question.                  
                                      '''}
    ]
    response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0)
    
    return response.choices[0].message.content
