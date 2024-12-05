import numpy as np

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


def cosine_similarity(a, b):
    breakpoint()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(client, input):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=input,
    )
    return response.data[0].embedding
    
    
if __name__ == '__main__':
    client = OpenAI()
    doc = get_embedding(
        client, "The food was delicious and the waiter..."
    )
    query_a = get_embedding(
        client, "The pod was awful and the wait..."
    )
    query_b = get_embedding(
        client, "Delicious Food!"
    )
    cosine_similarity(doc, query_a)