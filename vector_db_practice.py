import os

from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_core.vectorstores.in_memory import InMemoryVectorStore


load_dotenv()


def get_embedding(client, input):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=input,
    )
    return response.data[0].embedding
    

if __name__ == '__main__':
    ## Pinecone 연동 예시
    ai_client = OpenAI()
    db_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = db_client.Index('arxiv')
    docs = "The food was delicious and the waiter..."
    upsert_response = index.upsert(
        vectors=[{
            'id': 'doc',
            'values': get_embedding(ai_client, docs),
            'metadata': {'category': 'food'},
        }]
    )
    
    response_a = index.query(
        vector=get_embedding(ai_client, "The pod was awful and the wait..."),
        top_k=1,
    )
    response_b = index.query(
        vector=get_embedding(ai_client, "Delicious Food!"),
        top_k=1,
    )

    ## InMemoryVectorStore 연동 예시
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    # vectorstore = PineconeVectorStore(index_name='arxiv', embedding=embeddings)
    vectorstore = InMemoryVectorStore(embedding=embeddings)
    docs = "The food was delicious and the waiter..."
    response = vectorstore.add_texts(
        texts=[docs],
        metadatas=[{'category': 'food'}],
        ids=['doc'],
    )

    response_a = vectorstore.similarity_search_with_score(
        query="The pod was awful and the wait...",
        k=1,
    )
    response_b = vectorstore.similarity_search_with_score(
        query="Delicious Food!",
        k=1,
)