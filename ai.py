from typing import List

from dotenv import load_dotenv
from arxiv import Result
from langchain_core.documents.base import Document
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_core.runnables import RunnablePassthrough

from pdf import extract_text_from_pdf, split_pdf


load_dotenv()


def get_vectorstore():
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    # return PineconeVectorStore(index_name='arxiv', embedding=embeddings)
    return InMemoryVectorStore(embedding=embeddings)


def load_pdf_texts(vectorstore, papers: List[Result]):
    # vectorstore.delete(delete_all=True)
    for paper in papers:
        print(paper.pdf_url)
        texts, metadatas, ids = [], [], []   
        for i, text in enumerate(split_pdf(paper.pdf_url)):
            texts.append(text)
            metadatas.append({
                'title': paper.title,
                'pdf_url': paper.pdf_url,
                'category': paper.primary_category,
                'authors': ", ".join(author.name for author in paper.authors),
            })
            ids.append(paper.pdf_url.split('/')[-1] + f':{i}')
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def format_docs(docs: List[Document]):
    return "\n\n".join(
        f'Context:{doc.page_content}\n'
        f'PDF:{doc.metadata["pdf_url"]}\n'
        f'Title:{doc.metadata["title"]}\n'
        f'Authors:{doc.metadata["authors"]}\n'
        f'Category:{doc.metadata["category"]}'
        for doc in docs
    )


def get_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate([
        ("system", """
        당신은 최신 연구 논문 분석에 능통한 논문 리뷰 전문가입니다.
        당신의 임무는 최신 논문들을 바탕으로 연구 트렌드를 파악하고,
        사용자의 질문에 대한 심도 있는 답변을 제공하는 것입니다.
        
        **제공된 논문 정보를 바탕으로 다음과 같은 내용을 포함하여 답변해 주세요:**
        1. 질문에 대한 구체적인 답변
        2. 논문 간의 연관성 (필요 시 인용)
        3. 논문에서 발견된 새로운 연구 동향이나 중요한 발견
        4. 관련된 연구 주제나 추가적인 참고 자료

        **제공된 논문 정보:**
        '''{context}'''

        사용자의 질문:
        {question}
        
        **출력 형식:**
        - **답변**: 질문에 대한 명확한 답변을 서술합니다.
        - **논문 인용**: 관련 논문을 인용하여 논문 제목, 저자, 링크를 포함합니다.
        - **연구 동향**: 분석된 논문들로부터 추출한 연구 트렌드와 주요 발견 사항을 요약합니다.
        """)
    ])
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


if __name__ == '__main__':
    from integrate import fetch_recent_papers
    vectorstore = get_vectorstore()
    load_pdf_texts(vectorstore, fetch_recent_papers(max_results=5))
    ai_response = get_chain(vectorstore).invoke('토러스 임베딩에대해 알려주세요.')