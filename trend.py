from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


def get_chain():
    # ChatGPT 모델 설정
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )

    # 프롬프트 템플릿 정의
    prompt = ChatPromptTemplate([
        ("system", """
        당신은 대학 교수입니다. 최신 논문을 분석하고 연구 트렌드를 파악하는 전문가입니다.

        최근 발행된 논문을 제공해 드립니다.
        아래 제공된 논문 요약을 바탕으로 최근 연구의 핵심 트렌드, 주요 발견,
        그리고 향후 연구 방향에 대해 한국어로 분석해 주세요.
        가능한 한 구체적인 예시를 언급해 주세요.
        각 논문에 대한 출처를 함께 명시해 주세요.

        논문 정보: '''{context}'''
        """)
    ])

    # LLM 체인 구성
    llm_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return llm_chain


if __name__ == '__main__':
    from integrate import fetch_recent_papers, format_docs

    # 질문 입력 및 요약 생성
    papers = fetch_recent_papers(max_results=5)
    trend_summary = get_chain().invoke(
        {"context": format_docs(papers)}
    )
    print(trend_summary)