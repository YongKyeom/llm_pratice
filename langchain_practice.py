import getpass
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0)

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

# Model 직접 Invoke
model.invoke(messages)
model.invoke("Hello")
model.invoke([{"role": "user", "content": "Hello"}])
model.invoke([HumanMessage("Hello")])


# Template 사용 예제
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
response = model.invoke(prompt)

# chaining 예제 
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
chain = prompt | model
response = chain.invoke({'context': '논문검색 테스트'})