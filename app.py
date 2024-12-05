import streamlit as st

import ai, trend
from integrate import fetch_recent_papers, format_docs


st.title("최근 Arxiv 논문 트렌드 요약")

# 한 번만 실행되는 papers 변수 
if 'trends' not in st.session_state:
    with st.spinner('논문 가져오는 중...'):
        papers = list(fetch_recent_papers(max_results=3))
    with st.spinner('트렌드 분석하는 중...'):
        st.session_state.trends = trend.get_chain().invoke(
            {'context': format_docs(papers)}
        )
    with st.spinner('벡터 데이터베이스 초기화하는 중...'):
        st.session_state.vectorstore = ai.get_vectorstore()
        ai.load_pdf_texts(st.session_state.vectorstore, papers)

if st.session_state.trends:
    # 요약 기능
    st.subheader("최근 트렌드 요약")
    st.markdown(st.session_state.trends)
    # 질의 기능
    query = st.text_input("최근 트렌드에 대한 질문이 있으신가요?")
    if st.button("알아보기"):
        if query:
            answer = ai.get_chain(st.session_state.vectorstore).invoke(query)
            # answer = ai.get_chain(st.session_state.vectorstore).stream(query)
            st.markdown(answer)
