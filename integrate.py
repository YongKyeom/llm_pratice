import arxiv
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()


def fetch_recent_papers(days=7, max_results=30):
    """
    Arxiv API 검색 쿼리 (최근 일주일 논문)
    https://groups.google.com/g/arxiv-api/c/mAFYT2VRpK0?pli=1
    """
    # 최근 일주일 날짜 계산
    today = datetime.now(tz=timezone.utc)
    past_date = today - timedelta(days=days)
    
    # 날짜 형식 변환 (YYYYMMDDHHMM)
    start_date = past_date.strftime("%Y%m%d%H%M")
    end_date = today.strftime("%Y%m%d%H%M")

    # query = f"submittedDate:[202411141859 TO 202411141900]"
    query = f"submittedDate:[{start_date} TO {end_date}]"
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    return arxiv.Client().results(search)


def format_docs(papers):
    """
    논문 목록을 포맷팅하여 LLM에 전달할 텍스트로 변환
    """
    return "\n\n".join(
        f'제목: {paper.title}\n'
        f'저자: {", ".join(author.name for author in paper.authors)}\n'
        f'요약: {paper.summary}\n'
        f'PDF 링크: {paper.pdf_url}'
        for paper in papers
    )


if __name__ == '__main__':
    for paper in fetch_recent_papers(max_results=3):
        print(vars(paper).keys())
