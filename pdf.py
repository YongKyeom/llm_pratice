from io import BytesIO
import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_pdf(pdf_url):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
    )
    docs = extract_text_from_pdf(pdf_url)
    yield from text_splitter.split_text(docs)


def extract_text_from_pdf(url):
    response = requests.get(url)
    with BytesIO(response.content) as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


if __name__ == '__main__':
    # 논문 선택 후 텍스트 추출
    url = 'http://arxiv.org/pdf/2411.09702v1'
    for text in split_pdf(url):
        breakpoint()