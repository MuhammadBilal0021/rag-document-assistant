FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org

COPY requirements.txt .

RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    fastapi uvicorn python-multipart

RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    pydantic>=2.7.4 pydantic-settings python-dotenv requests groq

RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    langchain==0.3.0 langchain-groq langchain-community langchain-text-splitters

RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    'numpy<2.0.0' faiss-cpu pymupdf python-docx beautifulsoup4

RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    sentence-transformers

COPY . .
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
