import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Carrega o PDF
current_dir = Path(__file__).parent
pdf_path = current_dir / "gpt5.pdf"

print(f"Carregando PDF: {pdf_path}")
print(f"Arquivo existe: {pdf_path.exists()}")

try:
    docs = PyPDFLoader(str(pdf_path)).load()
    print(f"Número de páginas carregadas: {len(docs)}")
    
    if docs:
        print(f"Primeira página - tamanho do conteúdo: {len(docs[0].page_content)}")
        print(f"Primeiros 200 caracteres: {docs[0].page_content[:200]}")
        
        # Testa o text splitter
        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150, 
            add_start_index=False
        ).split_documents(docs)
        
        print(f"Número de chunks criados: {len(splits)}")
        
        if splits:
            print(f"Primeiro chunk - tamanho: {len(splits[0].page_content)}")
            print(f"Primeiro chunk - conteúdo: {splits[0].page_content[:200]}")
    else:
        print("❌ Nenhum documento foi carregado do PDF!")
        
except Exception as e:
    print(f"❌ Erro ao carregar PDF: {e}") 