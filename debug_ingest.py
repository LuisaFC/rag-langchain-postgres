import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

print("🔍 Verificando variáveis de ambiente...")
for k in ("GOOGLE_API_KEY", "PGVECTOR_URL","PGVECTOR_COLLECTION"):
    value = os.getenv(k)
    if not value:
        print(f"❌ Variável {k} não está definida!")
        exit(1)
    else:
        if k == "GOOGLE_API_KEY":
            print(f"✅ {k}: {value[:10]}...")  # Mostra apenas os primeiros caracteres
        else:
            print(f"✅ {k}: {value}")

print("\n📄 Carregando PDF...")
current_dir = Path(__file__).parent
pdf_path = current_dir / "gpt5.pdf"

docs = PyPDFLoader(str(pdf_path)).load()
print(f"✅ PDF carregado: {len(docs)} páginas")

print("\n✂️ Dividindo em chunks...")
splits = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=150, add_start_index=False).split_documents(docs)

print(f"✅ Chunks criados: {len(splits)}")

if not splits:
    print("❌ Nenhum chunk foi criado!")
    exit(1)

print("\n📝 Enriquecendo documentos...")
enriched = [
    Document(
        page_content=d.page_content,
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
    )
    for d in splits
]    

print(f"✅ Documentos enriquecidos: {len(enriched)}")

# Cria os ids para os documentos
ids = [f"doc-{i}" for i in range(len(enriched))]
print(f"✅ IDs criados: {len(ids)}")

print("\n🤖 Configurando embeddings...")
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("✅ Embeddings configurados")
except Exception as e:
    print(f"❌ Erro ao configurar embeddings: {e}")
    exit(1)

print("\n🗄️ Conectando ao banco...")
try:
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION"),
        connection=os.getenv("PGVECTOR_URL"),
        use_jsonb=True,
    )
    print("✅ Conexão com banco estabelecida")
except Exception as e:
    print(f"❌ Erro ao conectar com banco: {e}")
    exit(1)

print("\n💾 Adicionando documentos ao banco...")
try:
    # Vamos testar com apenas os primeiros 5 documentos para debug
    test_docs = enriched[:5]
    test_ids = ids[:5]
    
    print(f"🧪 Testando com {len(test_docs)} documentos...")
    store.add_documents(documents=test_docs, ids=test_ids)
    print("✅ Documentos adicionados com sucesso!")
    
except Exception as e:
    print(f"❌ Erro ao adicionar documentos: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n🎉 Ingestão concluída com sucesso!") 