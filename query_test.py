import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

print("🔍 Configurando sistema de busca...")

# Configura embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Conecta ao banco
store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

print("✅ Conexão estabelecida!")

# Testa algumas consultas
queries = [
    "O que é GPT-5?",
    "Quais são as capacidades do modelo?",
    "Como funciona o treinamento?",
    "Questões de segurança e ética"
]

print("\n🔎 Testando consultas semânticas:")
print("=" * 60)

for query in queries:
    print(f"\n📝 Pergunta: {query}")
    print("-" * 40)
    
    # Busca os documentos mais similares
    results = store.similarity_search(query, k=2)
    
    for i, doc in enumerate(results, 1):
        print(f"\n📄 Resultado {i}:")
        print(f"Página: {doc.metadata.get('page', 'N/A')}")
        print(f"Conteúdo: {doc.page_content[:200]}...")
        
print("\n🎉 Teste de consultas concluído!") 