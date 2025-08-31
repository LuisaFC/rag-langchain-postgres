import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

# Verificar variáveis de ambiente
for k in ("GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

class SemanticSearch:
    def __init__(self):
        # Configurar embeddings (mesmo modelo usado na ingestão)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Conectar ao banco vetorial
        self.store = PGVector(
            embeddings=self.embeddings,
            collection_name=os.getenv("PGVECTOR_COLLECTION"),
            connection=os.getenv("PGVECTOR_URL"),
            use_jsonb=True,
        )
    
    def search(self, query: str, k: int = 10):
        """
        Busca semântica no banco vetorial
        
        Args:
            query (str): Pergunta do usuário
            k (int): Número de resultados mais relevantes (padrão: 10)
            
        Returns:
            list: Lista de tuplas (documento, score)
        """
        try:
            # Buscar documentos similares com score
            results = self.store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Erro na busca: {e}")
            return []
    
    def get_context(self, query: str, k: int = 10):
        """
        Busca e concatena o contexto para o LLM
        
        Args:
            query (str): Pergunta do usuário
            k (int): Número de resultados
            
        Returns:
            str: Contexto concatenado dos documentos mais relevantes
        """
        results = self.search(query, k)
        
        if not results:
            return ""
        
        # Concatenar o conteúdo dos documentos
        context_parts = []
        for doc, score in results:
            context_parts.append(doc.page_content)
        
        return "\n\n".join(context_parts)

# Função auxiliar para teste
def test_search():
    """Função para testar a busca"""
    searcher = SemanticSearch()
    
    test_query = "Qual o faturamento da empresa?"
    print(f"Testando busca para: '{test_query}'")
    
    results = searcher.search(test_query, k=3)
    
    if results:
        print(f"\nEncontrados {len(results)} resultados:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n--- Resultado {i} (Score: {score:.4f}) ---")
            print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    else:
        print("Nenhum resultado encontrado.")

if __name__ == "__main__":
    test_search()
