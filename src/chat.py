import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from search import SemanticSearch

load_dotenv()

# Verificar variáveis de ambiente
for k in ("GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

class RAGChat:
    def __init__(self):
        # Configurar o LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        # Configurar busca semântica
        self.searcher = SemanticSearch()
        
        # Template do prompt obrigatório
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO\""""
        )
    
    def get_answer(self, question: str):
        """
        Processa uma pergunta e retorna a resposta baseada no contexto
        
        Args:
            question (str): Pergunta do usuário
            
        Returns:
            str: Resposta do LLM baseada no contexto
        """
        try:
            # Buscar contexto relevante
            context = self.searcher.get_context(question, k=10)
            
            if not context:
                return "Não tenho informações necessárias para responder sua pergunta."
            
            # Montar o prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # Chamar o LLM
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            print(f"Erro ao processar pergunta: {e}")
            return "Desculpe, ocorreu um erro ao processar sua pergunta."
    
    def start_chat(self):
        """Inicia o chat interativo"""
        print("=" * 60)
        print("🤖 CHAT RAG - Pergunte sobre o conteúdo do PDF")
        print("=" * 60)
        print("Digite 'sair' para encerrar o chat\n")
        
        while True:
            try:
                # Receber pergunta do usuário
                question = input("Faça sua pergunta: ").strip()
                
                # Verificar se quer sair
                if question.lower() in ['sair', 'exit', 'quit', '']:
                    print("\n👋 Chat encerrado. Até logo!")
                    break
                
                # Processar pergunta
                print("\n🔍 Buscando informações...")
                answer = self.get_answer(question)
                
                # Exibir resposta
                print(f"\n📋 RESPOSTA: {answer}")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat encerrado. Até logo!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")
                print("-" * 60)

def main():
    """Função principal"""
    try:
        chat = RAGChat()
        chat.start_chat()
    except Exception as e:
        print(f"Erro ao inicializar o chat: {e}")

if __name__ == "__main__":
    main()
