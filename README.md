# 🤖 Ingestão e Busca Semântica com LangChain e PostgreSQL

Sistema de RAG (Retrieval-Augmented Generation) que permite fazer ingestão de documentos PDF e realizar buscas semânticas com respostas baseadas exclusivamente no conteúdo dos documentos.

## 📋 Funcionalidades

- **Ingestão de PDF**: Lê arquivos PDF e armazena em banco vetorial PostgreSQL com pgVector
- **Busca Semântica**: Encontra informações relevantes usando embeddings
- **Chat Interativo**: Interface CLI para fazer perguntas sobre o conteúdo do PDF
- **Respostas Contextualizadas**: Respostas baseadas apenas no conteúdo do documento

## 🛠️ Tecnologias Utilizadas

- **Python 3.10+**
- **LangChain** - Framework para aplicações com LLM
- **PostgreSQL + pgVector** - Banco de dados vetorial
- **Google Gemini** - Embeddings e LLM
- **Docker & Docker Compose** - Containerização do banco

## 📁 Estrutura do Projeto

```
├── docker-compose.yaml   # Configuração PostgreSQL + pgVector
├── requirements.txt      # Dependências Python
├── .env.example         # Template das variáveis de ambiente
├── src/
│   ├── ingest.py        # Script de ingestão do PDF
│   ├── search.py        # Módulo de busca semântica
│   └── chat.py          # Interface CLI interativa
├── investimentos.pdf            # Documento PDF para ingestão
└── README.md           # Este arquivo
```

## 🚀 Instalação e Configuração

### 1. Clonar o Repositório

```bash
git clone <url-do-repositorio>
cd desafioLangchain
```

### 2. Criar Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar Variáveis de Ambiente

Copie o arquivo `.env.example` para `.env` e configure suas chaves:

```bash
cp .env.example .env
```

Edite o arquivo `.env`:

```env
# Configurações do Banco de Dados PostgreSQL
PGVECTOR_URL=postgresql://postgres:postgres@localhost:5432/rag
PGVECTOR_COLLECTION=pdf_documents

# API Key do Google Gemini
GOOGLE_API_KEY=sua_api_key_do_gemini_aqui
```

### 5. Obter API Key do Google Gemini

1. Acesse [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crie uma nova API Key
3. Cole a chave no arquivo `.env`

## 🐳 Executando o Sistema

### 1. Subir o Banco de Dados

```bash
docker compose up -d
```

Isso irá:
- Subir PostgreSQL com extensão pgVector
- Configurar PgAdmin (opcional) em http://localhost:8081
- Criar a extensão vector automaticamente

### 2. Executar Ingestão do PDF

```bash
python src/ingest.py
```

Este comando irá:
- Carregar o arquivo `investimentos.pdf`
- Dividir em chunks de 1000 caracteres (overlap 150)
- Gerar embeddings usando Gemini
- Armazenar no banco vetorial PostgreSQL

### 3. Iniciar o Chat Interativo

```bash
python src/chat.py
```

## 💬 Exemplos de Uso

### Perguntas Dentro do Contexto
```
Faça sua pergunta: Qual o faturamento da empresa?
🔍 Buscando informações...
📋 RESPOSTA: O faturamento foi de 10 milhões de reais.
```

### Perguntas Fora do Contexto
```
Faça sua pergunta: Qual é a capital da França?
🔍 Buscando informações...
📋 RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## 🔧 Testando Componentes Individualmente

### Testar Busca Semântica
```bash
python src/search.py
```

### Verificar Conexão com Banco
```bash
docker exec -it postgres_rag_desafio psql -U postgres -d rag -c "\dt"
```

## 📊 Monitoramento

### PgAdmin (Opcional)
- URL: http://localhost:8081
- Email: admin@admin.com
- Senha: admin

### Logs do Docker
```bash
docker compose logs -f postgres
```

## ⚙️ Configurações Avançadas

### Ajustar Parâmetros de Busca

No arquivo `src/search.py`, você pode modificar:

```python
# Número de resultados retornados (padrão: 10)
results = self.store.similarity_search_with_score(query, k=10)
```

### Ajustar Chunking

No arquivo `src/ingest.py`:

```python
splits = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Tamanho do chunk
    chunk_overlap=150     # Sobreposição entre chunks
)
```

### Trocar Modelo de LLM

No arquivo `src/chat.py`:

```python
self.llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Ou outro modelo
    temperature=0,             # Criatividade (0-1)
)
```

## 🐛 Solução de Problemas

### Erro de Conexão com Banco
```bash
# Verificar se containers estão rodando
docker compose ps

# Reiniciar containers
docker compose down && docker compose up -d
```

### Erro de API Key
```bash
# Verificar se variáveis estão carregadas
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GOOGLE_API_KEY'))"
```

### Erro de Dependências
```bash
# Reinstalar dependências
pip install --upgrade -r requirements.txt
```

## 📝 Requisitos Atendidos

- ✅ **Ingestão**: PDF dividido em chunks de 1000 caracteres com overlap de 150
- ✅ **Embeddings**: Usando `models/embedding-001` do Gemini
- ✅ **Banco Vetorial**: PostgreSQL com pgVector
- ✅ **Busca**: `similarity_search_with_score(query, k=10)`
- ✅ **LLM**: Gemini 1.5 Flash para respostas
- ✅ **Prompt Template**: Implementado conforme especificação
- ✅ **CLI**: Interface interativa no terminal
- ✅ **Docker**: PostgreSQL rodando em container

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👨‍💻 Autor

Desenvolvido como parte do desafio de Ingestão e Busca Semântica com LangChain e PostgreSQL.

---

**🚀 Pronto para usar! Execute os comandos na ordem e comece a fazer perguntas sobre seu PDF!**
