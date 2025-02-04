# Policy Summarizer using LLM & RAG with HuggingFace

A question-answering system that uses Language Models and Retrieval Augmented Generation (RAG) to analyze and answer questions about company policies.

## Overview

This project implements a policy document analyzer that can:
- Process and understand company policy documents
- Split documents into manageable chunks for analysis
- Create embeddings for semantic search
- Answer questions about policies using a combination of retrieval and language model generation
- Maintain conversation context for follow-up questions

## Technical Components

### Document Processing
- Uses `TextLoader` from LangChain to load policy documents
- Implements `CharacterTextSplitter` to break documents into digestible chunks
- Chunk size: 900 characters with no overlap

### Embeddings and Vector Storage
- Utilizes HuggingFace embeddings for text vectorization
- Stores vectors in ChromaDB for efficient retrieval
- Enables semantic search capabilities

### Language Model
- Uses OpenAI GPT-2 model through HuggingFace
- Implements text generation pipeline with configurable parameters
- Maximum new tokens: 100

### Question-Answering System
- Employs RetrievalQA and ConversationalRetrievalChain from LangChain
- Maintains conversation history using ConversationBufferMemory
- Custom prompt template for consistent and contextual responses

## Dependencies

```
langchain
langchain-openai
huggingface
huggingface-hub
sentence-transformers
chromadb
wget==3.2
openai
langchain-community
bitsandbytes
transformers
```

## Features

1. **Document Loading**: Ability to load and process text documents containing company policies

2. **Semantic Search**: Efficient retrieval of relevant policy sections based on questions

3. **Contextual Understanding**: Maintains conversation history for more coherent follow-up responses

4. **Custom Prompting**: Template-based prompting for consistent response formats

5. **Interactive Interface**: Command-line interface for real-time Q&A about policies

## Usage

1. Load your policy document:
```python
filename = 'companyPolicies.txt'
loader = TextLoader(filename)
documents = loader.load()
```

2. Initialize the QA system:
```python
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    memory=memory,
    get_chat_history=lambda h: h,
    return_source_documents=False
)
```

3. Ask questions about policies:
```python
query = "What is the company's smoking policy?"
result = qa({"question": query}, {"chat_history": history})
```

## Limitations

- Model responses are based on GPT-2, which may not always provide complete or accurate answers
- Performance depends on the quality and structure of input policy documents
- Limited to text-based policy documents
- Response generation time may vary based on document size and complexity

## Future Improvements

1. Integration with more advanced language models
2. Enhanced document preprocessing capabilities
3. Support for different document formats
4. Improved response accuracy and consistency
5. Web-based user interface
6. Multi-language support
7. Document version tracking

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

SKR
