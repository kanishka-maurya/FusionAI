from backend.core.logging import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from crewai import LLM 
from backend.core.exceptions import CustomException
import sys
import os
from services.research_service.embeddings.embedding_generator import EmbeddingGenerator
from services.research_service.vector_database.vector_database import ChromaVectorDatabase


@dataclass
class RAGResult:
    """Represents the result of RAG generation with citations"""
    query: str
    response: str
    sources_used: List[Dict[str, Any]]
    retrieval_count: int
    generation_tokens: Optional[int] = None

    def get_citation_summary(self) -> str:
        if not self.sources_used:
            return "No sources cited"
        
        source_summary = []
        for source in self.sources_used:
            source_info = f"•{source.get('source_file', 'Unknown')} ({source.get('source_type', 'unknown')})"
            if source.get('page_number'):
                source_info += f" - Page {source['page_number']}"
            source_summary.append(source_info)

        return "\n".join(source_summary)
    
class RAGGenerator:
    def __init__(self, 
                embedding_generator: EmbeddingGenerator, 
                vector_db: ChromaVectorDatabase,
                api_key: str,
                model_name: str = "groq/llama-3.1-8b-instant",
                temperature: float = 0.1,
                max_tokens: int = 2000):
        
        self.embedding_generator = embedding_generator
        self.ChromaDB = vector_db
        self.model_name = model_name

        self.llm = LLM(
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
        logging.info("RAG Generator initialized....")
        logging.info(f"RAG Generator initialized with {model_name}")
        
    def generate_results(self,
                query: str,
                memory: dict,
                max_chunks: int = 8,
                max_context_chars: int = 4000,
                top_k: int = 10,
            ) -> RAGResult:
        
        if not query.strip():
            logging.info("Empty query is recieved....")
            return RAGResult(
                query=query,
                response="Please provide a valid question.",
                sources_used=[],
                retrieval_count=0
            )
       
        
        try:
            logging.info(f"Generating response for: '{query[:50]}...'")
            
            query_vector = self.embedding_generator.generate_query_embedding(query)
            logging.info(f"query vector:{query_vector}")
            search_results =self.ChromaDB.search(
                query_vector=query_vector.tolist(),
                limit=top_k
            )
            print("search results",search_results)
            if not search_results:
                logging.info("Nothing retrieved from the vector DB....")
                return RAGResult(
                    query=query,
                    response="I couldn't find any relevant information in the available documents to answer your question.",
                    sources_used=[],
                    retrieval_count=0
                )
            
            logging.info("Retrieval results from vector DB are non-empty....")
            context, sources_info = self._format_context_with_citations(
                search_results, max_chunks, max_context_chars
            )
            
            prompt = self._create_rag_prompt(memory, context, query)
            
            try:
                response = self.llm.call(prompt)
                logging.info("Results obtained from LLM....")
                logging.info(response[:20])
            except Exception as e:
                error = CustomException(e, sys)
                logging.error(error)
                raise error

           
            rag_result = RAGResult(
                query=query,
                response=response,
                sources_used=sources_info,
                retrieval_count=len(search_results)
            )

            logging.info(f"Response generated successfully using {len(sources_info)} sources")
            return rag_result
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return RAGResult(
                query=query,
                response=f"I encountered an error while processing your question: {str(e)}",
                sources_used=[],
                retrieval_count=0
            )
            
    def _format_context_with_citations(
                self,
                search_results: List[Dict[str, Any]],
                max_chunks: int,
                max_context_chars: int
            ) -> Tuple[str, List[Dict[str, Any]]]:

                context_parts = []
                sources_info = []
                total_chars = 0
                
                for i, result in enumerate(search_results[:max_chunks]):
                    citation_info = result['citation']
                    source_file = citation_info.get('source_file', 'Unknown Source')
                    source_type = citation_info.get('source_type', 'unknown')
                    page_number = citation_info.get('page_number')
                    
                    citation_ref = f"[{i+1}]"
                    chunk_content = result['content']
                    chunk_text = f"{citation_ref} {chunk_content}"
                    
                    if total_chars + len(chunk_text) > max_context_chars and context_parts:
                        break
                    
                    context_parts.append(chunk_text)
                    total_chars += len(chunk_text)
                    
                    source_info = {
                        'reference': citation_ref,
                        'source_file': source_file,
                        'source_type': source_type,
                        'page_number': page_number,
                        'chunk_id': result['id'],
                        'relevance_score': result['score']
                    }
                    sources_info.append(source_info)
                
                formatted_context = '\n\n'.join(context_parts)

                return (formatted_context, sources_info)
    
    def _create_rag_prompt(self, memory: dict, context: str, query: str) -> str:
        prompt = f"""You are an AI assistant that answers questions based on provided source material. You must follow these citation rules:

CITATION REQUIREMENTS:
1. For each factual claim in your answer, include the citation reference number in square brackets [1], [2], etc.
2. Only use information from the provided context - do not add external knowledge
3. If you cannot find relevant information in the context, say so clearly
4. Be precise and accurate in your citations
5. When multiple sources support the same point, list all relevant citations like this [1], [2], [3].

CONTEXT (with citation references):
{context}

MEMORY: {memory}
QUESTION: {query}

Please provide a comprehensive answer with proper citations. Make sure every factual statement is supported by a citation reference."""
        
        return prompt
    


    def generate_summary(
        self,
        max_chunks: int = 15,
        summary_length: str = "medium"
    ) -> RAGResult:
        try:
            summary_query = "main topics key findings important information overview"
            query_vector = self.embedding_generator.generate_query_embedding(summary_query)
            search_results = self.ChromaDB.search(
                query_vector=query_vector.tolist(),
                limit=max_chunks
            )
            
            if not search_results:
                return RAGResult(
                    query="Document Summary",
                    response="No documents available for summarization.",
                    sources_used=[],
                    retrieval_count=0
                )
            
            context, sources_info = self._format_context_with_citations(
                search_results, max_chunks, 6000
            )
            
            length_instructions = {
                'short': "Provide a concise 2-3 paragraph summary highlighting the most important points.",
                'medium': "Provide a comprehensive 4-5 paragraph summary covering key topics and findings.",
                'long': "Provide a detailed summary with multiple sections covering all major topics and supporting details."
            }
            
            summary_prompt = f"""You are tasked with creating a summary of the provided document content. Follow these guidelines:

1. {length_instructions.get(summary_length, length_instructions['medium'])}
2. Include citations [1], [2], etc. for all factual claims
3. Organize information logically with clear topics
4. Focus on the most important and relevant information
5. Maintain accuracy and cite sources properly

DOCUMENT CONTENT (with citation references):
{context}

Please provide a well-structured summary with proper citations:"""
            
            response = self.llm.call(summary_prompt)
            
            return RAGResult(
                query="Document Summary",
                response=response,
                sources_used=sources_info,
                retrieval_count=len(search_results)
            )
            
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return RAGResult(
                query="Document Summary",
                response=f"Error generating summary: {str(e)}",
                sources_used=[],
                retrieval_count=0
            )

if __name__ == "__main__":
    import os
    from services.research_service.data_processing.doc_processing.doc_processor import DocumentProcessor
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Please set GROQ_API_KEY environment variable")
        exit(1)
    
    try:
        doc_processor = DocumentProcessor()
        embedding_gen = EmbeddingGenerator()
        vector_db = ChromaVectorDatabase()

        chunks = doc_processor.process_document(r"C:\Users\kanis\FusionAI\services\research_service\vector_database\CRAG Paper.pdf")
        embedded_chunks = embedding_gen.generate_embeddings(chunks)
        inserted_ids = vector_db.insert_embeddings(embedded_chunks)

        rag_generator = RAGGenerator(
            embedding_generator=embedding_gen,
            vector_db=vector_db,
            api_key=api_key,
            temperature=0.1
        )
        
        

        test_query = "What are the main findings discussed in the documents?"
        result = rag_generator.generate_results(test_query)
        
        print(f"Query: {result.query}")
        print(f"Response: {result.response}")
        print(f"\nSources Used ({len(result.sources_used)}):")
        print(result.get_citation_summary())
        
        summary_result = rag_generator.generate_summary(summary_length="medium")
        print(f"\nDocument Summary:")
        print(summary_result.response)
        
    except Exception as e:
        print(f"Error in RAG pipeline example: {e}")


