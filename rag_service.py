import os
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import structlog
import time

# Document processing
import PyPDF2
from docx import Document
import pandas as pd
import openpyxl
import numpy as np

# Vector operations
import faiss
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument

# Ollama integration
import ollama

logger = structlog.get_logger()

class RAGService:
    """RAG service for document processing and retrieval"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.embedding_model = "nomic-embed-text:v1.5"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        self.vector_store = None
        self.documents = {}
        self.document_embeddings = {}
        
        # Supported file types
        self.supported_extensions = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.txt': self._process_txt,
            '.csv': self._process_csv,
            '.xlsx': self._process_xlsx,
            '.xls': self._process_xlsx
        }
    
    async def initialize(self):
        """Initialize the RAG service"""
        try:
            # Test Ollama connection
            ollama.list()
            logger.info("RAG service initialized successfully")
            return True
        except Exception as e:
            logger.error("Failed to initialize RAG service", error=str(e))
            return False
    
    def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}", error=str(e))
            return ""
    
    def _process_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}", error=str(e))
            return ""
    
    def _process_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error processing TXT {file_path}", error=str(e))
            return ""
    
    def _process_csv(self, file_path: str) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df.to_string(index=False)
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}", error=str(e))
            return ""
    
    def _process_xlsx(self, file_path: str) -> str:
        """Extract text from Excel file"""
        try:
            df = pd.read_excel(file_path)
            return df.to_string(index=False)
        except Exception as e:
            logger.error(f"Error processing Excel {file_path}", error=str(e))
            return ""
    
    async def process_document(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """Process uploaded document and create embeddings"""
        try:
            # Get file extension
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Extract text
            text = self.supported_extensions[file_ext](file_path)
            
            if not text.strip():
                raise ValueError("No text content extracted from document")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Store document metadata
            self.documents[doc_id] = {
                "id": doc_id,
                "name": file_name,
                "path": file_path,
                "extension": file_ext,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "total_length": len(text),
                "uploaded_at": time.time()
            }
            
            # Generate embeddings for chunks
            embeddings = await self._generate_embeddings(chunks)
            
            # Store embeddings
            self.document_embeddings[doc_id] = embeddings
            
            logger.info(f"Document processed successfully", 
                       doc_id=doc_id, 
                       file_name=file_name, 
                       chunks=len(chunks))
            
            return {
                "success": True,
                "document_id": doc_id,
                "file_name": file_name,
                "chunk_count": len(chunks),
                "total_length": len(text)
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_name}", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama"""
        try:
            embeddings = []
            for text in texts:
                # Use Ollama for embeddings
                response = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=text
                )
                embeddings.append(response['embedding'])
            
            return embeddings
            
        except Exception as e:
            logger.error("Error generating embeddings", error=str(e))
            raise
    
    async def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents using semantic similarity"""
        try:
            if not self.documents:
                return []
            
            # Generate query embedding
            query_embedding = await self._generate_embeddings([query])
            query_vector = np.array(query_embedding[0])
            
            results = []
            
            # Calculate similarity scores for all documents
            for doc_id, doc_data in self.documents.items():
                if doc_id not in self.document_embeddings:
                    continue
                
                doc_embeddings = self.document_embeddings[doc_id]
                
                # Calculate cosine similarity for each chunk
                for i, chunk_embedding in enumerate(doc_embeddings):
                    chunk_vector = np.array(chunk_embedding)
                    
                    # Cosine similarity
                    similarity = np.dot(query_vector, chunk_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector)
                    )
                    
                    results.append({
                        "document_id": doc_id,
                        "document_name": doc_data["name"],
                        "chunk_index": i,
                        "chunk_text": doc_data["chunks"][i],
                        "similarity_score": float(similarity),
                        "chunk_length": len(doc_data["chunks"][i])
                    })
            
            # Sort by similarity score and return top_k
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error("Error searching documents", error=str(e))
            return []
    
    async def get_document_context(self, query: str, max_chunks: int = 3) -> str:
        """Get relevant document context for a query"""
        try:
            search_results = await self.search_documents(query, top_k=max_chunks)
            
            if not search_results:
                return ""
            
            # Combine relevant chunks
            context_parts = []
            for result in search_results:
                context_parts.append(f"From '{result['document_name']}':\n{result['chunk_text']}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error("Error getting document context", error=str(e))
            return ""
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of all processed documents"""
        documents = []
        for doc_id, doc_data in self.documents.items():
            documents.append({
                "id": doc_id,
                "name": doc_data["name"],
                "extension": doc_data["extension"],
                "chunk_count": doc_data["chunk_count"],
                "total_length": doc_data["total_length"],
                "uploaded_at": doc_data["uploaded_at"]
            })
        
        return sorted(documents, key=lambda x: x["uploaded_at"], reverse=True)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its embeddings"""
        try:
            if doc_id in self.documents:
                del self.documents[doc_id]
            
            if doc_id in self.document_embeddings:
                del self.document_embeddings[doc_id]
            
            logger.info(f"Document deleted", doc_id=doc_id)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}", error=str(e))
            return False
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        total_documents = len(self.documents)
        total_chunks = sum(doc["chunk_count"] for doc in self.documents.values())
        total_size = sum(doc["total_length"] for doc in self.documents.values())
        
        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_size": total_size,
            "supported_formats": list(self.supported_extensions.keys())
        }

