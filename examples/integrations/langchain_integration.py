#!/usr/bin/env python3
"""
LangChain + RudraDB-Opin Advanced RAG with Auto-Features

This example demonstrates how to integrate LangChain with RudraDB-Opin to create
an advanced RAG system with auto-dimension detection, intelligent chunking,
and sophisticated relationship-aware search.

Requirements:
    pip install rudradb-opin langchain sentence-transformers

Usage:
    python langchain_integration.py
"""

import numpy as np
import rudradb
from typing import List, Dict, Any, Optional, Tuple
import time

try:
    from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("❌ LangChain not found. Install with: pip install langchain")
    print("💡 For demo purposes, will use simulated chunking and embeddings")
    LANGCHAIN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ sentence-transformers not available - using simulated embeddings")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class LangChain_RudraDB_AutoRAG:
    """LangChain + RudraDB-Opin integration with auto-intelligence for advanced RAG"""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.langchain_available = LANGCHAIN_AVAILABLE
        self.sentence_transformers_available = SENTENCE_TRANSFORMERS_AVAILABLE
        
        # Initialize components based on availability
        if self.langchain_available and self.sentence_transformers_available:
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", " ", ""]
                )
                self.real_components = True
            except Exception as e:
                print(f"⚠️ Error initializing LangChain components: {e}")
                self.real_components = False
        else:
            self.real_components = False
        
        if not self.real_components:
            print("🔄 Using simulated LangChain components")
        
        # Initialize RudraDB-Opin with auto-dimension detection  
        self.db = rudradb.RudraDB()  # 🎯 Auto-detects embedding dimensions
        
        print(f"🦜 LangChain + RudraDB-Opin Auto-RAG initialized")
        print(f"   🎯 Embedding model: {embedding_model_name}")
        print(f"   🤖 Auto-dimension detection enabled")
        print(f"   📄 Real components: {self.real_components}")
    
    def _simulate_embedding(self, text: str) -> np.ndarray:
        """Simulate embedding for demo purposes"""
        # Simulate 384-dimensional embedding (all-MiniLM-L6-v2 size)
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(384).astype(np.float32)
    
    def _simulate_text_splitting(self, text: str) -> List[str]:
        """Simulate text splitting for demo purposes"""
        # Simple simulation: split by sentences, keep chunks ~500 chars
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) > 500 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
            else:
                current_chunk += sentence + ". "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using real or simulated method"""
        if self.real_components:
            try:
                return np.array(self.embeddings.embed_query(text), dtype=np.float32)
            except Exception as e:
                print(f"⚠️ Real embedding failed: {e}, using simulation")
        
        return self._simulate_embedding(text)
    
    def split_text(self, text: str) -> List[str]:
        """Split text using real or simulated method"""
        if self.real_components:
            try:
                doc = Document(page_content=text)
                chunks = self.text_splitter.split_documents([doc])
                return [chunk.page_content for chunk in chunks]
            except Exception as e:
                print(f"⚠️ Real text splitting failed: {e}, using simulation")
        
        return self._simulate_text_splitting(text)
    
    def add_documents_with_chunking(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents with LangChain chunking + RudraDB auto-relationship detection"""
        
        all_chunks = []
        chunk_metadata = []
        
        print("📄 Processing documents with LangChain + Auto-Intelligence...")
        
        # Process each document through LangChain-style processing
        for doc in documents:
            print(f"   Processing: {doc['id']}")
            
            # Split into chunks
            chunks = self.split_text(doc["content"])
            print(f"      Created {len(chunks)} chunks")
            
            # Process each chunk
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{doc['id']}_chunk_{i}"
                
                # Create embeddings
                embedding = self.get_embedding(chunk_text)
                
                # Enhanced metadata for auto-relationship detection
                enhanced_metadata = {
                    "chunk_id": chunk_id,
                    "source_document": doc["id"], 
                    "chunk_index": i,
                    "chunk_content": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    "embedding_model": self.embedding_model_name,
                    "langchain_processed": True,
                    "chunk_length": len(chunk_text),
                    **doc.get("metadata", {})
                }
                
                # Add to RudraDB with auto-dimension detection
                self.db.add_vector(chunk_id, embedding, enhanced_metadata)
                
                all_chunks.append(chunk_id)
                chunk_metadata.append(enhanced_metadata)
        
        # 🧠 Auto-detect relationships between chunks after all are added
        print("   🧠 Auto-detecting sophisticated chunk relationships...")
        relationships_created = self._auto_detect_document_relationships(chunk_metadata)
        
        return {
            "total_chunks": len(all_chunks),
            "auto_detected_dimension": self.db.dimension(),
            "auto_relationships": relationships_created,
            "documents_processed": len(documents),
            "real_processing": self.real_components
        }
    
    def _auto_detect_document_relationships(self, chunk_metadata: List[Dict[str, Any]]) -> int:
        """Auto-detect sophisticated relationships between document chunks"""
        relationships = 0
        
        for i, chunk_meta in enumerate(chunk_metadata):
            chunk_id = chunk_meta["chunk_id"]
            source_doc = chunk_meta["source_document"]
            chunk_index = chunk_meta["chunk_index"]
            category = chunk_meta.get("category")
            topics = set(chunk_meta.get("topics", []))
            
            for j, other_meta in enumerate(chunk_metadata[i+1:], i+1):
                if relationships >= 20:  # Limit for demo
                    break
                    
                other_chunk_id = other_meta["chunk_id"]
                other_source_doc = other_meta["source_document"]
                other_chunk_index = other_meta["chunk_index"]
                other_category = other_meta.get("category")
                other_topics = set(other_meta.get("topics", []))
                
                # 📊 Hierarchical: Sequential chunks from same document
                if (source_doc == other_source_doc and 
                    abs(chunk_index - other_chunk_index) == 1):
                    self.db.add_relationship(chunk_id, other_chunk_id, "hierarchical", 0.9,
                        {"auto_detected": True, "reason": "sequential_chunks", "method": "langchain_chunking"})
                    relationships += 1
                    print(f"      📊 Sequential: {chunk_id} → {other_chunk_id}")
                
                # 🔗 Semantic: Same category, different documents
                elif (category and category == other_category and 
                      source_doc != other_source_doc):
                    self.db.add_relationship(chunk_id, other_chunk_id, "semantic", 0.8,
                        {"auto_detected": True, "reason": "cross_document_category", "category": category})
                    relationships += 1
                    print(f"      🔗 Cross-doc semantic: {chunk_id} ↔ {other_chunk_id}")
                
                # 🏷️ Associative: Shared topics across documents
                elif len(topics & other_topics) >= 2 and source_doc != other_source_doc:
                    shared = topics & other_topics
                    strength = min(0.75, len(shared) * 0.25)
                    self.db.add_relationship(chunk_id, other_chunk_id, "associative", strength,
                        {"auto_detected": True, "reason": "shared_topics", "topics": list(shared)})
                    relationships += 1
                    print(f"      🏷️ Topic association: {chunk_id} ↔ {other_chunk_id} ({shared})")
                
                # ⏰ Temporal: Learning progression detection
                elif (chunk_meta.get("difficulty") and other_meta.get("difficulty") and
                      category == other_category):
                    levels = {"beginner": 1, "intermediate": 2, "advanced": 3}
                    level_diff = levels.get(other_meta["difficulty"], 2) - levels.get(chunk_meta["difficulty"], 2)
                    if level_diff == 1:  # Progressive difficulty
                        self.db.add_relationship(chunk_id, other_chunk_id, "temporal", 0.85,
                            {"auto_detected": True, "reason": "learning_progression", 
                             "from": chunk_meta["difficulty"], "to": other_meta["difficulty"]})
                        relationships += 1
                        print(f"      ⏰ Learning: {chunk_id} → {other_chunk_id}")
        
        return relationships
    
    def auto_enhanced_rag_search(self, query: str, top_k: int = 5, 
                                strategy: str = "balanced") -> Dict[str, Any]:
        """Advanced RAG search with auto-relationship enhancement"""
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Configure search strategy
        if strategy == "precise":
            params = rudradb.SearchParams(
                top_k=top_k,
                include_relationships=False,
                similarity_threshold=0.4
            )
        elif strategy == "discovery":
            params = rudradb.SearchParams(
                top_k=top_k * 2,  # Get more results for relationship expansion
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.6,
                similarity_threshold=0.1
            )
        else:  # balanced
            params = rudradb.SearchParams(
                top_k=top_k,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.35
            )
        
        # 🧠 Auto-enhanced relationship-aware search
        results = self.db.search(query_embedding, params)
        
        # Process and enhance results
        enhanced_results = []
        seen_documents = set()
        
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            metadata = vector['metadata']
            
            # Avoid duplicate chunks from same document (take best scoring)
            source_doc = metadata.get("source_document")
            if source_doc in seen_documents:
                continue
            seen_documents.add(source_doc)
            
            # Determine connection type and relevance
            if result.hop_count == 0:
                connection_type = "Direct similarity match"
                relevance = "high"
            elif result.hop_count == 1:
                connection_type = "1-hop relationship connection" 
                relevance = "medium-high"
            else:
                connection_type = f"{result.hop_count}-hop relationship chain"
                relevance = "medium"
            
            enhanced_results.append({
                "chunk_id": result.vector_id,
                "source_document": source_doc,
                "content": metadata.get("chunk_content", ""),
                "similarity_score": result.similarity_score,
                "combined_score": result.combined_score,
                "connection_type": connection_type,
                "relevance": relevance,
                "hop_count": result.hop_count,
                "category": metadata.get("category", ""),
                "chunk_index": metadata.get("chunk_index", 0)
            })
            
            if len(enhanced_results) >= top_k:
                break
        
        return {
            "query": query,
            "strategy": strategy,
            "total_results": len(enhanced_results),
            "relationship_enhanced": sum(1 for r in enhanced_results if r["hop_count"] > 0),
            "dimension": self.db.dimension(),
            "results": enhanced_results,
            "database_stats": {
                "total_chunks": self.db.vector_count(),
                "total_relationships": self.db.relationship_count()
            },
            "processing_method": "real_langchain" if self.real_components else "simulated"
        }


def demo_langchain_rag():
    """Demo LangChain + RudraDB-Opin Auto-RAG"""
    
    print("🦜 LangChain + RudraDB-Opin Auto-RAG Demo")
    print("=" * 50)
    
    rag_system = LangChain_RudraDB_AutoRAG("sentence-transformers/all-MiniLM-L6-v2")
    
    # Add documents with automatic chunking and relationship detection
    documents = [
        {
            "id": "ai_foundations",
            "content": """Artificial Intelligence Foundations

Introduction to AI:
Artificial Intelligence represents the simulation of human intelligence in machines. These systems are designed to think and learn like humans, performing tasks that traditionally require human cognition such as visual perception, speech recognition, decision-making, and language translation.

Core AI Concepts:
The foundation of AI lies in machine learning algorithms that can process vast amounts of data to identify patterns and make predictions. These systems improve their performance over time through experience, much like human learning processes.

AI Applications:
Modern AI applications span across industries including healthcare for medical diagnosis, finance for fraud detection, transportation for autonomous vehicles, and entertainment for recommendation systems.""",
            "metadata": {
                "category": "AI", 
                "topics": ["ai", "foundations", "introduction"], 
                "difficulty": "beginner",
                "author": "AI Research Team"
            }
        },
        {
            "id": "machine_learning_deep_dive",
            "content": """Machine Learning Deep Dive

ML Fundamentals:
Machine Learning is a subset of artificial intelligence that focuses on the development of algorithms that can learn from and make decisions based on data. Unlike traditional programming where humans write explicit instructions, ML systems learn patterns from data to make predictions or decisions.

Types of Machine Learning:
Supervised learning uses labeled data to train models for prediction tasks. Unsupervised learning finds hidden patterns in data without labels. Reinforcement learning learns through interaction with an environment, receiving rewards or penalties for actions.

ML in Practice:
Practical machine learning involves data preprocessing, feature engineering, model selection, training, validation, and deployment. The process requires careful attention to data quality, model evaluation metrics, and avoiding overfitting to ensure good generalization to new data.""",
            "metadata": {
                "category": "AI", 
                "topics": ["ml", "algorithms", "data"], 
                "difficulty": "intermediate",
                "author": "ML Experts"
            }
        },
        {
            "id": "neural_networks_advanced",
            "content": """Advanced Neural Networks

Deep Learning Architecture:
Neural networks with multiple hidden layers, known as deep neural networks, can learn complex hierarchical representations of data. Each layer learns increasingly abstract features, from simple edges and textures in lower layers to complex objects and concepts in higher layers.

Training Deep Networks:
Training deep neural networks requires specialized techniques including backpropagation for gradient computation, various optimization algorithms like Adam and SGD, regularization methods like dropout and batch normalization, and careful initialization strategies.

Modern Applications:
Advanced neural network architectures like convolutional neural networks excel at computer vision tasks, recurrent neural networks handle sequential data, and transformer models have revolutionized natural language processing with attention mechanisms enabling parallel processing of sequences.""",
            "metadata": {
                "category": "AI", 
                "topics": ["neural", "deep", "learning"], 
                "difficulty": "advanced",
                "author": "Deep Learning Lab"
            }
        }
    ]
    
    print("\n🦜 Processing documents with LangChain + RudraDB Auto-Intelligence:")
    processing_result = rag_system.add_documents_with_chunking(documents)
    
    print(f"\n✅ Document processing complete:")
    print(f"   📄 Documents processed: {processing_result['documents_processed']}")
    print(f"   📝 Total chunks created: {processing_result['total_chunks']}")
    print(f"   🎯 Auto-detected dimension: {processing_result['auto_detected_dimension']}D")
    print(f"   🧠 Auto-relationships created: {processing_result['auto_relationships']}")
    print(f"   🔄 Processing method: {'Real LangChain' if processing_result['real_processing'] else 'Simulated'}")
    
    # Advanced RAG search with relationship enhancement
    queries = [
        "What are the fundamentals of artificial intelligence?",
        "How do neural networks learn from data?",
        "What's the difference between supervised and unsupervised learning?"
    ]
    
    print(f"\n🔍 Advanced Auto-Enhanced RAG Search:")
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. ❓ Query: {query}")
        
        # Try different search strategies
        for strategy in ["balanced", "discovery"]:
            search_result = rag_system.auto_enhanced_rag_search(query, top_k=3, strategy=strategy)
            
            print(f"\n   📊 Strategy: {strategy.title()}")
            print(f"      Results: {search_result['total_results']} documents found")
            print(f"      Enhanced: {search_result['relationship_enhanced']} through auto-detected connections")
            print(f"      Dimension: {search_result['dimension']}D")
            
            print(f"      📋 Top Results:")
            for j, result in enumerate(search_result['results'][:2], 1):
                print(f"         {j}. Source: {result['source_document']} (Chunk {result['chunk_index']})")
                print(f"            Connection: {result['connection_type']} | Relevance: {result['relevance']}")
                print(f"            Score: {result['combined_score']:.3f} | Category: {result['category']}")
                print(f"            Content: {result['content']}")
    
    # Final statistics
    final_stats = rag_system.db.get_statistics()
    print(f"\n📈 Final System Statistics:")
    print(f"   📝 Total chunks: {final_stats['vector_count']}")
    print(f"   🔗 Total relationships: {final_stats['relationship_count']}")
    print(f"   🎯 Embedding dimension: {final_stats['dimension']}D")
    print(f"   💾 Capacity usage: {final_stats['capacity_usage']['vector_usage_percent']:.1f}% vectors, {final_stats['capacity_usage']['relationship_usage_percent']:.1f}% relationships")
    
    print(f"\n🎉 LangChain + RudraDB-Opin Auto-RAG successful!")
    print("    ✨ Auto-dimension detection seamlessly handled embeddings")
    print("    ✨ Auto-relationship detection created sophisticated document connections")
    print("    ✨ Multi-hop relationship traversal enhanced search relevance")
    print("    ✨ LangChain chunking + RudraDB relationships = powerful RAG!")


if __name__ == "__main__":
    demo_langchain_rag()
