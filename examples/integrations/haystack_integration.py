#!/usr/bin/env python3
"""
RudraDB-Opin + Haystack Integration Demo
Demonstrates document processing with auto-relationship detection
"""

import numpy as np
import rudradb
import time
import json
from typing import List, Dict, Any

# Mock Haystack components for demonstration
# In real usage: pip install farm-haystack
class MockDocument:
    def __init__(self, content, meta=None):
        self.content = content
        self.meta = meta or {}
        self.id = meta.get("id", str(hash(content)))

class MockInMemoryDocumentStore:
    def __init__(self):
        self.documents = {}
        self.embeddings = {}
    
    def write_documents(self, documents):
        for doc in documents:
            self.documents[doc.id] = doc
    
    def update_embeddings(self, retriever):
        # Mock embedding generation
        for doc_id, doc in self.documents.items():
            # Simulate DPR embeddings (768D)
            embedding = np.random.rand(768).astype(np.float32)
            self.embeddings[doc_id] = embedding.tolist()
    
    def get_embedding_by_id(self, doc_id):
        return self.embeddings.get(doc_id)

class MockDensePassageRetriever:
    def __init__(self, document_store):
        self.document_store = document_store
        self.dimension = 768  # DPR dimension
    
    def embed_queries(self, queries):
        # Mock query embedding
        return [np.random.rand(self.dimension).astype(np.float32) for _ in queries]
    
    def retrieve(self, query, top_k=10):
        # Mock retrieval - return some documents
        docs = list(self.document_store.documents.values())[:top_k]
        for doc in docs:
            doc.score = np.random.rand()  # Mock relevance score
        return docs

class Haystack_RudraDB_Pipeline:
    """Haystack + RudraDB-Opin integration with auto-intelligence"""
    
    def __init__(self):
        # Initialize mock Haystack components
        self.haystack_store = MockInMemoryDocumentStore()
        self.retriever = MockDensePassageRetriever(self.haystack_store)
        
        # Initialize RudraDB-Opin with auto-dimension detection
        self.rudra_db = rudradb.RudraDB()  # 🎯 Auto-detects DPR dimensions (768D)
        
        print("🔍 Haystack + RudraDB-Opin pipeline initialized")
        print("   🤖 Auto-dimension detection enabled for DPR embeddings")
    
    def process_documents(self, documents):
        """Process documents through Haystack and add to RudraDB with auto-relationships"""
        
        # Convert to Haystack documents
        haystack_docs = []
        for i, doc in enumerate(documents):
            haystack_doc = MockDocument(
                content=doc["text"],
                meta={
                    "id": doc["id"],
                    "title": doc.get("title", f"Document {i+1}"),
                    **doc.get("metadata", {})
                }
            )
            haystack_docs.append(haystack_doc)
        
        # Add to Haystack document store and create embeddings
        self.haystack_store.write_documents(haystack_docs)
        self.haystack_store.update_embeddings(self.retriever)
        
        print(f"📄 Processed {len(haystack_docs)} documents through Haystack")
        
        # Add to RudraDB-Opin with auto-dimension detection and relationship building
        relationships_created = 0
        for doc in haystack_docs:
            # Get embedding from Haystack
            embedding = self.haystack_store.get_embedding_by_id(doc.id)
            if embedding is not None:
                embedding_array = np.array(embedding, dtype=np.float32)
                
                # Add to RudraDB with enhanced metadata
                enhanced_meta = {
                    "haystack_id": doc.id,
                    "title": doc.meta["title"],
                    "content": doc.content,
                    "embedding_model": "facebook/dpr-ctx_encoder-single-nq-base",
                    **doc.meta
                }
                
                self.rudra_db.add_vector(doc.meta["id"], embedding_array, enhanced_meta)
                
                # 🧠 Auto-detect relationships based on Haystack processing + content analysis
                doc_relationships = self._auto_detect_haystack_relationships(doc.meta["id"], enhanced_meta)
                relationships_created += doc_relationships
        
        return {
            "processed_docs": len(haystack_docs),
            "rudra_dimension": self.rudra_db.dimension(),
            "auto_relationships": relationships_created,
            "total_vectors": self.rudra_db.vector_count()
        }
    
    def _auto_detect_haystack_relationships(self, doc_id, metadata):
        """Auto-detect relationships using Haystack embeddings + metadata"""
        relationships = 0
        doc_content = metadata.get('content', '')
        doc_title = metadata.get('title', '')
        doc_category = metadata.get('category')
        doc_topics = set(metadata.get('topics', []))
        
        # Analyze against existing documents
        for existing_id in self.rudra_db.list_vectors():
            if existing_id == doc_id or relationships >= 4:
                continue
            
            existing = self.rudra_db.get_vector(existing_id)
            existing_meta = existing['metadata']
            existing_content = existing_meta.get('content', '')
            existing_category = existing_meta.get('category')
            existing_topics = set(existing_meta.get('topics', []))
            
            # 🎯 Content-based semantic relationships (using Haystack embeddings)
            if doc_category and doc_category == existing_category:
                self.rudra_db.add_relationship(doc_id, existing_id, "semantic", 0.85,
                    {"auto_detected": True, "reason": "haystack_same_category", "method": "dpr_embeddings"})
                relationships += 1
                print(f"   🔗 Haystack semantic: {doc_id} ↔ {existing_id}")
            
            # 🏷️ Topic overlap relationships  
            shared_topics = doc_topics & existing_topics
            if len(shared_topics) >= 1:
                strength = min(0.8, len(shared_topics) * 0.3)
                self.rudra_db.add_relationship(doc_id, existing_id, "associative", strength,
                    {"auto_detected": True, "reason": "shared_topics", "topics": list(shared_topics), "method": "haystack_analysis"})
                relationships += 1
                print(f"   🏷️ Haystack associative: {doc_id} ↔ {existing_id} (topics: {shared_topics})")
            
            # 📊 Hierarchical relationships through title analysis
            if "introduction" in doc_title.lower() and existing_category == doc_category:
                self.rudra_db.add_relationship(doc_id, existing_id, "hierarchical", 0.7,
                    {"auto_detected": True, "reason": "introduction_hierarchy", "method": "haystack_title_analysis"})
                relationships += 1
                print(f"   📊 Haystack hierarchical: {doc_id} → {existing_id}")
        
        return relationships
    
    def hybrid_search(self, question, top_k=5):
        """Hybrid search using Haystack retrieval + RudraDB relationship-aware search"""
        
        # 1. Haystack dense retrieval
        haystack_results = self.retriever.retrieve(question, top_k=top_k*2)
        
        # 2. RudraDB-Opin relationship-aware search
        question_embedding = self.retriever.embed_queries([question])[0]
        question_embedding = np.array(question_embedding, dtype=np.float32)
        
        rudra_results = self.rudra_db.search(question_embedding, rudradb.SearchParams(
            top_k=top_k,
            include_relationships=True,  # 🧠 Use auto-detected relationships
            max_hops=2,
            relationship_weight=0.4
        ))
        
        # 3. Combine and deduplicate results
        combined_results = []
        seen_docs = set()
        
        # Add Haystack results
        for doc in haystack_results[:top_k]:
            if doc.meta["id"] not in seen_docs:
                combined_results.append({
                    "id": doc.meta["id"],
                    "title": doc.meta.get("title", ""),
                    "content": doc.content[:200] + "...",
                    "source": "haystack_dense",
                    "score": doc.score,
                    "method": "DPR retrieval"
                })
                seen_docs.add(doc.meta["id"])
        
        # Add RudraDB relationship-enhanced results
        for result in rudra_results:
            if result.vector_id not in seen_docs:
                vector = self.rudra_db.get_vector(result.vector_id)
                connection = "direct" if result.hop_count == 0 else f"{result.hop_count}-hop auto-connection"
                combined_results.append({
                    "id": result.vector_id,
                    "title": vector['metadata'].get('title', ''),
                    "content": vector['metadata'].get('content', '')[:200] + "...",
                    "source": "rudradb_relationships",
                    "score": result.combined_score,
                    "method": f"Relationship-aware ({connection})",
                    "hop_count": result.hop_count
                })
                seen_docs.add(result.vector_id)
        
        # Sort by score
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "question": question,
            "total_results": len(combined_results),
            "haystack_results": len([r for r in combined_results if r["source"] == "haystack_dense"]),
            "rudra_relationship_results": len([r for r in combined_results if r["source"] == "rudradb_relationships"]),
            "relationship_enhanced": len([r for r in combined_results if r.get("hop_count", 0) > 0]),
            "results": combined_results[:top_k],
            "dimension": self.rudra_db.dimension()
        }

def demo_haystack_integration():
    """Demo: Haystack + RudraDB Auto-Intelligence"""
    print("🚀 Haystack + RudraDB-Opin Integration Demo")
    print("=" * 50)
    
    pipeline = Haystack_RudraDB_Pipeline()
    
    # Process documents with auto-dimension detection and relationship building
    documents = [
        {
            "id": "ai_intro_doc",
            "text": "Artificial Intelligence Introduction: AI systems can perform tasks that typically require human intelligence, including learning, reasoning, and problem-solving.",
            "title": "AI Introduction",
            "metadata": {"category": "AI", "topics": ["ai", "introduction", "basics"], "difficulty": "beginner"}
        },
        {
            "id": "machine_learning_fundamentals", 
            "text": "Machine Learning Fundamentals: ML algorithms enable computers to learn from data without being explicitly programmed for every task.",
            "title": "ML Fundamentals",
            "metadata": {"category": "AI", "topics": ["ml", "algorithms", "data"], "difficulty": "intermediate"}
        },
        {
            "id": "neural_networks_deep",
            "text": "Neural Networks and Deep Learning: Deep neural networks with multiple layers can learn complex patterns and representations from large datasets.",
            "title": "Neural Networks",
            "metadata": {"category": "AI", "topics": ["neural", "deep", "learning"], "difficulty": "advanced"}
        },
        {
            "id": "nlp_processing",
            "text": "Natural Language Processing: NLP enables computers to understand, interpret, and generate human language in a valuable way.",
            "title": "NLP Overview", 
            "metadata": {"category": "NLP", "topics": ["nlp", "language", "text"], "difficulty": "intermediate"}
        },
        {
            "id": "computer_vision_intro",
            "text": "Computer Vision Introduction: CV systems can automatically identify, analyze, and understand visual content from images and videos.",
            "title": "Computer Vision",
            "metadata": {"category": "CV", "topics": ["vision", "images", "analysis"], "difficulty": "intermediate"}
        }
    ]
    
    print("🔍 Processing documents through Haystack + RudraDB pipeline:")
    processing_result = pipeline.process_documents(documents)
    
    print(f"✅ Processing complete:")
    print(f"   📄 Documents processed: {processing_result['processed_docs']}")
    print(f"   🎯 Auto-detected dimension: {processing_result['rudra_dimension']}D (DPR embeddings)")
    print(f"   🧠 Auto-relationships created: {processing_result['auto_relationships']}")
    print(f"   📊 Total vectors in RudraDB: {processing_result['total_vectors']}")
    
    # Hybrid search with relationship enhancement
    questions = [
        "What are the fundamentals of machine learning?",
        "How do neural networks work in AI systems?"
    ]
    
    print(f"\n🔍 Hybrid Search Demonstrations:")
    for question in questions:
        results = pipeline.hybrid_search(question, top_k=4)
        
        print(f"\n❓ Question: {question}")
        print(f"   📊 Results: {results['total_results']} total ({results['haystack_results']} Haystack + {results['rudra_relationship_results']} RudraDB)")
        print(f"   🧠 Relationship-enhanced: {results['relationship_enhanced']} documents found through auto-detected connections")
        print(f"   🎯 Search dimension: {results['dimension']}D")
        
        print("   Top Results:")
        for i, result in enumerate(results['results'][:3], 1):
            print(f"      {i}. {result['title']}")
            print(f"         Method: {result['method']} (score: {result['score']:.3f})")
            print(f"         Preview: {result['content']}")
    
    print(f"\n🎉 Haystack + RudraDB-Opin integration successful!")
    print("    Auto-dimension detection handled DPR embeddings seamlessly!")
    print("    Auto-relationship detection enhanced search with intelligent connections!")

if __name__ == "__main__":
    demo_haystack_integration()
