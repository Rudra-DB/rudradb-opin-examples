#!/usr/bin/env python3
"""
OpenAI + RudraDB-Opin Integration with Auto-Dimension Detection

This example demonstrates how to integrate OpenAI embeddings with RudraDB-Opin,
showcasing auto-dimension detection for OpenAI's 1536-dimensional embeddings
and building intelligent relationships.

Requirements:
    pip install rudradb-opin openai

Usage:
    python openai_integration.py
"""

import os
import numpy as np
import rudradb
from typing import Dict, List, Any, Optional

try:
    import openai
except ImportError:
    print("❌ OpenAI package not found. Install with: pip install openai")
    print("💡 For demo purposes, will use simulated embeddings")
    openai = None


class OpenAI_RudraDB_RAG:
    """Complete OpenAI + RudraDB-Opin integration with auto-features"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key"""
        if openai and api_key:
            openai.api_key = api_key
            self.use_real_openai = True
            print("🤖 OpenAI + RudraDB-Opin initialized with real OpenAI embeddings")
        else:
            self.use_real_openai = False
            print("🤖 OpenAI + RudraDB-Opin initialized with simulated embeddings (demo mode)")
        
        # 🎯 Auto-detects OpenAI's 1536 dimensions
        self.db = rudradb.RudraDB()
        print("   🎯 Auto-dimension detection enabled")
        
    def get_openai_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding (real or simulated)"""
        if self.use_real_openai:
            try:
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return np.array(response['data'][0]['embedding'], dtype=np.float32)
            except Exception as e:
                print(f"⚠️ OpenAI API error: {e}")
                print("   Falling back to simulated embeddings")
                self.use_real_openai = False
        
        # Simulated OpenAI-style embedding (1536 dimensions)
        np.random.seed(hash(text) % (2**32))  # Deterministic based on text
        return np.random.rand(1536).astype(np.float32)
    
    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Add document with OpenAI embeddings + auto-relationship detection"""
        
        # Get OpenAI embedding
        embedding = self.get_openai_embedding(text)
        
        # Add with auto-intelligence
        enhanced_metadata = {
            "text": text[:500],  # Store preview
            "embedding_model": "text-embedding-ada-002",
            "auto_detected_dim": self.db.dimension() if self.db.dimension() else "pending",
            "char_count": len(text),
            "word_count": len(text.split()),
            **(metadata or {})
        }
        
        self.db.add_vector(doc_id, embedding, enhanced_metadata)
        
        # 🧠 Auto-build relationships based on content analysis
        relationships_created = self._auto_detect_relationships(doc_id, enhanced_metadata)
        
        return {
            "dimension": self.db.dimension(),
            "relationships_created": relationships_created,
            "total_vectors": self.db.vector_count(),
            "embedding_model": "openai-ada-002" if self.use_real_openai else "simulated-openai"
        }
    
    def _auto_detect_relationships(self, new_doc_id: str, metadata: Dict[str, Any]) -> int:
        """Auto-detect relationships using OpenAI embeddings + metadata analysis"""
        relationships = 0
        new_text = metadata.get('text', '')
        new_category = metadata.get('category')
        new_tags = set(metadata.get('tags', []))
        
        for existing_id in self.db.list_vectors():
            if existing_id == new_doc_id or relationships >= 3:
                continue
                
            existing = self.db.get_vector(existing_id)
            existing_meta = existing['metadata']
            existing_text = existing_meta.get('text', '')
            existing_category = existing_meta.get('category')
            existing_tags = set(existing_meta.get('tags', []))
            
            # 🎯 Semantic similarity through category matching
            if new_category and new_category == existing_category:
                self.db.add_relationship(new_doc_id, existing_id, "semantic", 0.8, 
                                       {"reason": "same_category", "auto_detected": True})
                relationships += 1
                print(f"   🔗 Auto-linked {new_doc_id} ↔ {existing_id} (semantic: same category)")
            
            # 🏷️ Associative through tag overlap
            shared_tags = new_tags & existing_tags
            if len(shared_tags) >= 1:
                strength = min(0.7, len(shared_tags) * 0.3)
                self.db.add_relationship(new_doc_id, existing_id, "associative", strength,
                                       {"reason": "shared_tags", "tags": list(shared_tags), "auto_detected": True})
                relationships += 1
                print(f"   🏷️ Auto-linked {new_doc_id} ↔ {existing_id} (associative: {shared_tags})")
        
        return relationships
    
    def intelligent_qa(self, question: str) -> Dict[str, Any]:
        """Answer questions using relationship-aware search + GPT simulation"""
        
        # Get question embedding with auto-dimension compatibility
        query_embedding = self.get_openai_embedding(question)
        
        # 🧠 Auto-enhanced relationship-aware search
        results = self.db.search(query_embedding, rudradb.SearchParams(
            top_k=5,
            include_relationships=True,  # Use auto-detected relationships
            max_hops=2,                 # Multi-hop discovery
            relationship_weight=0.3     # Balance similarity + relationships
        ))
        
        # Build context from auto-enhanced results
        context_pieces = []
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            text = vector['metadata']['text']
            connection_type = "Direct match" if result.hop_count == 0 else f"{result.hop_count}-hop connection"
            context_pieces.append(f"[{connection_type}] {text}")
        
        context = "\n".join(context_pieces)
        
        # Simulate GPT response (in real usage, you'd call OpenAI ChatCompletion)
        simulated_answer = self._simulate_gpt_response(question, context)
        
        return {
            "answer": simulated_answer,
            "sources_found": len(results),
            "relationship_enhanced": sum(1 for r in results if r.hop_count > 0),
            "context_dimension": self.db.dimension(),
            "context_used": len(context_pieces)
        }
    
    def _simulate_gpt_response(self, question: str, context: str) -> str:
        """Simulate GPT response based on context"""
        if not context:
            return f"I don't have enough information to answer '{question}' based on the current knowledge base."
        
        # Simple simulation based on context
        context_lower = context.lower()
        question_lower = question.lower()
        
        if "ai" in question_lower or "artificial intelligence" in question_lower:
            if "ai" in context_lower or "artificial" in context_lower:
                return f"Based on the knowledge base, AI involves machine learning and intelligent systems. The context shows connections between AI concepts and related technologies."
        
        if "machine learning" in question_lower or "ml" in question_lower:
            if "machine" in context_lower or "learning" in context_lower:
                return f"Machine learning is a subset of AI that enables computers to learn from data. The knowledge base contains related information about algorithms and applications."
        
        if "python" in question_lower:
            if "python" in context_lower:
                return f"Python is a popular programming language for AI and data science. The knowledge base shows connections between Python and various AI/ML applications."
        
        # Generic response
        return f"Based on the relationship-aware search of the knowledge base, I found {len(context.split('['))} related documents that may contain information relevant to your question about '{question}'."


def demo_openai_integration():
    """Demo OpenAI + RudraDB-Opin integration"""
    
    print("🤖 OpenAI + RudraDB-Opin Integration Demo")
    print("=" * 50)
    
    # Initialize (try with environment variable for API key)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("💡 No OPENAI_API_KEY found - running in demo mode with simulated embeddings")
    
    rag = OpenAI_RudraDB_RAG(api_key)
    
    # Add AI knowledge with auto-relationship detection
    documents = [
        {
            "id": "ai_overview", 
            "text": "Artificial Intelligence is transforming industries through automation and intelligent decision making.", 
            "category": "AI", 
            "tags": ["ai", "automation", "industry"]
        },
        {
            "id": "ml_subset", 
            "text": "Machine Learning is a subset of AI that enables computers to learn from data without explicit programming.",
            "category": "AI", 
            "tags": ["ml", "data", "learning"]
        },
        {
            "id": "dl_neural", 
            "text": "Deep Learning uses neural networks with multiple layers to process complex patterns in data.",
            "category": "AI", 
            "tags": ["dl", "neural", "patterns"]
        },
        {
            "id": "nlp_language", 
            "text": "Natural Language Processing helps computers understand and generate human language.",
            "category": "AI", 
            "tags": ["nlp", "language", "text"]
        },
        {
            "id": "cv_vision", 
            "text": "Computer Vision enables machines to interpret and analyze visual information from images and videos.",
            "category": "AI", 
            "tags": ["cv", "vision", "images"]
        }
    ]
    
    print("\n🤖 Building AI Knowledge Base with OpenAI + Auto-Intelligence:")
    for doc in documents:
        result = rag.add_document(doc["id"], doc["text"], 
                                {"category": doc["category"], "tags": doc["tags"]})
        print(f"   📄 {doc['id']}: {result['relationships_created']} auto-relationships, {result['dimension']}D embedding")
    
    print(f"\n✅ Knowledge base ready: {rag.db.vector_count()} vectors, {rag.db.relationship_count()} auto-relationships")
    print(f"   🎯 Auto-detected dimension: {rag.db.dimension()}D (OpenAI embeddings)")
    
    # Intelligent Q&A with relationship-aware context
    questions = [
        "How does machine learning relate to other AI technologies?",
        "What are the applications of artificial intelligence?",
        "How do neural networks work in deep learning?"
    ]
    
    print(f"\n🧠 Intelligent Q&A with Relationship-Aware Search:")
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. ❓ Question: {question}")
        
        answer = rag.intelligent_qa(question)
        
        print(f"   📊 Found {answer['sources_found']} sources ({answer['relationship_enhanced']} through relationships)")
        print(f"   💬 Answer: {answer['answer']}")
        print(f"   🎯 Search dimension: {answer['context_dimension']}D")
    
    # Show database statistics
    stats = rag.db.get_statistics()
    print(f"\n📈 Final Statistics:")
    print(f"   📄 Vectors: {stats['vector_count']}/{rudradb.MAX_VECTORS}")
    print(f"   🔗 Relationships: {stats['relationship_count']}/{rudradb.MAX_RELATIONSHIPS}")
    print(f"   🎯 Dimension: {stats['dimension']}D")
    print(f"   🧠 Auto-features: Dimension detection ✅, Relationship detection ✅")
    
    print(f"\n🎉 OpenAI + RudraDB-Opin integration successful!")
    print("    ✨ Auto-dimension detection handled OpenAI embeddings seamlessly!")
    print("    ✨ Auto-relationship detection created intelligent connections!")
    print("    ✨ Relationship-aware search enhanced Q&A with contextual discovery!")


if __name__ == "__main__":
    demo_openai_integration()
