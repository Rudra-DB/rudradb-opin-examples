#!/usr/bin/env python3
"""
HuggingFace + RudraDB-Opin Multi-Model Auto-Dimension Detection

This example demonstrates how to use multiple HuggingFace models with RudraDB-Opin's
auto-dimension detection, showcasing seamless switching between different embedding
dimensions and intelligent relationship building.

Requirements:
    pip install rudradb-opin transformers torch sentence-transformers

Usage:
    python huggingface_integration.py
"""

import numpy as np
import rudradb
from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    import torch
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("❌ Required packages not found. Install with:")
    print("   pip install transformers torch sentence-transformers")
    print("💡 For demo purposes, will use simulated embeddings")
    TRANSFORMERS_AVAILABLE = False


class HuggingFace_RudraDB_MultiModel:
    """HuggingFace + RudraDB-Opin with multi-model auto-dimension detection"""
    
    def __init__(self):
        self.models = {}
        self.databases = {}
        self.available_models = TRANSFORMERS_AVAILABLE
        print("🤗 HuggingFace + RudraDB-Opin Multi-Model System initialized")
        
        if not self.available_models:
            print("   🎯 Running in demo mode with simulated embeddings")
        
    def add_model(self, model_name: str, model_type: str = "sentence-transformer") -> Dict[str, Any]:
        """Add a HuggingFace model with auto-dimension detection"""
        
        if self.available_models:
            try:
                if model_type == "sentence-transformer":
                    model = SentenceTransformer(model_name)
                    dimension = model.get_sentence_embedding_dimension()
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                    # Get dimension from config
                    dimension = model.config.hidden_size
                    model = {"tokenizer": tokenizer, "model": model}
                
                self.models[model_name] = {
                    "model": model,
                    "type": model_type, 
                    "expected_dimension": dimension,
                    "real_model": True
                }
                
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {e}")
                print("   Falling back to simulated model")
                self.available_models = False
        
        if not self.available_models:
            # Simulated model dimensions
            dimension_map = {
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "sentence-transformers/all-mpnet-base-v2": 768,
                "distilbert-base-uncased": 768,
                "bert-base-uncased": 768
            }
            dimension = dimension_map.get(model_name, 384)
            
            self.models[model_name] = {
                "model": None,
                "type": model_type,
                "expected_dimension": dimension,
                "real_model": False
            }
        
        # Create database with auto-dimension detection
        self.databases[model_name] = rudradb.RudraDB()  # 🎯 Auto-detects dimension
        
        print(f"✅ Added {model_name} (expected: {self.models[model_name]['expected_dimension']}D, auto-detection enabled)")
        
        return {
            "model_name": model_name,
            "expected_dimension": self.models[model_name]["expected_dimension"],
            "auto_detection_enabled": True,
            "real_model": self.models[model_name]["real_model"]
        }
        
    def encode_text(self, model_name: str, text: str) -> np.ndarray:
        """Encode text with specified model"""
        model_info = self.models[model_name]
        
        if model_info["real_model"] and self.available_models:
            try:
                if model_info["type"] == "sentence-transformer":
                    embedding = model_info["model"].encode([text])[0]
                else:
                    tokenizer = model_info["model"]["tokenizer"]
                    model = model_info["model"]["model"]
                    
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                return embedding.astype(np.float32)
                
            except Exception as e:
                print(f"⚠️ Error encoding with {model_name}: {e}")
                print("   Falling back to simulated embedding")
        
        # Simulated embedding
        expected_dim = model_info["expected_dimension"]
        np.random.seed(hash(text + model_name) % (2**32))  # Deterministic
        return np.random.rand(expected_dim).astype(np.float32)
    
    def add_document_multimodel(self, doc_id: str, text: str, metadata: Dict[str, Any], 
                               model_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Add document to multiple model databases with auto-relationship detection"""
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        results = {}
        for model_name in model_names:
            if model_name not in self.databases:
                continue
                
            db = self.databases[model_name]
            
            # Encode with current model
            embedding = self.encode_text(model_name, text)
            
            # Add to database - auto-dimension detection in action
            enhanced_metadata = {
                "text": text[:500],
                "model": model_name,
                "expected_dim": self.models[model_name]["expected_dimension"],
                "real_model": self.models[model_name]["real_model"],
                **metadata
            }
            
            db.add_vector(doc_id, embedding, enhanced_metadata)
            
            # Auto-detect relationships within this model's space
            relationships = self._auto_build_relationships(db, doc_id, enhanced_metadata)
            
            results[model_name] = {
                "expected_dim": self.models[model_name]["expected_dimension"],
                "detected_dim": db.dimension(),
                "relationships_created": relationships,
                "match": db.dimension() == self.models[model_name]["expected_dimension"],
                "model_type": self.models[model_name]["type"],
                "real_model": self.models[model_name]["real_model"]
            }
        
        return results
    
    def _auto_build_relationships(self, db: rudradb.RudraDB, doc_id: str, metadata: Dict[str, Any]) -> int:
        """Auto-build relationships based on metadata analysis"""
        relationships_created = 0
        doc_tags = set(metadata.get('tags', []))
        doc_category = metadata.get('category')
        doc_difficulty = metadata.get('difficulty')
        
        for other_id in db.list_vectors():
            if other_id == doc_id or relationships_created >= 3:
                continue
                
            other_vector = db.get_vector(other_id)
            other_meta = other_vector['metadata']
            other_tags = set(other_meta.get('tags', []))
            other_category = other_meta.get('category')
            other_difficulty = other_meta.get('difficulty')
            
            # Auto-detect relationship type and strength
            if doc_category == other_category:
                # Same category → semantic relationship
                db.add_relationship(doc_id, other_id, "semantic", 0.8,
                                  {"auto_detected": True, "reason": "same_category"})
                relationships_created += 1
                print(f"      🔗 {doc_id} ↔ {other_id} (semantic: same category)")
                
            elif len(doc_tags & other_tags) >= 1:
                # Shared tags → associative relationship  
                shared = doc_tags & other_tags
                strength = min(0.7, len(shared) * 0.25)
                db.add_relationship(doc_id, other_id, "associative", strength,
                                  {"auto_detected": True, "reason": "shared_tags", "tags": list(shared)})
                relationships_created += 1
                print(f"      🏷️ {doc_id} ↔ {other_id} (associative: {shared})")
                
            elif doc_difficulty and other_difficulty:
                # Learning progression → temporal relationship
                levels = {"beginner": 1, "intermediate": 2, "advanced": 3}
                if abs(levels.get(doc_difficulty, 2) - levels.get(other_difficulty, 2)) == 1:
                    db.add_relationship(doc_id, other_id, "temporal", 0.85,
                                      {"auto_detected": True, "reason": "learning_progression"})
                    relationships_created += 1
                    print(f"      ⏰ {doc_id} ↔ {other_id} (temporal: learning progression)")
        
        return relationships_created
    
    def cross_model_search(self, query: str, model_names: Optional[List[str]] = None, 
                          top_k: int = 5) -> Dict[str, Dict[str, Any]]:
        """Search across multiple models with auto-enhanced results"""
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        all_results = {}
        for model_name in model_names:
            if model_name not in self.databases:
                continue
                
            db = self.databases[model_name]
            if db.vector_count() == 0:
                continue
                
            query_embedding = self.encode_text(model_name, query)
            
            # Auto-enhanced relationship-aware search
            results = db.search(query_embedding, rudradb.SearchParams(
                top_k=top_k,
                include_relationships=True,  # Use auto-detected relationships
                max_hops=2,
                relationship_weight=0.3
            ))
            
            model_results = []
            for result in results:
                vector = db.get_vector(result.vector_id)
                model_results.append({
                    "document": result.vector_id,
                    "text": vector['metadata']['text'][:100] + "...",
                    "similarity": result.similarity_score,
                    "combined_score": result.combined_score,
                    "connection": "direct" if result.hop_count == 0 else f"{result.hop_count}-hop",
                    "model_dimension": db.dimension()
                })
            
            all_results[model_name] = {
                "results": model_results,
                "dimension": db.dimension(),
                "total_docs": db.vector_count(),
                "total_relationships": db.relationship_count(),
                "model_info": self.models[model_name]
            }
        
        return all_results


def demo_huggingface_multimodel():
    """Demo multi-model HuggingFace integration"""
    
    print("🤗 HuggingFace + RudraDB-Opin Multi-Model Demo")
    print("=" * 55)
    
    system = HuggingFace_RudraDB_MultiModel()
    
    # Add multiple HuggingFace models - each gets auto-dimension detection
    models_to_test = [
        ("sentence-transformers/all-MiniLM-L6-v2", "sentence-transformer"),  # 384D
        ("sentence-transformers/all-mpnet-base-v2", "sentence-transformer"),   # 768D  
        ("distilbert-base-uncased", "transformer")  # 768D
    ]
    
    print("\n🤗 Adding multiple HuggingFace models with auto-dimension detection:")
    for model_name, model_type in models_to_test:
        result = system.add_model(model_name, model_type)
        print(f"   Model: {model_name.split('/')[-1]}")
        print(f"   Type: {model_type}, Expected: {result['expected_dimension']}D")
        print(f"   Real model: {result['real_model']}")
    
    # Add documents to all models - watch auto-dimension detection work
    documents = [
        {
            "id": "transformers_paper", 
            "text": "Attention Is All You Need introduced the Transformer architecture revolutionizing NLP", 
            "category": "NLP", 
            "tags": ["transformers", "attention", "nlp"], 
            "difficulty": "advanced"
        },
        {
            "id": "bert_paper", 
            "text": "BERT Bidirectional Encoder Representations from Transformers for language understanding",
            "category": "NLP", 
            "tags": ["bert", "bidirectional", "nlp"], 
            "difficulty": "intermediate"
        },  
        {
            "id": "gpt_intro", 
            "text": "GPT Generative Pre-trained Transformers for text generation and completion",
            "category": "NLP", 
            "tags": ["gpt", "generative", "nlp"], 
            "difficulty": "intermediate"
        },
        {
            "id": "vision_transformer", 
            "text": "Vision Transformer ViT applies transformer architecture to computer vision tasks",
            "category": "CV", 
            "tags": ["vit", "transformers", "vision"], 
            "difficulty": "advanced"
        }
    ]
    
    print(f"\n📄 Adding documents with multi-model auto-dimension detection:")
    for doc in documents:
        results = system.add_document_multimodel(
            doc["id"], doc["text"], 
            {"category": doc["category"], "tags": doc["tags"], "difficulty": doc["difficulty"]}
        )
        
        print(f"\n   📄 {doc['id']}:")
        for model_name, result in results.items():
            status = "✅" if result["match"] else "⚠️"
            model_short = model_name.split('/')[-1] if '/' in model_name else model_name
            print(f"      {status} {model_short}: Expected {result['expected_dim']}D → Detected {result['detected_dim']}D")
            print(f"         Relationships: {result['relationships_created']} auto-created")
    
    # Cross-model search with auto-enhanced results
    query = "transformer architecture for language and vision"
    print(f"\n🔍 Cross-Model Search: '{query}'")
    search_results = system.cross_model_search(query, top_k=3)
    
    for model_name, results in search_results.items():
        model_short = model_name.split('/')[-1] if '/' in model_name else model_name
        print(f"\n📊 {model_short} ({results['dimension']}D, {results['total_relationships']} auto-relationships):")
        
        for result in results['results'][:2]:
            print(f"   📄 {result['document']}")
            print(f"      Connection: {result['connection']} (score: {result['combined_score']:.3f})")
            print(f"      Text: {result['text']}")
    
    # Summary statistics
    print(f"\n📈 Multi-Model Summary:")
    total_vectors = sum(results['total_docs'] for results in search_results.values())
    total_relationships = sum(results['total_relationships'] for results in search_results.values())
    dimensions_used = set(results['dimension'] for results in search_results.values())
    
    print(f"   📄 Total vectors: {total_vectors}")
    print(f"   🔗 Total relationships: {total_relationships}")
    print(f"   🎯 Dimensions handled: {sorted(dimensions_used)}")
    print(f"   🤗 Models integrated: {len(search_results)}")
    
    print(f"\n🎉 Multi-model auto-dimension detection successful!")
    print("    ✨ RudraDB-Opin seamlessly adapted to each model's dimensions automatically!")
    print("    ✨ Auto-relationship detection worked across different embedding spaces!")
    print("    ✨ Cross-model search enabled comprehensive discovery!")


if __name__ == "__main__":
    demo_huggingface_multimodel()
