#!/usr/bin/env python3
"""
Multi-Model Comparison with RudraDB-Opin
========================================

This example demonstrates using multiple embedding models simultaneously with
RudraDB-Opin's auto-dimension detection, comparing their effectiveness for
different types of content and use cases.

Features demonstrated:
- Multiple embedding models with different dimensions
- Auto-dimension detection for each model
- Cross-model performance comparison
- Model-specific relationship patterns
- Optimal model selection strategies

Requirements:
pip install sentence-transformers transformers torch numpy rudradb-opin
"""

import rudradb
import numpy as np
import time
from typing import Dict, List, Tuple, Any
import json

# Try importing actual models, fall back to mock if not available
try:
    from sentence_transformers import SentenceTransformer
    REAL_MODELS = True
except ImportError:
    REAL_MODELS = False
    print("⚠️ sentence-transformers not available, using mock embeddings")

class MultiModelComparison:
    """Compare multiple embedding models with RudraDB-Opin auto-dimension detection"""
    
    def __init__(self):
        self.models = {}
        self.databases = {}
        self.performance_metrics = {}
        
        print("🤖 Multi-Model Comparison System")
        print("=" * 40)
    
    def add_model(self, model_name: str, model_config: Dict[str, Any]):
        """Add an embedding model to the comparison"""
        
        print(f"📥 Adding model: {model_name}")
        
        if REAL_MODELS and model_config["type"] == "sentence_transformer":
            try:
                model = SentenceTransformer(model_config["model_id"])
                dimension = model.get_sentence_embedding_dimension()
                
                self.models[model_name] = {
                    "model": model,
                    "type": model_config["type"],
                    "dimension": dimension,
                    "description": model_config.get("description", ""),
                    "use_cases": model_config.get("use_cases", [])
                }
                
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {e}, using mock")
                self._add_mock_model(model_name, model_config)
        else:
            self._add_mock_model(model_name, model_config)
        
        # Create database with auto-dimension detection
        self.databases[model_name] = rudradb.RudraDB()  # 🎯 Auto-detects dimension
        
        print(f"   Expected dimension: {self.models[model_name]['dimension']}D")
        print(f"   Use cases: {', '.join(self.models[model_name]['use_cases'])}")
    
    def _add_mock_model(self, model_name: str, model_config: Dict[str, Any]):
        """Add a mock model for demonstration"""
        
        class MockModel:
            def __init__(self, dimension):
                self.dimension = dimension
            
            def encode(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                return [np.random.rand(self.dimension).astype(np.float32) for _ in texts]
        
        dimension = model_config.get("dimension", 384)
        self.models[model_name] = {
            "model": MockModel(dimension),
            "type": "mock",
            "dimension": dimension,
            "description": model_config.get("description", f"Mock {dimension}D model"),
            "use_cases": model_config.get("use_cases", ["demo"])
        }
    
    def encode_text(self, model_name: str, text: str) -> np.ndarray:
        """Encode text with specified model"""
        
        model_info = self.models[model_name]
        
        if REAL_MODELS and model_info["type"] == "sentence_transformer":
            embedding = model_info["model"].encode([text])[0]
        else:
            embedding = model_info["model"].encode([text])[0]
        
        return embedding.astype(np.float32)
    
    def add_documents_to_all_models(self, documents: List[Dict[str, Any]]):
        """Add documents to all model databases with auto-dimension detection"""
        
        print(f"\n📚 Adding {len(documents)} documents to all models...")
        
        results = {}
        
        for model_name in self.models.keys():
            db = self.databases[model_name]
            model_results = {
                "documents_added": 0,
                "relationships_created": 0,
                "auto_detected_dimension": None,
                "dimension_match": False
            }
            
            print(f"\n🔧 Processing with {model_name}:")
            
            for doc in documents:
                try:
                    # Encode with current model
                    embedding = self.encode_text(model_name, doc["text"])
                    
                    # Enhanced metadata
                    metadata = {
                        "text": doc["text"],
                        "model": model_name,
                        "expected_dim": self.models[model_name]["dimension"],
                        **doc.get("metadata", {})
                    }
                    
                    # Add to database - auto-dimension detection in action
                    db.add_vector(doc["id"], embedding, metadata)
                    model_results["documents_added"] += 1
                    
                    # Record auto-detected dimension
                    if model_results["auto_detected_dimension"] is None:
                        model_results["auto_detected_dimension"] = db.dimension()
                        model_results["dimension_match"] = (
                            db.dimension() == self.models[model_name]["dimension"]
                        )
                        print(f"   🎯 Auto-detected: {db.dimension()}D "
                              f"(expected: {self.models[model_name]['dimension']}D)")
                    
                    # Auto-build relationships
                    relationships = self._auto_build_relationships(db, doc["id"], metadata)
                    model_results["relationships_created"] += relationships
                    
                except Exception as e:
                    print(f"   ❌ Failed to add {doc['id']} to {model_name}: {e}")
            
            results[model_name] = model_results
            
            print(f"   ✅ Added {model_results['documents_added']} documents")
            print(f"   🔗 Created {model_results['relationships_created']} relationships")
            print(f"   📊 Dimension match: {'✅' if model_results['dimension_match'] else '❌'}")
        
        return results
    
    def _auto_build_relationships(self, db: rudradb.RudraDB, doc_id: str, metadata: Dict[str, Any]) -> int:
        """Auto-build relationships based on metadata"""
        
        relationships_created = 0
        doc_category = metadata.get("category")
        doc_tags = set(metadata.get("tags", []))
        doc_difficulty = metadata.get("difficulty")
        
        for other_id in db.list_vectors():
            if other_id == doc_id or relationships_created >= 3:
                continue
            
            other_vector = db.get_vector(other_id)
            other_meta = other_vector["metadata"]
            other_category = other_meta.get("category")
            other_tags = set(other_meta.get("tags", []))
            other_difficulty = other_meta.get("difficulty")
            
            # Same category → semantic relationship
            if doc_category and doc_category == other_category:
                db.add_relationship(doc_id, other_id, "semantic", 0.8, {
                    "auto_detected": True, 
                    "reason": "same_category",
                    "model": metadata["model"]
                })
                relationships_created += 1
            
            # Shared tags → associative relationship
            elif len(doc_tags & other_tags) >= 1:
                shared = doc_tags & other_tags
                strength = min(0.7, len(shared) * 0.25)
                db.add_relationship(doc_id, other_id, "associative", strength, {
                    "auto_detected": True,
                    "reason": "shared_tags",
                    "tags": list(shared),
                    "model": metadata["model"]
                })
                relationships_created += 1
            
            # Learning progression → temporal relationship
            elif doc_difficulty and other_difficulty:
                levels = {"beginner": 1, "intermediate": 2, "advanced": 3}
                if abs(levels.get(doc_difficulty, 2) - levels.get(other_difficulty, 2)) == 1:
                    db.add_relationship(doc_id, other_id, "temporal", 0.85, {
                        "auto_detected": True,
                        "reason": "learning_progression",
                        "model": metadata["model"]
                    })
                    relationships_created += 1
        
        return relationships_created
    
    def compare_search_performance(self, test_queries: List[str]) -> Dict[str, Any]:
        """Compare search performance across all models"""
        
        print(f"\n🔍 Comparing Search Performance")
        print("=" * 35)
        
        comparison_results = {}
        
        for query in test_queries:
            print(f"\n❓ Query: '{query}'")
            query_results = {}
            
            for model_name in self.models.keys():
                db = self.databases[model_name]
                
                if db.vector_count() == 0:
                    print(f"   ⚠️ {model_name}: No documents to search")
                    continue
                
                # Encode query with model
                query_embedding = self.encode_text(model_name, query)
                
                # Perform search with relationship awareness
                start_time = time.time()
                results = db.search(query_embedding, rudradb.SearchParams(
                    top_k=5,
                    include_relationships=True,
                    max_hops=2,
                    relationship_weight=0.3
                ))
                search_time = time.time() - start_time
                
                # Analyze results
                direct_results = [r for r in results if r.hop_count == 0]
                relationship_results = [r for r in results if r.hop_count > 0]
                
                query_results[model_name] = {
                    "total_results": len(results),
                    "direct_results": len(direct_results),
                    "relationship_results": len(relationship_results),
                    "search_time": search_time,
                    "avg_score": np.mean([r.combined_score for r in results]) if results else 0,
                    "dimension": db.dimension()
                }
                
                print(f"   📊 {model_name} ({db.dimension()}D):")
                print(f"      Results: {len(results)} ({len(relationship_results)} via relationships)")
                print(f"      Time: {search_time*1000:.1f}ms")
                print(f"      Avg Score: {query_results[model_name]['avg_score']:.3f}")
            
            comparison_results[query] = query_results
        
        return comparison_results
    
    def analyze_model_strengths(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze strengths and weaknesses of each model"""
        
        print(f"\n📈 Model Strengths Analysis")
        print("=" * 30)
        
        model_analysis = {}
        
        for model_name in self.models.keys():
            if model_name not in self.databases or self.databases[model_name].vector_count() == 0:
                continue
            
            # Aggregate performance metrics
            total_queries = len(comparison_results)
            avg_results = 0
            avg_relationship_results = 0
            avg_search_time = 0
            avg_score = 0
            
            for query_results in comparison_results.values():
                if model_name in query_results:
                    result = query_results[model_name]
                    avg_results += result["total_results"]
                    avg_relationship_results += result["relationship_results"]
                    avg_search_time += result["search_time"]
                    avg_score += result["avg_score"]
            
            if total_queries > 0:
                avg_results /= total_queries
                avg_relationship_results /= total_queries
                avg_search_time /= total_queries
                avg_score /= total_queries
            
            model_info = self.models[model_name]
            db = self.databases[model_name]
            
            model_analysis[model_name] = {
                "dimension": db.dimension(),
                "documents": db.vector_count(),
                "relationships": db.relationship_count(),
                "avg_results_per_query": avg_results,
                "avg_relationship_discoveries": avg_relationship_results,
                "avg_search_time_ms": avg_search_time * 1000,
                "avg_relevance_score": avg_score,
                "use_cases": model_info["use_cases"],
                "strengths": [],
                "recommendations": []
            }
            
            # Determine strengths
            if avg_relationship_results > 1:
                model_analysis[model_name]["strengths"].append("Good relationship discovery")
            
            if avg_search_time < 0.01:  # < 10ms
                model_analysis[model_name]["strengths"].append("Fast search")
            
            if avg_score > 0.5:
                model_analysis[model_name]["strengths"].append("High relevance scores")
            
            # Generate recommendations
            if db.dimension() < 500:
                model_analysis[model_name]["recommendations"].append("Good for fast retrieval")
            else:
                model_analysis[model_name]["recommendations"].append("Better for semantic understanding")
            
            if db.relationship_count() > db.vector_count():
                model_analysis[model_name]["recommendations"].append("Excellent for relationship modeling")
        
        # Print analysis
        for model_name, analysis in model_analysis.items():
            print(f"\n🤖 {model_name}:")
            print(f"   📊 {analysis['dimension']}D, {analysis['documents']} docs, {analysis['relationships']} relationships")
            print(f"   ⚡ Avg search time: {analysis['avg_search_time_ms']:.1f}ms")
            print(f"   🎯 Avg relevance: {analysis['avg_relevance_score']:.3f}")
            print(f"   🔗 Avg relationship discoveries: {analysis['avg_relationship_discoveries']:.1f}")
            
            if analysis["strengths"]:
                print(f"   💪 Strengths: {', '.join(analysis['strengths'])}")
            
            if analysis["recommendations"]:
                print(f"   💡 Best for: {', '.join(analysis['recommendations'])}")
        
        return model_analysis

def demo_multi_model_comparison():
    """Comprehensive multi-model comparison demo"""
    
    print("🚀 Multi-Model Comparison with Auto-Dimension Detection")
    print("=" * 60)
    
    # Initialize comparison system
    comparison = MultiModelComparison()
    
    # Add different models
    models_config = [
        {
            "name": "MiniLM",
            "config": {
                "type": "sentence_transformer",
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
                "description": "Fast, lightweight model",
                "use_cases": ["quick_retrieval", "general_purpose", "prototyping"]
            }
        },
        {
            "name": "MPNet", 
            "config": {
                "type": "sentence_transformer",
                "model_id": "sentence-transformers/all-mpnet-base-v2",
                "dimension": 768,
                "description": "High quality embeddings",
                "use_cases": ["high_accuracy", "semantic_search", "research"]
            }
        },
        {
            "name": "DistilBERT",
            "config": {
                "type": "sentence_transformer",
                "model_id": "sentence-transformers/msmarco-distilbert-base-tas-b",
                "dimension": 768,
                "description": "Optimized for passage ranking",
                "use_cases": ["passage_ranking", "question_answering", "domain_specific"]
            }
        }
    ]
    
    # Add models to comparison
    for model_config in models_config:
        comparison.add_model(model_config["name"], model_config["config"])
    
    # Prepare test documents
    documents = [
        {
            "id": "ai_intro",
            "text": "Artificial Intelligence is transforming technology and society through advanced algorithms.",
            "metadata": {"category": "AI", "difficulty": "beginner", "tags": ["ai", "technology"]}
        },
        {
            "id": "ml_algorithms",
            "text": "Machine Learning algorithms enable computers to learn patterns from data automatically.",
            "metadata": {"category": "AI", "difficulty": "intermediate", "tags": ["ml", "algorithms", "data"]}
        },
        {
            "id": "deep_learning",
            "text": "Deep Learning uses neural networks with multiple layers for complex pattern recognition.",
            "metadata": {"category": "AI", "difficulty": "advanced", "tags": ["dl", "neural", "patterns"]}
        },
        {
            "id": "nlp_overview",
            "text": "Natural Language Processing enables computers to understand and generate human language.",
            "metadata": {"category": "NLP", "difficulty": "intermediate", "tags": ["nlp", "language", "text"]}
        },
        {
            "id": "computer_vision",
            "text": "Computer Vision allows machines to interpret and understand visual information from images.",
            "metadata": {"category": "CV", "difficulty": "intermediate", "tags": ["cv", "vision", "images"]}
        }
    ]
    
    # Add documents to all models
    results = comparison.add_documents_to_all_models(documents)
    
    print(f"\n📊 Document Addition Results:")
    for model_name, model_results in results.items():
        status = "✅" if model_results["dimension_match"] else "⚠️"
        print(f"   {status} {model_name}: {model_results['documents_added']} docs, "
              f"{model_results['relationships_created']} relationships")
    
    # Compare search performance
    test_queries = [
        "artificial intelligence and machine learning",
        "neural networks for pattern recognition", 
        "natural language understanding technologies"
    ]
    
    search_results = comparison.compare_search_performance(test_queries)
    
    # Analyze model strengths
    analysis = comparison.analyze_model_strengths(search_results)
    
    print(f"\n🏆 Comparison Summary")
    print("=" * 25)
    
    # Find best model for different criteria
    best_speed = min(analysis.keys(), key=lambda m: analysis[m]["avg_search_time_ms"])
    best_relevance = max(analysis.keys(), key=lambda m: analysis[m]["avg_relevance_score"])
    best_relationships = max(analysis.keys(), key=lambda m: analysis[m]["avg_relationship_discoveries"])
    
    print(f"🏃 Fastest: {best_speed} ({analysis[best_speed]['avg_search_time_ms']:.1f}ms)")
    print(f"🎯 Most relevant: {best_relevance} (score: {analysis[best_relevance]['avg_relevance_score']:.3f})")
    print(f"🔗 Best relationships: {best_relationships} ({analysis[best_relationships]['avg_relationship_discoveries']:.1f} per query)")
    
    print(f"\n💡 Model Selection Guide:")
    print(f"   • For speed: Choose lower-dimensional models (384D)")
    print(f"   • For accuracy: Choose higher-dimensional models (768D+)")
    print(f"   • For relationships: Models with good semantic understanding")
    print(f"   • RudraDB-Opin: Auto-dimension detection works with all!")

def benchmark_dimension_impact():
    """Benchmark impact of different dimensions on performance"""
    
    print(f"\n⚡ Dimension Impact Benchmark")
    print("=" * 35)
    
    # Test different dimensions
    dimensions_to_test = [128, 256, 384, 512, 768, 1024]
    benchmark_results = {}
    
    for dimension in dimensions_to_test:
        print(f"\n🧪 Testing {dimension}D embeddings:")
        
        db = rudradb.RudraDB()  # Auto-dimension detection
        
        # Add test vectors
        start_time = time.time()
        for i in range(20):  # Within Opin limits
            embedding = np.random.rand(dimension).astype(np.float32)
            db.add_vector(f"vec_{i}", embedding, {"index": i})
        
        add_time = time.time() - start_time
        
        # Test search performance
        query = np.random.rand(dimension).astype(np.float32)
        
        start_time = time.time()
        results = db.search(query, rudradb.SearchParams(top_k=5))
        search_time = time.time() - start_time
        
        benchmark_results[dimension] = {
            "auto_detected_dim": db.dimension(),
            "add_time": add_time,
            "search_time": search_time,
            "dimension_match": db.dimension() == dimension
        }
        
        match_status = "✅" if benchmark_results[dimension]["dimension_match"] else "❌"
        print(f"   {match_status} Auto-detected: {db.dimension()}D")
        print(f"   📝 Add time: {add_time*1000:.1f}ms")
        print(f"   🔍 Search time: {search_time*1000:.1f}ms")
    
    print(f"\n📈 Dimension Performance Summary:")
    for dim, results in benchmark_results.items():
        efficiency = results["search_time"] * 1000  # Convert to ms
        print(f"   {dim}D: {efficiency:.1f}ms search time")
    
    # Find optimal dimension
    optimal_dim = min(benchmark_results.keys(), 
                     key=lambda d: benchmark_results[d]["search_time"])
    print(f"\n🏆 Most efficient: {optimal_dim}D ({benchmark_results[optimal_dim]['search_time']*1000:.1f}ms)")

if __name__ == "__main__":
    try:
        demo_multi_model_comparison()
        benchmark_dimension_impact()
        
        print(f"\n✅ Multi-model comparison completed!")
        print(f"\n🎯 Key Insights:")
        print(f"   • Auto-dimension detection works with any model")
        print(f"   • Different dimensions offer different trade-offs")
        print(f"   • Relationship-aware search enhances all models")
        print(f"   • Model choice depends on your specific use case")
        print(f"   • RudraDB-Opin makes model experimentation easy!")
        
    except Exception as e:
        print(f"❌ Error in multi-model comparison: {e}")
        print(f"💡 This example works with or without real ML libraries")
        print(f"📚 Check troubleshooting/debug_guide.py for help")
