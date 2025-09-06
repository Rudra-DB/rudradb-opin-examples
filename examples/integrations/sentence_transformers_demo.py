#!/usr/bin/env python3
"""
🤗 Sentence Transformers + RudraDB-Opin Integration

This example demonstrates seamless integration between Sentence Transformers
and RudraDB-Opin with auto-dimension detection and relationship-aware search.

Features demonstrated:
- Auto-dimension detection with Sentence Transformers
- Multiple model compatibility
- Batch processing with different models
- Semantic relationship building
- Advanced search strategies
- Performance optimization
- Model comparison and switching
"""

import rudradb
import numpy as np
import time
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

def create_sentence_transformer_demo():
    """Demonstrate basic Sentence Transformers + RudraDB-Opin integration"""
    
    print("🤗 Sentence Transformers + RudraDB-Opin Integration")
    print("=" * 60)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("⚠️  sentence-transformers not installed")
        print("   Install with: pip install sentence-transformers")
        print("   Using simulated embeddings for demo...")
        return create_simulated_demo()
    
    # Initialize popular Sentence Transformer model
    print("\n📥 Loading Sentence Transformer model...")
    model_name = 'all-MiniLM-L6-v2'  # Fast, high-quality model
    start_time = time.time()
    
    model = SentenceTransformer(model_name)
    load_time = time.time() - start_time
    
    print(f"   ✅ Loaded {model_name} in {load_time:.2f}s")
    print(f"   🎯 Model dimension: {model.get_sentence_embedding_dimension()}D")
    
    # Create RudraDB-Opin instance (auto-detects dimension!)
    db = rudradb.RudraDB()
    print(f"   ✅ RudraDB-Opin created with auto-dimension detection")
    
    # Sample documents about AI and technology
    documents = [
        {
            "id": "ai_revolution",
            "text": "Artificial Intelligence is revolutionizing industries through automation and intelligent decision making",
            "category": "AI",
            "tags": ["ai", "automation", "industry"],
            "difficulty": "beginner"
        },
        {
            "id": "machine_learning",
            "text": "Machine Learning enables computers to learn from data without explicit programming",
            "category": "AI", 
            "tags": ["ml", "data", "learning"],
            "difficulty": "intermediate"
        },
        {
            "id": "deep_learning",
            "text": "Deep Learning uses neural networks with multiple layers to process complex patterns",
            "category": "AI",
            "tags": ["deep", "neural", "patterns"], 
            "difficulty": "advanced"
        },
        {
            "id": "nlp_processing",
            "text": "Natural Language Processing helps computers understand and generate human language",
            "category": "NLP",
            "tags": ["nlp", "language", "text"],
            "difficulty": "intermediate"
        },
        {
            "id": "computer_vision",
            "text": "Computer Vision enables machines to interpret and analyze visual information",
            "category": "CV",
            "tags": ["vision", "images", "analysis"],
            "difficulty": "intermediate"
        }
    ]
    
    print(f"\n📄 Processing {len(documents)} documents with Sentence Transformers...")
    
    # Process documents
    embeddings_generated = 0
    start_time = time.time()
    
    for doc in documents:
        # Generate embedding using Sentence Transformers
        embedding = model.encode([doc["text"]])[0]
        
        # Add to RudraDB-Opin (auto-detects dimension on first add)
        db.add_vector(
            doc["id"], 
            embedding.astype(np.float32),  # Ensure float32 for efficiency
            {
                "text": doc["text"],
                "category": doc["category"],
                "tags": doc["tags"],
                "difficulty": doc["difficulty"],
                "model_used": model_name,
                "embedding_dim": len(embedding),
                "processed_at": datetime.now().isoformat()
            }
        )
        embeddings_generated += 1
        
        # Show auto-dimension detection on first document
        if embeddings_generated == 1:
            print(f"   🎯 Auto-detected dimension: {db.dimension()}D (matches {model_name})")
    
    processing_time = time.time() - start_time
    print(f"   ✅ Processed {embeddings_generated} documents in {processing_time:.3f}s")
    print(f"   ⚡ Rate: {embeddings_generated/processing_time:.1f} docs/second")
    
    return db, model, documents

def build_semantic_relationships(db, model, documents):
    """Build relationships based on semantic similarity"""
    
    print(f"\n🔗 Building semantic relationships...")
    
    # Build relationships based on categories and semantic similarity
    relationships = [
        # Hierarchical: AI field structure
        ("ai_revolution", "machine_learning", "hierarchical", 0.9, "AI encompasses ML"),
        ("machine_learning", "deep_learning", "hierarchical", 0.8, "ML includes deep learning"),
        
        # Semantic: Related AI fields
        ("deep_learning", "nlp_processing", "semantic", 0.7, "Both use neural approaches"),
        ("deep_learning", "computer_vision", "semantic", 0.7, "Both use deep networks"),
        
        # Temporal: Learning progression
        ("ai_revolution", "machine_learning", "temporal", 0.8, "ML follows AI introduction"),
        ("machine_learning", "deep_learning", "temporal", 0.7, "DL builds on ML"),
        
        # Associative: Cross-domain connections
        ("nlp_processing", "computer_vision", "associative", 0.5, "Both perception domains")
    ]
    
    relationships_built = 0
    for source, target, rel_type, strength, reason in relationships:
        try:
            db.add_relationship(source, target, rel_type, strength, {
                "reason": reason,
                "auto_generated": True,
                "semantic_basis": True,
                "created_at": datetime.now().isoformat()
            })
            relationships_built += 1
        except RuntimeError as e:
            if "capacity" in str(e).lower():
                print(f"   ⚠️  Relationship capacity reached at {relationships_built}")
                break
    
    print(f"   ✅ Built {relationships_built} semantic relationships")
    return relationships_built

def demonstrate_search_strategies(db, model, documents):
    """Demonstrate different search strategies"""
    
    print(f"\n🔍 Demonstrating Search Strategies...")
    
    # Test queries
    queries = [
        "How do machines learn from data?",
        "What is artificial intelligence?", 
        "Computer understanding of human language"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}️⃣ Query: '{query}'")
        
        # Generate query embedding
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            query_embedding = model.encode([query])[0].astype(np.float32)
        else:
            query_embedding = np.random.rand(db.dimension()).astype(np.float32)
        
        # Strategy 1: Pure similarity search
        print(f"   📊 Pure Similarity Search:")
        similarity_params = rudradb.SearchParams(
            top_k=3,
            include_relationships=False,
            similarity_threshold=0.0
        )
        
        start_time = time.time()
        similarity_results = db.search(query_embedding, similarity_params)
        similarity_time = time.time() - start_time
        
        print(f"      Found {len(similarity_results)} results in {similarity_time*1000:.2f}ms")
        for j, result in enumerate(similarity_results, 1):
            vector = db.get_vector(result.vector_id)
            title = vector['metadata']['text'][:40] + "..."
            print(f"      {j}. {title} (score: {result.similarity_score:.3f})")
        
        # Strategy 2: Relationship-aware search
        print(f"   🧠 Relationship-Aware Search:")
        relationship_params = rudradb.SearchParams(
            top_k=5,
            include_relationships=True,
            max_hops=2,
            relationship_weight=0.3,
            similarity_threshold=0.0
        )
        
        start_time = time.time()
        relationship_results = db.search(query_embedding, relationship_params)
        relationship_time = time.time() - start_time
        
        print(f"      Found {len(relationship_results)} results in {relationship_time*1000:.2f}ms")
        direct_results = [r for r in relationship_results if r.hop_count == 0]
        indirect_results = [r for r in relationship_results if r.hop_count > 0]
        
        print(f"      Direct matches: {len(direct_results)}, Relationship discoveries: {len(indirect_results)}")
        
        for j, result in enumerate(relationship_results[:3], 1):
            vector = db.get_vector(result.vector_id)
            title = vector['metadata']['text'][:35] + "..."
            connection = "direct" if result.hop_count == 0 else f"{result.hop_count}-hop"
            print(f"      {j}. {title}")
            print(f"         Connection: {connection} (score: {result.combined_score:.3f})")

def compare_sentence_transformer_models():
    """Compare different Sentence Transformer models with RudraDB-Opin"""
    
    print(f"\n\n🏆 Multi-Model Comparison")
    print("=" * 30)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("⚠️  Sentence Transformers not available - using simulation")
        return simulate_multi_model_comparison()
    
    # Different models to test (commented out larger models to avoid long download times)
    models_to_test = [
        ("all-MiniLM-L6-v2", 384),      # Fast, good quality
        # ("all-mpnet-base-v2", 768),   # Slower, higher quality  
        # ("all-distilroberta-v1", 768) # Alternative approach
    ]
    
    sample_text = "Machine learning enables computers to learn patterns from data"
    
    for model_name, expected_dim in models_to_test:
        print(f"\n🔬 Testing {model_name} ({expected_dim}D):")
        
        try:
            # Load model
            start_time = time.time()
            model = SentenceTransformer(model_name)
            load_time = time.time() - start_time
            
            # Create fresh database for this model
            db = rudradb.RudraDB()
            
            # Generate embedding
            start_time = time.time()
            embedding = model.encode([sample_text])[0]
            encoding_time = time.time() - start_time
            
            # Add to database
            db.add_vector("test_doc", embedding.astype(np.float32), {
                "model": model_name,
                "text": sample_text
            })
            
            actual_dim = db.dimension()
            
            print(f"   ✅ Model loaded in {load_time:.2f}s")
            print(f"   ✅ Encoding time: {encoding_time*1000:.2f}ms")
            print(f"   🎯 Expected: {expected_dim}D, Auto-detected: {actual_dim}D")
            
            if actual_dim == expected_dim:
                print(f"   🎉 Perfect match!")
            else:
                print(f"   ⚠️  Dimension mismatch")
                
        except Exception as e:
            print(f"   ❌ Error with {model_name}: {e}")

def simulate_multi_model_comparison():
    """Simulate multi-model comparison without sentence-transformers"""
    
    models_to_simulate = [
        ("all-MiniLM-L6-v2", 384),
        ("all-mpnet-base-v2", 768),
        ("all-distilroberta-v1", 768)
    ]
    
    sample_text = "Machine learning enables computers to learn patterns from data"
    
    for model_name, dimension in models_to_simulate:
        print(f"\n🔬 Simulating {model_name} ({dimension}D):")
        
        # Create database and simulate embedding
        db = rudradb.RudraDB()
        embedding = np.random.rand(dimension).astype(np.float32)
        
        db.add_vector("test_doc", embedding, {
            "model": model_name,
            "text": sample_text,
            "simulated": True
        })
        
        actual_dim = db.dimension()
        print(f"   🎯 Expected: {dimension}D, Auto-detected: {actual_dim}D")
        print(f"   ✅ Auto-dimension detection working perfectly")

def create_simulated_demo():
    """Create demo with simulated embeddings when sentence-transformers not available"""
    
    print("🔄 Creating simulated Sentence Transformers demo...")
    
    # Simulate all-MiniLM-L6-v2 (384 dimensions)
    db = rudradb.RudraDB()
    
    documents = [
        {
            "id": "ai_revolution",
            "text": "Artificial Intelligence is revolutionizing industries",
            "category": "AI"
        },
        {
            "id": "machine_learning", 
            "text": "Machine Learning enables computers to learn from data",
            "category": "AI"
        },
        {
            "id": "deep_learning",
            "text": "Deep Learning uses neural networks with multiple layers", 
            "category": "AI"
        }
    ]
    
    # Add documents with simulated 384D embeddings
    for doc in documents:
        embedding = np.random.rand(384).astype(np.float32)
        db.add_vector(doc["id"], embedding, {
            "text": doc["text"],
            "category": doc["category"],
            "model_used": "all-MiniLM-L6-v2 (simulated)",
            "simulated": True
        })
    
    print(f"   ✅ Created simulated demo with {db.vector_count()} documents")
    print(f"   🎯 Auto-detected dimension: {db.dimension()}D")
    
    return db, None, documents

def demonstrate_batch_processing(db, model, documents):
    """Demonstrate efficient batch processing"""
    
    print(f"\n📦 Batch Processing Demo...")
    
    # Additional documents for batch processing
    batch_documents = [
        "Natural language understanding in AI systems",
        "Computer vision applications in autonomous vehicles", 
        "Reinforcement learning for game playing",
        "Generative AI and creative applications",
        "Ethical considerations in artificial intelligence"
    ]
    
    print(f"   Processing {len(batch_documents)} additional documents...")
    
    if SENTENCE_TRANSFORMERS_AVAILABLE and model:
        # Batch encode for efficiency
        start_time = time.time()
        batch_embeddings = model.encode(batch_documents)
        encoding_time = time.time() - start_time
        
        print(f"   ⚡ Batch encoding: {encoding_time*1000:.2f}ms ({len(batch_documents)/encoding_time:.1f} docs/sec)")
        
        # Add to database
        start_time = time.time()
        for i, (text, embedding) in enumerate(zip(batch_documents, batch_embeddings)):
            db.add_vector(f"batch_doc_{i}", embedding.astype(np.float32), {
                "text": text,
                "batch_processed": True,
                "category": "AI"
            })
        add_time = time.time() - start_time
        
        print(f"   📊 Database insertion: {add_time*1000:.2f}ms")
        
    else:
        # Simulated batch processing
        for i, text in enumerate(batch_documents):
            embedding = np.random.rand(db.dimension()).astype(np.float32)
            db.add_vector(f"batch_doc_{i}", embedding, {
                "text": text,
                "batch_processed": True,
                "category": "AI",
                "simulated": True
            })
        
        print(f"   ✅ Simulated batch processing complete")
    
    print(f"   📊 Total documents: {db.vector_count()}")

def main():
    """Run complete Sentence Transformers + RudraDB-Opin demo"""
    
    try:
        # Basic integration demo
        db, model, documents = create_sentence_transformer_demo()
        
        # Build semantic relationships
        build_semantic_relationships(db, model, documents)
        
        # Demonstrate search strategies
        demonstrate_search_strategies(db, model, documents)
        
        # Multi-model comparison
        compare_sentence_transformer_models()
        
        # Batch processing demo
        demonstrate_batch_processing(db, model, documents)
        
        # Final statistics
        print(f"\n📊 Final Database Statistics:")
        stats = db.get_statistics()
        usage = stats['capacity_usage']
        
        print(f"   Vectors: {stats['vector_count']}/{rudradb.MAX_VECTORS} ({usage['vector_usage_percent']:.1f}%)")
        print(f"   Relationships: {stats['relationship_count']}/{rudradb.MAX_RELATIONSHIPS} ({usage['relationship_usage_percent']:.1f}%)")
        print(f"   Dimension: {stats['dimension']}D")
        
        # Summary
        print(f"\n🎉 Sentence Transformers + RudraDB-Opin Integration Complete!")
        print("=" * 70)
        
        key_benefits = [
            "Zero configuration - auto-dimension detection works seamlessly",
            "High-quality embeddings from Sentence Transformers", 
            "Relationship-aware search discovers hidden connections",
            "Efficient batch processing for large document sets",
            "Multiple model support with consistent API",
            "Production-ready performance within Opin limits"
        ]
        
        print(f"\n🔑 Key Benefits:")
        for benefit in key_benefits:
            print(f"   ✅ {benefit}")
        
        print(f"\n🚀 Ready to build powerful semantic search applications!")
        
        if usage['vector_usage_percent'] > 50:
            print(f"\n💡 Scaling Up:")
            print(f"   Current usage: {usage['vector_usage_percent']:.1f}% of Opin capacity")
            print(f"   Ready for production scale? Upgrade to full RudraDB!")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("💡 Install dependencies:")
        print("   pip install rudradb-opin sentence-transformers")

if __name__ == "__main__":
    main()
