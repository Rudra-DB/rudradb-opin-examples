#!/usr/bin/env python3
"""
🚀 RudraDB-Opin Quick Start Example

This is the famous "30-second demo" that shows how to get started with 
relationship-aware vector search in under a minute.

Features demonstrated:
- Auto-dimension detection (works with any embedding model)
- Vector storage with rich metadata
- Relationship building (5 types available)
- Relationship-aware search vs traditional similarity search
- Multi-hop discovery through connection chains
"""

import rudradb
import numpy as np
import time
from datetime import datetime

def main():
    print("🧬 RudraDB-Opin Quick Start Demo")
    print("=" * 50)
    
    # 1. Create database (auto-detects embedding dimensions!)
    print("\n1️⃣ Creating RudraDB-Opin database...")
    db = rudradb.RudraDB()  # No dimension specification needed!
    print(f"   ✅ Database created with auto-dimension detection")
    
    # 2. Add vectors with rich metadata
    print("\n2️⃣ Adding AI/ML documents...")
    
    # Sample documents about AI/ML
    documents = [
        {
            "id": "ai_intro",
            "title": "Introduction to Artificial Intelligence",
            "content": "AI fundamentals and core concepts for beginners",
            "category": "AI",
            "difficulty": "beginner",
            "tags": ["ai", "introduction", "basics"]
        },
        {
            "id": "ml_basics", 
            "title": "Machine Learning Fundamentals",
            "content": "ML algorithms and supervised learning techniques",
            "category": "AI", 
            "difficulty": "intermediate",
            "tags": ["ml", "algorithms", "supervised"]
        },
        {
            "id": "deep_learning",
            "title": "Deep Learning with Neural Networks", 
            "content": "Neural networks and deep learning architectures",
            "category": "AI",
            "difficulty": "advanced", 
            "tags": ["deep", "neural", "networks"]
        },
        {
            "id": "python_ml",
            "title": "Python for Machine Learning",
            "content": "Using Python libraries for ML development",
            "category": "Programming",
            "difficulty": "intermediate",
            "tags": ["python", "ml", "programming"]
        },
        {
            "id": "data_science",
            "title": "Data Science Fundamentals",
            "content": "Data analysis and statistical methods",
            "category": "Data Science", 
            "difficulty": "beginner",
            "tags": ["data", "statistics", "analysis"]
        }
    ]
    
    # Add documents with random embeddings (in real use, use actual embeddings)
    embeddings_added = 0
    for doc in documents:
        # Simulate 384-dimensional embeddings (common for sentence transformers)
        embedding = np.random.rand(384).astype(np.float32)
        
        # Add vector - dimension will be auto-detected from first vector!
        db.add_vector(doc["id"], embedding, {
            "title": doc["title"],
            "content": doc["content"], 
            "category": doc["category"],
            "difficulty": doc["difficulty"],
            "tags": doc["tags"],
            "added_at": datetime.now().isoformat()
        })
        embeddings_added += 1
    
    print(f"   ✅ Added {embeddings_added} documents")
    print(f"   🎯 Auto-detected dimension: {db.dimension()}D")
    
    # 3. Build relationships between documents
    print("\n3️⃣ Building intelligent relationships...")
    
    relationships = [
        # Learning progression (hierarchical)
        ("ai_intro", "ml_basics", "hierarchical", 0.9),
        ("ml_basics", "deep_learning", "hierarchical", 0.8),
        
        # Sequential learning (temporal)
        ("ai_intro", "ml_basics", "temporal", 0.8),
        ("ml_basics", "deep_learning", "temporal", 0.7),
        
        # Implementation connection (causal)
        ("ml_basics", "python_ml", "causal", 0.7),
        
        # Related fields (semantic) 
        ("ml_basics", "data_science", "semantic", 0.6),
        
        # Cross-domain (associative)
        ("python_ml", "data_science", "associative", 0.5)
    ]
    
    relationships_added = 0
    for source, target, rel_type, strength in relationships:
        db.add_relationship(source, target, rel_type, strength, {
            "created_at": datetime.now().isoformat(),
            "reason": f"Learning connection: {rel_type}"
        })
        relationships_added += 1
    
    print(f"   ✅ Built {relationships_added} relationships")
    print(f"   📊 Types: hierarchical, temporal, causal, semantic, associative")
    
    # 4. Compare traditional vs relationship-aware search
    print("\n4️⃣ Comparing search approaches...")
    
    # Create a query (looking for machine learning content)
    query_embedding = np.random.rand(384).astype(np.float32)
    
    # Traditional similarity-only search
    print("\n   🔍 Traditional Similarity Search:")
    start_time = time.time()
    traditional_params = rudradb.SearchParams(
        top_k=5,
        include_relationships=False,  # No relationships
        similarity_threshold=0.1
    )
    traditional_results = db.search(query_embedding, traditional_params)
    traditional_time = time.time() - start_time
    
    print(f"     Found {len(traditional_results)} results in {traditional_time*1000:.1f}ms")
    for i, result in enumerate(traditional_results[:3], 1):
        vector = db.get_vector(result.vector_id)
        title = vector['metadata']['title']
        print(f"     {i}. {title} (similarity: {result.similarity_score:.3f})")
    
    # Relationship-aware search
    print("\n   🧠 Relationship-Aware Search:")
    start_time = time.time()
    relationship_params = rudradb.SearchParams(
        top_k=5,
        include_relationships=True,   # Include relationships!
        max_hops=2,                   # Multi-hop discovery
        relationship_weight=0.4,      # Balance similarity + relationships
        similarity_threshold=0.1
    )
    relationship_results = db.search(query_embedding, relationship_params)
    relationship_time = time.time() - start_time
    
    print(f"     Found {len(relationship_results)} results in {relationship_time*1000:.1f}ms")
    for i, result in enumerate(relationship_results[:3], 1):
        vector = db.get_vector(result.vector_id)
        title = vector['metadata']['title']
        connection = "Direct match" if result.hop_count == 0 else f"{result.hop_count}-hop connection"
        print(f"     {i}. {title}")
        print(f"        Connection: {connection} (combined score: {result.combined_score:.3f})")
    
    # 5. Multi-hop relationship discovery
    print("\n5️⃣ Multi-hop relationship discovery...")
    
    # Find all documents connected to "ai_intro" through relationships
    connected_vectors = db.get_connected_vectors("ai_intro", max_hops=2)
    
    print(f"   📈 Documents connected to 'AI Introduction':")
    for vector_data, hop_count in connected_vectors:
        title = vector_data['metadata']['title']
        if hop_count == 0:
            print(f"     📄 {title} (starting document)")
        else:
            print(f"     🔗 {title} ({hop_count} hop{'s' if hop_count > 1 else ''} away)")
    
    # 6. Database statistics
    print("\n6️⃣ Database statistics...")
    stats = db.get_statistics()
    usage = stats['capacity_usage']
    
    print(f"   📊 Vectors: {stats['vector_count']}/{rudradb.MAX_VECTORS} ({usage['vector_usage_percent']:.1f}%)")
    print(f"   🔗 Relationships: {stats['relationship_count']}/{rudradb.MAX_RELATIONSHIPS} ({usage['relationship_usage_percent']:.1f}%)")
    print(f"   🎯 Dimension: {stats['dimension']}")
    print(f"   📏 Max hops: {rudradb.MAX_HOPS}")
    
    # 7. Key insights
    print("\n✨ Key Insights:")
    additional_discoveries = len([r for r in relationship_results if r.hop_count > 0])
    
    insights = [
        f"Auto-dimension detection worked seamlessly with {db.dimension()}D embeddings",
        f"Built {relationships_added} relationships using 5 different types",
        f"Relationship-aware search discovered {additional_discoveries} additional relevant documents",
        f"Multi-hop traversal found {len(connected_vectors)-1} connected documents",
        f"Perfect learning capacity: {usage['vector_usage_percent']:.1f}% vectors, {usage['relationship_usage_percent']:.1f}% relationships used"
    ]
    
    for insight in insights:
        print(f"   💡 {insight}")
    
    print(f"\n🎉 Quick Start Complete!")
    print(f"   You've experienced the power of relationship-aware vector search!")
    print(f"   Ready to build amazing AI applications with RudraDB-Opin!")
    
    # Upgrade suggestion if approaching limits
    if usage['vector_usage_percent'] > 50 or usage['relationship_usage_percent'] > 50:
        print(f"\n🚀 Ready for More?")
        print(f"   When you need more capacity: {rudradb.UPGRADE_MESSAGE}")
        print(f"   Same API, 1000x more capacity!")

if __name__ == "__main__":
    # Check if RudraDB-Opin is available
    try:
        import rudradb
        print(f"🎯 Using RudraDB-Opin v{rudradb.__version__}")
        main()
    except ImportError:
        print("❌ RudraDB-Opin not found!")
        print("   Install with: pip install rudradb-opin")
        print("   Then run this example again")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have numpy installed: pip install numpy")
