#!/usr/bin/env python3
"""
🎯 Auto-Dimension Detection Demo

This example demonstrates RudraDB-Opin's revolutionary auto-dimension detection
that works with any ML model automatically - no manual configuration needed!

Features demonstrated:
- Zero-configuration setup
- Works with any embedding model 
- Seamless model switching
- Multi-model compatibility
- Dimension validation and error handling
"""

import rudradb
import numpy as np
import time

def demonstrate_auto_detection():
    """Demonstrate auto-dimension detection with different embedding sizes"""
    
    print("🎯 RudraDB-Opin Auto-Dimension Detection Demo")
    print("=" * 60)
    
    print("\n🤖 Revolutionary Feature: Works with ANY ML model automatically!")
    print("   No dimension specification needed - just start adding vectors!")
    
    # Common embedding dimensions used by popular models
    model_dimensions = {
        "OpenAI text-embedding-ada-002": 1536,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "BERT-base": 768,
        "Word2Vec Google News": 300,
        "GloVe": 300,
        "Custom small model": 128,
        "Custom large model": 2048
    }
    
    print(f"\n📊 Testing with {len(model_dimensions)} different model dimensions:")
    
    for model_name, dimension in model_dimensions.items():
        print(f"\n🔬 Testing: {model_name} ({dimension}D)")
        
        # Create fresh database for each model
        db = rudradb.RudraDB()
        
        # Check initial state
        print(f"   Initial dimension: {db.dimension()} (None - awaiting auto-detection)")
        
        # Generate simulated embedding for this "model"
        embedding = np.random.rand(dimension).astype(np.float32)
        
        # Add vector - dimension will be auto-detected!
        start_time = time.time()
        db.add_vector(f"test_doc_{dimension}", embedding, {
            "model": model_name,
            "dimension": dimension,
            "test": "auto_detection"
        })
        detection_time = time.time() - start_time
        
        # Verify auto-detection worked
        detected_dim = db.dimension()
        print(f"   ✅ Auto-detected: {detected_dim}D in {detection_time*1000:.2f}ms")
        
        if detected_dim == dimension:
            print(f"   🎉 Perfect match! Database configured for {model_name}")
        else:
            print(f"   ❌ Mismatch: expected {dimension}D, got {detected_dim}D")
        
        # Test dimension consistency
        try:
            # Try to add another vector with same dimension
            embedding2 = np.random.rand(dimension).astype(np.float32)
            db.add_vector(f"test_doc2_{dimension}", embedding2, {"second": "vector"})
            print(f"   ✅ Consistency check passed: second {dimension}D vector accepted")
        except Exception as e:
            print(f"   ❌ Consistency error: {e}")
        
        # Test dimension validation
        try:
            # Try to add vector with wrong dimension
            wrong_dim = dimension + 100
            wrong_embedding = np.random.rand(wrong_dim).astype(np.float32)
            db.add_vector(f"wrong_dim_{wrong_dim}", wrong_embedding)
            print(f"   ❌ Should have rejected {wrong_dim}D vector!")
        except Exception as e:
            print(f"   ✅ Correctly rejected wrong dimension: {type(e).__name__}")

def demonstrate_model_switching():
    """Show how to switch between different models"""
    
    print(f"\n\n🔄 Model Switching Demo")
    print("=" * 30)
    
    print("Scenario: You want to use different embedding models for different tasks")
    print("Solution: Create separate RudraDB-Opin instances for each model\n")
    
    # Model 1: Small, fast model for quick prototyping
    print("1️⃣ Quick Prototyping Model (384D):")
    db_small = rudradb.RudraDB()
    small_embedding = np.random.rand(384).astype(np.float32)
    db_small.add_vector("prototype_doc", small_embedding, {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "use_case": "rapid_prototyping",
        "speed": "fast"
    })
    print(f"   ✅ Created: {db_small.dimension()}D database for prototyping")
    
    # Model 2: High-quality model for production
    print("\n2️⃣ Production Model (1536D):")
    db_large = rudradb.RudraDB()
    large_embedding = np.random.rand(1536).astype(np.float32)
    db_large.add_vector("production_doc", large_embedding, {
        "model": "text-embedding-ada-002",
        "use_case": "production",
        "quality": "high"
    })
    print(f"   ✅ Created: {db_large.dimension()}D database for production")
    
    # Model 3: Specialized model for domain-specific tasks
    print("\n3️⃣ Domain-Specific Model (768D):")
    db_domain = rudradb.RudraDB()
    domain_embedding = np.random.rand(768).astype(np.float32)
    db_domain.add_vector("domain_doc", domain_embedding, {
        "model": "sentence-transformers/all-mpnet-base-v2",
        "use_case": "domain_specific",
        "specialization": "scientific_papers"
    })
    print(f"   ✅ Created: {db_domain.dimension()}D database for domain tasks")
    
    print(f"\n💡 Key Benefits:")
    benefits = [
        "Each database automatically configured for its model",
        "No manual dimension management needed",
        "Models isolated - no interference",
        "Same API for all databases",
        "Easy to switch between models for different tasks"
    ]
    
    for benefit in benefits:
        print(f"   • {benefit}")

def demonstrate_real_world_integration():
    """Show integration patterns with popular libraries"""
    
    print(f"\n\n🌍 Real-World Integration Patterns")
    print("=" * 40)
    
    print("Here's how auto-dimension detection works with popular ML libraries:\n")
    
    # Simulate different library patterns
    integrations = [
        {
            "name": "Sentence Transformers",
            "code": """
# With sentence transformers - 384D
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
db = rudradb.RudraDB()  # Auto-detects 384D

text = "Machine learning is fascinating"
embedding = model.encode([text])[0]
db.add_vector("doc1", embedding.astype(np.float32))
# ✅ Dimension: 384D auto-detected!
            """,
            "dimension": 384
        },
        {
            "name": "OpenAI Embeddings", 
            "code": """
# With OpenAI - 1536D
import openai

response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input="Your text here"
)

db = rudradb.RudraDB()  # Auto-detects 1536D
embedding = np.array(response['data'][0]['embedding'])
db.add_vector("doc1", embedding.astype(np.float32))
# ✅ Dimension: 1536D auto-detected!
            """,
            "dimension": 1536
        },
        {
            "name": "HuggingFace Transformers",
            "code": """
# With HuggingFace - 768D
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

db = rudradb.RudraDB()  # Auto-detects 768D
# ... process text and get embedding ...
db.add_vector("doc1", embedding.astype(np.float32))
# ✅ Dimension: 768D auto-detected!
            """,
            "dimension": 768
        }
    ]
    
    for integration in integrations:
        print(f"📚 {integration['name']} ({integration['dimension']}D):")
        print(f"```python{integration['code']}```")
        print()
        
        # Simulate the integration
        db = rudradb.RudraDB()
        embedding = np.random.rand(integration['dimension']).astype(np.float32)
        db.add_vector("sample_doc", embedding, {"library": integration['name']})
        print(f"   ✅ Simulation: {db.dimension()}D auto-detected for {integration['name']}")
        print()

def test_error_handling():
    """Test error handling and edge cases"""
    
    print(f"\n\n🛡️ Error Handling & Edge Cases")
    print("=" * 35)
    
    print("Testing robust error handling for edge cases:\n")
    
    # Test 1: Invalid embedding types
    print("1️⃣ Invalid embedding type handling:")
    db = rudradb.RudraDB()
    
    invalid_embeddings = [
        ("Python list", [1, 2, 3, 4]),
        ("Wrong dtype", np.array([1, 2, 3], dtype=np.int32)),
        ("2D array", np.array([[1, 2], [3, 4]], dtype=np.float32)),
        ("Empty array", np.array([], dtype=np.float32))
    ]
    
    for test_name, invalid_embedding in invalid_embeddings:
        try:
            db.add_vector("test", invalid_embedding)
            print(f"   ❌ {test_name}: Should have been rejected!")
        except Exception as e:
            print(f"   ✅ {test_name}: Correctly rejected ({type(e).__name__})")
    
    # Test 2: Dimension consistency
    print(f"\n2️⃣ Dimension consistency enforcement:")
    db2 = rudradb.RudraDB()
    
    # Add first vector (sets dimension)
    first_embedding = np.random.rand(384).astype(np.float32)
    db2.add_vector("first", first_embedding)
    print(f"   ✅ First vector: {db2.dimension()}D established")
    
    # Try incompatible dimensions
    incompatible_dims = [128, 512, 768, 1536]
    for dim in incompatible_dims:
        try:
            wrong_embedding = np.random.rand(dim).astype(np.float32)
            db2.add_vector(f"wrong_{dim}", wrong_embedding)
            print(f"   ❌ {dim}D: Should have been rejected!")
        except Exception as e:
            print(f"   ✅ {dim}D: Correctly rejected (dimension mismatch)")
    
    # Test 3: Valid same dimension
    try:
        compatible_embedding = np.random.rand(384).astype(np.float32)
        db2.add_vector("compatible", compatible_embedding)
        print(f"   ✅ 384D: Correctly accepted (same dimension)")
    except Exception as e:
        print(f"   ❌ 384D: Unexpected rejection: {e}")

def main():
    """Run all auto-dimension detection demonstrations"""
    
    try:
        # Core auto-detection demo
        demonstrate_auto_detection()
        
        # Model switching patterns
        demonstrate_model_switching()
        
        # Real-world integration examples
        demonstrate_real_world_integration()
        
        # Error handling tests
        test_error_handling()
        
        # Summary
        print(f"\n\n🎉 Auto-Dimension Detection Demo Complete!")
        print("=" * 50)
        
        key_takeaways = [
            "Works with ANY embedding model - no configuration needed",
            "Automatically detects dimension from first vector added", 
            "Enforces dimension consistency for data integrity",
            "Robust error handling for invalid inputs",
            "Perfect for multi-model workflows",
            "Same API regardless of embedding dimension"
        ]
        
        print(f"\n🔑 Key Takeaways:")
        for takeaway in key_takeaways:
            print(f"   • {takeaway}")
        
        print(f"\n🚀 Ready to use with your favorite ML models!")
        print(f"   Just add vectors - RudraDB-Opin handles the rest!")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("💡 Make sure rudradb-opin is installed: pip install rudradb-opin")

if __name__ == "__main__":
    main()
