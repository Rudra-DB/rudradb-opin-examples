#!/usr/bin/env python3
"""
Dimension Troubleshooting Guide for RudraDB-Opin
===============================================

This script demonstrates common dimension-related issues and their solutions.
Covers auto-dimension detection, dimension mismatches, and best practices.

Run this script to understand and resolve dimension problems.
"""

import rudradb
import numpy as np
import sys
import traceback

def test_auto_dimension_detection():
    """Test 1: Auto-dimension detection functionality"""
    
    print("🔍 Test 1: Auto-Dimension Detection")
    print("=" * 40)
    
    try:
        # Create database with auto-detection
        db = rudradb.RudraDB()
        print(f"✅ Database created with auto-dimension detection")
        
        # Check initial state
        initial_dim = db.dimension()
        print(f"📊 Initial dimension: {initial_dim} (None is expected)")
        
        if initial_dim is not None:
            print("⚠️ Warning: Expected None before adding vectors")
        
        # Add first vector - triggers auto-detection
        test_embedding = np.random.rand(384).astype(np.float32)
        db.add_vector("test_vector", test_embedding)
        
        detected_dim = db.dimension()
        print(f"🎯 Auto-detected dimension: {detected_dim}")
        
        if detected_dim == 384:
            print("✅ Auto-dimension detection working correctly!")
        else:
            print(f"❌ Expected 384, got {detected_dim}")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-dimension detection failed: {e}")
        traceback.print_exc()
        return False

def test_dimension_mismatch_scenarios():
    """Test 2: Common dimension mismatch scenarios"""
    
    print("\n🔍 Test 2: Dimension Mismatch Scenarios")
    print("=" * 45)
    
    db = rudradb.RudraDB()
    
    # Establish dimension with first vector
    first_embedding = np.random.rand(256).astype(np.float32)
    db.add_vector("first", first_embedding)
    established_dim = db.dimension()
    
    print(f"📊 Established dimension: {established_dim}")
    
    # Test different dimension scenarios
    test_dimensions = [128, 256, 384, 512, 768, 1536]
    
    for test_dim in test_dimensions:
        try:
            test_embedding = np.random.rand(test_dim).astype(np.float32)
            db.add_vector(f"test_{test_dim}", test_embedding)
            
            if test_dim == established_dim:
                print(f"✅ {test_dim}D vector added successfully (matches established dimension)")
            else:
                print(f"❌ Unexpected: {test_dim}D vector should have been rejected")
                
        except Exception as e:
            if test_dim == established_dim:
                print(f"❌ Unexpected error for matching dimension {test_dim}: {e}")
            else:
                print(f"✅ Correctly rejected {test_dim}D vector (expected {established_dim}D)")

def test_embedding_shape_validation():
    """Test 3: Embedding shape and type validation"""
    
    print("\n🔍 Test 3: Embedding Shape and Type Validation")
    print("=" * 50)
    
    db = rudradb.RudraDB()
    
    # Test cases for different embedding formats
    test_cases = [
        ("Correct format", np.random.rand(384).astype(np.float32)),
        ("Wrong dtype (float64)", np.random.rand(384).astype(np.float64)),
        ("Wrong shape (2D)", np.random.rand(384, 1).astype(np.float32)),
        ("List instead of array", np.random.rand(384).tolist()),
        ("Contains NaN", np.array([np.nan] * 384, dtype=np.float32)),
        ("Contains Infinity", np.array([np.inf] * 384, dtype=np.float32)),
    ]
    
    for test_name, embedding in test_cases:
        try:
            db.add_vector(f"test_{test_name.replace(' ', '_')}", embedding)
            
            if test_name == "Correct format":
                print(f"✅ {test_name}: Added successfully")
            else:
                print(f"⚠️ {test_name}: Added but might cause issues")
                
        except Exception as e:
            if test_name == "Correct format":
                print(f"❌ {test_name}: Should have been accepted - {e}")
            else:
                print(f"✅ {test_name}: Correctly rejected - {type(e).__name__}")

def demonstrate_dimension_best_practices():
    """Demonstrate best practices for handling dimensions"""
    
    print("\n💡 Dimension Best Practices")
    print("=" * 35)
    
    # Practice 1: Use auto-detection for flexibility
    print("1. ✅ Use auto-detection for flexibility:")
    db1 = rudradb.RudraDB()  # Will auto-detect from first vector
    embedding = np.random.rand(768).astype(np.float32)
    db1.add_vector("auto_detect", embedding)
    print(f"   Auto-detected: {db1.dimension()}D")
    
    # Practice 2: Specify dimension when you know it
    print("\n2. ✅ Specify dimension when you know it:")
    db2 = rudradb.RudraDB(dimension=384)  # Explicit dimension
    print(f"   Explicitly set: {db2.dimension()}D")
    
    # Practice 3: Validate embedding shapes before adding
    print("\n3. ✅ Validate embedding shapes:")
    def safe_add_vector(db, vec_id, embedding, metadata=None):
        """Safely add vector with shape validation"""
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        if len(embedding.shape) != 1:
            raise ValueError(f"Expected 1D array, got shape {embedding.shape}")
        
        expected_dim = db.dimension()
        if expected_dim is not None and embedding.shape[0] != expected_dim:
            raise ValueError(f"Dimension mismatch: expected {expected_dim}, got {embedding.shape[0]}")
        
        db.add_vector(vec_id, embedding, metadata)
        return True
    
    # Test the safe function
    test_embedding = np.random.rand(768).astype(np.float32)
    try:
        safe_add_vector(db1, "safe_test", test_embedding)
        print("   ✅ Safe add function works correctly")
    except Exception as e:
        print(f"   ❌ Safe add function failed: {e}")
    
    # Practice 4: Handle different embedding sources
    print("\n4. ✅ Handle different embedding sources:")
    
    embedding_sources = [
        ("OpenAI ada-002", 1536),
        ("Sentence Transformers MiniLM", 384),
        ("Sentence Transformers MPNet", 768),
        ("Custom model", 512)
    ]
    
    for source_name, dimension in embedding_sources:
        db = rudradb.RudraDB()  # Fresh DB for each source
        test_emb = np.random.rand(dimension).astype(np.float32)
        db.add_vector(f"{source_name.replace(' ', '_')}", test_emb)
        print(f"   {source_name}: {db.dimension()}D")

def debug_common_dimension_errors():
    """Debug common dimension-related errors"""
    
    print("\n🔧 Common Dimension Error Debugging")
    print("=" * 40)
    
    # Error 1: Dimension mismatch after auto-detection
    print("1. 🐛 Dimension mismatch after auto-detection:")
    db = rudradb.RudraDB()
    
    # Establish with 384D
    db.add_vector("first", np.random.rand(384).astype(np.float32))
    print(f"   Established dimension: {db.dimension()}")
    
    # Try to add different dimension
    try:
        db.add_vector("second", np.random.rand(768).astype(np.float32))
    except Exception as e:
        print(f"   ✅ Correctly caught dimension mismatch: {type(e).__name__}")
        print(f"      Solution: Use consistent embedding model or create new DB")
    
    # Error 2: Shape issues
    print("\n2. 🐛 Shape issues:")
    try:
        # 2D array instead of 1D
        bad_embedding = np.random.rand(384, 1).astype(np.float32)
        db.add_vector("bad_shape", bad_embedding)
    except Exception as e:
        print(f"   ✅ Correctly caught shape issue: {type(e).__name__}")
        print(f"      Solution: Use embedding.flatten() or embedding.squeeze()")
    
    # Error 3: Wrong data type
    print("\n3. 🐛 Data type issues:")
    try:
        # Float64 instead of float32
        bad_dtype = np.random.rand(384).astype(np.float64)
        db.add_vector("bad_dtype", bad_dtype)
        print("   ⚠️ Float64 accepted but should use float32 for efficiency")
    except Exception as e:
        print(f"   Issue with dtype: {e}")
    
    print(f"      Solution: Always use .astype(np.float32)")

def create_dimension_diagnostic_tool():
    """Create a comprehensive dimension diagnostic tool"""
    
    print("\n🔬 Dimension Diagnostic Tool")
    print("=" * 35)
    
    def diagnose_embedding(embedding, db=None):
        """Comprehensive embedding diagnosis"""
        print(f"🔍 Diagnosing embedding...")
        
        # Type check
        print(f"   Type: {type(embedding)}")
        
        # Convert to numpy if needed
        if not isinstance(embedding, np.ndarray):
            try:
                embedding = np.array(embedding)
                print(f"   ✅ Converted to numpy array")
            except:
                print(f"   ❌ Cannot convert to numpy array")
                return False
        
        # Shape check
        print(f"   Shape: {embedding.shape}")
        if len(embedding.shape) != 1:
            print(f"   ❌ Wrong shape: expected 1D, got {len(embedding.shape)}D")
            return False
        
        # Dtype check
        print(f"   Dtype: {embedding.dtype}")
        if embedding.dtype != np.float32:
            print(f"   ⚠️ Recommended: convert to float32 for efficiency")
        
        # Value checks
        if np.any(np.isnan(embedding)):
            print(f"   ❌ Contains NaN values")
            return False
        
        if np.any(np.isinf(embedding)):
            print(f"   ❌ Contains infinite values")
            return False
        
        # Dimension check against database
        if db is not None:
            db_dim = db.dimension()
            if db_dim is not None:
                if embedding.shape[0] != db_dim:
                    print(f"   ❌ Dimension mismatch: embedding {embedding.shape[0]}D, database {db_dim}D")
                    return False
                else:
                    print(f"   ✅ Dimension matches database: {db_dim}D")
            else:
                print(f"   ✅ Will establish database dimension: {embedding.shape[0]}D")
        
        print(f"   ✅ Embedding passes all checks!")
        return True
    
    # Test the diagnostic tool
    test_embeddings = [
        ("Good embedding", np.random.rand(384).astype(np.float32)),
        ("Bad shape", np.random.rand(384, 1).astype(np.float32)),
        ("Bad dtype", np.random.rand(384).astype(np.float64)),
        ("Contains NaN", np.array([np.nan] * 384, dtype=np.float32)),
    ]
    
    db = rudradb.RudraDB()
    
    for test_name, embedding in test_embeddings:
        print(f"\n📊 Testing: {test_name}")
        is_valid = diagnose_embedding(embedding, db)
        
        if is_valid and test_name == "Good embedding":
            db.add_vector(test_name.replace(" ", "_"), embedding)
            print(f"   ✅ Successfully added to database")

def main():
    """Run all dimension troubleshooting tests"""
    
    print("🧬 RudraDB-Opin Dimension Troubleshooting Guide")
    print("=" * 55)
    print("This guide helps you understand and resolve dimension-related issues.\n")
    
    # Run all tests
    tests = [
        test_auto_dimension_detection,
        test_dimension_mismatch_scenarios,
        test_embedding_shape_validation,
        demonstrate_dimension_best_practices,
        debug_common_dimension_errors,
        create_dimension_diagnostic_tool
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
    
    # Summary
    print(f"\n📋 Troubleshooting Summary")
    print("=" * 30)
    print(f"✅ All dimension concepts demonstrated")
    print(f"🔧 Common issues and solutions covered")
    print(f"💡 Best practices provided")
    
    print(f"\n🎯 Key Takeaways:")
    takeaways = [
        "Use auto-dimension detection for flexibility",
        "Always use np.float32 for embeddings", 
        "Validate embedding shapes before adding",
        "Keep embedding models consistent within a database",
        "Use the diagnostic tool when troubleshooting"
    ]
    
    for takeaway in takeaways:
        print(f"   • {takeaway}")
    
    print(f"\n📚 Need more help? Check:")
    print(f"   • debug_guide.py for general troubleshooting")
    print(f"   • comprehensive_test_suite.py for system validation")
    print(f"   • RudraDB documentation at docs.rudradb.com")

if __name__ == "__main__":
    main()
