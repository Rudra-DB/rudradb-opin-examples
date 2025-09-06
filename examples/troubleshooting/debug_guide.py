#!/usr/bin/env python3
"""
RudraDB-Opin: Troubleshooting and Debugging Guide

This example demonstrates common issues and their solutions:

1. Installation and import problems
2. Dimension mismatch errors  
3. Capacity limit handling
4. Relationship building issues
5. Search optimization problems
6. Performance debugging
7. Data export/import issues

Perfect for learning how to debug and resolve common problems!
"""

import rudradb
import numpy as np
import sys
import time
import json
from typing import Dict, List, Any, Optional


class TroubleshootingGuide:
    """Comprehensive troubleshooting and debugging guide"""
    
    def __init__(self):
        print("🔧 RudraDB-Opin Troubleshooting Guide")
        print("=" * 45)
        self._check_system_info()
    
    def _check_system_info(self):
        """Check and display system information"""
        print(f"\n📊 System Information:")
        print(f"   Python Version: {sys.version}")
        print(f"   RudraDB-Opin Version: {rudradb.__version__}")
        print(f"   NumPy Version: {np.__version__}")
        
        # Check RudraDB-Opin configuration
        print(f"\n🧬 RudraDB-Opin Configuration:")
        print(f"   Edition: {rudradb.EDITION}")
        print(f"   Max Vectors: {rudradb.MAX_VECTORS}")
        print(f"   Max Relationships: {rudradb.MAX_RELATIONSHIPS}")
        print(f"   Max Hops: {rudradb.MAX_HOPS}")
    
    def troubleshoot_installation_issues(self):
        """Diagnose and resolve installation problems"""
        print(f"\n🔧 1. INSTALLATION TROUBLESHOOTING")
        print(f"=" * 40)
        
        print(f"\n✅ Testing Basic Import...")
        try:
            import rudradb
            print(f"   ✅ RudraDB-Opin imported successfully")
            print(f"   Version: {rudradb.__version__}")
        except ImportError as e:
            print(f"   ❌ Import failed: {e}")
            print(f"\n💡 Solutions:")
            print(f"   1. Check installation: pip list | grep rudradb")
            print(f"   2. Reinstall: pip uninstall rudradb-opin && pip install rudradb-opin")
            print(f"   3. Check virtual environment: which python")
            print(f"   4. Try fresh environment: python -m venv test_env")
            return False
        
        print(f"\n✅ Testing Database Creation...")
        try:
            db = rudradb.RudraDB()
            print(f"   ✅ Database created successfully")
            print(f"   Auto-dimension detection: Ready")
        except Exception as e:
            print(f"   ❌ Database creation failed: {e}")
            print(f"\n💡 This might indicate a deeper installation issue")
            return False
        
        print(f"\n✅ Testing Basic Operations...")
        try:
            # Test vector addition
            test_embedding = np.random.rand(384).astype(np.float32)
            db.add_vector("test_vector", test_embedding, {"test": True})
            print(f"   ✅ Vector addition: Working")
            print(f"   Auto-detected dimension: {db.dimension()}")
            
            # Test relationship addition
            test_embedding2 = np.random.rand(384).astype(np.float32)
            db.add_vector("test_vector2", test_embedding2, {"test": True})
            db.add_relationship("test_vector", "test_vector2", "semantic", 0.8)
            print(f"   ✅ Relationship addition: Working")
            
            # Test search
            results = db.search(test_embedding)
            print(f"   ✅ Search functionality: Working ({len(results)} results)")
            
        except Exception as e:
            print(f"   ❌ Basic operations failed: {e}")
            return False
        
        print(f"\n🎉 Installation verification complete!")
        return True
    
    def troubleshoot_dimension_issues(self):
        """Diagnose and resolve dimension-related problems"""
        print(f"\n🎯 2. DIMENSION TROUBLESHOOTING")
        print(f"=" * 35)
        
        print(f"\n🔍 Testing Auto-Dimension Detection...")
        
        # Test 1: Fresh database with auto-detection
        db = rudradb.RudraDB()
        print(f"   Initial dimension: {db.dimension()} (None = auto-detection ready)")
        
        # Test 2: Add first vector
        print(f"\n✅ Testing First Vector Addition...")
        try:
            first_embedding = np.random.rand(256).astype(np.float32)
            db.add_vector("first_doc", first_embedding, {"order": 1})
            print(f"   ✅ First vector added successfully")
            print(f"   Auto-detected dimension: {db.dimension()}")
        except Exception as e:
            print(f"   ❌ First vector failed: {e}")
            return False
        
        # Test 3: Try adding vector with same dimension
        print(f"\n✅ Testing Same Dimension Vector...")
        try:
            same_dim_embedding = np.random.rand(256).astype(np.float32)
            db.add_vector("same_dim_doc", same_dim_embedding, {"order": 2})
            print(f"   ✅ Same dimension vector added successfully")
        except Exception as e:
            print(f"   ❌ Same dimension vector failed: {e}")
        
        # Test 4: Try adding vector with different dimension (should fail)
        print(f"\n⚠️ Testing Different Dimension Vector (should fail)...")
        try:
            different_dim_embedding = np.random.rand(512).astype(np.float32)
            db.add_vector("different_dim_doc", different_dim_embedding, {"order": 3})
            print(f"   ❌ ERROR: Different dimension vector should have been rejected!")
        except Exception as e:
            if "dimension" in str(e).lower():
                print(f"   ✅ Correctly rejected different dimension: {type(e).__name__}")
                print(f"   Error message: {str(e)[:80]}...")
            else:
                print(f"   ⚠️ Unexpected error: {e}")
        
        # Test 5: Common dimension issues and solutions
        print(f"\n💡 Common Dimension Issues and Solutions:")
        
        issues_solutions = [
            ("Wrong data type", "Use np.float32: embedding.astype(np.float32)"),
            ("Wrong shape", "Ensure 1D array: embedding.shape should be (N,)"),
            ("Mixed models", "Use consistent embedding model or separate databases"),
            ("Dimension mismatch", "Check embedding.shape[0] before adding"),
            ("Model switching", "Create new database for different dimensions")
        ]
        
        for issue, solution in issues_solutions:
            print(f"   • {issue}: {solution}")
        
        # Test 6: Demonstrate correct usage patterns
        print(f"\n✅ Best Practices for Dimension Management:")
        
        # Different embedding dimensions in separate databases
        test_dimensions = [128, 256, 384, 512, 768, 1536]
        
        for dim in test_dimensions[:3]:  # Test first 3 to avoid capacity limits
            try:
                test_db = rudradb.RudraDB()  # Fresh database for each dimension
                test_embedding = np.random.rand(dim).astype(np.float32)
                test_db.add_vector(f"test_{dim}d", test_embedding)
                print(f"   ✅ {dim}D embeddings: Working (auto-detected: {test_db.dimension()})")
            except Exception as e:
                print(f"   ❌ {dim}D embeddings failed: {e}")
        
        return True
    
    def troubleshoot_capacity_limits(self):
        """Demonstrate capacity limit handling and solutions"""
        print(f"\n📊 3. CAPACITY LIMIT TROUBLESHOOTING")
        print(f"=" * 40)
        
        print(f"📋 RudraDB-Opin Limits (by design):")
        print(f"   Max Vectors: {rudradb.MAX_VECTORS}")
        print(f"   Max Relationships: {rudradb.MAX_RELATIONSHIPS}")
        print(f"   Max Hops: {rudradb.MAX_HOPS}")
        print(f"   Purpose: Perfect for learning and tutorials!")
        
        # Create test database
        db = rudradb.RudraDB()
        
        # Test vector capacity behavior
        print(f"\n🔍 Testing Vector Capacity Behavior...")
        
        try:
            # Add vectors up to a reasonable number for testing
            test_count = min(15, rudradb.MAX_VECTORS)
            
            for i in range(test_count):
                embedding = np.random.rand(384).astype(np.float32)
                db.add_vector(f"capacity_test_{i}", embedding, {"index": i})
            
            print(f"   ✅ Added {test_count} vectors successfully")
            print(f"   Current usage: {db.vector_count()}/{rudradb.MAX_VECTORS}")
            
        except RuntimeError as e:
            if "RudraDB-Opin Vector Limit Reached" in str(e):
                print(f"   📊 Vector capacity limit reached at {db.vector_count()}")
                print(f"   This is expected behavior for learning!")
                self._show_capacity_solutions()
            else:
                print(f"   ❌ Unexpected error: {e}")
        
        # Test relationship capacity behavior
        print(f"\n🔍 Testing Relationship Capacity...")
        
        try:
            # Add some relationships for testing
            vectors = db.list_vectors()
            relationships_added = 0
            
            for i in range(min(10, len(vectors) - 1)):
                source = vectors[i]
                target = vectors[i + 1]
                db.add_relationship(source, target, "semantic", 0.8)
                relationships_added += 1
            
            print(f"   ✅ Added {relationships_added} relationships successfully")
            print(f"   Current usage: {db.relationship_count()}/{rudradb.MAX_RELATIONSHIPS}")
            
        except RuntimeError as e:
            if "RudraDB-Opin Relationship Limit Reached" in str(e):
                print(f"   📊 Relationship capacity limit reached")
                print(f"   This demonstrates the Opin learning limits!")
                self._show_capacity_solutions()
            else:
                print(f"   ❌ Unexpected error: {e}")
        
        # Show capacity monitoring
        self._demonstrate_capacity_monitoring(db)
        
        return True
    
    def _show_capacity_solutions(self):
        """Show solutions for capacity limits"""
        print(f"\n💡 Capacity Limit Solutions:")
        solutions = [
            "🎓 Perfect for learning: You've explored the full capacity!",
            "🧹 Clean up: Remove test/temporary vectors",
            "📦 Export data: Use db.export_data() to save progress",
            "🆕 Fresh start: Create new database for different experiments", 
            f"🚀 Production ready: Upgrade to full RudraDB: {rudradb.UPGRADE_MESSAGE}",
            "📚 Focus quality: Build fewer, higher-quality relationships"
        ]
        
        for solution in solutions:
            print(f"   {solution}")
    
    def _demonstrate_capacity_monitoring(self, db):
        """Show how to monitor capacity usage"""
        print(f"\n📊 Capacity Monitoring Example:")
        
        stats = db.get_statistics()
        if 'capacity_usage' in stats:
            usage = stats['capacity_usage']
            
            print(f"   Current Usage:")
            print(f"     Vectors: {stats['vector_count']}/{rudradb.MAX_VECTORS} ({usage['vector_usage_percent']:.1f}%)")
            print(f"     Relationships: {stats['relationship_count']}/{rudradb.MAX_RELATIONSHIPS} ({usage['relationship_usage_percent']:.1f}%)")
            
            # Visual progress bars
            self._print_progress_bar("Vectors", usage['vector_usage_percent'])
            self._print_progress_bar("Relationships", usage['relationship_usage_percent'])
            
            # Warnings
            if usage['vector_usage_percent'] > 80:
                print(f"   ⚠️ Vector capacity warning: {usage['vector_usage_percent']:.1f}%")
            
            if usage['relationship_usage_percent'] > 80:
                print(f"   ⚠️ Relationship capacity warning: {usage['relationship_usage_percent']:.1f}%")
    
    def _print_progress_bar(self, label, percentage, width=20):
        """Print a visual progress bar"""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        print(f"     {label}: [{bar}] {percentage:.1f}%")
    
    def troubleshoot_relationship_issues(self):
        """Diagnose relationship building problems"""
        print(f"\n🔗 4. RELATIONSHIP TROUBLESHOOTING")
        print(f"=" * 40)
        
        # Create test database
        db = rudradb.RudraDB()
        
        # Add test vectors
        test_vectors = ["vec1", "vec2", "vec3", "vec4"]
        for i, vec_id in enumerate(test_vectors):
            embedding = np.random.rand(384).astype(np.float32)
            db.add_vector(vec_id, embedding, {"index": i})
        
        print(f"✅ Added {len(test_vectors)} test vectors")
        
        # Test valid relationship types
        print(f"\n🔍 Testing Valid Relationship Types...")
        
        valid_types = ["semantic", "hierarchical", "temporal", "causal", "associative"]
        for rel_type in valid_types:
            try:
                db.add_relationship("vec1", "vec2", rel_type, 0.8)
                print(f"   ✅ {rel_type}: Valid")
                # Clean up for next test
                db.remove_relationship("vec1", "vec2")
            except Exception as e:
                print(f"   ❌ {rel_type}: Failed - {e}")
        
        # Test invalid relationship scenarios
        print(f"\n⚠️ Testing Invalid Relationship Scenarios...")
        
        invalid_scenarios = [
            ("invalid_type", "vec1", "vec2", "invalid_relationship_type", 0.8, "Invalid relationship type"),
            ("self_reference", "vec1", "vec1", "semantic", 0.8, "Self-referencing relationship"),
            ("nonexistent_source", "nonexistent", "vec2", "semantic", 0.8, "Non-existent source vector"),
            ("nonexistent_target", "vec1", "nonexistent", "semantic", 0.8, "Non-existent target vector"),
            ("invalid_strength_high", "vec1", "vec3", "semantic", 1.5, "Strength too high"),
            ("invalid_strength_low", "vec1", "vec3", "semantic", -0.5, "Strength too low"),
            ("invalid_strength_nan", "vec1", "vec3", "semantic", float('nan'), "NaN strength"),
        ]
        
        for test_name, source, target, rel_type, strength, description in invalid_scenarios:
            try:
                db.add_relationship(source, target, rel_type, strength)
                print(f"   ❌ {test_name}: Should have been rejected ({description})")
            except Exception as e:
                print(f"   ✅ {test_name}: Correctly rejected ({type(e).__name__})")
        
        # Show relationship best practices
        print(f"\n💡 Relationship Best Practices:")
        
        best_practices = [
            ("Choose appropriate types", "semantic for similar content, hierarchical for parent-child"),
            ("Use valid strengths", "0.0 to 1.0, with 0.8-1.0 for strong connections"),
            ("Avoid self-references", "Don't connect a vector to itself"),
            ("Check vector existence", "Ensure both source and target vectors exist"),
            ("Build strategically", "Focus on meaningful connections, not quantity"),
            ("Monitor capacity", "Stay within 500 relationship limit for Opin")
        ]
        
        for practice, explanation in best_practices:
            print(f"   • {practice}: {explanation}")
        
        # Demonstrate relationship querying
        print(f"\n🔍 Testing Relationship Querying...")
        
        # Add some test relationships
        db.add_relationship("vec1", "vec2", "semantic", 0.9, {"test": "semantic_connection"})
        db.add_relationship("vec2", "vec3", "hierarchical", 0.8, {"test": "hierarchical_connection"})
        db.add_relationship("vec1", "vec3", "temporal", 0.7, {"test": "temporal_connection"})
        
        # Query relationships
        vec1_relationships = db.get_relationships("vec1")
        print(f"   vec1 relationships: {len(vec1_relationships)}")
        
        for rel in vec1_relationships:
            print(f"     • {rel['source_id']} → {rel['target_id']} ({rel['relationship_type']}, {rel['strength']})")
        
        # Test relationship existence
        print(f"   Relationship exists (vec1 → vec2): {db.has_relationship('vec1', 'vec2')}")
        print(f"   Relationship exists (vec1 → vec4): {db.has_relationship('vec1', 'vec4')}")
        
        return True
    
    def troubleshoot_search_issues(self):
        """Diagnose search problems and optimize performance"""
        print(f"\n🔍 5. SEARCH TROUBLESHOOTING")
        print(f"=" * 35)
        
        # Create test database with content
        db = rudradb.RudraDB()
        
        # Add test content
        print(f"📚 Setting up test content...")
        test_content = [
            ("doc1", "Machine learning fundamentals and basic concepts"),
            ("doc2", "Deep learning with neural networks"),
            ("doc3", "Python programming for data science"),
            ("doc4", "Data visualization techniques and tools"),
            ("doc5", "Advanced AI algorithms and optimization")
        ]
        
        for doc_id, text in test_content:
            # Use simple random embeddings for testing
            embedding = np.random.rand(384).astype(np.float32)
            db.add_vector(doc_id, embedding, {"text": text})
        
        # Add some relationships
        db.add_relationship("doc1", "doc2", "hierarchical", 0.8)
        db.add_relationship("doc2", "doc5", "semantic", 0.7)
        db.add_relationship("doc3", "doc4", "associative", 0.6)
        
        print(f"   ✅ Added {len(test_content)} documents and 3 relationships")
        
        # Test basic search functionality
        print(f"\n🔍 Testing Basic Search...")
        
        query_embedding = np.random.rand(384).astype(np.float32)
        
        try:
            # Basic similarity search
            results = db.search(query_embedding)
            print(f"   ✅ Basic search: {len(results)} results")
            
            # Search with parameters
            params = rudradb.SearchParams(
                top_k=3,
                include_relationships=True,
                similarity_threshold=0.1
            )
            results_with_params = db.search(query_embedding, params)
            print(f"   ✅ Parameterized search: {len(results_with_params)} results")
            
        except Exception as e:
            print(f"   ❌ Basic search failed: {e}")
            return False
        
        # Test search performance
        print(f"\n⚡ Testing Search Performance...")
        
        search_configurations = [
            ("Basic similarity", {"top_k": 5, "include_relationships": False}),
            ("With relationships", {"top_k": 5, "include_relationships": True, "max_hops": 1}),
            ("Multi-hop", {"top_k": 5, "include_relationships": True, "max_hops": 2}),
            ("Filtered", {"top_k": 5, "include_relationships": True, "max_hops": 2, "similarity_threshold": 0.3})
        ]
        
        for config_name, config_params in search_configurations:
            params = rudradb.SearchParams(**config_params)
            
            # Measure search time
            start_time = time.time()
            results = db.search(query_embedding, params)
            search_time = time.time() - start_time
            
            print(f"   {config_name}:")
            print(f"     Time: {search_time*1000:.2f}ms")
            print(f"     Results: {len(results)}")
            
            if search_time > 0.1:
                print(f"     ⚠️ Slower than expected")
            else:
                print(f"     ✅ Good performance")
        
        # Test search edge cases
        print(f"\n⚠️ Testing Search Edge Cases...")
        
        edge_cases = [
            ("Empty database search", rudradb.RudraDB(), "Search with no vectors"),
            ("Invalid parameters", db, "Search with invalid top_k"),
            ("Wrong dimension", db, "Search with wrong embedding dimension")
        ]
        
        for case_name, test_db, description in edge_cases:
            try:
                if case_name == "Empty database search":
                    empty_results = test_db.search(query_embedding)
                    print(f"   ✅ {case_name}: {len(empty_results)} results (expected 0)")
                
                elif case_name == "Invalid parameters":
                    invalid_params = rudradb.SearchParams(top_k=0)  # Invalid
                    invalid_results = test_db.search(query_embedding, invalid_params)
                    print(f"   ❌ {case_name}: Should have failed")
                
                elif case_name == "Wrong dimension":
                    wrong_embedding = np.random.rand(256).astype(np.float32)  # Different dimension
                    wrong_results = test_db.search(wrong_embedding)
                    print(f"   ❌ {case_name}: Should have failed")
                    
            except Exception as e:
                if "parameter" in str(e).lower() or "dimension" in str(e).lower():
                    print(f"   ✅ {case_name}: Correctly handled ({type(e).__name__})")
                else:
                    print(f"   ⚠️ {case_name}: Unexpected error - {e}")
        
        # Show search optimization tips
        print(f"\n💡 Search Optimization Tips:")
        
        optimization_tips = [
            ("Reduce top_k", "Use smaller top_k if you don't need many results"),
            ("Use thresholds", "Set similarity_threshold to filter low-quality matches"),
            ("Limit hops", "Use max_hops=1 for faster relationship search"),
            ("Filter relationships", "Specify relationship_types for focused search"),
            ("Monitor performance", "Measure search times and optimize as needed"),
            ("Batch queries", "Process multiple queries efficiently")
        ]
        
        for tip, explanation in optimization_tips:
            print(f"   • {tip}: {explanation}")
        
        return True
    
    def troubleshoot_performance_issues(self):
        """Diagnose and resolve performance problems"""
        print(f"\n⚡ 6. PERFORMANCE TROUBLESHOOTING")
        print(f"=" * 40)
        
        # Create test database
        db = rudradb.RudraDB()
        
        print(f"📊 Performance Baseline Test...")
        
        # Test vector addition performance
        print(f"\n🔍 Vector Addition Performance...")
        
        vector_counts = [10, 25, 50]
        
        for count in vector_counts:
            if db.vector_count() + count > rudradb.MAX_VECTORS:
                print(f"   ⚠️ Skipping {count} vectors (would exceed capacity)")
                continue
            
            start_time = time.time()
            
            for i in range(count):
                embedding = np.random.rand(384).astype(np.float32)
                metadata = {"batch": count, "index": i}
                db.add_vector(f"perf_test_{db.vector_count()}_{i}", embedding, metadata)
            
            add_time = time.time() - start_time
            rate = count / add_time if add_time > 0 else float('inf')
            
            print(f"   {count} vectors: {add_time:.3f}s ({rate:.0f}/sec)")
            
            if add_time > 2.0:
                print(f"     ⚠️ Slower than expected - check system resources")
            else:
                print(f"     ✅ Good performance")
        
        # Test search performance  
        print(f"\n🔍 Search Performance...")
        
        query_embedding = np.random.rand(384).astype(np.float32)
        search_iterations = [10, 25, 50]
        
        for iterations in search_iterations:
            start_time = time.time()
            
            for _ in range(iterations):
                results = db.search(query_embedding)
            
            search_time = time.time() - start_time
            rate = iterations / search_time if search_time > 0 else float('inf')
            
            print(f"   {iterations} searches: {search_time:.3f}s ({rate:.0f}/sec)")
            
            if search_time > 1.0:
                print(f"     ⚠️ Search performance concern")
            else:
                print(f"     ✅ Good search performance")
        
        # Memory usage estimation
        print(f"\n💾 Memory Usage Estimation...")
        
        stats = db.get_statistics()
        vector_memory = stats['vector_count'] * stats['dimension'] * 4  # 4 bytes per float32
        relationship_memory = stats['relationship_count'] * 200  # Estimated bytes per relationship
        total_memory = vector_memory + relationship_memory
        
        print(f"   Vectors: ~{vector_memory / (1024*1024):.2f} MB")
        print(f"   Relationships: ~{relationship_memory / (1024*1024):.2f} MB") 
        print(f"   Total estimated: ~{total_memory / (1024*1024):.2f} MB")
        
        if total_memory > 100 * 1024 * 1024:  # 100 MB
            print(f"     ⚠️ High memory usage - consider optimization")
        else:
            print(f"     ✅ Memory usage is reasonable")
        
        # Performance optimization suggestions
        print(f"\n💡 Performance Optimization Strategies:")
        
        optimizations = [
            ("Use np.float32", "Ensure embeddings are float32 for memory efficiency"),
            ("Minimize metadata", "Store only essential metadata to reduce memory"),
            ("Batch operations", "Add multiple vectors/relationships in batches"),
            ("Optimize search params", "Use appropriate top_k and similarity_threshold"),
            ("Monitor capacity", "Stay within Opin limits for optimal performance"),
            ("Profile bottlenecks", "Identify slow operations and optimize them")
        ]
        
        for optimization, explanation in optimizations:
            print(f"   • {optimization}: {explanation}")
        
        return True
    
    def troubleshoot_data_issues(self):
        """Diagnose data export/import and persistence problems"""
        print(f"\n💾 7. DATA PERSISTENCE TROUBLESHOOTING")
        print(f"=" * 45)
        
        # Create test database with sample data
        db = rudradb.RudraDB()
        
        print(f"📊 Setting up test data...")
        
        # Add sample vectors and relationships
        sample_data = [
            ("test_doc1", "Sample document about machine learning", {"category": "AI"}),
            ("test_doc2", "Tutorial on Python programming", {"category": "Programming"}),
            ("test_doc3", "Data science best practices guide", {"category": "Data Science"})
        ]
        
        for doc_id, text, metadata in sample_data:
            embedding = np.random.rand(384).astype(np.float32)
            db.add_vector(doc_id, embedding, metadata)
        
        db.add_relationship("test_doc1", "test_doc2", "associative", 0.7)
        db.add_relationship("test_doc2", "test_doc3", "semantic", 0.8)
        
        original_stats = db.get_statistics()
        print(f"   ✅ Created test data: {original_stats['vector_count']} vectors, {original_stats['relationship_count']} relationships")
        
        # Test data export
        print(f"\n📤 Testing Data Export...")
        
        try:
            export_data = db.export_data()
            print(f"   ✅ Data exported successfully")
            print(f"   Export contains:")
            print(f"     Vectors: {len(export_data.get('vectors', []))}")
            print(f"     Relationships: {len(export_data.get('relationships', []))}")
            print(f"     Metadata: {len(export_data.get('metadata', {}))}")
            
            # Verify export structure
            if 'vectors' in export_data and 'relationships' in export_data:
                print(f"   ✅ Export structure is valid")
            else:
                print(f"   ❌ Export structure is invalid")
                return False
                
        except Exception as e:
            print(f"   ❌ Data export failed: {e}")
            return False
        
        # Test data import
        print(f"\n📥 Testing Data Import...")
        
        try:
            # Create new database and import data
            new_db = rudradb.RudraDB()
            new_db.import_data(export_data)
            
            imported_stats = new_db.get_statistics()
            print(f"   ✅ Data imported successfully")
            print(f"   Imported: {imported_stats['vector_count']} vectors, {imported_stats['relationship_count']} relationships")
            
            # Verify import integrity
            if (imported_stats['vector_count'] == original_stats['vector_count'] and
                imported_stats['relationship_count'] == original_stats['relationship_count']):
                print(f"   ✅ Import integrity verified")
            else:
                print(f"   ⚠️ Import integrity issue:")
                print(f"     Original: {original_stats['vector_count']} vectors, {original_stats['relationship_count']} relationships")
                print(f"     Imported: {imported_stats['vector_count']} vectors, {imported_stats['relationship_count']} relationships")
                
        except Exception as e:
            print(f"   ❌ Data import failed: {e}")
            return False
        
        # Test file-based persistence
        print(f"\n💾 Testing File-Based Persistence...")
        
        try:
            # Save to file
            with open("test_export.json", "w") as f:
                json.dump(export_data, f, indent=2)
            print(f"   ✅ Data saved to file")
            
            # Load from file
            with open("test_export.json", "r") as f:
                file_data = json.load(f)
            
            # Import from file
            file_db = rudradb.RudraDB()
            file_db.import_data(file_data)
            
            file_stats = file_db.get_statistics()
            print(f"   ✅ Data loaded from file: {file_stats['vector_count']} vectors")
            
            # Clean up test file
            import os
            os.remove("test_export.json")
            print(f"   ✅ Test file cleaned up")
            
        except Exception as e:
            print(f"   ❌ File persistence failed: {e}")
            return False
        
        # Show data management best practices
        print(f"\n💡 Data Management Best Practices:")
        
        best_practices = [
            ("Regular backups", "Export data regularly: export_data = db.export_data()"),
            ("Version control", "Include version info in exports for compatibility"),
            ("Validate imports", "Check imported data integrity after import_data()"),
            ("Handle large exports", "For large data, consider chunked export/import"),
            ("Secure storage", "Protect exported data files appropriately"),
            ("Migration planning", "Plan data migration when upgrading to full RudraDB")
        ]
        
        for practice, explanation in best_practices:
            print(f"   • {practice}: {explanation}")
        
        return True
    
    def run_comprehensive_diagnostic(self):
        """Run complete diagnostic suite"""
        print(f"\n🏥 COMPREHENSIVE DIAGNOSTIC SUITE")
        print(f"=" * 45)
        
        diagnostic_results = {}
        
        # Run all diagnostic tests
        tests = [
            ("Installation", self.troubleshoot_installation_issues),
            ("Dimensions", self.troubleshoot_dimension_issues),
            ("Capacity", self.troubleshoot_capacity_limits),
            ("Relationships", self.troubleshoot_relationship_issues),
            ("Search", self.troubleshoot_search_issues),
            ("Performance", self.troubleshoot_performance_issues),
            ("Data Persistence", self.troubleshoot_data_issues)
        ]
        
        for test_name, test_function in tests:
            print(f"\n🔍 Running {test_name} diagnostics...")
            try:
                result = test_function()
                diagnostic_results[test_name] = "✅ PASS" if result else "❌ FAIL"
            except Exception as e:
                diagnostic_results[test_name] = f"❌ ERROR: {e}"
                print(f"   ❌ Diagnostic error: {e}")
        
        # Summary report
        print(f"\n📊 DIAGNOSTIC SUMMARY")
        print(f"=" * 25)
        
        for test_name, result in diagnostic_results.items():
            print(f"   {test_name}: {result}")
        
        # Overall health assessment
        passed_tests = sum(1 for result in diagnostic_results.values() if result.startswith("✅"))
        total_tests = len(diagnostic_results)
        
        health_score = (passed_tests / total_tests) * 100
        
        print(f"\n🏥 Overall System Health: {health_score:.0f}%")
        
        if health_score >= 90:
            print(f"   🎉 Excellent! RudraDB-Opin is working perfectly")
        elif health_score >= 70:
            print(f"   ✅ Good! Minor issues may exist but system is functional")
        elif health_score >= 50:
            print(f"   ⚠️ Caution! Several issues detected, review failed tests")
        else:
            print(f"   ❌ Critical! Major issues detected, check installation")
        
        return diagnostic_results


def main():
    """Run the complete troubleshooting guide"""
    troubleshooter = TroubleshootingGuide()
    
    print(f"\n🎯 Choose troubleshooting mode:")
    print(f"   1. Quick diagnostic (recommended)")
    print(f"   2. Comprehensive diagnostic suite")
    print(f"   3. Specific issue troubleshooting")
    
    try:
        # For demo purposes, run quick diagnostic
        print(f"\n🚀 Running Quick Diagnostic...")
        
        # Quick tests
        print(f"\n⚡ Quick System Check:")
        installation_ok = troubleshooter.troubleshoot_installation_issues()
        
        if installation_ok:
            print(f"   ✅ Installation: Working")
            
            # Quick dimension test
            db_test = rudradb.RudraDB()
            test_emb = np.random.rand(384).astype(np.float32)
            db_test.add_vector("quick_test", test_emb)
            print(f"   ✅ Auto-dimension detection: Working ({db_test.dimension()}D)")
            
            # Quick relationship test
            test_emb2 = np.random.rand(384).astype(np.float32)
            db_test.add_vector("quick_test2", test_emb2)
            db_test.add_relationship("quick_test", "quick_test2", "semantic", 0.8)
            print(f"   ✅ Relationships: Working")
            
            # Quick search test
            results = db_test.search(test_emb)
            print(f"   ✅ Search: Working ({len(results)} results)")
            
            print(f"\n🎉 Quick diagnostic complete - RudraDB-Opin is working!")
            print(f"   🔧 For detailed troubleshooting, run individual diagnostic functions")
            
        else:
            print(f"   ❌ Installation issues detected")
            print(f"   🔧 Please fix installation before using other features")
        
    except Exception as e:
        print(f"❌ Diagnostic error: {e}")
        print(f"💡 This might indicate a deeper system issue")
        print(f"   Check Python version, package installation, and system resources")


if __name__ == "__main__":
    main()
