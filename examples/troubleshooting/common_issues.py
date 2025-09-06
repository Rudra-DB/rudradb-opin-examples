#!/usr/bin/env python3
"""
RudraDB-Opin: Common Issues Solutions

This example demonstrates solutions to the most common issues users encounter:

1. Performance optimization
2. Memory management
3. Capacity planning
4. Search tuning
5. Relationship optimization
6. Error recovery
7. Data migration

Perfect for learning how to solve real-world problems!
"""

import rudradb
import numpy as np
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import gc
import psutil
import os


class CommonIssuesSolver:
    """Solutions for common RudraDB-Opin issues"""
    
    def __init__(self):
        print("🛠️ RudraDB-Opin Common Issues Solver")
        print("=" * 45)
    
    def solve_slow_search_performance(self):
        """Solve slow search performance issues"""
        print("\n⚡ SOLVING: Slow Search Performance")
        print("=" * 40)
        
        # Create test database with performance issues
        db = rudradb.RudraDB()
        
        print("📊 Setting up performance test scenario...")
        
        # Add vectors (simulate larger dataset)
        for i in range(50):  # Within Opin limits
            embedding = np.random.rand(384).astype(np.float32)
            metadata = {
                "text": f"Document {i} with content about various topics",
                "category": f"category_{i % 5}",
                "tags": [f"tag_{j}" for j in range(i % 3 + 1)],
                "complexity": np.random.choice(["low", "medium", "high"])
            }
            db.add_vector(f"doc_{i}", embedding, metadata)
        
        # Add many relationships (can slow search)
        relationships_added = 0
        for i in range(40):
            for j in range(i+1, min(i+6, 50)):  # Connect to next 5 documents
                if relationships_added >= 200:  # Stay well under limit
                    break
                try:
                    db.add_relationship(f"doc_{i}", f"doc_{j}", "semantic", 0.3 + np.random.rand() * 0.5)
                    relationships_added += 1
                except:
                    break
        
        print(f"   Test data: {db.vector_count()} vectors, {db.relationship_count()} relationships")
        
        # Demonstrate slow search (unoptimized)
        print("\n🐌 Problematic Search Configuration:")
        
        query = np.random.rand(384).astype(np.float32)
        
        # Slow parameters
        slow_params = rudradb.SearchParams(
            top_k=20,                    # Too many results
            include_relationships=True,
            max_hops=2,                  # Maximum traversal
            similarity_threshold=0.0,    # No filtering
            relationship_weight=0.8      # Heavy relationship processing
        )
        
        start_time = time.time()
        slow_results = db.search(query, slow_params)
        slow_time = time.time() - start_time
        
        print(f"   Slow search: {slow_time*1000:.2f}ms, {len(slow_results)} results")
        
        # Show optimization solutions
        print("\n🚀 SOLUTIONS for Search Performance:")
        
        optimizations = [
            ("Reduce top_k", rudradb.SearchParams(top_k=5, include_relationships=True, max_hops=2)),
            ("Add threshold", rudradb.SearchParams(top_k=10, include_relationships=True, max_hops=2, similarity_threshold=0.3)),
            ("Limit hops", rudradb.SearchParams(top_k=10, include_relationships=True, max_hops=1)),
            ("Balance weights", rudradb.SearchParams(top_k=10, include_relationships=True, max_hops=2, relationship_weight=0.3)),
            ("Filter types", rudradb.SearchParams(top_k=10, include_relationships=True, max_hops=2, relationship_types=["semantic"]))
        ]
        
        for opt_name, params in optimizations:
            start_time = time.time()
            results = db.search(query, params)
            opt_time = time.time() - start_time
            
            improvement = ((slow_time - opt_time) / slow_time) * 100 if slow_time > 0 else 0
            print(f"   {opt_name}: {opt_time*1000:.2f}ms ({improvement:.0f}% faster, {len(results)} results)")
    
    def solve_memory_issues(self):
        """Solve memory usage and optimization issues"""
        print("\n💾 SOLVING: Memory Usage Issues")
        print("=" * 35)
        
        print("📊 Memory Usage Analysis and Solutions:")
        
        # Get current memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   Initial memory: {initial_memory:.2f} MB")
        
        # Create database and monitor memory
        db = rudradb.RudraDB()
        
        # Memory-heavy scenario (large embeddings, rich metadata)
        print("\n🔍 Memory-Heavy Scenario:")
        
        memory_measurements = []
        
        for batch in range(5):  # 5 batches of 10 vectors each
            batch_start_memory = process.memory_info().rss / 1024 / 1024
            
            for i in range(10):
                # Large embedding (high dimension for demo)
                embedding = np.random.rand(384).astype(np.float32)  # Keep reasonable for Opin
                
                # Rich metadata (can consume memory)
                metadata = {
                    "title": f"Document {batch * 10 + i}",
                    "content": "Lorem ipsum " * 100,  # Large text content
                    "tags": [f"tag_{j}" for j in range(20)],  # Many tags
                    "features": np.random.rand(50).tolist(),  # Additional numeric features
                    "timestamp": datetime.now().isoformat(),
                    "category": f"category_{i % 3}",
                    "metadata_size": "large"
                }
                
                db.add_vector(f"heavy_doc_{batch}_{i}", embedding, metadata)
            
            batch_end_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = batch_end_memory - batch_start_memory
            
            memory_measurements.append({
                "batch": batch + 1,
                "vectors_added": 10,
                "memory_before": batch_start_memory,
                "memory_after": batch_end_memory,
                "memory_increase": memory_increase
            })
            
            print(f"   Batch {batch + 1}: +{memory_increase:.2f} MB ({batch_end_memory:.2f} MB total)")
        
        # Show memory optimization solutions
        print("\n🚀 SOLUTIONS for Memory Optimization:")
        
        solutions = [
            ("Optimize embeddings", "Use np.float32 instead of float64", "4 bytes vs 8 bytes per value"),
            ("Minimize metadata", "Store only essential metadata", "Reduce per-vector overhead"),
            ("Batch processing", "Process data in batches", "Control memory growth"),
            ("Garbage collection", "Manual gc.collect()", "Free unused memory"),
            ("Efficient data types", "Use appropriate data types", "str vs list for single values"),
            ("Remove unused data", "Clean up temporary vectors", "Free memory explicitly")
        ]
        
        for solution, technique, benefit in solutions:
            print(f"   • {solution}: {technique} → {benefit}")
        
        # Demonstrate memory optimization
        print("\n✅ Memory Optimization Demo:")
        
        # Before optimization
        before_memory = process.memory_info().rss / 1024 / 1024
        
        # Remove some vectors to free memory
        vectors_to_remove = [f"heavy_doc_0_{i}" for i in range(5)]
        for vec_id in vectors_to_remove:
            if db.vector_exists(vec_id):
                db.remove_vector(vec_id)
        
        # Force garbage collection
        gc.collect()
        
        after_memory = process.memory_info().rss / 1024 / 1024
        memory_freed = before_memory - after_memory
        
        print(f"   Before cleanup: {before_memory:.2f} MB")
        print(f"   After cleanup: {after_memory:.2f} MB")
        print(f"   Memory freed: {memory_freed:.2f} MB")
        
        # Show capacity-aware memory management
        print("\n📊 Capacity-Aware Memory Management:")
        
        stats = db.get_statistics()
        estimated_memory = (stats['vector_count'] * stats['dimension'] * 4) / (1024 * 1024)  # Vectors only
        
        print(f"   Current vectors: {stats['vector_count']}/{rudradb.MAX_VECTORS}")
        print(f"   Estimated vector memory: {estimated_memory:.2f} MB")
        print(f"   Memory per vector: {estimated_memory / stats['vector_count']:.4f} MB")
        
        if estimated_memory > 50:
            print(f"   💡 Consider: Reduce embedding dimension or metadata size")
    
    def solve_capacity_planning_issues(self):
        """Solve capacity planning and limit management"""
        print("\n📊 SOLVING: Capacity Planning Issues")
        print("=" * 40)
        
        db = rudradb.RudraDB()
        
        # Simulate approaching capacity limits
        print("🔍 Capacity Planning Simulation:")
        
        # Add vectors strategically
        vector_batches = [
            (20, "Initial content"),
            (30, "Growing dataset"),
            (25, "Peak usage"),
            (25, "Approaching limit")
        ]
        
        total_added = 0
        for batch_size, phase in vector_batches:
            if total_added + batch_size > rudradb.MAX_VECTORS:
                actual_batch = rudradb.MAX_VECTORS - total_added
                print(f"   {phase}: Adding {actual_batch} vectors (limited by capacity)")
            else:
                actual_batch = batch_size
                print(f"   {phase}: Adding {actual_batch} vectors")
            
            for i in range(actual_batch):
                embedding = np.random.rand(384).astype(np.float32)
                metadata = {
                    "phase": phase,
                    "batch_index": i,
                    "priority": np.random.choice(["low", "medium", "high"]),
                    "category": f"category_{i % 5}"
                }
                
                try:
                    db.add_vector(f"capacity_test_{total_added + i}", embedding, metadata)
                except RuntimeError as e:
                    if "Vector Limit Reached" in str(e):
                        print(f"     ⚠️ Vector capacity reached at {db.vector_count()}")
                        break
                    else:
                        raise
            
            total_added = db.vector_count()
            
            # Show capacity status
            stats = db.get_statistics()
            if 'capacity_usage' in stats:
                usage = stats['capacity_usage']
                print(f"     Status: {usage['vector_usage_percent']:.1f}% used ({usage['vector_capacity_remaining']} remaining)")
        
        print("\n🚀 SOLUTIONS for Capacity Planning:")
        
        capacity_strategies = [
            ("Monitor usage", "Check stats regularly", "stats = db.get_statistics()"),
            ("Set thresholds", "Alert at 80% capacity", "if usage > 80: plan_upgrade()"),
            ("Prioritize content", "Keep high-priority vectors", "Remove test/temp data"),
            ("Quality over quantity", "Focus on meaningful relationships", "Avoid weak connections"),
            ("Export before limits", "Save data for migration", "export_data = db.export_data()"),
            ("Plan upgrade path", "Prepare for full RudraDB", "pip install rudradb")
        ]
        
        for strategy, description, example in capacity_strategies:
            print(f"   • {strategy}: {description}")
            print(f"     Example: {example}")
        
        # Demonstrate capacity-aware operations
        print("\n✅ Capacity-Aware Operations Demo:")
        
        def safe_add_vector(db, vec_id, embedding, metadata):
            """Add vector with capacity awareness"""
            stats = db.get_statistics()
            if 'capacity_usage' in stats:
                usage = stats['capacity_usage']
                
                if usage['vector_usage_percent'] > 95:
                    return {"success": False, "reason": "Capacity critical"}
                elif usage['vector_usage_percent'] > 85:
                    print(f"     ⚠️ Warning: {usage['vector_usage_percent']:.1f}% capacity used")
            
            try:
                db.add_vector(vec_id, embedding, metadata)
                return {"success": True, "reason": "Added successfully"}
            except RuntimeError as e:
                if "Limit Reached" in str(e):
                    return {"success": False, "reason": "Capacity limit reached"}
                else:
                    raise
        
        # Test safe addition
        test_embedding = np.random.rand(384).astype(np.float32)
        result = safe_add_vector(db, "safe_test", test_embedding, {"test": True})
        print(f"   Safe addition result: {result}")
    
    def solve_search_tuning_issues(self):
        """Solve search result quality and relevance issues"""
        print("\n🎯 SOLVING: Search Tuning Issues")
        print("=" * 35)
        
        # Create database with diverse content
        db = rudradb.RudraDB()
        
        print("📚 Setting up diverse content for search tuning...")
        
        # Add content with varying relevance
        content_types = [
            ("highly_relevant", "Machine learning algorithms and neural networks", {"relevance": "high", "category": "ML"}),
            ("somewhat_relevant", "Data science and statistical analysis", {"relevance": "medium", "category": "DS"}),
            ("loosely_relevant", "Programming fundamentals and software development", {"relevance": "low", "category": "Programming"}),
            ("unrelated", "Cooking recipes and kitchen techniques", {"relevance": "none", "category": "Cooking"}),
            ("noisy", "Random text with no clear meaning or structure", {"relevance": "noise", "category": "Noise"})
        ]
        
        # Create multiple documents of each type
        for i in range(5):  # 5 documents per type
            for content_type, text_template, metadata in content_types:
                doc_id = f"{content_type}_{i}"
                text = f"{text_template} - variation {i}"
                embedding = np.random.rand(384).astype(np.float32)
                
                enhanced_metadata = {
                    "text": text,
                    "content_type": content_type,
                    "variation": i,
                    **metadata
                }
                
                db.add_vector(doc_id, embedding, enhanced_metadata)
        
        # Add relationships for some content
        db.add_relationship("highly_relevant_0", "highly_relevant_1", "semantic", 0.9)
        db.add_relationship("highly_relevant_1", "somewhat_relevant_0", "associative", 0.7)
        db.add_relationship("somewhat_relevant_0", "loosely_relevant_0", "associative", 0.5)
        
        print(f"   Created {db.vector_count()} documents with varying relevance")
        
        # Demonstrate search tuning problems and solutions
        print("\n🔍 Search Tuning Problems & Solutions:")
        
        query = np.random.rand(384).astype(np.float32)
        
        # Problem 1: Too many irrelevant results
        print("\n❌ Problem: Too many irrelevant results")
        
        loose_params = rudradb.SearchParams(
            top_k=15,
            include_relationships=True,
            similarity_threshold=0.0,  # No filtering
            relationship_weight=0.2
        )
        
        loose_results = db.search(query, loose_params)
        irrelevant_count = sum(1 for r in loose_results 
                              if db.get_vector(r.vector_id)['metadata'].get('relevance') in ['none', 'noise'])
        
        print(f"   Loose search: {len(loose_results)} results, {irrelevant_count} irrelevant")
        
        print("✅ Solution: Add similarity threshold")
        
        filtered_params = rudradb.SearchParams(
            top_k=15,
            include_relationships=True,
            similarity_threshold=0.3,  # Filter low similarity
            relationship_weight=0.2
        )
        
        filtered_results = db.search(query, filtered_params)
        filtered_irrelevant = sum(1 for r in filtered_results 
                                 if db.get_vector(r.vector_id)['metadata'].get('relevance') in ['none', 'noise'])
        
        print(f"   Filtered search: {len(filtered_results)} results, {filtered_irrelevant} irrelevant")
        
        # Problem 2: Missing related content
        print("\n❌ Problem: Missing related content")
        
        strict_params = rudradb.SearchParams(
            top_k=5,
            include_relationships=False,  # No relationship discovery
            similarity_threshold=0.6
        )
        
        strict_results = db.search(query, strict_params)
        print(f"   Strict search: {len(strict_results)} results (may miss related content)")
        
        print("✅ Solution: Enable relationship discovery")
        
        discovery_params = rudradb.SearchParams(
            top_k=8,
            include_relationships=True,  # Enable relationships
            max_hops=2,
            similarity_threshold=0.4,
            relationship_weight=0.4
        )
        
        discovery_results = db.search(query, discovery_params)
        relationship_discoveries = sum(1 for r in discovery_results if r.hop_count > 0)
        print(f"   Discovery search: {len(discovery_results)} results, {relationship_discoveries} via relationships")
        
        # Show adaptive search strategy
        print("\n🎯 Adaptive Search Strategy:")
        
        def adaptive_search(db, query_embedding, target_results=5, max_attempts=3):
            """Adaptive search that adjusts parameters based on results"""
            
            # Strategy progression: strict → balanced → discovery
            strategies = [
                ("Strict", rudradb.SearchParams(top_k=target_results, similarity_threshold=0.5, include_relationships=False)),
                ("Balanced", rudradb.SearchParams(top_k=target_results, similarity_threshold=0.3, include_relationships=True, max_hops=1)),
                ("Discovery", rudradb.SearchParams(top_k=target_results*2, similarity_threshold=0.1, include_relationships=True, max_hops=2, relationship_weight=0.5))
            ]
            
            for i, (strategy_name, params) in enumerate(strategies):
                results = db.search(query_embedding, params)
                
                print(f"   {strategy_name}: {len(results)} results")
                
                if len(results) >= target_results or i == len(strategies) - 1:
                    return results[:target_results]
            
            return []
        
        adaptive_results = adaptive_search(db, query, target_results=6)
        print(f"   Final adaptive results: {len(adaptive_results)}")
    
    def solve_relationship_optimization_issues(self):
        """Solve relationship building and optimization issues"""
        print("\n🔗 SOLVING: Relationship Optimization Issues")
        print("=" * 45)
        
        db = rudradb.RudraDB()
        
        print("🔍 Relationship Quality Issues & Solutions:")
        
        # Add test content
        docs = [
            ("ai_intro", "Introduction to Artificial Intelligence", {"category": "AI", "level": "beginner"}),
            ("ml_basics", "Machine Learning Basics", {"category": "AI", "level": "intermediate"}),
            ("dl_advanced", "Deep Learning Advanced Topics", {"category": "AI", "level": "advanced"}),
            ("python_prog", "Python Programming", {"category": "Programming", "level": "beginner"}),
            ("data_viz", "Data Visualization", {"category": "Data", "level": "intermediate"})
        ]
        
        for doc_id, text, metadata in docs:
            embedding = np.random.rand(384).astype(np.float32)
            db.add_vector(doc_id, embedding, metadata)
        
        # Problem 1: Weak/meaningless relationships
        print("\n❌ Problem: Weak relationships dilute search quality")
        
        # Add weak relationships (bad practice)
        weak_relationships = [
            ("ai_intro", "python_prog", "associative", 0.2),  # Very weak
            ("ml_basics", "data_viz", "semantic", 0.1),       # Almost meaningless
            ("dl_advanced", "python_prog", "temporal", 0.15)  # Wrong type + weak
        ]
        
        for source, target, rel_type, strength in weak_relationships:
            db.add_relationship(source, target, rel_type, strength)
        
        print("   Added weak relationships (strength < 0.3)")
        
        # Show impact on search
        query = np.random.rand(384).astype(np.float32)
        results_with_weak = db.search(query, rudradb.SearchParams(
            top_k=5, include_relationships=True, max_hops=2
        ))
        
        weak_connections = [r for r in results_with_weak if r.hop_count > 0 and r.combined_score < 0.4]
        print(f"   Search found {len(weak_connections)} weak relationship connections")
        
        # Solution: Remove weak relationships
        print("✅ Solution: Remove weak relationships")
        
        for source, target, _, _ in weak_relationships:
            db.remove_relationship(source, target)
        
        # Add strong, meaningful relationships
        strong_relationships = [
            ("ai_intro", "ml_basics", "hierarchical", 0.9),      # Strong prerequisite
            ("ml_basics", "dl_advanced", "temporal", 0.85),      # Clear progression
            ("ai_intro", "dl_advanced", "hierarchical", 0.7)     # Broader context
        ]
        
        for source, target, rel_type, strength in strong_relationships:
            db.add_relationship(source, target, rel_type, strength)
        
        print("   Added strong relationships (strength > 0.7)")
        
        results_with_strong = db.search(query, rudradb.SearchParams(
            top_k=5, include_relationships=True, max_hops=2
        ))
        
        strong_connections = [r for r in results_with_strong if r.hop_count > 0]
        avg_strength = np.mean([r.combined_score for r in strong_connections]) if strong_connections else 0
        
        print(f"   Improved search: {len(strong_connections)} connections, avg score: {avg_strength:.3f}")
        
        # Show relationship optimization strategies
        print("\n🚀 Relationship Optimization Strategies:")
        
        strategies = [
            ("Strength thresholds", "Use strength > 0.6 for meaningful relationships", "Strong connections only"),
            ("Type accuracy", "Choose correct relationship type", "semantic vs hierarchical"),
            ("Quality over quantity", "Fewer strong relationships > many weak ones", "Focus on value"),
            ("Regular cleanup", "Remove outdated/weak relationships", "Maintain quality"),
            ("Strategic building", "Plan relationship networks", "Support use cases"),
            ("Monitor impact", "Test search quality after changes", "Verify improvements")
        ]
        
        for strategy, technique, benefit in strategies:
            print(f"   • {strategy}: {technique} → {benefit}")
        
        # Demonstrate relationship audit
        print("\n🔍 Relationship Quality Audit:")
        
        all_relationships = []
        for vec_id in db.list_vectors():
            relationships = db.get_relationships(vec_id)
            all_relationships.extend(relationships)
        
        # Analyze relationship quality
        strength_distribution = {
            "strong (0.8+)": len([r for r in all_relationships if r["strength"] >= 0.8]),
            "good (0.6-0.8)": len([r for r in all_relationships if 0.6 <= r["strength"] < 0.8]),
            "weak (0.4-0.6)": len([r for r in all_relationships if 0.4 <= r["strength"] < 0.6]),
            "poor (<0.4)": len([r for r in all_relationships if r["strength"] < 0.4])
        }
        
        print("   Relationship strength distribution:")
        for category, count in strength_distribution.items():
            percentage = (count / len(all_relationships) * 100) if all_relationships else 0
            print(f"     {category}: {count} ({percentage:.1f}%)")
        
        if strength_distribution["poor (<0.4)"] > 0:
            print("   💡 Recommendation: Remove poor relationships")
        if strength_distribution["strong (0.8+)"] < len(all_relationships) * 0.3:
            print("   💡 Recommendation: Add more strong relationships")
    
    def solve_error_recovery_issues(self):
        """Solve error handling and recovery issues"""
        print("\n🚨 SOLVING: Error Recovery Issues")
        print("=" * 35)
        
        print("🔍 Common Error Scenarios & Recovery:")
        
        # Scenario 1: Dimension mismatch recovery
        print("\n❌ Scenario: Dimension mismatch error")
        
        db = rudradb.RudraDB()
        
        # Add first vector
        correct_embedding = np.random.rand(384).astype(np.float32)
        db.add_vector("correct_doc", correct_embedding)
        
        print(f"   Database dimension locked to: {db.dimension()}")
        
        # Try to add wrong dimension (will fail)
        try:
            wrong_embedding = np.random.rand(512).astype(np.float32)
            db.add_vector("wrong_doc", wrong_embedding)
            print("   ❌ ERROR: Should have failed!")
        except Exception as e:
            print(f"   ✅ Error caught: {type(e).__name__}")
            print(f"   Error message: {str(e)[:60]}...")
        
        print("✅ Recovery strategies:")
        recovery_strategies = [
            "Check embedding dimension before adding",
            "Use consistent embedding model",
            "Create new database for different dimensions",
            "Implement dimension validation wrapper"
        ]
        
        for i, strategy in enumerate(recovery_strategies, 1):
            print(f"   {i}. {strategy}")
        
        # Demonstrate recovery wrapper
        def safe_add_vector_with_validation(db, vec_id, embedding, metadata=None):
            """Safe vector addition with validation"""
            try:
                # Check dimension compatibility
                if db.dimension() is not None and embedding.shape[0] != db.dimension():
                    return {
                        "success": False,
                        "error": f"Dimension mismatch: expected {db.dimension()}, got {embedding.shape[0]}",
                        "recovery": "Use correct dimension or create new database"
                    }
                
                # Check data type
                if embedding.dtype != np.float32:
                    print(f"     💡 Converting {embedding.dtype} to float32")
                    embedding = embedding.astype(np.float32)
                
                # Add vector
                db.add_vector(vec_id, embedding, metadata)
                return {"success": True, "message": f"Added {vec_id} successfully"}
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "recovery": "Check error message for specific solution"
                }
        
        # Test recovery wrapper
        test_embedding = np.random.rand(384).astype(np.float64)  # Wrong type
        result = safe_add_vector_with_validation(db, "recovery_test", test_embedding)
        print(f"   Recovery result: {result}")
        
        # Scenario 2: Capacity limit recovery
        print("\n❌ Scenario: Capacity limit reached")
        
        # Simulate capacity limit (we'll mock this for demo)
        print("   Simulating capacity limit scenario...")
        
        def handle_capacity_limit_error(db, vec_id, embedding, metadata):
            """Handle capacity limit with graceful degradation"""
            try:
                db.add_vector(vec_id, embedding, metadata)
                return {"success": True, "action": "added_normally"}
            
            except RuntimeError as e:
                if "Limit Reached" in str(e):
                    # Capacity management strategies
                    stats = db.get_statistics()
                    
                    return {
                        "success": False,
                        "error": "Capacity limit reached",
                        "current_usage": f"{stats['vector_count']}/{rudradb.MAX_VECTORS}",
                        "recovery_options": [
                            "Export data and start fresh",
                            "Remove low-priority vectors",
                            "Upgrade to full RudraDB",
                            "Optimize existing content"
                        ]
                    }
                else:
                    raise
        
        # Test capacity handling
        capacity_result = handle_capacity_limit_error(
            db, "capacity_test", np.random.rand(384).astype(np.float32), {"test": True}
        )
        print(f"   Capacity handling: {capacity_result}")
        
        # Scenario 3: Data corruption recovery
        print("\n❌ Scenario: Data corruption or invalid state")
        
        def diagnose_and_repair_database(db):
            """Diagnose and attempt to repair database issues"""
            issues_found = []
            repairs_made = []
            
            # Check 1: Vector integrity
            try:
                vector_count = db.vector_count()
                relationship_count = db.relationship_count()
                
                print(f"   Database state: {vector_count} vectors, {relationship_count} relationships")
                
                # Check for orphaned relationships
                valid_vectors = set(db.list_vectors())
                orphaned_relationships = 0
                
                for vec_id in list(valid_vectors)[:5]:  # Check sample
                    relationships = db.get_relationships(vec_id)
                    for rel in relationships:
                        if (rel["source_id"] not in valid_vectors or 
                            rel["target_id"] not in valid_vectors):
                            orphaned_relationships += 1
                
                if orphaned_relationships > 0:
                    issues_found.append(f"Found {orphaned_relationships} orphaned relationships")
                    repairs_made.append("Would remove orphaned relationships in full repair")
                
            except Exception as e:
                issues_found.append(f"Vector access error: {e}")
            
            # Check 2: Search functionality
            try:
                if db.vector_count() > 0:
                    test_vector = db.get_vector(db.list_vectors()[0])
                    test_results = db.search(test_vector["embedding"])
                    
                    if len(test_results) == 0:
                        issues_found.append("Search returns no results")
                    else:
                        repairs_made.append("Search functionality verified")
                        
            except Exception as e:
                issues_found.append(f"Search error: {e}")
            
            return {
                "issues_found": issues_found,
                "repairs_made": repairs_made,
                "health_status": "healthy" if not issues_found else "issues_detected"
            }
        
        # Run database diagnosis
        diagnosis = diagnose_and_repair_database(db)
        print(f"   Diagnosis result: {diagnosis['health_status']}")
        
        if diagnosis["issues_found"]:
            print("   Issues found:")
            for issue in diagnosis["issues_found"]:
                print(f"     • {issue}")
        
        if diagnosis["repairs_made"]:
            print("   Repairs/verifications:")
            for repair in diagnosis["repairs_made"]:
                print(f"     • {repair}")
        
        # Show comprehensive error handling pattern
        print("\n✅ Comprehensive Error Handling Pattern:")
        
        error_handling_code = """
def robust_database_operation(db, operation, *args, **kwargs):
    try:
        return operation(db, *args, **kwargs)
    
    except RuntimeError as e:
        if "Limit Reached" in str(e):
            return handle_capacity_limit(db, e)
        elif "dimension" in str(e).lower():
            return handle_dimension_mismatch(db, e)
        else:
            return handle_generic_runtime_error(db, e)
    
    except ValueError as e:
        return handle_value_error(db, e)
    
    except Exception as e:
        return handle_unexpected_error(db, e)
"""
        
        print("   Error handling pattern:")
        for line in error_handling_code.strip().split('\n'):
            print(f"     {line}")


def main():
    """Run the complete common issues solver"""
    solver = CommonIssuesSolver()
    
    try:
        print("\n🎯 Running Common Issues Solutions Demo...")
        
        # Solve each common issue category
        solver.solve_slow_search_performance()
        solver.solve_memory_issues()
        solver.solve_capacity_planning_issues()
        solver.solve_search_tuning_issues()
        solver.solve_relationship_optimization_issues()
        solver.solve_error_recovery_issues()
        
        print("\n🎉 Common Issues Solutions Demo Complete!")
        print("\n📚 Key Takeaways:")
        takeaways = [
            "⚡ Optimize search with thresholds and hop limits",
            "💾 Manage memory with efficient data types and cleanup",
            "📊 Plan capacity usage and monitor limits proactively", 
            "🎯 Tune search parameters for quality vs coverage",
            "🔗 Build strong, meaningful relationships over quantity",
            "🚨 Implement robust error handling and recovery"
        ]
        
        for takeaway in takeaways:
            print(f"   {takeaway}")
        
        print(f"\n💡 Ready to handle any RudraDB-Opin challenges!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print(f"💡 This demonstrates the importance of error handling!")


if __name__ == "__main__":
    main()
