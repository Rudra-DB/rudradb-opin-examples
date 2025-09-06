#!/usr/bin/env python3
"""
🧪 RudraDB-Opin Comprehensive Test Suite

This example provides a complete testing framework for validating all RudraDB-Opin
functionality. It includes:

1. Auto-dimension detection testing across different models
2. Relationship building and traversal validation
3. Search accuracy and performance testing
4. Capacity limit testing and handling
5. Data integrity and consistency checks
6. Integration testing with ML frameworks
7. Stress testing within Opin limits
8. Regression testing for updates

Perfect for ensuring RudraDB-Opin works correctly in your environment!
"""

import rudradb
import numpy as np
import time
import json
import traceback
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import sys

class RudraDB_Test_Suite:
    """Comprehensive test suite for RudraDB-Opin functionality"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "test_details": {},
            "start_time": datetime.now(),
            "end_time": None
        }
        
        print("🧪 RudraDB-Opin Comprehensive Test Suite")
        print("=" * 50)
        print(f"   🎯 RudraDB-Opin Version: {rudradb.__version__}")
        print(f"   📊 Max Vectors: {rudradb.MAX_VECTORS}")
        print(f"   🔗 Max Relationships: {rudradb.MAX_RELATIONSHIPS}")
        print(f"   🎪 Edition: {rudradb.EDITION}")
        
    def run_all_tests(self):
        """Run the complete test suite"""
        
        print(f"\n🚀 Starting comprehensive test suite...")
        
        test_categories = [
            ("Basic Functionality", self.test_basic_functionality),
            ("Auto-Dimension Detection", self.test_auto_dimension_detection),
            ("Relationship Management", self.test_relationship_management),
            ("Search Functionality", self.test_search_functionality),
            ("Capacity Limits", self.test_capacity_limits),
            ("Data Integrity", self.test_data_integrity),
            ("Performance Characteristics", self.test_performance),
            ("Error Handling", self.test_error_handling),
            ("Advanced Features", self.test_advanced_features)
        ]
        
        for category_name, test_method in test_categories:
            print(f"\n📋 Testing Category: {category_name}")
            print("-" * 40)
            
            try:
                test_method()
            except Exception as e:
                self.record_test_result(f"{category_name}_CRITICAL", False, 
                                       f"Category failed with critical error: {e}")
                if self.verbose:
                    traceback.print_exc()
        
        self._generate_test_report()
        return self.test_results
    
    def test_basic_functionality(self):
        """Test basic RudraDB-Opin operations"""
        
        # Test 1: Database creation
        try:
            db = rudradb.RudraDB()
            self.record_test_result("database_creation", True, "Database created successfully")
        except Exception as e:
            self.record_test_result("database_creation", False, f"Failed to create database: {e}")
            return
        
        # Test 2: Vector addition
        try:
            embedding = np.random.rand(384).astype(np.float32)
            db.add_vector("test_vector", embedding, {"test": True})
            self.record_test_result("vector_addition", True, "Vector added successfully")
        except Exception as e:
            self.record_test_result("vector_addition", False, f"Failed to add vector: {e}")
        
        # Test 3: Vector retrieval
        try:
            vector = db.get_vector("test_vector")
            if vector and vector['metadata']['test']:
                self.record_test_result("vector_retrieval", True, "Vector retrieved successfully")
            else:
                self.record_test_result("vector_retrieval", False, "Vector data mismatch")
        except Exception as e:
            self.record_test_result("vector_retrieval", False, f"Failed to retrieve vector: {e}")
        
        # Test 4: Vector existence check
        try:
            exists = db.vector_exists("test_vector")
            non_exists = db.vector_exists("non_existent_vector")
            if exists and not non_exists:
                self.record_test_result("vector_existence", True, "Vector existence check works")
            else:
                self.record_test_result("vector_existence", False, "Vector existence check failed")
        except Exception as e:
            self.record_test_result("vector_existence", False, f"Vector existence check error: {e}")
        
        # Test 5: Basic relationship
        try:
            embedding2 = np.random.rand(384).astype(np.float32)
            db.add_vector("test_vector2", embedding2, {"test": True})
            db.add_relationship("test_vector", "test_vector2", "semantic", 0.8)
            self.record_test_result("basic_relationship", True, "Basic relationship added")
        except Exception as e:
            self.record_test_result("basic_relationship", False, f"Failed to add relationship: {e}")
        
        # Test 6: Basic search
        try:
            query = np.random.rand(384).astype(np.float32)
            results = db.search(query)
            if len(results) >= 1:
                self.record_test_result("basic_search", True, f"Search returned {len(results)} results")
            else:
                self.record_test_result("basic_search", False, "Search returned no results")
        except Exception as e:
            self.record_test_result("basic_search", False, f"Search failed: {e}")
    
    def test_auto_dimension_detection(self):
        """Test auto-dimension detection with various embedding sizes"""
        
        test_dimensions = [128, 256, 384, 512, 768, 1024, 1536]
        
        for dim in test_dimensions:
            try:
                db = rudradb.RudraDB()  # Fresh instance for each test
                
                # Check initial state
                if db.dimension() is not None:
                    self.record_test_result(f"auto_dim_initial_{dim}", False, 
                                           f"Dimension already set before first vector")
                    continue
                
                # Add first vector
                embedding = np.random.rand(dim).astype(np.float32)
                db.add_vector(f"dim_test_{dim}", embedding)
                
                # Check dimension detection
                detected_dim = db.dimension()
                if detected_dim == dim:
                    self.record_test_result(f"auto_dim_detection_{dim}", True, 
                                           f"Correctly detected {dim}D embeddings")
                else:
                    self.record_test_result(f"auto_dim_detection_{dim}", False, 
                                           f"Expected {dim}D, got {detected_dim}D")
                
                # Test dimension consistency
                try:
                    wrong_embedding = np.random.rand(dim + 1).astype(np.float32)
                    db.add_vector(f"wrong_dim_{dim}", wrong_embedding)
                    self.record_test_result(f"dim_consistency_{dim}", False, 
                                           "Should reject different dimension")
                except Exception:
                    self.record_test_result(f"dim_consistency_{dim}", True, 
                                           "Correctly rejected different dimension")
                
            except Exception as e:
                self.record_test_result(f"auto_dim_error_{dim}", False, 
                                       f"Auto-dimension detection failed for {dim}D: {e}")
    
    def test_relationship_management(self):
        """Test relationship creation, retrieval, and management"""
        
        db = rudradb.RudraDB()
        
        # Setup test vectors
        test_vectors = {}
        for i in range(5):
            embedding = np.random.rand(384).astype(np.float32)
            vector_id = f"rel_test_{i}"
            db.add_vector(vector_id, embedding, {"index": i})
            test_vectors[vector_id] = embedding
        
        # Test 1: All relationship types
        relationship_types = ["semantic", "hierarchical", "temporal", "causal", "associative"]
        
        for i, rel_type in enumerate(relationship_types):
            try:
                source = f"rel_test_{i}"
                target = f"rel_test_{(i+1) % 5}"
                strength = 0.8 + (i * 0.02)  # Vary strength slightly
                
                db.add_relationship(source, target, rel_type, strength)
                
                # Verify relationship exists
                relationships = db.get_relationships(source)
                found = any(r['target_id'] == target and r['relationship_type'] == rel_type 
                           for r in relationships)
                
                if found:
                    self.record_test_result(f"relationship_type_{rel_type}", True, 
                                           f"{rel_type} relationship created successfully")
                else:
                    self.record_test_result(f"relationship_type_{rel_type}", False, 
                                           f"{rel_type} relationship not found")
                    
            except Exception as e:
                self.record_test_result(f"relationship_type_{rel_type}", False, 
                                       f"Failed to create {rel_type} relationship: {e}")
        
        # Test 2: Relationship strength validation
        try:
            invalid_strengths = [-0.5, 1.5, float('nan'), float('inf')]
            valid_rejections = 0
            
            for strength in invalid_strengths:
                try:
                    db.add_relationship("rel_test_0", "rel_test_1", "semantic", strength)
                except (ValueError, RuntimeError):
                    valid_rejections += 1
            
            if valid_rejections == len(invalid_strengths):
                self.record_test_result("relationship_strength_validation", True, 
                                       "Invalid strengths correctly rejected")
            else:
                self.record_test_result("relationship_strength_validation", False, 
                                       f"Only {valid_rejections}/{len(invalid_strengths)} invalid strengths rejected")
        except Exception as e:
            self.record_test_result("relationship_strength_validation", False, 
                                   f"Strength validation test failed: {e}")
        
        # Test 3: Multi-hop traversal
        try:
            connected = db.get_connected_vectors("rel_test_0", max_hops=2)
            
            if len(connected) > 0:
                # Check hop counts
                hop_counts = [hop_count for _, hop_count in connected]
                max_hops = max(hop_counts)
                
                if max_hops <= 2:
                    self.record_test_result("multi_hop_traversal", True, 
                                           f"Multi-hop traversal found {len(connected)} connections")
                else:
                    self.record_test_result("multi_hop_traversal", False, 
                                           f"Max hops exceeded: {max_hops}")
            else:
                self.record_test_result("multi_hop_traversal", False, 
                                       "No connected vectors found")
        except Exception as e:
            self.record_test_result("multi_hop_traversal", False, 
                                   f"Multi-hop traversal failed: {e}")
        
        # Test 4: Relationship removal
        try:
            db.remove_relationship("rel_test_0", "rel_test_1")
            relationships_after = db.get_relationships("rel_test_0")
            
            # Check if relationship was removed
            still_exists = any(r['target_id'] == "rel_test_1" for r in relationships_after)
            
            if not still_exists:
                self.record_test_result("relationship_removal", True, "Relationship successfully removed")
            else:
                self.record_test_result("relationship_removal", False, "Relationship still exists after removal")
        except Exception as e:
            self.record_test_result("relationship_removal", False, f"Relationship removal failed: {e}")
    
    def test_search_functionality(self):
        """Test various search patterns and parameters"""
        
        db = rudradb.RudraDB()
        
        # Setup test data with known relationships
        test_data = []
        for i in range(10):
            # Create somewhat predictable embeddings
            base_embedding = np.ones(384) * (i * 0.1)
            noise = np.random.rand(384) * 0.1
            embedding = (base_embedding + noise).astype(np.float32)
            
            vector_id = f"search_test_{i}"
            metadata = {
                "index": i,
                "category": "A" if i < 5 else "B",
                "value": i * 10
            }
            
            db.add_vector(vector_id, embedding, metadata)
            test_data.append((vector_id, embedding, metadata))
        
        # Add some relationships
        for i in range(8):
            db.add_relationship(f"search_test_{i}", f"search_test_{i+1}", "temporal", 0.8)
        
        # Test 1: Basic similarity search
        try:
            query = np.ones(384) * 0.15  # Should be closest to search_test_1 or search_test_2
            results = db.search(query.astype(np.float32))
            
            if len(results) > 0:
                self.record_test_result("similarity_search", True, 
                                       f"Similarity search returned {len(results)} results")
            else:
                self.record_test_result("similarity_search", False, "No similarity search results")
        except Exception as e:
            self.record_test_result("similarity_search", False, f"Similarity search failed: {e}")
        
        # Test 2: Relationship-aware search
        try:
            params = rudradb.SearchParams(
                top_k=5,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.3
            )
            
            results = db.search(query.astype(np.float32), params)
            
            relationship_enhanced = sum(1 for r in results if r.hop_count > 0)
            if relationship_enhanced > 0:
                self.record_test_result("relationship_aware_search", True, 
                                       f"Found {relationship_enhanced} relationship-enhanced results")
            else:
                self.record_test_result("relationship_aware_search", False, 
                                       "No relationship-enhanced results found")
        except Exception as e:
            self.record_test_result("relationship_aware_search", False, 
                                   f"Relationship-aware search failed: {e}")
        
        # Test 3: Search parameter validation
        try:
            invalid_params_tests = [
                {"top_k": 0},
                {"top_k": -1},
                {"similarity_threshold": -0.1},
                {"similarity_threshold": 1.1},
                {"relationship_weight": -0.1},
                {"relationship_weight": 1.1},
                {"max_hops": -1}
            ]
            
            valid_rejections = 0
            for invalid_params in invalid_params_tests:
                try:
                    params = rudradb.SearchParams(**invalid_params)
                    db.search(query.astype(np.float32), params)
                except (ValueError, RuntimeError):
                    valid_rejections += 1
            
            if valid_rejections >= len(invalid_params_tests) // 2:  # At least half should be rejected
                self.record_test_result("search_parameter_validation", True, 
                                       f"{valid_rejections}/{len(invalid_params_tests)} invalid params rejected")
            else:
                self.record_test_result("search_parameter_validation", False, 
                                       f"Only {valid_rejections}/{len(invalid_params_tests)} invalid params rejected")
        except Exception as e:
            self.record_test_result("search_parameter_validation", False, 
                                   f"Parameter validation test failed: {e}")
        
        # Test 4: Search consistency
        try:
            # Same query should return same results (order might vary slightly due to ties)
            results1 = db.search(query.astype(np.float32))
            results2 = db.search(query.astype(np.float32))
            
            ids1 = set(r.vector_id for r in results1)
            ids2 = set(r.vector_id for r in results2)
            
            if ids1 == ids2:
                self.record_test_result("search_consistency", True, "Search results are consistent")
            else:
                self.record_test_result("search_consistency", False, 
                                       f"Search inconsistency: {len(ids1 & ids2)} common out of {len(ids1)}, {len(ids2)}")
        except Exception as e:
            self.record_test_result("search_consistency", False, f"Search consistency test failed: {e}")
    
    def test_capacity_limits(self):
        """Test capacity limits and graceful handling"""
        
        # Test 1: Vector capacity limit
        try:
            db = rudradb.RudraDB()
            vectors_added = 0
            limit_reached = False
            
            # Try to add more vectors than the limit
            for i in range(rudradb.MAX_VECTORS + 5):
                try:
                    embedding = np.random.rand(384).astype(np.float32)
                    db.add_vector(f"capacity_test_{i}", embedding)
                    vectors_added += 1
                except RuntimeError as e:
                    if "RudraDB-Opin Vector Limit Reached" in str(e):
                        limit_reached = True
                        break
                    else:
                        raise  # Different error
            
            if limit_reached and vectors_added == rudradb.MAX_VECTORS:
                self.record_test_result("vector_capacity_limit", True, 
                                       f"Vector limit correctly enforced at {vectors_added}")
            else:
                self.record_test_result("vector_capacity_limit", False, 
                                       f"Vector limit not enforced properly: added {vectors_added}, limit reached: {limit_reached}")
        
        except Exception as e:
            self.record_test_result("vector_capacity_limit", False, 
                                   f"Vector capacity test failed: {e}")
        
        # Test 2: Relationship capacity limit (using smaller test for speed)
        try:
            db = rudradb.RudraDB()
            
            # Add minimum vectors needed for relationships
            vectors_for_relationships = min(10, rudradb.MAX_VECTORS)
            for i in range(vectors_for_relationships):
                embedding = np.random.rand(384).astype(np.float32)
                db.add_vector(f"rel_capacity_{i}", embedding)
            
            relationships_added = 0
            limit_reached = False
            
            # Try to add relationships up to limit
            test_limit = min(50, rudradb.MAX_RELATIONSHIPS)  # Test subset for speed
            
            for i in range(test_limit + 5):
                try:
                    source = f"rel_capacity_{i % vectors_for_relationships}"
                    target = f"rel_capacity_{(i + 1) % vectors_for_relationships}"
                    
                    if source != target:  # Avoid self-referencing
                        db.add_relationship(source, target, "semantic", 0.5)
                        relationships_added += 1
                except RuntimeError as e:
                    if "RudraDB-Opin Relationship Limit Reached" in str(e):
                        limit_reached = True
                        break
                    elif "already exists" in str(e).lower():
                        continue  # Skip duplicate relationships
                    else:
                        raise  # Different error
            
            # We should hit the limit or complete the test
            if relationships_added > 0:
                self.record_test_result("relationship_capacity_test", True, 
                                       f"Relationship capacity test completed: {relationships_added} added")
            else:
                self.record_test_result("relationship_capacity_test", False, 
                                       "No relationships added in capacity test")
                
        except Exception as e:
            self.record_test_result("relationship_capacity_test", False, 
                                   f"Relationship capacity test failed: {e}")
        
        # Test 3: Graceful capacity handling
        try:
            stats = db.get_statistics()
            usage = stats.get('capacity_usage', {})
            
            required_fields = ['vector_usage_percent', 'relationship_usage_percent', 
                             'vector_capacity_remaining', 'relationship_capacity_remaining']
            
            if all(field in usage for field in required_fields):
                self.record_test_result("capacity_monitoring", True, 
                                       "Capacity monitoring fields available")
            else:
                missing = [field for field in required_fields if field not in usage]
                self.record_test_result("capacity_monitoring", False, 
                                       f"Missing capacity fields: {missing}")
        except Exception as e:
            self.record_test_result("capacity_monitoring", False, 
                                   f"Capacity monitoring test failed: {e}")
    
    def test_data_integrity(self):
        """Test data integrity and consistency"""
        
        db = rudradb.RudraDB()
        
        # Test 1: Vector data integrity
        try:
            original_embedding = np.array([1.0, 2.0, 3.0] + [0.0] * 381, dtype=np.float32)
            original_metadata = {"test": "integrity", "number": 42}
            
            db.add_vector("integrity_test", original_embedding, original_metadata)
            retrieved = db.get_vector("integrity_test")
            
            # Check embedding integrity
            embedding_match = np.allclose(retrieved['embedding'], original_embedding, rtol=1e-5)
            
            # Check metadata integrity
            metadata_match = (retrieved['metadata']['test'] == "integrity" and 
                            retrieved['metadata']['number'] == 42)
            
            if embedding_match and metadata_match:
                self.record_test_result("vector_data_integrity", True, "Vector data preserved correctly")
            else:
                self.record_test_result("vector_data_integrity", False, 
                                       f"Data integrity failed: embedding_match={embedding_match}, metadata_match={metadata_match}")
        except Exception as e:
            self.record_test_result("vector_data_integrity", False, f"Vector integrity test failed: {e}")
        
        # Test 2: Relationship integrity
        try:
            embedding2 = np.random.rand(384).astype(np.float32)
            db.add_vector("integrity_test2", embedding2, {"test": "relationship"})
            
            db.add_relationship("integrity_test", "integrity_test2", "semantic", 0.75, 
                               {"relationship_metadata": "test"})
            
            relationships = db.get_relationships("integrity_test")
            target_relationship = None
            
            for rel in relationships:
                if rel['target_id'] == "integrity_test2":
                    target_relationship = rel
                    break
            
            if (target_relationship and 
                target_relationship['relationship_type'] == "semantic" and
                abs(target_relationship['strength'] - 0.75) < 0.001):
                self.record_test_result("relationship_integrity", True, "Relationship data preserved correctly")
            else:
                self.record_test_result("relationship_integrity", False, "Relationship data integrity failed")
        except Exception as e:
            self.record_test_result("relationship_integrity", False, f"Relationship integrity test failed: {e}")
        
        # Test 3: Database statistics accuracy
        try:
            initial_stats = db.get_statistics()
            
            # Add known quantities
            test_vectors = 3
            test_relationships = 2
            
            for i in range(test_vectors):
                embedding = np.random.rand(384).astype(np.float32)
                db.add_vector(f"stats_test_{i}", embedding)
            
            for i in range(test_relationships):
                source = f"stats_test_{i}"
                target = f"stats_test_{(i+1) % test_vectors}"
                db.add_relationship(source, target, "semantic", 0.5)
            
            final_stats = db.get_statistics()
            
            vector_increase = final_stats['vector_count'] - initial_stats['vector_count']
            relationship_increase = final_stats['relationship_count'] - initial_stats['relationship_count']
            
            if vector_increase == test_vectors and relationship_increase == test_relationships:
                self.record_test_result("statistics_accuracy", True, "Database statistics are accurate")
            else:
                self.record_test_result("statistics_accuracy", False, 
                                       f"Statistics mismatch: vectors +{vector_increase} (expected {test_vectors}), relationships +{relationship_increase} (expected {test_relationships})")
        except Exception as e:
            self.record_test_result("statistics_accuracy", False, f"Statistics accuracy test failed: {e}")
    
    def test_performance(self):
        """Test performance characteristics within Opin limits"""
        
        db = rudradb.RudraDB()
        
        # Test 1: Vector addition performance
        try:
            test_count = 20
            start_time = time.time()
            
            for i in range(test_count):
                embedding = np.random.rand(384).astype(np.float32)
                db.add_vector(f"perf_test_{i}", embedding)
            
            elapsed_time = time.time() - start_time
            vectors_per_second = test_count / elapsed_time
            
            if vectors_per_second > 10:  # Should be much faster than this
                self.record_test_result("vector_addition_performance", True, 
                                       f"Vector addition: {vectors_per_second:.0f} vectors/sec")
            else:
                self.record_test_result("vector_addition_performance", False, 
                                       f"Vector addition too slow: {vectors_per_second:.1f} vectors/sec")
        except Exception as e:
            self.record_test_result("vector_addition_performance", False, 
                                   f"Vector addition performance test failed: {e}")
        
        # Test 2: Search performance
        try:
            query = np.random.rand(384).astype(np.float32)
            test_searches = 10
            
            start_time = time.time()
            for _ in range(test_searches):
                results = db.search(query)
            elapsed_time = time.time() - start_time
            
            searches_per_second = test_searches / elapsed_time
            avg_latency_ms = (elapsed_time / test_searches) * 1000
            
            if avg_latency_ms < 100:  # Should be fast
                self.record_test_result("search_performance", True, 
                                       f"Search performance: {avg_latency_ms:.1f}ms avg latency")
            else:
                self.record_test_result("search_performance", False, 
                                       f"Search too slow: {avg_latency_ms:.1f}ms avg latency")
        except Exception as e:
            self.record_test_result("search_performance", False, 
                                   f"Search performance test failed: {e}")
        
        # Test 3: Memory usage estimation
        try:
            stats = db.get_statistics()
            
            # Rough memory estimation (very approximate)
            vector_memory_mb = (stats['vector_count'] * stats['dimension'] * 4) / (1024 * 1024)
            relationship_memory_mb = (stats['relationship_count'] * 100) / (1024 * 1024)  # Rough estimate
            total_memory_mb = vector_memory_mb + relationship_memory_mb
            
            if total_memory_mb < 100:  # Should be reasonable for Opin limits
                self.record_test_result("memory_usage", True, 
                                       f"Estimated memory usage: {total_memory_mb:.1f} MB")
            else:
                self.record_test_result("memory_usage", False, 
                                       f"High memory usage: {total_memory_mb:.1f} MB")
        except Exception as e:
            self.record_test_result("memory_usage", False, f"Memory usage test failed: {e}")
    
    def test_error_handling(self):
        """Test proper error handling for invalid operations"""
        
        db = rudradb.RudraDB()
        
        # Test 1: Invalid vector operations
        error_cases = [
            ("get_nonexistent_vector", lambda: db.get_vector("nonexistent")),
            ("remove_nonexistent_vector", lambda: db.remove_vector("nonexistent")),
            ("invalid_embedding_type", lambda: db.add_vector("test", "not_an_array")),
            ("self_referencing_relationship", lambda: self._test_self_relationship(db)),
        ]
        
        for test_name, operation in error_cases:
            try:
                result = operation()
                if result is None:  # Expected for get_nonexistent_vector
                    self.record_test_result(f"error_handling_{test_name}", True, 
                                           f"{test_name} handled gracefully")
                else:
                    self.record_test_result(f"error_handling_{test_name}", False, 
                                           f"{test_name} should have failed")
            except Exception as e:
                # Most operations should raise exceptions
                if test_name == "get_nonexistent_vector":
                    self.record_test_result(f"error_handling_{test_name}", False, 
                                           f"{test_name} raised exception when it should return None")
                else:
                    self.record_test_result(f"error_handling_{test_name}", True, 
                                           f"{test_name} properly raised exception")
    
    def _test_self_relationship(self, db):
        """Helper to test self-referencing relationship"""
        embedding = np.random.rand(384).astype(np.float32)
        db.add_vector("self_test", embedding)
        db.add_relationship("self_test", "self_test", "semantic", 0.8)  # Should fail
    
    def test_advanced_features(self):
        """Test advanced RudraDB-Opin features"""
        
        db = rudradb.RudraDB()
        
        # Test 1: Export/Import functionality
        try:
            # Add some test data
            for i in range(3):
                embedding = np.random.rand(384).astype(np.float32)
                db.add_vector(f"export_test_{i}", embedding, {"index": i})
            
            db.add_relationship("export_test_0", "export_test_1", "semantic", 0.8)
            
            # Export data
            exported_data = db.export_data()
            
            # Verify export structure
            required_keys = ["vectors", "relationships", "metadata"]
            if all(key in exported_data for key in required_keys):
                self.record_test_result("data_export", True, "Data export successful")
                
                # Test import (create new database)
                db2 = rudradb.RudraDB()
                db2.import_data(exported_data)
                
                # Verify import
                if (db2.vector_count() == 3 and db2.relationship_count() == 1):
                    self.record_test_result("data_import", True, "Data import successful")
                else:
                    self.record_test_result("data_import", False, 
                                           f"Import mismatch: {db2.vector_count()} vectors, {db2.relationship_count()} relationships")
            else:
                missing_keys = [key for key in required_keys if key not in exported_data]
                self.record_test_result("data_export", False, f"Export missing keys: {missing_keys}")
        
        except Exception as e:
            self.record_test_result("export_import", False, f"Export/import test failed: {e}")
        
        # Test 2: Complex search scenarios
        try:
            # Test filtered search by relationship type
            params = rudradb.SearchParams(
                top_k=5,
                include_relationships=True,
                relationship_types=["semantic"]
            )
            
            query = np.random.rand(384).astype(np.float32)
            results = db.search(query, params)
            
            # Should work without error
            self.record_test_result("filtered_relationship_search", True, 
                                   f"Filtered search returned {len(results)} results")
        except Exception as e:
            self.record_test_result("filtered_relationship_search", False, 
                                   f"Filtered search failed: {e}")
        
        # Test 3: Metadata updates
        try:
            embedding = np.random.rand(384).astype(np.float32)
            db.add_vector("metadata_test", embedding, {"original": "value"})
            
            # Update metadata
            db.update_vector_metadata("metadata_test", {"updated": "new_value", "additional": 123})
            
            # Retrieve and verify
            vector = db.get_vector("metadata_test")
            if (vector and 
                vector['metadata'].get('updated') == "new_value" and
                vector['metadata'].get('additional') == 123):
                self.record_test_result("metadata_update", True, "Metadata update successful")
            else:
                self.record_test_result("metadata_update", False, "Metadata update failed")
        except Exception as e:
            self.record_test_result("metadata_update", False, f"Metadata update test failed: {e}")
    
    def record_test_result(self, test_name: str, passed: bool, message: str = ""):
        """Record the result of a test"""
        
        if passed:
            self.test_results["passed"] += 1
            status = "✅"
        else:
            self.test_results["failed"] += 1
            status = "❌"
            self.test_results["errors"].append(f"{test_name}: {message}")
        
        self.test_results["test_details"][test_name] = {
            "passed": passed,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.verbose:
            print(f"   {status} {test_name}: {message}")
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        
        self.test_results["end_time"] = datetime.now()
        duration = (self.test_results["end_time"] - self.test_results["start_time"]).total_seconds()
        
        print(f"\n📊 TEST SUITE RESULTS")
        print("=" * 50)
        
        total_tests = self.test_results["passed"] + self.test_results["failed"]
        pass_rate = (self.test_results["passed"] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📈 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {self.test_results['passed']} ✅")
        print(f"   Failed: {self.test_results['failed']} ❌")
        print(f"   Pass Rate: {pass_rate:.1f}%")
        print(f"   Duration: {duration:.2f} seconds")
        
        # Category summary
        category_results = {}
        for test_name, details in self.test_results["test_details"].items():
            category = test_name.split("_")[0] if "_" in test_name else "other"
            if category not in category_results:
                category_results[category] = {"passed": 0, "failed": 0}
            
            if details["passed"]:
                category_results[category]["passed"] += 1
            else:
                category_results[category]["failed"] += 1
        
        print(f"\n📋 Results by Category:")
        for category, results in category_results.items():
            total = results["passed"] + results["failed"]
            rate = (results["passed"] / total * 100) if total > 0 else 0
            print(f"   {category.title()}: {results['passed']}/{total} ({rate:.0f}%)")
        
        # Failed tests detail
        if self.test_results["failed"] > 0:
            print(f"\n❌ Failed Tests:")
            for error in self.test_results["errors"][:5]:  # Show first 5
                print(f"   • {error}")
            if len(self.test_results["errors"]) > 5:
                print(f"   ... and {len(self.test_results['errors']) - 5} more")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if pass_rate >= 95:
            print(f"   🎉 Excellent! RudraDB-Opin is working perfectly in your environment")
        elif pass_rate >= 85:
            print(f"   ✅ Good! Minor issues detected, but core functionality works")
        elif pass_rate >= 70:
            print(f"   ⚠️  Some issues detected. Review failed tests and environment")
        else:
            print(f"   🚨 Significant issues detected. Check installation and environment")
        
        if self.test_results["failed"] > 0:
            print(f"   🔧 For failed tests, check:")
            print(f"      - RudraDB-Opin installation: pip install --upgrade rudradb-opin")
            print(f"      - NumPy compatibility: pip install --upgrade numpy")
            print(f"      - Python version: {sys.version}")
        
        print(f"\n📄 Full test report saved to test results object")

def run_integration_tests():
    """Run integration tests with common ML libraries"""
    
    print(f"\n🔌 Integration Tests")
    print("-" * 30)
    
    integration_results = {"passed": 0, "failed": 0, "skipped": 0}
    
    # Test sentence-transformers integration
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        db = rudradb.RudraDB()
        
        text = "This is a test sentence for integration testing"
        embedding = model.encode([text])[0].astype(np.float32)
        
        db.add_vector("sentence_test", embedding, {"text": text})
        
        if db.dimension() == 384:  # all-MiniLM-L6-v2 dimension
            print("   ✅ Sentence Transformers integration: Working")
            integration_results["passed"] += 1
        else:
            print(f"   ❌ Sentence Transformers integration: Wrong dimension {db.dimension()}")
            integration_results["failed"] += 1
            
    except ImportError:
        print("   ⏭️ Sentence Transformers: Not installed (optional)")
        integration_results["skipped"] += 1
    except Exception as e:
        print(f"   ❌ Sentence Transformers integration: Failed - {e}")
        integration_results["failed"] += 1
    
    # Test numpy array handling
    try:
        db = rudradb.RudraDB()
        
        # Test different numpy array types
        array_types = [
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float64).astype(np.float32),
            np.array([1.0] * 384, dtype=np.float32)
        ]
        
        for i, arr in enumerate(array_types):
            db.add_vector(f"numpy_test_{i}", arr)
        
        print("   ✅ NumPy array handling: Working")
        integration_results["passed"] += 1
        
    except Exception as e:
        print(f"   ❌ NumPy array handling: Failed - {e}")
        integration_results["failed"] += 1
    
    return integration_results

def main():
    """Run the comprehensive RudraDB-Opin test suite"""
    
    # Run main test suite
    test_suite = RudraDB_Test_Suite(verbose=True)
    results = test_suite.run_all_tests()
    
    # Run integration tests
    integration_results = run_integration_tests()
    
    # Final summary
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 30)
    
    total_main = results["passed"] + results["failed"]
    total_integration = integration_results["passed"] + integration_results["failed"]
    total_all = total_main + total_integration
    
    if total_all > 0:
        overall_pass_rate = ((results["passed"] + integration_results["passed"]) / total_all * 100)
        
        print(f"Overall Test Results:")
        print(f"   Main Tests: {results['passed']}/{total_main} passed")
        print(f"   Integration Tests: {integration_results['passed']}/{total_integration} passed")
        print(f"   Overall Pass Rate: {overall_pass_rate:.1f}%")
        
        if overall_pass_rate >= 90:
            print(f"\n🎉 RudraDB-Opin is working excellently in your environment!")
            print(f"   Ready for relationship-aware vector search development!")
        elif overall_pass_rate >= 75:
            print(f"\n✅ RudraDB-Opin is working well with minor issues")
            print(f"   Check failed tests for optimization opportunities")
        else:
            print(f"\n⚠️ RudraDB-Opin has some issues in your environment")
            print(f"   Review failed tests and check installation")
        
        print(f"\n🔗 Next Steps:")
        print(f"   • Explore the examples in the examples/ directory")
        print(f"   • Try the tutorial examples for hands-on learning")
        print(f"   • Join the community: https://discord.gg/rudradb")
        print(f"   • Upgrade when ready: pip install rudradb")
    
    return results, integration_results

if __name__ == "__main__":
    try:
        print(f"🧪 RudraDB-Opin Test Suite v1.0")
        print(f"Testing RudraDB-Opin v{rudradb.__version__}")
        
        main()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure RudraDB-Opin is installed: pip install rudradb-opin")
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        print("   Check your Python environment and RudraDB-Opin installation")
        traceback.print_exc()
