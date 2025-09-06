#!/usr/bin/env python3
"""
RudraDB-Opin Best Practices Guide
=================================

Complete guide to best practices for using RudraDB-Opin effectively,
including capacity management, relationship design, and search optimization.
"""

import rudradb
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class RudraDB_BestPractices:
    """Comprehensive best practices for RudraDB-Opin"""
    
    def __init__(self):
        self.db = rudradb.RudraDB()
        
        print("💡 RudraDB-Opin Best Practices Guide")
        print("=" * 40)
        print("   🎯 Designed for 100 vectors, 500 relationships")
        print("   🧠 Focus on quality over quantity")
        print("   ⚡ Optimize for learning and prototyping")
    
    def demonstrate_dimension_management(self):
        """Best practices for dimension management"""
        
        print("\n1️⃣ Dimension Management Best Practices")
        print("=" * 45)
        
        print("📖 Theory:")
        print("   • Use auto-detection for flexibility")
        print("   • Consistent embeddings from same model")
        print("   • Check dimension before adding vectors")
        
        print("\n💻 Implementation:")
        
        # ✅ DO: Use auto-detection
        print("   ✅ DO: Use auto-detection for flexibility")
        db_auto = rudradb.RudraDB()  # Will auto-detect from first embedding
        print(f"      Auto-detection enabled: {db_auto.dimension() is None}")
        
        # Add first vector to trigger auto-detection
        first_embedding = np.random.rand(384).astype(np.float32)
        db_auto.add_vector("first", first_embedding, {"type": "test"})
        print(f"      Auto-detected dimension: {db_auto.dimension()}D")
        
        # ✅ DO: Use explicit dimension when you know it
        print("\n   ✅ DO: Use explicit dimension when certain")
        db_explicit = rudradb.RudraDB(dimension=384)  # For sentence-transformers/all-MiniLM-L6-v2
        print(f"      Explicit dimension set: {db_explicit.dimension()}D")
        
        # ✅ DO: Validate embeddings before adding
        print("\n   ✅ DO: Validate embeddings before adding")
        def safe_add_vector(db, vec_id, embedding, metadata=None):
            """Safe vector addition with validation"""
            if db.dimension() and len(embedding) != db.dimension():
                raise ValueError(f"Dimension mismatch: expected {db.dimension()}, got {len(embedding)}")
            
            db.add_vector(vec_id, embedding.astype(np.float32), metadata or {})
            return True
        
        # Test validation
        try:
            wrong_dim_embedding = np.random.rand(512).astype(np.float32)
            safe_add_vector(db_auto, "wrong", wrong_dim_embedding)
        except ValueError as e:
            print(f"      ✅ Correctly caught dimension error: {e}")
        
        # ❌ DON'T: Mix different embedding dimensions
        print("\n   ❌ DON'T: Mix different embedding dimensions")
        print("      This will cause dimension mismatch errors")
        
        return {"auto_dimension": db_auto.dimension(), "explicit_dimension": db_explicit.dimension()}
    
    def demonstrate_metadata_design(self):
        """Best practices for metadata design"""
        
        print("\n2️⃣ Metadata Design Best Practices")
        print("=" * 40)
        
        print("📖 Theory:")
        print("   • Use consistent metadata structure")
        print("   • Keep metadata searchable and meaningful")
        print("   • Store summaries, not full content")
        
        print("\n💻 Implementation:")
        
        # ✅ DO: Use consistent metadata structure
        print("   ✅ DO: Use consistent metadata structure")
        
        def create_standard_metadata(title: str, category: str, tags: List[str], **kwargs) -> Dict[str, Any]:
            """Standard metadata structure for consistency"""
            return {
                "title": title,
                "category": category,
                "tags": tags or [],
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                **kwargs
            }
        
        # Examples of good metadata
        good_examples = [
            ("doc1", "Machine Learning Basics", "education", ["ml", "basics", "tutorial"]),
            ("doc2", "Python Programming Guide", "programming", ["python", "guide", "coding"]),
            ("doc3", "Data Science Introduction", "education", ["data", "science", "intro"])
        ]
        
        for doc_id, title, category, tags in good_examples:
            embedding = np.random.rand(384).astype(np.float32)
            metadata = create_standard_metadata(
                title=title,
                category=category,
                tags=tags,
                difficulty="beginner",
                author="Best Practices Guide"
            )
            self.db.add_vector(doc_id, embedding, metadata)
            
        print(f"      ✅ Added {len(good_examples)} documents with consistent metadata")
        
        # ✅ DO: Keep metadata searchable
        print("\n   ✅ DO: Keep metadata searchable and filterable")
        searchable_fields = ["title", "category", "tags", "difficulty", "author"]
        print(f"      Searchable fields: {searchable_fields}")
        
        # ❌ DON'T: Store large text in metadata
        print("\n   ❌ DON'T: Store large text in metadata")
        print("      Store summaries or previews instead of full content")
        
        # Show good vs bad metadata
        bad_metadata_example = {
            "full_content": "This is a very long document content that should not be stored in metadata because it wastes memory and makes the database slower..." * 10,
            "raw_html": "<html><body><p>Full HTML content...</p></body></html>",
            "binary_data": b"binary content should not be in metadata"
        }
        
        good_metadata_example = {
            "title": "Document Title",
            "summary": "Brief 2-3 sentence summary of the document",
            "word_count": 150,
            "content_type": "text/plain",
            "preview": "First 200 characters of content..."
        }
        
        print("      Bad metadata example:")
        for key in bad_metadata_example.keys():
            print(f"         ❌ {key}: <large or inappropriate data>")
        
        print("      Good metadata example:")
        for key, value in good_metadata_example.items():
            print(f"         ✅ {key}: {str(value)[:50]}...")
        
        return {"vectors_with_good_metadata": len(good_examples)}
    
    def demonstrate_relationship_strategy(self):
        """Best practices for relationship building"""
        
        print("\n3️⃣ Relationship Strategy Best Practices")
        print("=" * 45)
        
        print("📖 Theory:")
        print("   • Build strategic, meaningful relationships")
        print("   • Use appropriate relationship types")
        print("   • Manage capacity within 500-relationship limit")
        
        print("\n💻 Implementation:")
        
        def build_smart_relationships(db, doc_id: str, metadata: Dict[str, Any], max_connections: int = 5) -> int:
            """Build relationships strategically"""
            category = metadata.get("category")
            tags = metadata.get("tags", [])
            difficulty = metadata.get("difficulty")
            
            relationships_added = 0
            
            # Strategy 1: Connect to same category (semantic)
            if category and relationships_added < max_connections:
                similar_docs = [
                    vid for vid in db.list_vectors()
                    if db.get_vector(vid)["metadata"].get("category") == category and vid != doc_id
                ]
                
                # Connect to most recent in category (limit to avoid over-connection)
                for other_doc in similar_docs[-2:]:  # Only last 2 documents
                    if relationships_added < max_connections:
                        db.add_relationship(doc_id, other_doc, "semantic", 0.7, 
                                          {"reason": "same_category", "auto_created": True})
                        relationships_added += 1
                        print(f"         🔗 Semantic: {doc_id} ↔ {other_doc} (same category)")
            
            # Strategy 2: Connect by shared tags (associative)
            for tag in tags[:2]:  # Limit tag connections to top 2 tags
                if relationships_added >= max_connections:
                    break
                    
                tagged_docs = [
                    vid for vid in db.list_vectors()
                    if tag in db.get_vector(vid)["metadata"].get("tags", []) and vid != doc_id
                ]
                
                # Connect to one document per tag
                if tagged_docs and relationships_added < max_connections:
                    other_doc = tagged_docs[-1]  # Most recent with this tag
                    db.add_relationship(doc_id, other_doc, "associative", 0.5,
                                      {"reason": "shared_tag", "tag": tag, "auto_created": True})
                    relationships_added += 1
                    print(f"         🏷️ Associative: {doc_id} ↔ {other_doc} (tag: {tag})")
            
            # Strategy 3: Learning progression (hierarchical/temporal)
            if difficulty:
                difficulty_order = {"beginner": 1, "intermediate": 2, "advanced": 3}
                current_level = difficulty_order.get(difficulty, 2)
                
                for other_id in db.list_vectors():
                    if other_id == doc_id or relationships_added >= max_connections:
                        continue
                    
                    other_vector = db.get_vector(other_id)
                    other_difficulty = other_vector["metadata"].get("difficulty")
                    
                    if other_difficulty:
                        other_level = difficulty_order.get(other_difficulty, 2)
                        
                        # Create hierarchical relationship for learning progression
                        if abs(current_level - other_level) == 1 and category == other_vector["metadata"].get("category"):
                            rel_type = "hierarchical" if current_level < other_level else "temporal"
                            db.add_relationship(doc_id, other_id, rel_type, 0.8,
                                              {"reason": "learning_progression", "auto_created": True})
                            relationships_added += 1
                            print(f"         📚 {rel_type.title()}: {doc_id} → {other_id} (learning progression)")
                            break
            
            return relationships_added
        
        # Apply strategic relationship building to existing vectors
        print("   ✅ Building strategic relationships:")
        total_relationships = 0
        
        for vec_id in self.db.list_vectors():
            vector = self.db.get_vector(vec_id)
            if vector:
                connections = build_smart_relationships(self.db, vec_id, vector["metadata"], max_connections=3)
                total_relationships += connections
        
        print(f"      ✅ Created {total_relationships} strategic relationships")
        
        # Show relationship type distribution
        relationship_types = {}
        for vec_id in self.db.list_vectors():
            relationships = self.db.get_relationships(vec_id)
            for rel in relationships:
                rel_type = rel["relationship_type"]
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        print("      📊 Relationship type distribution:")
        for rel_type, count in relationship_types.items():
            print(f"         {rel_type}: {count}")
        
        return {"total_relationships": total_relationships, "relationship_types": relationship_types}
    
    def demonstrate_search_optimization(self):
        """Best practices for search optimization"""
        
        print("\n4️⃣ Search Optimization Best Practices")
        print("=" * 42)
        
        print("📖 Theory:")
        print("   • Adapt search parameters to use case")
        print("   • Use progressive search strategies")
        print("   • Balance precision vs discovery")
        
        print("\n💻 Implementation:")
        
        class SearchOptimizer:
            def __init__(self, db):
                self.db = db
            
            def adaptive_search(self, query_embedding: np.ndarray, context: str = "general") -> List[Any]:
                """Adapt search parameters based on context"""
                
                base_params = {
                    "top_k": 10,
                    "include_relationships": True,
                    "max_hops": 2
                }
                
                if context == "precise":
                    # High precision, low recall
                    base_params.update({
                        "top_k": 5,
                        "similarity_threshold": 0.5,
                        "include_relationships": False
                    })
                    
                elif context == "discovery":
                    # High recall, discovery-focused
                    base_params.update({
                        "top_k": 15,
                        "similarity_threshold": 0.1,
                        "relationship_weight": 0.6
                    })
                    
                elif context == "recommendation":
                    # Balanced with relationship emphasis
                    base_params.update({
                        "top_k": 10,
                        "similarity_threshold": 0.2,
                        "relationship_weight": 0.5,
                        "relationship_types": ["semantic", "associative"]
                    })
                
                params = rudradb.SearchParams(**base_params)
                return self.db.search(query_embedding, params)
            
            def progressive_search(self, query_embedding: np.ndarray, target_results: int = 5) -> List[Any]:
                """Try different search strategies until target met"""
                
                strategies = [
                    ("precise", 0.5, False, 0.2),
                    ("balanced", 0.3, True, 0.3), 
                    ("broad", 0.1, True, 0.5)
                ]
                
                for strategy_name, threshold, use_rels, rel_weight in strategies:
                    params = rudradb.SearchParams(
                        top_k=target_results * 2,  # Get more to allow for filtering
                        similarity_threshold=threshold,
                        include_relationships=use_rels,
                        relationship_weight=rel_weight,
                        max_hops=2
                    )
                    
                    results = self.db.search(query_embedding, params)
                    
                    if len(results) >= target_results:
                        print(f"      ✅ Found {len(results)} results using {strategy_name} strategy")
                        return results[:target_results]
                
                return results  # Return whatever we got
        
        # Demonstrate search optimization
        optimizer = SearchOptimizer(self.db)
        
        # Test query
        test_query = np.random.rand(384).astype(np.float32)
        
        print("   ✅ Testing adaptive search strategies:")
        
        # Precise search
        precise_results = optimizer.adaptive_search(test_query, context="precise")
        print(f"      Precise search: {len(precise_results)} high-confidence results")
        
        # Discovery search
        discovery_results = optimizer.adaptive_search(test_query, context="discovery")
        print(f"      Discovery search: {len(discovery_results)} results (including relationships)")
        
        # Recommendation search
        recommendation_results = optimizer.adaptive_search(test_query, context="recommendation")
        print(f"      Recommendation search: {len(recommendation_results)} balanced results")
        
        # Progressive search
        print("\n   ✅ Testing progressive search:")
        progressive_results = optimizer.progressive_search(test_query, target_results=5)
        print(f"      Progressive search: Found {len(progressive_results)} results")
        
        return {
            "precise_count": len(precise_results),
            "discovery_count": len(discovery_results),
            "recommendation_count": len(recommendation_results),
            "progressive_count": len(progressive_results)
        }
    
    def demonstrate_capacity_management(self):
        """Best practices for capacity management"""
        
        print("\n5️⃣ Capacity Management Best Practices")
        print("=" * 44)
        
        print("📖 Theory:")
        print("   • Monitor capacity usage proactively")
        print("   • Quality over quantity approach")
        print("   • Plan for upgrade when needed")
        
        print("\n💻 Implementation:")
        
        def monitor_capacity(db) -> Dict[str, Any]:
            """Monitor and report capacity usage"""
            stats = db.get_statistics()
            usage = stats['capacity_usage']
            
            capacity_info = {
                "vector_usage": usage['vector_usage_percent'],
                "relationship_usage": usage['relationship_usage_percent'],
                "vector_remaining": usage['vector_capacity_remaining'],
                "relationship_remaining": usage['relationship_capacity_remaining'],
                "status": "healthy"
            }
            
            # Determine status
            max_usage = max(usage['vector_usage_percent'], usage['relationship_usage_percent'])
            
            if max_usage > 90:
                capacity_info["status"] = "critical"
            elif max_usage > 75:
                capacity_info["status"] = "warning"
            elif max_usage > 50:
                capacity_info["status"] = "moderate"
            
            print(f"   📊 Capacity Status: {capacity_info['status'].upper()}")
            print(f"      Vectors: {stats['vector_count']}/{rudradb.MAX_VECTORS} ({usage['vector_usage_percent']:.1f}%)")
            print(f"      Relationships: {stats['relationship_count']}/{rudradb.MAX_RELATIONSHIPS} ({usage['relationship_usage_percent']:.1f}%)")
            
            # Provide recommendations based on status
            if capacity_info["status"] == "critical":
                print("   🚨 Critical: Consider upgrade or data cleanup")
            elif capacity_info["status"] == "warning":
                print("   ⚠️ Warning: Approaching capacity limits")
            elif capacity_info["status"] == "moderate":
                print("   💡 Good usage: Consider relationship quality")
            else:
                print("   ✅ Healthy: Plenty of capacity remaining")
            
            return capacity_info
        
        def capacity_aware_operations(db) -> Dict[str, Any]:
            """Demonstrate capacity-aware operations"""
            
            def safe_add_vector(db, vec_id, embedding, metadata):
                """Add vector with capacity awareness"""
                try:
                    db.add_vector(vec_id, embedding, metadata)
                    return {"success": True, "message": f"Vector '{vec_id}' added"}
                except RuntimeError as e:
                    if "Vector Limit Reached" in str(e):
                        return {
                            "success": False, 
                            "message": "🎓 Vector capacity reached - perfect for learning!",
                            "upgrade_hint": "Ready for production? Upgrade to full RudraDB!"
                        }
                    else:
                        raise
            
            def safe_add_relationship(db, source_id, target_id, rel_type, strength):
                """Add relationship with capacity awareness"""
                try:
                    db.add_relationship(source_id, target_id, rel_type, strength)
                    return {"success": True, "message": f"Relationship '{source_id}' -> '{target_id}' added"}
                except RuntimeError as e:
                    if "Relationship Limit Reached" in str(e):
                        return {
                            "success": False,
                            "message": "🧠 Relationship capacity reached - you've mastered relationship modeling!",
                            "upgrade_hint": "Ready for production? Upgrade to full RudraDB!"
                        }
                    else:
                        raise
            
            return {"safe_add_vector": safe_add_vector, "safe_add_relationship": safe_add_relationship}
        
        # Monitor current capacity
        capacity_info = monitor_capacity(self.db)
        
        # Demonstrate capacity-aware operations
        safe_ops = capacity_aware_operations(self.db)
        
        print("\n   ✅ Capacity-aware operation examples:")
        print("      Use safe_add_vector() and safe_add_relationship()")
        print("      These functions handle capacity limits gracefully")
        
        # Show optimization strategies
        print("\n   💡 Optimization Strategies:")
        strategies = [
            "Focus on high-quality, meaningful relationships",
            "Remove low-strength relationships (< 0.3) if needed",
            "Use strategic relationship building (max 3-5 per vector)",
            "Consider upgrading when you consistently hit limits",
            "Export/import to manage different datasets"
        ]
        
        for i, strategy in enumerate(strategies, 1):
            print(f"      {i}. {strategy}")
        
        return capacity_info
    
    def demonstrate_error_handling(self):
        """Best practices for error handling"""
        
        print("\n6️⃣ Error Handling Best Practices")
        print("=" * 38)
        
        print("📖 Theory:")
        print("   • Handle capacity limits gracefully")
        print("   • Provide helpful error messages")
        print("   • Implement retry strategies")
        
        print("\n💻 Implementation:")
        
        class RobustRudraDB:
            """RudraDB wrapper with robust error handling"""
            
            def __init__(self):
                self.db = rudradb.RudraDB()
                self.error_log = []
            
            def safe_add_vector(self, vec_id: str, embedding: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
                """Safely add vector with comprehensive error handling"""
                try:
                    self.db.add_vector(vec_id, embedding, metadata or {})
                    return {"success": True, "vector_id": vec_id}
                    
                except RuntimeError as e:
                    error_msg = str(e)
                    self.error_log.append({"type": "vector_add", "error": error_msg, "vector_id": vec_id})
                    
                    if "Vector Limit Reached" in error_msg:
                        return {
                            "success": False,
                            "error_type": "capacity_limit",
                            "message": "🎓 You've successfully explored 100 vectors with RudraDB-Opin!",
                            "suggestion": "Ready for production scale? Upgrade to full RudraDB!"
                        }
                    elif "dimension" in error_msg.lower():
                        return {
                            "success": False,
                            "error_type": "dimension_mismatch",
                            "message": f"Embedding dimension mismatch. Expected: {self.db.dimension()}",
                            "suggestion": "Ensure all embeddings have the same dimension"
                        }
                    else:
                        return {
                            "success": False,
                            "error_type": "unknown",
                            "message": error_msg
                        }
                        
                except Exception as e:
                    return {
                        "success": False,
                        "error_type": "unexpected",
                        "message": f"Unexpected error: {str(e)}"
                    }
            
            def safe_add_relationship(self, source_id: str, target_id: str, rel_type: str, strength: float) -> Dict[str, Any]:
                """Safely add relationship with comprehensive error handling"""
                try:
                    self.db.add_relationship(source_id, target_id, rel_type, strength)
                    return {"success": True, "relationship": f"{source_id} -> {target_id}"}
                    
                except RuntimeError as e:
                    error_msg = str(e)
                    self.error_log.append({"type": "relationship_add", "error": error_msg})
                    
                    if "Relationship Limit Reached" in error_msg:
                        return {
                            "success": False,
                            "error_type": "capacity_limit",
                            "message": "🧠 You've built 500 relationships - amazing!",
                            "suggestion": "Ready for unlimited relationships? Upgrade to full RudraDB!"
                        }
                    elif "not found" in error_msg.lower():
                        return {
                            "success": False,
                            "error_type": "vector_not_found",
                            "message": "One or both vectors don't exist",
                            "suggestion": "Add vectors before creating relationships"
                        }
                    else:
                        return {
                            "success": False,
                            "error_type": "unknown",
                            "message": error_msg
                        }
                        
                except Exception as e:
                    return {
                        "success": False,
                        "error_type": "unexpected",
                        "message": f"Unexpected error: {str(e)}"
                    }
            
            def get_error_summary(self) -> Dict[str, Any]:
                """Get summary of errors encountered"""
                error_types = {}
                for error in self.error_log:
                    error_type = error.get("type", "unknown")
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                return {
                    "total_errors": len(self.error_log),
                    "error_types": error_types,
                    "recent_errors": self.error_log[-5:]  # Last 5 errors
                }
        
        # Demonstrate robust error handling
        robust_db = RobustRudraDB()
        
        print("   ✅ Testing robust error handling:")
        
        # Test normal operation
        normal_result = robust_db.safe_add_vector(
            "test_vector", 
            np.random.rand(384).astype(np.float32),
            {"test": True}
        )
        print(f"      Normal operation: {'✅' if normal_result['success'] else '❌'}")
        
        # Test dimension mismatch
        dimension_error = robust_db.safe_add_vector(
            "wrong_dim",
            np.random.rand(512).astype(np.float32),  # Wrong dimension
            {"test": True}
        )
        print(f"      Dimension error handling: {'✅' if not dimension_error['success'] else '❌'}")
        if not dimension_error['success']:
            print(f"         Message: {dimension_error['message']}")
        
        # Test relationship with non-existent vector
        relationship_error = robust_db.safe_add_relationship(
            "test_vector", "non_existent", "semantic", 0.8
        )
        print(f"      Missing vector error: {'✅' if not relationship_error['success'] else '❌'}")
        
        # Show error summary
        error_summary = robust_db.get_error_summary()
        print(f"      Total errors logged: {error_summary['total_errors']}")
        
        return {"robust_operations": True, "error_summary": error_summary}
    
    def generate_best_practices_summary(self):
        """Generate comprehensive best practices summary"""
        
        print("\n📋 Best Practices Summary")
        print("=" * 30)
        
        best_practices = {
            "Dimension Management": [
                "Use auto-detection for flexibility",
                "Validate embeddings before adding",
                "Keep all embeddings the same dimension"
            ],
            "Metadata Design": [
                "Use consistent structure across all vectors",
                "Store searchable, meaningful information",
                "Use summaries instead of full content"
            ],
            "Relationship Strategy": [
                "Build strategic, meaningful connections",
                "Use appropriate relationship types",
                "Limit connections per vector (3-5 max)"
            ],
            "Search Optimization": [
                "Adapt parameters to use case",
                "Use progressive search strategies",
                "Balance precision vs discovery"
            ],
            "Capacity Management": [
                "Monitor usage proactively",
                "Focus on quality over quantity",
                "Plan upgrade when consistently hitting limits"
            ],
            "Error Handling": [
                "Handle capacity limits gracefully",
                "Provide helpful error messages",
                "Log errors for debugging"
            ]
        }
        
        for category, practices in best_practices.items():
            print(f"\n{category}:")
            for practice in practices:
                print(f"   ✅ {practice}")
        
        # Final recommendations
        print(f"\n🎯 Key Recommendations for RudraDB-Opin:")
        key_recommendations = [
            "Start with auto-dimension detection",
            "Build 3-5 strategic relationships per vector",
            "Monitor capacity usage regularly",
            "Focus on relationship quality over quantity",
            "Use semantic relationships for similar content",
            "Use hierarchical relationships for categories",
            "Use temporal relationships for sequences",
            "Use causal relationships for problem-solution pairs",
            "Upgrade to full RudraDB when ready for production"
        ]
        
        for i, rec in enumerate(key_recommendations, 1):
            print(f"   {i}. {rec}")
        
        return best_practices

def main():
    """Demonstrate all best practices"""
    
    print("🚀 RudraDB-Opin Best Practices Demonstration")
    print("=" * 50)
    
    # Initialize best practices guide
    guide = RudraDB_BestPractices()
    
    # Demonstrate each category
    dimension_result = guide.demonstrate_dimension_management()
    metadata_result = guide.demonstrate_metadata_design()
    relationship_result = guide.demonstrate_relationship_strategy()
    search_result = guide.demonstrate_search_optimization()
    capacity_result = guide.demonstrate_capacity_management()
    error_result = guide.demonstrate_error_handling()
    
    # Generate summary
    summary = guide.generate_best_practices_summary()
    
    print(f"\n🎉 Best Practices Demonstration Complete!")
    print(f"   📊 Database: {guide.db.vector_count()} vectors, {guide.db.relationship_count()} relationships")
    print(f"   🎯 Dimension: {guide.db.dimension()}D (auto-detected)")
    print(f"   💡 All best practices demonstrated successfully!")
    
    print(f"\n📚 Next Steps:")
    print(f"   • Apply these practices to your own data")
    print(f"   • Experiment with different relationship types")
    print(f"   • Monitor capacity usage as you build")
    print(f"   • Upgrade to full RudraDB when ready for production")
    
    return {
        "dimension_management": dimension_result,
        "metadata_design": metadata_result,
        "relationship_strategy": relationship_result,
        "search_optimization": search_result,
        "capacity_management": capacity_result,
        "error_handling": error_result,
        "best_practices": summary
    }

if __name__ == "__main__":
    main()
