#!/usr/bin/env python3
"""
Pinecone to RudraDB-Opin Migration Tool

This example demonstrates how to migrate from Pinecone to RudraDB-Opin,
showcasing the advantages of relationship-aware search and auto-dimension detection
while providing a seamless migration path.

Requirements:
    pip install rudradb-opin

Usage:
    python pinecone_migration.py
"""

import rudradb
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time


class Pinecone_to_RudraDB_Migration:
    """Seamless migration from Pinecone to RudraDB-Opin with enhanced auto-features"""
    
    def __init__(self, pinecone_dimension: Optional[int] = None):
        """Initialize migration tool with optional dimension hint"""
        
        # RudraDB-Opin with auto-dimension detection
        self.rudra_db = rudradb.RudraDB()  # 🎯 Auto-detects dimensions from migrated data
        
        # Migration tracking
        self.migration_stats = {
            "vectors_migrated": 0,
            "relationships_auto_created": 0,
            "dimension_detected": None,
            "pinecone_dimension_hint": pinecone_dimension,
            "migration_start_time": time.time()
        }
        
        print("🌊 Pinecone → RudraDB-Opin Migration Tool initialized")
        print("   🎯 Auto-dimension detection enabled (no manual dimension setting required)")
        print("   🧠 Auto-relationship detection will create intelligent connections")
        if pinecone_dimension:
            print(f"   💡 Pinecone dimension hint: {pinecone_dimension}D")
        
    def create_sample_pinecone_data(self) -> List[Dict[str, Any]]:
        """Create sample Pinecone-format data for demo purposes"""
        
        # Simulate typical Pinecone vector data format
        sample_data = [
            {
                "id": "doc_ai_intro",
                "values": np.random.rand(384).tolist(),  # Simulated 384D embedding
                "metadata": {
                    "category": "AI",
                    "tags": ["artificial intelligence", "introduction", "basics"],
                    "user_id": "researcher_1",
                    "source": "research_papers",
                    "title": "Introduction to Artificial Intelligence",
                    "created_at": "2024-01-01T00:00:00Z"
                }
            },
            {
                "id": "doc_ml_fundamentals", 
                "values": np.random.rand(384).tolist(),
                "metadata": {
                    "category": "AI",
                    "tags": ["machine learning", "algorithms", "data science"],
                    "user_id": "researcher_1",
                    "source": "research_papers", 
                    "title": "Machine Learning Fundamentals",
                    "created_at": "2024-01-15T00:00:00Z"
                }
            },
            {
                "id": "doc_neural_networks",
                "values": np.random.rand(384).tolist(),
                "metadata": {
                    "category": "AI",
                    "tags": ["neural networks", "deep learning", "backpropagation"],
                    "user_id": "researcher_2",
                    "source": "textbooks",
                    "title": "Neural Networks and Deep Learning",
                    "created_at": "2024-02-01T00:00:00Z"
                }
            },
            {
                "id": "doc_nlp_overview",
                "values": np.random.rand(384).tolist(),
                "metadata": {
                    "category": "NLP",
                    "tags": ["natural language processing", "text analysis", "linguistics"],
                    "user_id": "researcher_2", 
                    "source": "research_papers",
                    "title": "Natural Language Processing Overview",
                    "created_at": "2024-02-15T00:00:00Z"
                }
            },
            {
                "id": "doc_computer_vision",
                "values": np.random.rand(384).tolist(),
                "metadata": {
                    "category": "CV",
                    "tags": ["computer vision", "image processing", "pattern recognition"],
                    "user_id": "researcher_1",
                    "source": "textbooks",
                    "title": "Computer Vision Techniques",
                    "created_at": "2024-03-01T00:00:00Z"
                }
            }
        ]
        
        print(f"📄 Created {len(sample_data)} sample Pinecone vectors for migration demo")
        return sample_data
    
    def migrate_pinecone_data(self, pinecone_vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Migrate vectors from Pinecone format to RudraDB-Opin with auto-enhancements"""
        
        print(f"🔄 Starting migration of {len(pinecone_vectors)} vectors from Pinecone...")
        
        relationships_created = 0
        migration_errors = []
        
        for i, pinecone_vector in enumerate(pinecone_vectors):
            try:
                # Extract Pinecone data
                vector_id = pinecone_vector.get("id", f"migrated_vector_{i}")
                values = pinecone_vector.get("values", [])
                metadata = pinecone_vector.get("metadata", {})
                
                # Convert to numpy array
                embedding = np.array(values, dtype=np.float32)
                
                # Enhance metadata for auto-relationship detection
                enhanced_metadata = {
                    "migrated_from": "pinecone",
                    "original_id": vector_id,
                    "migration_order": i,
                    "migration_timestamp": datetime.now().isoformat(),
                    **metadata  # Preserve original Pinecone metadata
                }
                
                # Add to RudraDB-Opin with auto-dimension detection
                self.rudra_db.add_vector(vector_id, embedding, enhanced_metadata)
                self.migration_stats["vectors_migrated"] += 1
                
                # Update dimension info after first vector
                if self.migration_stats["dimension_detected"] is None:
                    self.migration_stats["dimension_detected"] = self.rudra_db.dimension()
                    print(f"   🎯 Auto-detected dimension: {self.migration_stats['dimension_detected']}D")
                    
                    # Validate against Pinecone hint
                    if (self.migration_stats["pinecone_dimension_hint"] and 
                        self.migration_stats["dimension_detected"] != self.migration_stats["pinecone_dimension_hint"]):
                        print(f"   ⚠️ Dimension mismatch detected!")
                        print(f"      Pinecone hint: {self.migration_stats['pinecone_dimension_hint']}D")
                        print(f"      Auto-detected: {self.migration_stats['dimension_detected']}D")
                
                # 🧠 Auto-create relationships based on migrated metadata
                if self.migration_stats["vectors_migrated"] > 1:  # Need at least 2 vectors
                    vector_relationships = self._auto_create_migration_relationships(vector_id, enhanced_metadata)
                    relationships_created += vector_relationships
                
                if (i + 1) % 2 == 0 or i == len(pinecone_vectors) - 1:  # Progress update
                    print(f"   📊 Migrated {i + 1}/{len(pinecone_vectors)} vectors...")
                    
            except Exception as e:
                error_info = f"Vector {vector_id}: {str(e)}"
                migration_errors.append(error_info)
                print(f"   ❌ Failed to migrate vector {vector_id}: {e}")
                continue
        
        self.migration_stats["relationships_auto_created"] = relationships_created
        self.migration_stats["migration_duration"] = time.time() - self.migration_stats["migration_start_time"]
        self.migration_stats["migration_errors"] = migration_errors
        
        return self.migration_stats
    
    def _auto_create_migration_relationships(self, new_vector_id: str, metadata: Dict[str, Any]) -> int:
        """Auto-create intelligent relationships based on migrated Pinecone metadata"""
        
        relationships_created = 0
        
        # Extract relationship indicators from metadata
        new_category = metadata.get("category") or metadata.get("type")
        new_tags = set(metadata.get("tags", []) if isinstance(metadata.get("tags"), list) else [])
        new_user = metadata.get("user_id") or metadata.get("user")
        new_source = metadata.get("source") or metadata.get("source_type")
        
        # Analyze existing vectors for relationship opportunities
        for existing_id in self.rudra_db.list_vectors():
            if existing_id == new_vector_id or relationships_created >= 3:
                continue
                
            existing_vector = self.rudra_db.get_vector(existing_id)
            existing_meta = existing_vector['metadata']
            
            existing_category = existing_meta.get("category") or existing_meta.get("type")
            existing_tags = set(existing_meta.get("tags", []) if isinstance(existing_meta.get("tags"), list) else [])
            existing_user = existing_meta.get("user_id") or existing_meta.get("user")
            existing_source = existing_meta.get("source") or existing_meta.get("source_type")
            
            # 🔗 Semantic relationship: Same category/type
            if new_category and new_category == existing_category:
                self.rudra_db.add_relationship(new_vector_id, existing_id, "semantic", 0.8,
                    {"auto_detected": True, "reason": "pinecone_same_category", "category": new_category})
                relationships_created += 1
                print(f"      🔗 Auto-linked: {new_vector_id} ↔ {existing_id} (same category: {new_category})")
            
            # 🏷️ Associative relationship: Shared tags
            elif len(new_tags & existing_tags) >= 1:
                shared_tags = new_tags & existing_tags
                strength = min(0.7, len(shared_tags) * 0.3)
                self.rudra_db.add_relationship(new_vector_id, existing_id, "associative", strength,
                    {"auto_detected": True, "reason": "pinecone_shared_tags", "tags": list(shared_tags)})
                relationships_created += 1
                print(f"      🏷️ Auto-linked: {new_vector_id} ↔ {existing_id} (shared tags: {shared_tags})")
            
            # 📊 Hierarchical relationship: Same user/owner
            elif new_user and new_user == existing_user:
                self.rudra_db.add_relationship(new_vector_id, existing_id, "hierarchical", 0.7,
                    {"auto_detected": True, "reason": "pinecone_same_user", "user": new_user})
                relationships_created += 1
                print(f"      📊 Auto-linked: {new_vector_id} ↔ {existing_id} (same user: {new_user})")
            
            # 🎯 Associative relationship: Same source
            elif new_source and new_source == existing_source:
                self.rudra_db.add_relationship(new_vector_id, existing_id, "associative", 0.6,
                    {"auto_detected": True, "reason": "pinecone_same_source", "source": new_source})
                relationships_created += 1
                print(f"      🎯 Auto-linked: {new_vector_id} ↔ {existing_id} (same source: {new_source})")
        
        return relationships_created
    
    def compare_capabilities(self) -> Dict[str, Any]:
        """Compare Pinecone vs RudraDB-Opin capabilities after migration"""
        
        stats = self.rudra_db.get_statistics()
        
        comparison = {
            "feature_comparison": {
                "Vector Storage": {"Pinecone": "✅ Yes", "RudraDB-Opin": "✅ Yes"},
                "Similarity Search": {"Pinecone": "✅ Yes", "RudraDB-Opin": "✅ Yes"},
                "Auto-Dimension Detection": {"Pinecone": "❌ Manual only", "RudraDB-Opin": "✅ Automatic"},
                "Relationship Modeling": {"Pinecone": "❌ None", "RudraDB-Opin": "✅ 5 types"},
                "Auto-Relationship Detection": {"Pinecone": "❌ None", "RudraDB-Opin": "✅ Intelligent"},
                "Multi-hop Discovery": {"Pinecone": "❌ None", "RudraDB-Opin": "✅ 2 hops"},
                "Metadata Filtering": {"Pinecone": "✅ Basic", "RudraDB-Opin": "✅ Advanced"},
                "Free Tier": {"Pinecone": "❌ Limited trial", "RudraDB-Opin": "✅ 100% free forever"},
                "Setup Complexity": {"Pinecone": "❌ API keys, config", "RudraDB-Opin": "✅ pip install"},
                "Relationship-Enhanced Search": {"Pinecone": "❌ Not available", "RudraDB-Opin": "✅ Automatic"}
            },
            "migration_results": {
                "vectors_migrated": self.migration_stats["vectors_migrated"],
                "relationships_auto_created": self.migration_stats["relationships_auto_created"],
                "dimension_auto_detected": self.migration_stats["dimension_detected"],
                "migration_duration": f"{self.migration_stats['migration_duration']:.2f}s",
                "capacity_remaining": {
                    "vectors": stats["capacity_usage"]["vector_capacity_remaining"],
                    "relationships": stats["capacity_usage"]["relationship_capacity_remaining"]
                },
                "migration_success_rate": (self.migration_stats["vectors_migrated"] / 
                                         (self.migration_stats["vectors_migrated"] + len(self.migration_stats.get("migration_errors", [])))) * 100
            }
        }
        
        return comparison
    
    def demonstrate_enhanced_search(self, query_vector: np.ndarray, query_description: str) -> Dict[str, Any]:
        """Demonstrate RudraDB-Opin's enhanced search vs Pinecone-style search"""
        
        print(f"🔍 Search Comparison: {query_description}")
        
        # Pinecone-style similarity-only search
        basic_results = self.rudra_db.search(query_vector, rudradb.SearchParams(
            top_k=5,
            include_relationships=False  # Pinecone equivalent
        ))
        
        # RudraDB-Opin enhanced search with auto-relationships
        enhanced_results = self.rudra_db.search(query_vector, rudradb.SearchParams(
            top_k=5,
            include_relationships=True,  # Use auto-detected relationships
            max_hops=2,
            relationship_weight=0.3
        ))
        
        comparison_result = {
            "query_description": query_description,
            "pinecone_style_results": len(basic_results),
            "rudradb_enhanced_results": len(enhanced_results),
            "additional_discoveries": len([r for r in enhanced_results if r.hop_count > 0]),
            "results_preview": []
        }
        
        print(f"   📊 Pinecone-style search: {len(basic_results)} results")
        print(f"   🧠 RudraDB-Opin enhanced: {len(enhanced_results)} results")
        print(f"   ✨ Additional discoveries: {len([r for r in enhanced_results if r.hop_count > 0])} through relationships")
        
        # Show preview of enhanced results
        for result in enhanced_results[:3]:
            vector = self.rudra_db.get_vector(result.vector_id)
            connection = "Direct similarity" if result.hop_count == 0 else f"{result.hop_count}-hop relationship"
            
            result_info = {
                "vector_id": result.vector_id,
                "connection_type": connection,
                "combined_score": result.combined_score,
                "metadata_preview": {k: v for k, v in vector['metadata'].items() if k in ['category', 'tags', 'source', 'title']}
            }
            
            comparison_result["results_preview"].append(result_info)
            print(f"      📄 {result.vector_id}: {connection} (score: {result.combined_score:.3f})")
        
        return comparison_result


def demo_pinecone_migration():
    """Demo complete Pinecone to RudraDB-Opin migration"""
    
    print("🌊 Pinecone → RudraDB-Opin Migration Demo")
    print("=" * 50)
    
    # Initialize migration tool
    migration_tool = Pinecone_to_RudraDB_Migration(pinecone_dimension=384)
    
    # Create sample Pinecone data
    pinecone_vectors = migration_tool.create_sample_pinecone_data()
    
    print(f"\n🔄 Starting migration process...")
    
    # Perform migration
    migration_results = migration_tool.migrate_pinecone_data(pinecone_vectors)
    
    print(f"\n✅ Migration Complete!")
    print(f"   📊 Vectors migrated: {migration_results['vectors_migrated']}")
    print(f"   🎯 Auto-detected dimension: {migration_results['dimension_detected']}D")
    print(f"   🧠 Auto-relationships created: {migration_results['relationships_auto_created']}")
    print(f"   ⏱️ Migration duration: {migration_results['migration_duration']:.2f}s")
    if migration_results.get('migration_errors'):
        print(f"   ⚠️ Migration errors: {len(migration_results['migration_errors'])}")
    
    # Compare capabilities
    print(f"\n📈 Capability Comparison:")
    comparison = migration_tool.compare_capabilities()
    
    print("   🆚 Feature Comparison:")
    for feature, implementations in comparison["feature_comparison"].items():
        print(f"      {feature}:")
        print(f"         Pinecone: {implementations['Pinecone']}")
        print(f"         RudraDB-Opin: {implementations['RudraDB-Opin']}")
    
    print(f"\n   📊 Migration Results:")
    migration_res = comparison["migration_results"]
    print(f"      Success rate: {migration_res['migration_success_rate']:.1f}%")
    print(f"      Auto-created relationships: {migration_res['relationships_auto_created']}")
    print(f"      Remaining capacity: {migration_res['capacity_remaining']['vectors']} vectors, {migration_res['capacity_remaining']['relationships']} relationships")
    
    # Demonstrate enhanced search capabilities
    print(f"\n🔍 Search Enhancement Demo:")
    query_vector = np.random.rand(migration_results['dimension_detected']).astype(np.float32)
    search_demo = migration_tool.demonstrate_enhanced_search(
        query_vector, "AI and machine learning concepts"
    )
    
    # Show upgrade benefits
    print(f"\n🚀 Migration Benefits Summary:")
    benefits = [
        "✨ Gained relationship-aware search capabilities",
        "✨ Auto-dimension detection eliminated configuration",
        "✨ Auto-relationship detection created intelligent connections",
        f"✨ Enhanced search discovered {search_demo['additional_discoveries']} additional relevant results",
        "✨ 100% free forever vs Pinecone's usage-based pricing",
        "✨ Same API simplicity with advanced relationship intelligence",
        "✨ Perfect for learning, prototyping, and smaller-scale applications"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")
    
    # Show next steps
    print(f"\n💡 Next Steps:")
    print(f"   1. Explore relationship types: semantic, hierarchical, temporal, causal, associative")
    print(f"   2. Try multi-hop search with max_hops=2")
    print(f"   3. Build custom relationships for your specific use case")
    print(f"   4. When ready for production scale: upgrade to full RudraDB")
    
    print(f"\n🎉 Pinecone → RudraDB-Opin migration successful!")
    print("    Welcome to relationship-aware vector search! 🧠")


if __name__ == "__main__":
    demo_pinecone_migration()
