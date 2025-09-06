#!/usr/bin/env python3
"""
RudraDB-Opin: Search Patterns Demonstration

This example demonstrates different search patterns and strategies:
1. Basic similarity search
2. Relationship-enhanced search  
3. Discovery-focused search
4. Filtered search
5. Progressive search strategies
6. Multi-hop traversal patterns

Perfect for learning how to optimize search for different use cases!
"""

import rudradb
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import time


class SearchPatternsDemo:
    """Comprehensive demonstration of search patterns and strategies"""
    
    def __init__(self):
        self.db = rudradb.RudraDB()  # Auto-dimension detection
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("🔍 RudraDB-Opin Search Patterns Demo")
        print("=" * 50)
        self._setup_test_data()
        
    def _setup_test_data(self):
        """Set up comprehensive test dataset for search demonstrations"""
        print("\n📚 Setting up test dataset...")
        
        # AI/ML documents with rich metadata
        documents = [
            # Beginner content
            ("ai_intro", "Introduction to Artificial Intelligence concepts and applications",
             {"category": "AI", "level": "beginner", "tags": ["ai", "introduction"], "domain": "education"}),
            ("python_basics", "Python programming fundamentals for beginners",
             {"category": "Programming", "level": "beginner", "tags": ["python", "basics"], "domain": "education"}),
            ("data_intro", "Introduction to data science and analytics",
             {"category": "Data Science", "level": "beginner", "tags": ["data", "analytics"], "domain": "education"}),
            
            # Intermediate content
            ("ml_algorithms", "Machine Learning algorithms and their applications",
             {"category": "AI", "level": "intermediate", "tags": ["ml", "algorithms"], "domain": "technical"}),
            ("pandas_advanced", "Advanced Pandas techniques for data manipulation",
             {"category": "Programming", "level": "intermediate", "tags": ["pandas", "data"], "domain": "technical"}),
            ("visualization", "Data visualization techniques and best practices",
             {"category": "Data Science", "level": "intermediate", "tags": ["visualization", "charts"], "domain": "technical"}),
            
            # Advanced content
            ("deep_learning", "Deep Learning with neural networks and optimization",
             {"category": "AI", "level": "advanced", "tags": ["deep learning", "neural networks"], "domain": "research"}),
            ("distributed_computing", "Distributed computing for large-scale data processing",
             {"category": "Infrastructure", "level": "advanced", "tags": ["distributed", "scaling"], "domain": "research"}),
            ("model_optimization", "Advanced model optimization and hyperparameter tuning",
             {"category": "AI", "level": "advanced", "tags": ["optimization", "tuning"], "domain": "research"}),
            
            # Specialized content
            ("nlp_transformers", "Natural Language Processing with Transformer models",
             {"category": "NLP", "level": "advanced", "tags": ["nlp", "transformers"], "domain": "specialized"}),
            ("computer_vision", "Computer Vision techniques for image analysis",
             {"category": "CV", "level": "intermediate", "tags": ["cv", "images"], "domain": "specialized"}),
            ("recommendation_systems", "Building recommendation systems with collaborative filtering",
             {"category": "Systems", "level": "intermediate", "tags": ["recommendations", "filtering"], "domain": "specialized"}),
            
            # Problem-solution pairs
            ("overfitting_problem", "Understanding and identifying overfitting in ML models",
             {"category": "Problems", "level": "intermediate", "tags": ["overfitting", "problems"], "domain": "troubleshooting"}),
            ("regularization_solution", "Regularization techniques to prevent overfitting",
             {"category": "Solutions", "level": "intermediate", "tags": ["regularization", "solutions"], "domain": "troubleshooting"}),
            ("scaling_issues", "Performance and scaling issues in ML systems",
             {"category": "Problems", "level": "advanced", "tags": ["scaling", "performance"], "domain": "troubleshooting"}),
            ("optimization_techniques", "System optimization and performance tuning strategies",
             {"category": "Solutions", "level": "advanced", "tags": ["optimization", "performance"], "domain": "troubleshooting"})
        ]
        
        # Add all documents
        for doc_id, text, metadata in documents:
            embedding = self.model.encode([text])[0].astype(np.float32)
            self.db.add_vector(doc_id, embedding, metadata)
        
        # Build comprehensive relationships
        self._build_relationships()
        
        stats = self.db.get_statistics()
        print(f"   ✅ Setup complete: {stats['vector_count']} documents, {stats['relationship_count']} relationships")
        
    def _build_relationships(self):
        """Build relationships between documents"""
        relationships = [
            # Learning progression (temporal)
            ("ai_intro", "ml_algorithms", "temporal", 0.9),
            ("python_basics", "pandas_advanced", "temporal", 0.8),
            ("data_intro", "visualization", "temporal", 0.8),
            ("ml_algorithms", "deep_learning", "temporal", 0.9),
            
            # Category hierarchies (hierarchical)
            ("ai_intro", "ml_algorithms", "hierarchical", 0.8),
            ("ai_intro", "deep_learning", "hierarchical", 0.7),
            ("ml_algorithms", "nlp_transformers", "hierarchical", 0.7),
            ("ml_algorithms", "computer_vision", "hierarchical", 0.7),
            ("data_intro", "pandas_advanced", "hierarchical", 0.8),
            ("data_intro", "visualization", "hierarchical", 0.8),
            
            # Problem-solution pairs (causal)
            ("overfitting_problem", "regularization_solution", "causal", 0.95),
            ("scaling_issues", "optimization_techniques", "causal", 0.90),
            ("scaling_issues", "distributed_computing", "causal", 0.85),
            
            # Content similarity (semantic)
            ("deep_learning", "nlp_transformers", "semantic", 0.8),
            ("deep_learning", "computer_vision", "semantic", 0.8),
            ("pandas_advanced", "visualization", "semantic", 0.7),
            ("model_optimization", "optimization_techniques", "semantic", 0.8),
            
            # General associations (associative)
            ("python_basics", "data_intro", "associative", 0.6),
            ("recommendation_systems", "ml_algorithms", "associative", 0.7),
            ("distributed_computing", "scaling_issues", "associative", 0.6),
            ("visualization", "computer_vision", "associative", 0.5)
        ]
        
        for source, target, rel_type, strength in relationships:
            self.db.add_relationship(source, target, rel_type, strength)
    
    def demonstrate_basic_similarity_search(self):
        """Demonstrate pure similarity search without relationships"""
        print("\n🔍 1. BASIC SIMILARITY SEARCH")
        print("   Pure vector similarity without relationship enhancement")
        
        test_queries = [
            ("machine learning basics", "Finding ML fundamentals"),
            ("data visualization techniques", "Finding visualization content"),
            ("advanced AI concepts", "Finding advanced AI topics")
        ]
        
        for query_text, description in test_queries:
            print(f"\n   Query: '{query_text}' - {description}")
            query_embedding = self.model.encode([query_text])[0].astype(np.float32)
            
            # Pure similarity search
            params = rudradb.SearchParams(
                top_k=5,
                include_relationships=False,  # No relationship enhancement
                similarity_threshold=0.1
            )
            
            start_time = time.time()
            results = self.db.search(query_embedding, params)
            search_time = time.time() - start_time
            
            print(f"      Results: {len(results)} documents ({search_time*1000:.1f}ms)")
            
            for i, result in enumerate(results[:3], 1):
                vector = self.db.get_vector(result.vector_id)
                print(f"         {i}. {result.vector_id}")
                print(f"            Score: {result.similarity_score:.3f}")
                print(f"            Text: {vector['metadata']['text'][:50]}...")
    
    def demonstrate_relationship_enhanced_search(self):
        """Demonstrate relationship-enhanced search"""
        print("\n🔗 2. RELATIONSHIP-ENHANCED SEARCH")
        print("   Combining similarity + relationship intelligence")
        
        test_queries = [
            ("machine learning basics", "Finding ML + related content"),
            ("solving model problems", "Finding problems + solutions"),
            ("learning data science", "Finding learning progression")
        ]
        
        for query_text, description in test_queries:
            print(f"\n   Query: '{query_text}' - {description}")
            query_embedding = self.model.encode([query_text])[0].astype(np.float32)
            
            # Relationship-enhanced search
            params = rudradb.SearchParams(
                top_k=6,
                include_relationships=True,  # Enable relationship enhancement
                max_hops=2,
                relationship_weight=0.3
            )
            
            start_time = time.time()
            results = self.db.search(query_embedding, params)
            search_time = time.time() - start_time
            
            # Separate direct and relationship results
            direct_results = [r for r in results if r.hop_count == 0]
            relationship_results = [r for r in results if r.hop_count > 0]
            
            print(f"      Results: {len(results)} total ({search_time*1000:.1f}ms)")
            print(f"         Direct: {len(direct_results)}, Relationship-discovered: {len(relationship_results)}")
            
            # Show mix of direct and relationship results
            for i, result in enumerate(results[:4], 1):
                vector = self.db.get_vector(result.vector_id)
                connection = "Direct match" if result.hop_count == 0 else f"{result.hop_count}-hop connection"
                print(f"         {i}. {result.vector_id} ({connection})")
                print(f"            Combined Score: {result.combined_score:.3f}")
                print(f"            Text: {vector['metadata']['text'][:45]}...")
    
    def demonstrate_discovery_focused_search(self):
        """Demonstrate discovery-focused search for exploration"""
        print("\n🌟 3. DISCOVERY-FOCUSED SEARCH")
        print("   Emphasize relationship discovery and exploration")
        
        test_queries = [
            ("python programming", "Discovering Python ecosystem"),
            ("AI research topics", "Discovering AI research areas")
        ]
        
        for query_text, description in test_queries:
            print(f"\n   Query: '{query_text}' - {description}")
            query_embedding = self.model.encode([query_text])[0].astype(np.float32)
            
            # Discovery-focused parameters
            params = rudradb.SearchParams(
                top_k=8,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.6,  # High relationship influence
                similarity_threshold=0.1   # Lower similarity requirement
            )
            
            start_time = time.time()
            results = self.db.search(query_embedding, params)
            search_time = time.time() - start_time
            
            # Analyze discovery effectiveness
            discovered_through_relationships = [r for r in results if r.hop_count > 0]
            categories_found = set()
            
            for result in results:
                vector = self.db.get_vector(result.vector_id)
                categories_found.add(vector['metadata'].get('category', 'Unknown'))
            
            print(f"      Discovery Results: {len(results)} documents ({search_time*1000:.1f}ms)")
            print(f"         Relationship discoveries: {len(discovered_through_relationships)}")
            print(f"         Categories discovered: {', '.join(sorted(categories_found))}")
            
            # Show most interesting discoveries
            print(f"      Top discoveries:")
            for result in discovered_through_relationships[:3]:
                vector = self.db.get_vector(result.vector_id)
                print(f"         • {result.vector_id} ({result.hop_count}-hop)")
                print(f"           Category: {vector['metadata']['category']}")
                print(f"           Score: {result.combined_score:.3f}")
    
    def demonstrate_filtered_search(self):
        """Demonstrate filtered search with metadata and relationship types"""
        print("\n🎯 4. FILTERED SEARCH")
        print("   Filtering by metadata and relationship types")
        
        # Test different filtering strategies
        filter_strategies = [
            ("Learning Path Filter", ["temporal", "hierarchical"], "Finding learning progressions"),
            ("Problem-Solution Filter", ["causal"], "Finding solutions to problems"),
            ("Content Similarity Filter", ["semantic", "associative"], "Finding related content")
        ]
        
        query_text = "advanced machine learning techniques"
        query_embedding = self.model.encode([query_text])[0].astype(np.float32)
        
        print(f"\n   Base Query: '{query_text}'")
        
        for strategy_name, rel_types, description in filter_strategies:
            print(f"\n      {strategy_name}: {description}")
            
            params = rudradb.SearchParams(
                top_k=6,
                include_relationships=True,
                max_hops=2,
                relationship_types=rel_types,  # Filter relationship types
                relationship_weight=0.4
            )
            
            results = self.db.search(query_embedding, params)
            relationship_results = [r for r in results if r.hop_count > 0]
            
            print(f"         Results: {len(results)} documents")
            print(f"         Relationship types used: {', '.join(rel_types)}")
            print(f"         Filtered discoveries: {len(relationship_results)}")
            
            # Show filtered results
            for result in relationship_results[:2]:
                vector = self.db.get_vector(result.vector_id)
                print(f"           • {result.vector_id} ({result.hop_count}-hop)")
                print(f"             Level: {vector['metadata']['level']}")
    
    def demonstrate_progressive_search(self):
        """Demonstrate progressive search strategy"""
        print("\n📈 5. PROGRESSIVE SEARCH STRATEGY")
        print("   Progressively broaden search until target results found")
        
        query_text = "neural network optimization"
        target_results = 5
        
        print(f"\n   Query: '{query_text}' (target: {target_results} results)")
        query_embedding = self.model.encode([query_text])[0].astype(np.float32)
        
        # Progressive search strategies
        strategies = [
            ("Strict Similarity", {
                "top_k": target_results,
                "include_relationships": False,
                "similarity_threshold": 0.5
            }),
            ("Add Relationships", {
                "top_k": target_results,
                "include_relationships": True,
                "max_hops": 1,
                "similarity_threshold": 0.3,
                "relationship_weight": 0.2
            }),
            ("Broader Discovery", {
                "top_k": target_results,
                "include_relationships": True,
                "max_hops": 2,
                "similarity_threshold": 0.1,
                "relationship_weight": 0.4
            }),
            ("Maximum Discovery", {
                "top_k": target_results * 2,
                "include_relationships": True,
                "max_hops": 2,
                "similarity_threshold": 0.05,
                "relationship_weight": 0.6
            })
        ]
        
        for strategy_name, strategy_params in strategies:
            params = rudradb.SearchParams(**strategy_params)
            results = self.db.search(query_embedding, params)
            
            print(f"\n      {strategy_name}:")
            print(f"         Found: {len(results)} results")
            
            if len(results) >= target_results:
                relationship_results = len([r for r in results if r.hop_count > 0])
                print(f"         ✅ Target reached! ({relationship_results} via relationships)")
                
                # Show result mix
                categories = {}
                for result in results:
                    vector = self.db.get_vector(result.vector_id)
                    category = vector['metadata']['category']
                    categories[category] = categories.get(category, 0) + 1
                
                print(f"         Categories: {dict(categories)}")
                break
            else:
                print(f"         ❌ Target not reached, trying broader strategy...")
    
    def demonstrate_multi_hop_traversal(self):
        """Demonstrate multi-hop relationship traversal patterns"""
        print("\n🕸️ 6. MULTI-HOP TRAVERSAL PATTERNS")
        print("   Exploring relationship chains and connection paths")
        
        # Start from a specific document and explore connections
        start_documents = [
            ("ai_intro", "Starting from AI introduction"),
            ("overfitting_problem", "Starting from a specific problem")
        ]
        
        for start_doc, description in start_documents:
            print(f"\n   {description}: '{start_doc}'")
            
            # Test different hop depths
            for max_hops in [1, 2]:
                connected = self.db.get_connected_vectors(start_doc, max_hops=max_hops)
                
                print(f"\n      Max hops: {max_hops}")
                print(f"         Connected documents: {len(connected)}")
                
                # Group by hop count
                by_hops = {}
                for vector_data, hop_count in connected:
                    if hop_count not in by_hops:
                        by_hops[hop_count] = []
                    by_hops[hop_count].append(vector_data)
                
                for hop_count in sorted(by_hops.keys()):
                    docs = by_hops[hop_count]
                    print(f"           {hop_count}-hop: {len(docs)} documents")
                    
                    # Show examples
                    for doc in docs[:2]:
                        category = doc['metadata']['category']
                        level = doc['metadata']['level']
                        print(f"             • {doc['id']}: {category} ({level})")
    
    def demonstrate_search_optimization(self):
        """Demonstrate search optimization techniques"""
        print("\n⚡ 7. SEARCH OPTIMIZATION TECHNIQUES")
        print("   Performance tips and optimization strategies")
        
        query_text = "machine learning applications"
        query_embedding = self.model.encode([query_text])[0].astype(np.float32)
        
        # Test different optimization approaches
        optimizations = [
            ("Unoptimized", {
                "top_k": 10,
                "include_relationships": True,
                "max_hops": 2,
                "similarity_threshold": 0.0
            }),
            ("Threshold Filter", {
                "top_k": 10,
                "include_relationships": True,
                "max_hops": 2,
                "similarity_threshold": 0.3  # Filter low similarity
            }),
            ("Reduced Hops", {
                "top_k": 10,
                "include_relationships": True,
                "max_hops": 1,  # Limit traversal depth
                "similarity_threshold": 0.3
            }),
            ("Focused Results", {
                "top_k": 5,  # Fewer results
                "include_relationships": True,
                "max_hops": 1,
                "similarity_threshold": 0.4
            })
        ]
        
        print(f"\n   Query: '{query_text}'")
        
        for opt_name, opt_params in optimizations:
            params = rudradb.SearchParams(**opt_params)
            
            start_time = time.time()
            results = self.db.search(query_embedding, params)
            search_time = time.time() - start_time
            
            print(f"\n      {opt_name}:")
            print(f"         Time: {search_time*1000:.2f}ms")
            print(f"         Results: {len(results)}")
            print(f"         Relationships used: {len([r for r in results if r.hop_count > 0])}")
            
            if search_time < 0.01:  # Very fast
                print(f"         ✅ Excellent performance")
            elif search_time < 0.05:  # Good
                print(f"         ⚡ Good performance")
            else:
                print(f"         ⚠️ Consider optimization")
    
    def show_search_patterns_summary(self):
        """Show summary of all search patterns demonstrated"""
        stats = self.db.get_statistics()
        
        print(f"\n🔍 Search Patterns Demo Summary")
        print(f"=" * 45)
        print(f"📊 Database: {stats['vector_count']} documents, {stats['relationship_count']} relationships")
        print(f"🎯 Dimension: {stats['dimension']}")
        
        print(f"\n✅ Search Patterns Covered:")
        patterns = [
            "🔍 Basic Similarity Search - Pure vector similarity",
            "🔗 Relationship-Enhanced Search - Similarity + relationships",
            "🌟 Discovery-Focused Search - Exploration and discovery",
            "🎯 Filtered Search - Metadata and relationship filtering",
            "📈 Progressive Search - Adaptive search strategies",
            "🕸️ Multi-Hop Traversal - Relationship chain exploration",
            "⚡ Search Optimization - Performance optimization techniques"
        ]
        
        for pattern in patterns:
            print(f"   {pattern}")
        
        print(f"\n💡 Key Takeaways:")
        takeaways = [
            "Choose search strategy based on your use case",
            "Relationship search discovers content similarity misses", 
            "Use progressive search for adaptive applications",
            "Filter relationships for focused discovery",
            "Optimize with thresholds and hop limits",
            "Multi-hop traversal reveals indirect connections"
        ]
        
        for takeaway in takeaways:
            print(f"   • {takeaway}")


def main():
    """Run the complete search patterns demonstration"""
    demo = SearchPatternsDemo()
    
    try:
        # Demonstrate all search patterns
        demo.demonstrate_basic_similarity_search()
        demo.demonstrate_relationship_enhanced_search()
        demo.demonstrate_discovery_focused_search()
        demo.demonstrate_filtered_search()
        demo.demonstrate_progressive_search()
        demo.demonstrate_multi_hop_traversal()
        demo.demonstrate_search_optimization()
        
        # Show summary
        demo.show_search_patterns_summary()
        
        print(f"\n🎉 Search Patterns Demo Complete!")
        print(f"   You've mastered relationship-aware search strategies")
        print(f"   Ready to optimize search for any application!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print(f"💡 Make sure you have sentence-transformers installed:")
        print(f"   pip install sentence-transformers")


if __name__ == "__main__":
    main()
