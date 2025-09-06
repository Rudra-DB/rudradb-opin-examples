#!/usr/bin/env python3
"""
RudraDB-Opin: Relationship Types Demonstration

This example demonstrates all 5 relationship types supported by RudraDB-Opin:
1. Semantic - Content similarity, topical connections
2. Hierarchical - Parent-child structures, categorization
3. Temporal - Sequential content, time-based flow
4. Causal - Cause-effect, problem-solution pairs
5. Associative - General associations, loose connections

Perfect for learning how different relationship types work!
"""

import rudradb
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


class RelationshipTypesDemo:
    """Comprehensive demonstration of all 5 relationship types"""
    
    def __init__(self):
        self.db = rudradb.RudraDB()  # Auto-dimension detection
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("🧬 RudraDB-Opin Relationship Types Demo")
        print("=" * 50)
        
    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None) -> None:
        """Add document with embedding"""
        embedding = self.model.encode([text])[0].astype(np.float32)
        doc_metadata = {
            "text": text,
            "length": len(text),
            **(metadata or {})
        }
        self.db.add_vector(doc_id, embedding, doc_metadata)
    
    def demonstrate_semantic_relationships(self):
        """Demonstrate semantic relationships - content similarity"""
        print("\n1️⃣ SEMANTIC RELATIONSHIPS")
        print("   Use Case: Content similarity, topical connections")
        print("   Strength: How similar the content/meaning is")
        
        # Add related AI/ML documents
        docs = [
            ("ai_basics", "Artificial Intelligence fundamentals and core concepts", 
             {"category": "AI", "level": "beginner"}),
            ("ml_intro", "Machine Learning algorithms and data-driven approaches",
             {"category": "AI", "level": "intermediate"}),
            ("dl_neural", "Deep Learning with neural networks and backpropagation",
             {"category": "AI", "level": "advanced"}),
            ("nlp_overview", "Natural Language Processing for text understanding",
             {"category": "AI", "level": "intermediate"}),
            ("cv_vision", "Computer Vision for image and video analysis",
             {"category": "AI", "level": "intermediate"})
        ]
        
        for doc_id, text, metadata in docs:
            self.add_document(doc_id, text, metadata)
        
        # Build semantic relationships based on content similarity
        semantic_connections = [
            ("ai_basics", "ml_intro", 0.85, "AI contains ML concepts"),
            ("ml_intro", "dl_neural", 0.80, "DL is subset of ML"),
            ("ml_intro", "nlp_overview", 0.75, "NLP uses ML techniques"),
            ("ml_intro", "cv_vision", 0.75, "CV uses ML techniques"),
            ("nlp_overview", "cv_vision", 0.65, "Both are AI application domains")
        ]
        
        for source, target, strength, reason in semantic_connections:
            self.db.add_relationship(source, target, "semantic", strength, 
                                   {"reason": reason})
            print(f"   🔗 {source} ↔ {target} (strength: {strength:.2f}) - {reason}")
        
        # Test semantic search
        query_text = "machine learning and neural networks"
        self._test_relationship_search(query_text, "semantic", "Semantic")
    
    def demonstrate_hierarchical_relationships(self):
        """Demonstrate hierarchical relationships - parent-child structures"""
        print("\n2️⃣ HIERARCHICAL RELATIONSHIPS")
        print("   Use Case: Parent-child structures, categorization")
        print("   Direction: Parent → Child (broader → specific)")
        
        # Add hierarchical knowledge structure
        docs = [
            ("computer_science", "Computer Science - study of computation and systems", 
             {"level": 0, "type": "field"}),
            ("artificial_intelligence", "Artificial Intelligence - machines that mimic human intelligence",
             {"level": 1, "type": "subfield"}),
            ("machine_learning", "Machine Learning - algorithms that learn from data",
             {"level": 2, "type": "discipline"}),
            ("supervised_learning", "Supervised Learning - learning with labeled examples",
             {"level": 3, "type": "method"}),
            ("linear_regression", "Linear Regression - modeling linear relationships",
             {"level": 4, "type": "algorithm"}),
            ("neural_networks", "Neural Networks - interconnected processing nodes",
             {"level": 3, "type": "method"}),
            ("cnn_networks", "Convolutional Neural Networks - for image processing",
             {"level": 4, "type": "algorithm"})
        ]
        
        for doc_id, text, metadata in docs:
            self.add_document(doc_id, text, metadata)
        
        # Build hierarchical relationships (parent → child)
        hierarchical_connections = [
            ("computer_science", "artificial_intelligence", 0.95, "CS contains AI"),
            ("artificial_intelligence", "machine_learning", 0.90, "AI contains ML"),
            ("machine_learning", "supervised_learning", 0.85, "ML contains supervised learning"),
            ("machine_learning", "neural_networks", 0.85, "ML contains neural networks"),
            ("supervised_learning", "linear_regression", 0.80, "Supervised learning includes regression"),
            ("neural_networks", "cnn_networks", 0.80, "Neural networks include CNNs")
        ]
        
        for parent, child, strength, reason in hierarchical_connections:
            self.db.add_relationship(parent, child, "hierarchical", strength,
                                   {"reason": reason, "relationship": "parent_to_child"})
            print(f"   📊 {parent} → {child} (strength: {strength:.2f}) - {reason}")
        
        # Test hierarchical search
        query_text = "specific machine learning algorithms"
        self._test_relationship_search(query_text, "hierarchical", "Hierarchical")
    
    def demonstrate_temporal_relationships(self):
        """Demonstrate temporal relationships - sequential flow"""
        print("\n3️⃣ TEMPORAL RELATIONSHIPS")
        print("   Use Case: Sequential content, time-based flow")
        print("   Direction: Earlier → Later (prerequisite → follow-up)")
        
        # Add course sequence
        docs = [
            ("course_intro", "Course Introduction - Overview and prerequisites",
             {"week": 1, "type": "introduction"}),
            ("python_basics", "Python Programming Basics - Syntax and fundamentals",
             {"week": 2, "type": "foundation"}),
            ("data_structures", "Data Structures - Lists, dictionaries, and sets",
             {"week": 3, "type": "foundation"}),
            ("numpy_intro", "NumPy Introduction - Numerical computing with arrays",
             {"week": 4, "type": "tools"}),
            ("pandas_basics", "Pandas Basics - Data manipulation and analysis",
             {"week": 5, "type": "tools"}),
            ("visualization", "Data Visualization - Creating charts and graphs",
             {"week": 6, "type": "application"}),
            ("ml_project", "Machine Learning Project - Apply everything learned",
             {"week": 7, "type": "project"})
        ]
        
        for doc_id, text, metadata in docs:
            self.add_document(doc_id, text, metadata)
        
        # Build temporal relationships (prerequisite → next)
        temporal_connections = [
            ("course_intro", "python_basics", 0.95, "Introduction comes before basics"),
            ("python_basics", "data_structures", 0.90, "Basics before data structures"),
            ("data_structures", "numpy_intro", 0.85, "Data structures before NumPy"),
            ("numpy_intro", "pandas_basics", 0.85, "NumPy before Pandas"),
            ("pandas_basics", "visualization", 0.80, "Data manipulation before visualization"),
            ("visualization", "ml_project", 0.80, "Visualization before final project"),
            # Cross-dependencies
            ("python_basics", "numpy_intro", 0.75, "Python basics needed for NumPy"),
            ("data_structures", "pandas_basics", 0.70, "Data structures help with Pandas")
        ]
        
        for prereq, follow_up, strength, reason in temporal_connections:
            self.db.add_relationship(prereq, follow_up, "temporal", strength,
                                   {"reason": reason, "relationship": "prerequisite"})
            print(f"   ⏰ {prereq} → {follow_up} (strength: {strength:.2f}) - {reason}")
        
        # Test temporal search
        query_text = "learning path for data analysis"
        self._test_relationship_search(query_text, "temporal", "Temporal")
    
    def demonstrate_causal_relationships(self):
        """Demonstrate causal relationships - cause-effect pairs"""
        print("\n4️⃣ CAUSAL RELATIONSHIPS")
        print("   Use Case: Cause-effect, problem-solution pairs")
        print("   Direction: Problem → Solution (cause → effect)")
        
        # Add problem-solution pairs
        docs = [
            ("model_overfitting", "Model Overfitting - Poor generalization to new data",
             {"type": "problem", "domain": "machine_learning"}),
            ("regularization", "Regularization Techniques - Prevent overfitting with penalties",
             {"type": "solution", "domain": "machine_learning"}),
            ("slow_training", "Slow Training Speed - Model takes too long to converge",
             {"type": "problem", "domain": "optimization"}),
            ("batch_optimization", "Batch Size Optimization - Improve training efficiency",
             {"type": "solution", "domain": "optimization"}),
            ("data_leakage", "Data Leakage - Future information in training data",
             {"type": "problem", "domain": "data_quality"}),
            ("proper_splitting", "Proper Data Splitting - Clean train-test separation",
             {"type": "solution", "domain": "data_quality"}),
            ("vanishing_gradients", "Vanishing Gradients - Gradients become too small",
             {"type": "problem", "domain": "deep_learning"}),
            ("residual_connections", "Residual Connections - Skip connections in networks",
             {"type": "solution", "domain": "deep_learning"})
        ]
        
        for doc_id, text, metadata in docs:
            self.add_document(doc_id, text, metadata)
        
        # Build causal relationships (problem → solution)
        causal_connections = [
            ("model_overfitting", "regularization", 0.90, "Regularization solves overfitting"),
            ("slow_training", "batch_optimization", 0.85, "Batch optimization speeds training"),
            ("data_leakage", "proper_splitting", 0.95, "Proper splitting prevents leakage"),
            ("vanishing_gradients", "residual_connections", 0.88, "ResNet solves vanishing gradients")
        ]
        
        for problem, solution, strength, reason in causal_connections:
            self.db.add_relationship(problem, solution, "causal", strength,
                                   {"reason": reason, "relationship": "problem_solution"})
            print(f"   🎯 {problem} → {solution} (strength: {strength:.2f}) - {reason}")
        
        # Test causal search
        query_text = "machine learning problems and solutions"
        self._test_relationship_search(query_text, "causal", "Causal")
    
    def demonstrate_associative_relationships(self):
        """Demonstrate associative relationships - general associations"""
        print("\n5️⃣ ASSOCIATIVE RELATIONSHIPS")
        print("   Use Case: General associations, loose connections")
        print("   Nature: Bidirectional, general relatedness")
        
        # Add associated concepts
        docs = [
            ("python_programming", "Python Programming - Popular language for data science",
             {"category": "programming", "domain": "general"}),
            ("data_science", "Data Science - Extracting insights from data",
             {"category": "field", "domain": "analytics"}),
            ("statistics", "Statistics - Mathematical analysis of data patterns",
             {"category": "mathematics", "domain": "analytics"}),
            ("visualization_tools", "Data Visualization Tools - Charts, graphs, and plots",
             {"category": "tools", "domain": "presentation"}),
            ("jupyter_notebooks", "Jupyter Notebooks - Interactive computing environment",
             {"category": "tools", "domain": "development"}),
            ("big_data", "Big Data Processing - Handling large-scale datasets",
             {"category": "infrastructure", "domain": "scalability"}),
            ("cloud_computing", "Cloud Computing - Scalable computing resources",
             {"category": "infrastructure", "domain": "deployment"}),
            ("business_intelligence", "Business Intelligence - Data-driven decision making",
             {"category": "application", "domain": "business"})
        ]
        
        for doc_id, text, metadata in docs:
            self.add_document(doc_id, text, metadata)
        
        # Build associative relationships (general connections)
        associative_connections = [
            ("python_programming", "data_science", 0.80, "Python popular for data science"),
            ("data_science", "statistics", 0.75, "Data science uses statistics"),
            ("data_science", "visualization_tools", 0.70, "Visualization important for data science"),
            ("python_programming", "jupyter_notebooks", 0.85, "Jupyter popular with Python"),
            ("big_data", "cloud_computing", 0.75, "Cloud good for big data"),
            ("data_science", "business_intelligence", 0.65, "Data science enables BI"),
            ("visualization_tools", "business_intelligence", 0.60, "BI uses visualization"),
            ("statistics", "business_intelligence", 0.55, "BI benefits from statistics")
        ]
        
        for concept1, concept2, strength, reason in associative_connections:
            self.db.add_relationship(concept1, concept2, "associative", strength,
                                   {"reason": reason, "relationship": "general_association"})
            print(f"   🏷️ {concept1} ↔ {concept2} (strength: {strength:.2f}) - {reason}")
        
        # Test associative search
        query_text = "tools and technologies for data analysis"
        self._test_relationship_search(query_text, "associative", "Associative")
    
    def _test_relationship_search(self, query_text: str, rel_type: str, rel_name: str):
        """Test search with specific relationship type"""
        query_embedding = self.model.encode([query_text])[0].astype(np.float32)
        
        # Search using only this relationship type
        params = rudradb.SearchParams(
            top_k=5,
            include_relationships=True,
            max_hops=2,
            relationship_types=[rel_type],
            relationship_weight=0.5
        )
        
        results = self.db.search(query_embedding, params)
        
        print(f"\n   🔍 {rel_name} Search Test: '{query_text}'")
        print(f"   Found {len(results)} results:")
        
        for i, result in enumerate(results[:3], 1):
            vector = self.db.get_vector(result.vector_id)
            connection = "Direct" if result.hop_count == 0 else f"{result.hop_count}-hop {rel_type}"
            print(f"      {i}. {result.vector_id}: {connection} (score: {result.combined_score:.3f})")
            print(f"         Text: {vector['metadata']['text'][:60]}...")
    
    def demonstrate_mixed_relationships(self):
        """Demonstrate using multiple relationship types together"""
        print("\n6️⃣ MIXED RELATIONSHIP SEARCH")
        print("   Combining multiple relationship types for rich discovery")
        
        query_text = "comprehensive learning path for AI development"
        query_embedding = self.model.encode([query_text])[0].astype(np.float32)
        
        # Search strategies with different relationship combinations
        strategies = [
            ("Hierarchical + Temporal", ["hierarchical", "temporal"], "Learning progression"),
            ("Semantic + Associative", ["semantic", "associative"], "Content similarity"),
            ("All Types", ["semantic", "hierarchical", "temporal", "causal", "associative"], "Full relationship intelligence"),
            ("Problem-Solution Focus", ["causal", "hierarchical"], "Solution-oriented search")
        ]
        
        for strategy_name, rel_types, description in strategies:
            params = rudradb.SearchParams(
                top_k=8,
                include_relationships=True,
                max_hops=2,
                relationship_types=rel_types,
                relationship_weight=0.4
            )
            
            results = self.db.search(query_embedding, params)
            relationship_results = [r for r in results if r.hop_count > 0]
            
            print(f"\n   📊 {strategy_name}: {description}")
            print(f"      Total results: {len(results)}")
            print(f"      Relationship-discovered: {len(relationship_results)}")
            print(f"      Top discoveries:")
            
            for result in relationship_results[:2]:
                vector = self.db.get_vector(result.vector_id)
                print(f"         • {result.vector_id} ({result.hop_count}-hop)")
                print(f"           {vector['metadata']['text'][:50]}...")
    
    def show_database_summary(self):
        """Show final database statistics"""
        stats = self.db.get_statistics()
        
        print(f"\n🧬 RudraDB-Opin Database Summary")
        print(f"=" * 40)
        print(f"📊 Vectors: {stats['vector_count']}/{rudradb.MAX_VECTORS}")
        print(f"🔗 Relationships: {stats['relationship_count']}/{rudradb.MAX_RELATIONSHIPS}")
        print(f"🎯 Dimension: {stats['dimension']}")
        
        # Count relationship types
        relationship_types = {"semantic": 0, "hierarchical": 0, "temporal": 0, "causal": 0, "associative": 0}
        
        for vec_id in self.db.list_vectors():
            relationships = self.db.get_relationships(vec_id)
            for rel in relationships:
                rel_type = rel["relationship_type"]
                if rel_type in relationship_types:
                    relationship_types[rel_type] += 1
        
        print(f"\n🔗 Relationships by Type:")
        for rel_type, count in relationship_types.items():
            emoji_map = {
                "semantic": "🔗",
                "hierarchical": "📊", 
                "temporal": "⏰",
                "causal": "🎯",
                "associative": "🏷️"
            }
            emoji = emoji_map.get(rel_type, "🔗")
            print(f"   {emoji} {rel_type.capitalize()}: {count}")
        
        print(f"\n✅ All 5 relationship types demonstrated!")
        print(f"   Perfect foundation for relationship-aware vector search")


def main():
    """Run the complete relationship types demonstration"""
    demo = RelationshipTypesDemo()
    
    try:
        # Demonstrate each relationship type
        demo.demonstrate_semantic_relationships()
        demo.demonstrate_hierarchical_relationships()
        demo.demonstrate_temporal_relationships()
        demo.demonstrate_causal_relationships()
        demo.demonstrate_associative_relationships()
        
        # Show mixed relationship search
        demo.demonstrate_mixed_relationships()
        
        # Final summary
        demo.show_database_summary()
        
        print(f"\n🎉 Relationship Types Demo Complete!")
        print(f"   You've explored all 5 relationship types in RudraDB-Opin")
        print(f"   Ready to build relationship-aware applications!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print(f"💡 Make sure you have sentence-transformers installed:")
        print(f"   pip install sentence-transformers")


if __name__ == "__main__":
    main()
