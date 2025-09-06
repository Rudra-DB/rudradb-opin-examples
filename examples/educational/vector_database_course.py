#!/usr/bin/env python3
"""
Educational: Vector Database Course with RudraDB-Opin
====================================================

Perfect for teaching vector databases and AI concepts with auto-dimension detection
and relationship-aware search. Designed for educators and students.
"""

import rudradb
import numpy as np
from typing import List, Dict, Any
import json
import time

class VectorDatabase_Course:
    """Complete course for teaching vector databases with RudraDB-Opin"""
    
    def __init__(self):
        self.db = rudradb.RudraDB()  # Auto-detects any model students use
        self.lesson_count = 0
        self.student_progress = {}
        
        print("🎓 Vector Database Course with RudraDB-Opin")
        print("=" * 50)
        print("   🤖 Auto-dimension detection: Students can use any embedding model")
        print("   🧠 Auto-relationship detection: Focus on concepts, not configuration")
        print("   📚 Perfect 100-vector limit: Complete course fits in memory")
        
    def lesson_1_basics(self):
        """Lesson 1: Vector basics with auto-dimension detection"""
        print("\n🎓 Lesson 1: Understanding Vector Databases")
        print("=" * 45)
        
        print("📖 Theory:")
        print("   • Vectors represent data as numerical arrays")
        print("   • Similar vectors are close in vector space")
        print("   • Vector databases enable semantic search")
        
        print("\n💻 Hands-on Practice:")
        
        # Demonstrate with simple mock embeddings
        print("   Step 1: Creating our first vectors")
        
        # Simple text concepts and their mock embeddings
        concepts = {
            "ai": [0.8, 0.2, 0.9, 0.1],      # High on 'intelligence', low on 'manual'
            "manual": [0.1, 0.9, 0.2, 0.8],  # High on 'manual', low on 'intelligence'
            "learning": [0.7, 0.3, 0.8, 0.3], # Similar to AI
            "robot": [0.6, 0.4, 0.7, 0.4]    # Between AI and manual
        }
        
        for concept, embedding in concepts.items():
            embedding_array = np.array(embedding, dtype=np.float32)
            self.db.add_vector(concept, embedding_array, {
                "lesson": 1, 
                "topic": "basics", 
                "concept": concept,
                "description": f"Basic concept: {concept}"
            })
        
        print(f"   ✅ Added {len(concepts)} concept vectors")
        print(f"   🎯 Auto-detected embedding dimension: {self.db.dimension()}D")
        print("   💡 Students learn without worrying about configuration!")
        
        # Simple similarity demonstration
        print("\n   Step 2: Finding similar concepts")
        query = np.array([0.75, 0.25, 0.8, 0.2], dtype=np.float32)  # Similar to "ai"
        
        results = self.db.search(query, rudradb.SearchParams(top_k=3))
        
        print("   🔍 Search results (closest to our query):")
        for i, result in enumerate(results, 1):
            vector = self.db.get_vector(result.vector_id)
            print(f"      {i}. {result.vector_id} (similarity: {result.similarity_score:.3f})")
        
        print("\n📊 Key Learning:")
        print("   • Vectors close in space = similar meaning")
        print("   • Auto-dimension detection = no configuration needed")
        print("   • Vector databases enable semantic search")
        
        self.lesson_count += 1
        return {"lesson": 1, "vectors_created": len(concepts), "dimension": self.db.dimension()}
    
    def lesson_2_relationships(self):
        """Lesson 2: Beyond Similarity - Relationship Intelligence"""
        print("\n🎓 Lesson 2: Beyond Similarity - Relationship Intelligence")
        print("=" * 55)
        
        print("📖 Theory:")
        print("   • Traditional vector DBs only find similar items")
        print("   • RudraDB-Opin also models relationships between items")
        print("   • 5 relationship types: semantic, hierarchical, temporal, causal, associative")
        
        print("\n💻 Hands-on Practice:")
        
        # Add learning progression concepts
        learning_concepts = {
            "ai_intro": {
                "embedding": [0.8, 0.2, 0.9, 0.1, 0.7, 0.3],
                "difficulty": "beginner",
                "subject": "AI",
                "description": "Introduction to Artificial Intelligence"
            },
            "ml_basics": {
                "embedding": [0.7, 0.3, 0.8, 0.2, 0.8, 0.2],
                "difficulty": "intermediate", 
                "subject": "AI",
                "description": "Machine Learning Fundamentals"
            },
            "dl_advanced": {
                "embedding": [0.9, 0.1, 0.95, 0.05, 0.9, 0.1],
                "difficulty": "advanced",
                "subject": "AI", 
                "description": "Deep Learning Neural Networks"
            },
            "python_basics": {
                "embedding": [0.3, 0.7, 0.4, 0.6, 0.5, 0.5],
                "difficulty": "beginner",
                "subject": "Programming",
                "description": "Python Programming Fundamentals"
            },
            "python_ai": {
                "embedding": [0.6, 0.4, 0.7, 0.3, 0.8, 0.2],
                "difficulty": "intermediate",
                "subject": "Programming",
                "description": "Python for AI Development"
            }
        }
        
        print("   Step 1: Adding learning concepts")
        for concept_id, data in learning_concepts.items():
            embedding = np.array(data["embedding"], dtype=np.float32)
            metadata = {
                "lesson": 2,
                "difficulty": data["difficulty"],
                "subject": data["subject"],
                "description": data["description"],
                "type": "educational"
            }
            self.db.add_vector(concept_id, embedding, metadata)
        
        print(f"   ✅ Added {len(learning_concepts)} learning concepts")
        
        print("\n   Step 2: Creating educational relationships")
        
        # Hierarchical relationships (learning progression)
        hierarchical_rels = [
            ("ai_intro", "ml_basics", "AI introduction leads to ML basics"),
            ("ml_basics", "dl_advanced", "ML basics lead to deep learning")
        ]
        
        for source, target, description in hierarchical_rels:
            self.db.add_relationship(source, target, "hierarchical", 0.9, 
                                   {"learning_path": True, "description": description})
            print(f"      📊 Hierarchical: {source} → {target}")
        
        # Temporal relationships (learning sequence)
        temporal_rels = [
            ("python_basics", "python_ai", "Python basics before AI programming")
        ]
        
        for source, target, description in temporal_rels:
            self.db.add_relationship(source, target, "temporal", 0.8,
                                   {"sequence": True, "description": description})
            print(f"      ⏰ Temporal: {source} → {target}")
        
        # Causal relationships (prerequisite)
        causal_rels = [
            ("python_basics", "ml_basics", "Python knowledge enables ML learning")
        ]
        
        for source, target, description in causal_rels:
            self.db.add_relationship(source, target, "causal", 0.85,
                                   {"prerequisite": True, "description": description})
            print(f"      🎯 Causal: {source} → {target}")
        
        print(f"\n   🧠 Created learning progression relationships automatically")
        print(f"   📊 Database: {self.db.vector_count()} concepts, {self.db.relationship_count()} connections")
        
        self.lesson_count += 1
        return {"lesson": 2, "relationships_created": self.db.relationship_count()}
    
    def lesson_3_search_patterns(self):
        """Lesson 3: Advanced Search Patterns"""
        print("\n🎓 Lesson 3: Advanced Search Patterns")
        print("=" * 40)
        
        print("📖 Theory:")
        print("   • Basic search: Find similar vectors only")
        print("   • Relationship-aware search: Find similar + connected")
        print("   • Multi-hop discovery: Find indirect connections")
        
        print("\n💻 Hands-on Practice:")
        
        # Create query for comparison
        query_text = "learning artificial intelligence"
        # Mock embedding for this query
        query_embedding = np.array([0.75, 0.25, 0.85, 0.15, 0.8, 0.2], dtype=np.float32)
        
        print(f"   Query: '{query_text}'")
        
        # Search Pattern 1: Traditional similarity only
        print("\n   Pattern 1: Traditional similarity search")
        basic_results = self.db.search(query_embedding, rudradb.SearchParams(
            top_k=5, 
            include_relationships=False
        ))
        
        print(f"      Results: {len(basic_results)} similar items found")
        for result in basic_results:
            vector = self.db.get_vector(result.vector_id)
            print(f"         • {result.vector_id}: {vector['metadata'].get('description', 'N/A')}")
        
        # Search Pattern 2: Relationship-aware search
        print("\n   Pattern 2: Relationship-aware search")
        enhanced_results = self.db.search(query_embedding, rudradb.SearchParams(
            top_k=8,
            include_relationships=True,
            max_hops=2,
            relationship_weight=0.4
        ))
        
        print(f"      Results: {len(enhanced_results)} items found (including relationships)")
        direct_count = sum(1 for r in enhanced_results if r.hop_count == 0)
        relationship_count = sum(1 for r in enhanced_results if r.hop_count > 0)
        
        print(f"         Direct matches: {direct_count}")
        print(f"         Through relationships: {relationship_count}")
        
        for result in enhanced_results:
            vector = self.db.get_vector(result.vector_id)
            connection = "Direct" if result.hop_count == 0 else f"{result.hop_count}-hop"
            print(f"         • {result.vector_id} ({connection}): {vector['metadata'].get('description', 'N/A')}")
        
        # Search Pattern 3: Discovery-focused search
        print("\n   Pattern 3: Discovery-focused search")
        discovery_results = self.db.search(query_embedding, rudradb.SearchParams(
            top_k=10,
            include_relationships=True,
            max_hops=2,
            relationship_weight=0.6,  # Higher relationship influence
            similarity_threshold=0.1  # Lower similarity requirement
        ))
        
        discovered = sum(1 for r in discovery_results if r.hop_count > 0)
        print(f"      Discovery results: {discovered} additional connections found")
        
        print(f"\n📊 Search Pattern Comparison:")
        print(f"   Traditional search: {len(basic_results)} results")
        print(f"   Relationship-aware: {len(enhanced_results)} results")
        print(f"   Discovery-focused: {len(discovery_results)} results")
        print(f"   🎯 Additional discoveries: {len(enhanced_results) - len(basic_results)} through relationships")
        
        self.lesson_count += 1
        return {
            "lesson": 3,
            "basic_results": len(basic_results),
            "enhanced_results": len(enhanced_results),
            "additional_discoveries": len(enhanced_results) - len(basic_results)
        }
    
    def demonstrate_power(self, query="machine learning concepts"):
        """Show the power of relationship-aware search to students"""
        print("\n🔍 Final Demonstration: The Power of Relationship-Aware Search")
        print("=" * 65)
        
        # Mock query embedding
        query_emb = np.array([0.7, 0.3, 0.8, 0.2, 0.75, 0.25], dtype=np.float32)
        
        # Traditional search
        basic_results = self.db.search(query_emb, rudradb.SearchParams(
            include_relationships=False,
            top_k=5
        ))
        
        # Relationship-aware search  
        enhanced_results = self.db.search(query_emb, rudradb.SearchParams(
            include_relationships=True, 
            max_hops=2, 
            relationship_weight=0.4,
            top_k=8
        ))
        
        print(f"🔍 Search Comparison for: '{query}'")
        print(f"   Traditional search: {len(basic_results)} results")
        print(f"   Relationship-aware: {len(enhanced_results)} results")
        print(f"   🎯 Additional discoveries: {len(enhanced_results) - len(basic_results)} through relationships")
        
        # Show learning path discovery
        learning_paths = []
        for result in enhanced_results:
            if result.hop_count > 0:
                vector = self.db.get_vector(result.vector_id)
                learning_paths.append({
                    "concept": result.vector_id,
                    "hops": result.hop_count,
                    "description": vector['metadata'].get('description', 'N/A'),
                    "difficulty": vector['metadata'].get('difficulty', 'unknown')
                })
        
        if learning_paths:
            print(f"\n📚 Learning Path Discoveries:")
            for path in learning_paths:
                print(f"   • {path['concept']} ({path['difficulty']}) - {path['hops']} connection hops")
                print(f"     └── {path['description']}")
        
        return {
            "traditional": len(basic_results),
            "relationship_aware": len(enhanced_results),
            "improvement": len(enhanced_results) - len(basic_results),
            "learning_paths": len(learning_paths)
        }
    
    def course_summary(self):
        """Provide course completion summary"""
        print("\n🎉 Course Completion Summary")
        print("=" * 35)
        
        stats = self.db.get_statistics()
        
        print(f"📊 What Students Learned:")
        print(f"   ✅ Lessons completed: {self.lesson_count}")
        print(f"   ✅ Concepts explored: {stats['vector_count']} vectors")
        print(f"   ✅ Relationships modeled: {stats['relationship_count']} connections")
        print(f"   ✅ Embedding dimension: {stats['dimension']}D (auto-detected)")
        
        # Analyze relationship types used
        relationship_types = set()
        for vec_id in self.db.list_vectors():
            relationships = self.db.get_relationships(vec_id)
            for rel in relationships:
                relationship_types.add(rel["relationship_type"])
        
        print(f"   ✅ Relationship types mastered: {len(relationship_types)}/5")
        for rel_type in sorted(relationship_types):
            print(f"      • {rel_type}")
        
        capacity = stats['capacity_usage']
        print(f"\n🎯 Capacity Utilization:")
        print(f"   Vectors: {capacity['vector_usage_percent']:.1f}% ({stats['vector_count']}/100)")
        print(f"   Relationships: {capacity['relationship_usage_percent']:.1f}% ({stats['relationship_count']}/500)")
        
        print(f"\n🚀 Next Steps:")
        if capacity['vector_usage_percent'] > 80:
            print(f"   🎓 Excellent! You've explored most of RudraDB-Opin's capacity")
            print(f"   🎯 Ready for production? Upgrade to full RudraDB for 100,000+ vectors")
            print(f"   📚 Try building real applications with relationship-aware search")
        else:
            print(f"   📚 Keep exploring! Try adding more concepts and relationships")
            print(f"   🧪 Experiment with different search patterns")
            print(f"   🔬 Test with your own data and embedding models")
        
        return {
            "lessons_completed": self.lesson_count,
            "concepts_learned": stats['vector_count'],
            "relationships_mastered": stats['relationship_count'],
            "relationship_types": len(relationship_types),
            "mastery_percentage": (len(relationship_types) / 5) * 100
        }

def main():
    """Run the complete vector database course"""
    print("🎓 Welcome to Vector Database Course with RudraDB-Opin!")
    print("Perfect for teaching AI concepts with zero configuration complexity")
    
    # Initialize course
    course = VectorDatabase_Course()
    
    # Run lessons
    lesson1_result = course.lesson_1_basics()
    lesson2_result = course.lesson_2_relationships()
    lesson3_result = course.lesson_3_search_patterns()
    
    # Final demonstration
    demo_results = course.demonstrate_power()
    print(f"\n🎉 Students see {demo_results['improvement']} more relevant results with relationship intelligence!")
    
    # Course summary
    summary = course.course_summary()
    
    print(f"\n💡 Educational Impact:")
    print(f"   🎓 Students learned {summary['concepts_learned']} vector concepts")
    print(f"   🧠 Students mastered {summary['relationship_types']}/5 relationship types ({summary['mastery_percentage']:.0f}% complete)")
    print(f"   🚀 Students ready to build relationship-aware AI applications")
    
    print(f"\n✨ Perfect for:")
    print(f"   • Computer Science courses")
    print(f"   • AI/ML bootcamps") 
    print(f"   • Self-paced learning")
    print(f"   • Research methodology")
    
    return summary

if __name__ == "__main__":
    main()
