#!/usr/bin/env python3
"""
Educational RAG System with Auto-Learning Enhancement
Demonstrates RudraDB-Opin's educational capabilities
"""

import numpy as np
import rudradb
import time
from typing import List, Dict, Any

# Mock SentenceTransformer for demo purposes
class MockSentenceTransformer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.dimension = 384
        
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return [np.random.rand(self.dimension).astype(np.float32) for _ in texts]

class LearningRAG:
    """Educational RAG system with relationship-aware search"""
    
    def __init__(self):
        self.db = rudradb.RudraDB()  # Auto-detects embedding dimensions
        self.encoder = MockSentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
        
        print("🎓 Educational RAG System initialized")
        print("   🎯 Auto-dimension detection enabled")
        print("   🧠 Auto-relationship detection for learning paths")
    
    def add_learning_content(self, content_items):
        """Add educational content with relationships"""
        for item in content_items:
            embedding = self.encoder.encode(item['text'])[0]
            
            # Enhanced metadata for learning
            metadata = {
                "text": item['text'],
                "topic": item.get('topic', 'general'),
                "difficulty": item.get('difficulty', 'beginner'),
                "type": item.get('type', 'concept'),
                "learning_objectives": item.get('objectives', []),
                "prerequisites": item.get('prerequisites', []),
                "estimated_time": item.get('time_minutes', 30)
            }
            
            self.db.add_vector(item['id'], embedding, metadata)
        
        print(f"📚 Added {len(content_items)} learning items")
    
    def auto_build_learning_relationships(self):
        """Automatically build educational relationships"""
        print("🧠 Auto-building learning relationships...")
        
        relationships_created = 0
        vectors = self.db.list_vectors()
        
        for vec_id in vectors:
            vector = self.db.get_vector(vec_id)
            if not vector:
                continue
                
            metadata = vector['metadata']
            difficulty = metadata.get('difficulty', 'beginner')
            topic = metadata.get('topic', '')
            content_type = metadata.get('type', '')
            prerequisites = metadata.get('prerequisites', [])
            
            # Find relationships with other vectors
            for other_id in vectors:
                if other_id == vec_id or relationships_created >= 50:  # Stay within Opin limits
                    continue
                    
                other_vector = self.db.get_vector(other_id)
                if not other_vector:
                    continue
                    
                other_meta = other_vector['metadata']
                other_difficulty = other_meta.get('difficulty', 'beginner')
                other_topic = other_meta.get('topic', '')
                other_type = other_meta.get('type', '')
                
                # 1. Hierarchical: Same topic, different difficulty levels
                if (topic == other_topic and 
                    self._is_prerequisite(difficulty, other_difficulty)):
                    self.db.add_relationship(vec_id, other_id, "hierarchical", 0.9,
                        {"auto_detected": True, "reason": "learning_prerequisite"})
                    relationships_created += 1
                    print(f"   📊 Learning hierarchy: {vec_id} → {other_id} ({difficulty} → {other_difficulty})")
                
                # 2. Temporal: Sequential learning progression  
                elif (topic == other_topic and
                      content_type == "lesson" and other_type == "exercise"):
                    self.db.add_relationship(vec_id, other_id, "temporal", 0.8,
                        {"auto_detected": True, "reason": "lesson_to_practice"})
                    relationships_created += 1
                    print(f"   ⏰ Learning sequence: {vec_id} → {other_id} (theory → practice)")
                
                # 3. Causal: Problem-solution relationships
                elif content_type == "problem" and other_type == "solution":
                    self.db.add_relationship(vec_id, other_id, "causal", 0.95,
                        {"auto_detected": True, "reason": "problem_solution"})
                    relationships_created += 1
                    print(f"   🎯 Problem-solution: {vec_id} → {other_id}")
                
                # 4. Semantic: Related concepts in same domain
                elif (topic == other_topic and 
                      content_type == "concept" and other_type == "concept" and
                      difficulty == other_difficulty):
                    self.db.add_relationship(vec_id, other_id, "semantic", 0.7,
                        {"auto_detected": True, "reason": "related_concepts"})
                    relationships_created += 1
                    print(f"   🔗 Related concepts: {vec_id} ↔ {other_id}")
        
        print(f"✅ Created {relationships_created} learning relationships")
        return relationships_created
    
    def _is_prerequisite(self, current_level, other_level):
        """Check if current level is prerequisite to other level"""
        levels = {"beginner": 1, "intermediate": 2, "advanced": 3}
        current_num = levels.get(current_level, 1)
        other_num = levels.get(other_level, 1)
        return current_num < other_num and (other_num - current_num) <= 1
    
    def intelligent_learning_search(self, query, learning_style="comprehensive"):
        """Search with educational intelligence"""
        query_embedding = self.encoder.encode(query)[0]
        
        if learning_style == "comprehensive":
            # Include prerequisites and related concepts
            params = rudradb.SearchParams(
                top_k=10,
                include_relationships=True,
                max_hops=2,
                relationship_types=["hierarchical", "semantic"],
                relationship_weight=0.4
            )
        elif learning_style == "sequential":
            # Focus on learning sequence
            params = rudradb.SearchParams(
                top_k=8,
                include_relationships=True,
                max_hops=1,
                relationship_types=["temporal", "hierarchical"],
                relationship_weight=0.5
            )
        elif learning_style == "problem_solving":
            # Focus on problem-solution relationships
            params = rudradb.SearchParams(
                top_k=6,
                include_relationships=True,
                max_hops=2,
                relationship_types=["causal", "hierarchical"],
                relationship_weight=0.6
            )
        else:  # basic
            params = rudradb.SearchParams(
                top_k=5,
                include_relationships=False
            )
        
        results = self.db.search(query_embedding, params)
        
        # Organize results by learning path
        learning_path = []
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            if vector:
                metadata = vector['metadata']
                learning_path.append({
                    "id": result.vector_id,
                    "text": metadata.get('text', ''),
                    "topic": metadata.get('topic', ''),
                    "difficulty": metadata.get('difficulty', ''),
                    "type": metadata.get('type', ''),
                    "objectives": metadata.get('learning_objectives', []),
                    "time_minutes": metadata.get('estimated_time', 30),
                    "relevance_score": result.combined_score,
                    "connection": "Direct match" if result.hop_count == 0 else f"{result.hop_count}-hop relationship",
                    "hop_count": result.hop_count
                })
        
        # Sort by learning progression (prerequisites first)
        learning_path.sort(key=lambda x: (
            {"beginner": 1, "intermediate": 2, "advanced": 3}.get(x["difficulty"], 2),
            -x["relevance_score"]
        ))
        
        return learning_path
    
    def create_learning_path(self, topic, target_level="intermediate"):
        """Create a structured learning path for a topic"""
        print(f"📚 Creating learning path for: {topic} (target: {target_level})")
        
        # Find all content related to the topic
        query_results = self.intelligent_learning_search(topic, "comprehensive")
        
        # Filter and organize by difficulty progression
        levels = ["beginner", "intermediate", "advanced"]
        target_index = levels.index(target_level) if target_level in levels else 1
        
        learning_path = {
            "topic": topic,
            "target_level": target_level,
            "estimated_total_time": 0,
            "prerequisite_paths": [],
            "main_path": [],
            "practice_exercises": [],
            "assessment": []
        }
        
        for item in query_results:
            item_level = item["difficulty"]
            item_type = item["type"]
            
            # Only include items up to target level
            if levels.index(item_level) <= target_index:
                if item_type == "concept" or item_type == "lesson":
                    learning_path["main_path"].append(item)
                elif item_type == "exercise" or item_type == "practice":
                    learning_path["practice_exercises"].append(item)
                elif item_type == "assessment" or item_type == "quiz":
                    learning_path["assessment"].append(item)
                
                learning_path["estimated_total_time"] += item["time_minutes"]
        
        return learning_path
    
    def get_learning_analytics(self):
        """Get analytics about the learning database"""
        stats = self.db.get_statistics()
        
        # Analyze content by difficulty
        difficulty_distribution = {"beginner": 0, "intermediate": 0, "advanced": 0}
        topic_distribution = {}
        type_distribution = {}
        
        for vec_id in self.db.list_vectors():
            vector = self.db.get_vector(vec_id)
            if vector:
                metadata = vector['metadata']
                difficulty = metadata.get('difficulty', 'beginner')
                topic = metadata.get('topic', 'unknown')
                content_type = metadata.get('type', 'unknown')
                
                difficulty_distribution[difficulty] += 1
                topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
                type_distribution[content_type] = type_distribution.get(content_type, 0) + 1
        
        return {
            "total_content": stats['vector_count'],
            "total_relationships": stats['relationship_count'],
            "dimension": stats['dimension'],
            "difficulty_distribution": difficulty_distribution,
            "topic_distribution": topic_distribution,
            "type_distribution": type_distribution,
            "learning_paths_possible": sum(1 for r in self.db.get_relationships("") if r.get("metadata", {}).get("reason") == "learning_prerequisite"),
            "capacity_usage": stats['capacity_usage']
        }

def demo_educational_rag():
    """Demonstrate educational RAG capabilities"""
    print("🎓 Educational RAG Demo with Auto-Learning Enhancement")
    print("=" * 60)
    
    rag = LearningRAG()
    
    # Sample educational content
    learning_content = [
        {
            "id": "ml_intro",
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn from data",
            "topic": "machine_learning",
            "difficulty": "beginner",
            "type": "concept",
            "objectives": ["Understand ML definition", "Identify ML applications"],
            "time_minutes": 15
        },
        {
            "id": "supervised_learning",
            "text": "Supervised learning algorithms learn from labeled training data to make predictions",
            "topic": "machine_learning", 
            "difficulty": "intermediate",
            "type": "concept",
            "objectives": ["Understand supervised learning", "Identify supervised algorithms"],
            "prerequisites": ["ml_intro"],
            "time_minutes": 25
        },
        {
            "id": "linear_regression",
            "text": "Linear regression finds the best line through data points to predict continuous values",
            "topic": "machine_learning",
            "difficulty": "intermediate", 
            "type": "lesson",
            "objectives": ["Implement linear regression", "Interpret regression results"],
            "prerequisites": ["supervised_learning"],
            "time_minutes": 45
        },
        {
            "id": "regression_exercise",
            "text": "Practice exercise: Build a linear regression model to predict house prices",
            "topic": "machine_learning",
            "difficulty": "intermediate",
            "type": "exercise", 
            "objectives": ["Apply regression to real data", "Evaluate model performance"],
            "prerequisites": ["linear_regression"],
            "time_minutes": 60
        },
        {
            "id": "deep_learning",
            "text": "Deep learning uses neural networks with multiple layers to learn complex patterns",
            "topic": "machine_learning",
            "difficulty": "advanced",
            "type": "concept",
            "objectives": ["Understand neural networks", "Identify deep learning applications"],
            "prerequisites": ["supervised_learning"],
            "time_minutes": 40
        },
        {
            "id": "overfitting_problem",
            "text": "Overfitting occurs when a model memorizes training data but fails on new data",
            "topic": "machine_learning",
            "difficulty": "intermediate",
            "type": "problem",
            "objectives": ["Recognize overfitting", "Understand causes"],
            "time_minutes": 20
        },
        {
            "id": "regularization_solution",
            "text": "Regularization techniques like L1/L2 help prevent overfitting by penalizing complexity",
            "topic": "machine_learning", 
            "difficulty": "advanced",
            "type": "solution",
            "objectives": ["Apply regularization", "Choose regularization parameters"],
            "prerequisites": ["overfitting_problem"],
            "time_minutes": 35
        }
    ]
    
    # Add learning content
    rag.add_learning_content(learning_content)
    
    # Auto-build learning relationships
    relationships = rag.auto_build_learning_relationships()
    
    # Get learning analytics
    analytics = rag.get_learning_analytics()
    print(f"\n📊 Learning Database Analytics:")
    print(f"   Total content: {analytics['total_content']} items")
    print(f"   Auto-relationships: {analytics['total_relationships']}")
    print(f"   Embedding dimension: {analytics['dimension']}D")
    print(f"   Difficulty distribution: {analytics['difficulty_distribution']}")
    print(f"   Topic distribution: {analytics['topic_distribution']}")
    print(f"   Content types: {analytics['type_distribution']}")
    
    # Demonstrate different learning styles
    query = "learn linear regression for machine learning"
    
    print(f"\n🔍 Intelligent Learning Search Demonstrations:")
    print(f"Query: '{query}'\n")
    
    learning_styles = ["comprehensive", "sequential", "problem_solving", "basic"]
    
    for style in learning_styles:
        print(f"📚 {style.title()} Learning Style:")
        results = rag.intelligent_learning_search(query, style)
        
        for i, item in enumerate(results[:3], 1):
            connection_info = f" ({item['connection']})" if item['hop_count'] > 0 else ""
            print(f"   {i}. [{item['type'].upper()}] {item['text'][:60]}...")
            print(f"      Difficulty: {item['difficulty']} | Time: {item['time_minutes']}min{connection_info}")
        
        print()
    
    # Create a structured learning path
    print("🛤️ Creating Structured Learning Path:")
    learning_path = rag.create_learning_path("machine learning", "advanced")
    
    print(f"Topic: {learning_path['topic']} (Target: {learning_path['target_level']})")
    print(f"Estimated total time: {learning_path['estimated_total_time']} minutes")
    
    print("\nMain Learning Path:")
    for i, item in enumerate(learning_path['main_path'], 1):
        print(f"   {i}. [{item['difficulty'].upper()}] {item['text'][:50]}... ({item['time_minutes']}min)")
    
    print("\nPractice Exercises:")
    for i, item in enumerate(learning_path['practice_exercises'], 1):
        print(f"   {i}. {item['text'][:50]}... ({item['time_minutes']}min)")
    
    # Show capacity usage
    capacity = analytics['capacity_usage']
    print(f"\n📈 RudraDB-Opin Capacity Usage:")
    print(f"   Vectors: {capacity['vector_usage_percent']:.1f}% used")
    print(f"   Relationships: {capacity['relationship_usage_percent']:.1f}% used")
    print(f"   Perfect for educational prototyping!")
    
    print(f"\n🎉 Educational RAG demonstration complete!")
    print(f"   ✨ Auto-relationship detection created intelligent learning paths")
    print(f"   ✨ Multiple learning styles supported")
    print(f"   ✨ Structured progression from beginner to advanced")
    print(f"   ✨ Ready to scale with full RudraDB for complete curricula!")

if __name__ == "__main__":
    demo_educational_rag()
