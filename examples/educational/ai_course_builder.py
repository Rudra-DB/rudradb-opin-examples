#!/usr/bin/env python3
"""
🎓 AI Course Builder with RudraDB-Opin

This example demonstrates how to build an intelligent AI/ML course system using
relationship-aware vector search. It shows how to:

1. Create structured learning content with auto-dimension detection
2. Build intelligent learning progressions using relationship types
3. Generate personalized learning paths
4. Track student progress through relationship networks
5. Recommend next learning steps based on current knowledge

Perfect example of educational technology powered by relationship intelligence!
"""

import rudradb
import numpy as np
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

class AI_Course_Builder:
    """Intelligent AI/ML Course Builder using RudraDB-Opin"""
    
    def __init__(self, course_name: str = "AI/ML Fundamentals"):
        self.db = rudradb.RudraDB()  # Auto-dimension detection
        self.course_name = course_name
        self.student_progress = {}  # Track student progress
        
        print(f"🎓 Initializing AI Course Builder: {course_name}")
        print(f"   🤖 Auto-dimension detection enabled")
        
    def add_learning_content(self, content_items: List[Dict]):
        """Add course content with automatic relationship building"""
        
        print(f"\n📚 Adding {len(content_items)} learning items...")
        
        for item in content_items:
            # Simulate embedding (in real use, use actual embeddings from your model)
            embedding = np.random.rand(384).astype(np.float32)
            
            # Enhanced metadata for educational relationships
            enhanced_metadata = {
                "title": item["title"],
                "content": item["content"][:200],  # Preview
                "type": item["type"],  # lesson, exercise, quiz, project
                "difficulty": item["difficulty"],  # beginner, intermediate, advanced
                "topic": item["topic"],  # ai, ml, programming, math
                "prerequisites": item.get("prerequisites", []),
                "learning_objectives": item.get("learning_objectives", []),
                "estimated_time_hours": item.get("time_hours", 1),
                "tags": item.get("tags", []),
                "module": item.get("module", "general"),
                "order_in_module": item.get("order", 0),
                "interactive": item.get("interactive", False),
                "created_at": datetime.now().isoformat()
            }
            
            self.db.add_vector(item["id"], embedding, enhanced_metadata)
        
        print(f"   ✅ Added {len(content_items)} items")
        if self.db.dimension():
            print(f"   🎯 Auto-detected dimension: {self.db.dimension()}D")
    
    def build_learning_relationships(self):
        """Build intelligent learning relationships automatically"""
        
        print(f"\n🔗 Building intelligent learning relationships...")
        
        all_items = self.db.list_vectors()
        relationships_created = 0
        
        for item_id in all_items:
            item = self.db.get_vector(item_id)
            if not item:
                continue
                
            metadata = item['metadata']
            
            # Build relationships with other learning items
            for other_id in all_items:
                if other_id == item_id or relationships_created >= 200:  # Stay within Opin limits
                    continue
                    
                other_item = self.db.get_vector(other_id)
                if not other_item:
                    continue
                    
                other_metadata = other_item['metadata']
                
                # 1. Prerequisites -> Course content (hierarchical)
                if other_id in metadata.get("prerequisites", []):
                    self.db.add_relationship(other_id, item_id, "hierarchical", 0.95, {
                        "relationship_type": "prerequisite",
                        "learning_reason": "required_before"
                    })
                    relationships_created += 1
                    continue
                
                # 2. Sequential learning within same module (temporal)
                if (metadata["module"] == other_metadata["module"] and
                    metadata["type"] == other_metadata["type"] == "lesson"):
                    
                    order_diff = metadata["order_in_module"] - other_metadata["order_in_module"]
                    if order_diff == 1:  # Sequential lessons
                        self.db.add_relationship(other_id, item_id, "temporal", 0.9, {
                            "relationship_type": "sequential_learning",
                            "learning_reason": "next_in_sequence"
                        })
                        relationships_created += 1
                        continue
                
                # 3. Difficulty progression (temporal)
                if (metadata["topic"] == other_metadata["topic"] and
                    metadata["type"] == other_metadata["type"]):
                    
                    difficulty_order = {"beginner": 1, "intermediate": 2, "advanced": 3}
                    current_level = difficulty_order.get(metadata["difficulty"], 2)
                    other_level = difficulty_order.get(other_metadata["difficulty"], 2)
                    
                    if current_level - other_level == 1:  # Next difficulty level
                        self.db.add_relationship(other_id, item_id, "temporal", 0.8, {
                            "relationship_type": "difficulty_progression",
                            "learning_reason": "next_difficulty_level"
                        })
                        relationships_created += 1
                        continue
                
                # 4. Theory -> Practice connections (causal)
                if (metadata["type"] == "exercise" and other_metadata["type"] == "lesson" and
                    metadata["topic"] == other_metadata["topic"]):
                    
                    self.db.add_relationship(other_id, item_id, "causal", 0.85, {
                        "relationship_type": "theory_to_practice",
                        "learning_reason": "practice_opportunity"
                    })
                    relationships_created += 1
                    continue
                
                # 5. Topic clustering (semantic)
                if (metadata["topic"] == other_metadata["topic"] and
                    item_id != other_id):
                    
                    # Same topic, different types or modules
                    if (metadata["module"] != other_metadata["module"] or
                        metadata["type"] != other_metadata["type"]):
                        
                        self.db.add_relationship(item_id, other_id, "semantic", 0.6, {
                            "relationship_type": "topic_clustering",
                            "learning_reason": "related_topic_content"
                        })
                        relationships_created += 1
                        continue
                
                # 6. Cross-topic associations (associative)
                shared_tags = set(metadata.get("tags", [])) & set(other_metadata.get("tags", []))
                if len(shared_tags) >= 2:
                    strength = min(0.7, len(shared_tags) * 0.2)
                    self.db.add_relationship(item_id, other_id, "associative", strength, {
                        "relationship_type": "cross_topic_association",
                        "learning_reason": f"shared_concepts: {list(shared_tags)}",
                        "shared_tags": list(shared_tags)
                    })
                    relationships_created += 1
        
        print(f"   ✅ Created {relationships_created} educational relationships")
        print(f"   📊 Relationship types: prerequisite, sequential, difficulty progression, theory->practice, topic clustering")
        
        return relationships_created
    
    def generate_learning_path(self, student_id: str, current_topic: str, 
                              target_difficulty: str = "advanced", 
                              max_items: int = 10) -> List[Dict]:
        """Generate personalized learning path for a student"""
        
        print(f"\n🗺️ Generating learning path for student: {student_id}")
        print(f"   📍 Current topic: {current_topic}")
        print(f"   🎯 Target difficulty: {target_difficulty}")
        
        # Find starting point based on current topic
        starting_items = []
        for item_id in self.db.list_vectors():
            item = self.db.get_vector(item_id)
            if item and item['metadata']['topic'] == current_topic:
                starting_items.append(item_id)
        
        if not starting_items:
            print(f"   ❌ No content found for topic: {current_topic}")
            return []
        
        # Start from beginner level of the topic
        best_start = None
        for item_id in starting_items:
            item = self.db.get_vector(item_id)
            if (item and item['metadata']['difficulty'] == 'beginner' and 
                item['metadata']['type'] == 'lesson'):
                best_start = item_id
                break
        
        if not best_start:
            best_start = starting_items[0]  # Fallback to first item
        
        # Traverse learning relationships to build path
        learning_path = []
        visited = set()
        current_items = [best_start]
        
        difficulty_progression = {"beginner": 1, "intermediate": 2, "advanced": 3}
        target_level = difficulty_progression.get(target_difficulty, 3)
        
        while current_items and len(learning_path) < max_items:
            current_id = current_items.pop(0)
            
            if current_id in visited:
                continue
                
            visited.add(current_id)
            current_item = self.db.get_vector(current_id)
            
            if not current_item:
                continue
            
            metadata = current_item['metadata']
            current_level = difficulty_progression.get(metadata['difficulty'], 1)
            
            # Add to learning path
            learning_path.append({
                "item_id": current_id,
                "title": metadata['title'],
                "type": metadata['type'],
                "difficulty": metadata['difficulty'],
                "topic": metadata['topic'],
                "estimated_hours": metadata.get('estimated_time_hours', 1),
                "learning_objectives": metadata.get('learning_objectives', []),
                "step_number": len(learning_path) + 1
            })
            
            # If we've reached target difficulty, look for capstone projects
            if current_level >= target_level:
                # Look for projects or advanced exercises
                relationships = self.db.get_relationships(current_id)
                for rel in relationships:
                    if rel['relationship_type'] in ['causal', 'semantic']:
                        related_item = self.db.get_vector(rel['target_id'])
                        if (related_item and 
                            related_item['metadata']['type'] in ['project', 'exercise'] and
                            rel['target_id'] not in visited):
                            current_items.append(rel['target_id'])
                break
            
            # Find next items in learning progression
            relationships = self.db.get_relationships(current_id)
            
            # Priority order for learning relationships
            relationship_priority = {
                'temporal': 1,      # Next in sequence
                'hierarchical': 2,  # Prerequisites fulfilled
                'causal': 3,        # Theory to practice
                'semantic': 4,      # Related content
                'associative': 5    # Loose associations
            }
            
            # Sort relationships by learning priority
            sorted_relationships = sorted(relationships, 
                key=lambda r: relationship_priority.get(r['relationship_type'], 6))
            
            for rel in sorted_relationships[:3]:  # Top 3 most relevant next steps
                if rel['target_id'] not in visited:
                    current_items.append(rel['target_id'])
        
        # Calculate total estimated time
        total_hours = sum(item['estimated_hours'] for item in learning_path)
        
        print(f"   ✅ Generated learning path with {len(learning_path)} items")
        print(f"   ⏱️ Estimated completion time: {total_hours} hours")
        
        return learning_path
    
    def track_student_progress(self, student_id: str, completed_item_id: str, 
                              score: float = None, time_spent_hours: float = None):
        """Track student progress through learning path"""
        
        if student_id not in self.student_progress:
            self.student_progress[student_id] = {
                "completed_items": [],
                "scores": {},
                "time_spent": {},
                "current_topics": set(),
                "mastered_topics": set(),
                "started_at": datetime.now().isoformat()
            }
        
        progress = self.student_progress[student_id]
        
        if completed_item_id not in progress["completed_items"]:
            progress["completed_items"].append(completed_item_id)
            
            # Get item details
            item = self.db.get_vector(completed_item_id)
            if item:
                metadata = item['metadata']
                progress["current_topics"].add(metadata['topic'])
                
                # Check if topic is mastered (completed advanced content)
                if metadata['difficulty'] == 'advanced':
                    progress["mastered_topics"].add(metadata['topic'])
        
        if score is not None:
            progress["scores"][completed_item_id] = score
            
        if time_spent_hours is not None:
            progress["time_spent"][completed_item_id] = time_spent_hours
        
        print(f"📈 Updated progress for {student_id}: {len(progress['completed_items'])} items completed")
        
        return progress
    
    def recommend_next_steps(self, student_id: str, max_recommendations: int = 5) -> List[Dict]:
        """Recommend next learning steps based on student progress"""
        
        if student_id not in self.student_progress:
            print(f"❌ No progress found for student: {student_id}")
            return []
        
        progress = self.student_progress[student_id]
        completed_items = set(progress["completed_items"])
        
        print(f"\n🎯 Generating recommendations for {student_id}...")
        print(f"   📊 Completed: {len(completed_items)} items")
        
        recommendations = []
        candidate_items = set()
        
        # Find items connected to completed content
        for completed_id in completed_items:
            connected_vectors = self.db.get_connected_vectors(completed_id, max_hops=2)
            
            for vector_data, hop_count in connected_vectors:
                item_id = vector_data['id']
                
                if item_id not in completed_items:
                    candidate_items.add(item_id)
        
        # Score and rank candidates
        scored_candidates = []
        
        for item_id in candidate_items:
            item = self.db.get_vector(item_id)
            if not item:
                continue
                
            metadata = item['metadata']
            
            # Base relevance score
            relevance_score = 0.5
            
            # Boost for current topics
            if metadata['topic'] in progress["current_topics"]:
                relevance_score += 0.3
            
            # Boost for difficulty progression
            difficulty_scores = {"beginner": 0.1, "intermediate": 0.2, "advanced": 0.3}
            relevance_score += difficulty_scores.get(metadata['difficulty'], 0)
            
            # Boost for interactive content
            if metadata.get('interactive', False):
                relevance_score += 0.1
            
            # Check prerequisites are met
            prerequisites = metadata.get('prerequisites', [])
            prerequisites_met = all(prereq in completed_items for prereq in prerequisites)
            
            if prerequisites_met:
                relevance_score += 0.2
            else:
                relevance_score -= 0.3  # Penalize if prerequisites not met
            
            scored_candidates.append({
                "item_id": item_id,
                "title": metadata['title'],
                "type": metadata['type'],
                "difficulty": metadata['difficulty'],
                "topic": metadata['topic'],
                "estimated_hours": metadata.get('estimated_time_hours', 1),
                "prerequisites_met": prerequisites_met,
                "relevance_score": relevance_score,
                "recommendation_reason": self._get_recommendation_reason(metadata, progress)
            })
        
        # Sort by relevance score and return top recommendations
        scored_candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        recommendations = scored_candidates[:max_recommendations]
        
        print(f"   ✅ Generated {len(recommendations)} personalized recommendations")
        
        return recommendations
    
    def _get_recommendation_reason(self, metadata: Dict, progress: Dict) -> str:
        """Generate human-readable recommendation reasoning"""
        
        reasons = []
        
        if metadata['topic'] in progress["current_topics"]:
            reasons.append(f"Continue learning {metadata['topic']}")
            
        if metadata['difficulty'] == 'intermediate' and len(progress["completed_items"]) > 3:
            reasons.append("Ready for intermediate concepts")
            
        if metadata['type'] == 'exercise':
            reasons.append("Practice what you've learned")
            
        if metadata['type'] == 'project':
            reasons.append("Apply knowledge in real project")
            
        if metadata.get('interactive', False):
            reasons.append("Interactive learning experience")
        
        return " • ".join(reasons) if reasons else "Relevant to your learning journey"
    
    def get_course_analytics(self) -> Dict:
        """Get comprehensive course analytics"""
        
        stats = self.db.get_statistics()
        
        # Analyze content distribution
        content_types = {}
        difficulty_distribution = {}
        topic_distribution = {}
        
        for item_id in self.db.list_vectors():
            item = self.db.get_vector(item_id)
            if item:
                metadata = item['metadata']
                
                # Content types
                content_type = metadata['type']
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                # Difficulty distribution
                difficulty = metadata['difficulty']
                difficulty_distribution[difficulty] = difficulty_distribution.get(difficulty, 0) + 1
                
                # Topic distribution
                topic = metadata['topic']
                topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
        
        # Student progress analytics
        student_analytics = {}
        total_completions = 0
        
        for student_id, progress in self.student_progress.items():
            completed_count = len(progress["completed_items"])
            total_completions += completed_count
            
            avg_score = np.mean(list(progress["scores"].values())) if progress["scores"] else 0
            total_time = sum(progress["time_spent"].values()) if progress["time_spent"] else 0
            
            student_analytics[student_id] = {
                "completed_items": completed_count,
                "average_score": avg_score,
                "total_time_hours": total_time,
                "topics_explored": len(progress["current_topics"]),
                "topics_mastered": len(progress["mastered_topics"])
            }
        
        return {
            "course_info": {
                "name": self.course_name,
                "total_items": stats['vector_count'],
                "total_relationships": stats['relationship_count'],
                "dimension": stats['dimension']
            },
            "content_distribution": {
                "by_type": content_types,
                "by_difficulty": difficulty_distribution,
                "by_topic": topic_distribution
            },
            "student_progress": {
                "total_students": len(self.student_progress),
                "total_completions": total_completions,
                "student_details": student_analytics
            },
            "capacity_usage": stats['capacity_usage']
        }

def create_sample_ai_course():
    """Create a comprehensive AI/ML course for demonstration"""
    
    # Course content covering AI/ML fundamentals
    learning_content = [
        # AI Fundamentals Module
        {
            "id": "ai_intro_lesson",
            "title": "Introduction to Artificial Intelligence",
            "content": "What is AI? History, applications, and impact on society. Understanding different types of AI systems.",
            "type": "lesson",
            "difficulty": "beginner",
            "topic": "ai",
            "module": "ai_fundamentals",
            "order": 1,
            "time_hours": 2,
            "learning_objectives": ["Define AI", "Understand AI history", "Identify AI applications"],
            "tags": ["introduction", "history", "overview"]
        },
        {
            "id": "ai_ethics_lesson",
            "title": "AI Ethics and Responsible AI",
            "content": "Ethical considerations in AI development. Bias, fairness, transparency, and accountability.",
            "type": "lesson", 
            "difficulty": "intermediate",
            "topic": "ai",
            "module": "ai_fundamentals",
            "order": 2,
            "prerequisites": ["ai_intro_lesson"],
            "time_hours": 1.5,
            "learning_objectives": ["Understand AI ethics", "Identify bias sources", "Apply ethical frameworks"],
            "tags": ["ethics", "bias", "fairness", "responsibility"]
        },
        {
            "id": "ai_quiz_1",
            "title": "AI Fundamentals Quiz",
            "content": "Test your understanding of AI basics and ethics",
            "type": "quiz",
            "difficulty": "beginner", 
            "topic": "ai",
            "module": "ai_fundamentals",
            "prerequisites": ["ai_intro_lesson", "ai_ethics_lesson"],
            "time_hours": 0.5,
            "interactive": True,
            "tags": ["assessment", "fundamentals"]
        },
        
        # Machine Learning Module
        {
            "id": "ml_intro_lesson",
            "title": "Introduction to Machine Learning", 
            "content": "What is ML? Supervised, unsupervised, and reinforcement learning paradigms.",
            "type": "lesson",
            "difficulty": "beginner",
            "topic": "ml",
            "module": "machine_learning",
            "order": 1,
            "prerequisites": ["ai_intro_lesson"],
            "time_hours": 2,
            "learning_objectives": ["Define ML", "Compare learning types", "Understand ML workflow"],
            "tags": ["machine learning", "supervised", "unsupervised", "reinforcement"]
        },
        {
            "id": "ml_algorithms_lesson",
            "title": "Common ML Algorithms",
            "content": "Linear regression, decision trees, SVM, k-means clustering, neural networks overview.",
            "type": "lesson",
            "difficulty": "intermediate", 
            "topic": "ml",
            "module": "machine_learning",
            "order": 2,
            "prerequisites": ["ml_intro_lesson"],
            "time_hours": 3,
            "learning_objectives": ["Understand key algorithms", "Choose appropriate algorithms", "Compare algorithm strengths"],
            "tags": ["algorithms", "regression", "classification", "clustering"]
        },
        {
            "id": "ml_python_exercise",
            "title": "ML with Python - Hands-on Exercise",
            "content": "Implement linear regression and decision trees using scikit-learn. Data preprocessing and evaluation.",
            "type": "exercise",
            "difficulty": "intermediate",
            "topic": "ml", 
            "module": "machine_learning",
            "prerequisites": ["ml_algorithms_lesson"],
            "time_hours": 4,
            "interactive": True,
            "learning_objectives": ["Implement ML algorithms", "Use scikit-learn", "Evaluate models"],
            "tags": ["python", "scikit-learn", "hands-on", "implementation"]
        },
        
        # Deep Learning Module
        {
            "id": "dl_intro_lesson",
            "title": "Introduction to Deep Learning",
            "content": "Neural networks, layers, activation functions, backpropagation fundamentals.",
            "type": "lesson",
            "difficulty": "intermediate",
            "topic": "deep_learning",
            "module": "deep_learning", 
            "order": 1,
            "prerequisites": ["ml_algorithms_lesson"],
            "time_hours": 2.5,
            "learning_objectives": ["Understand neural networks", "Learn backpropagation", "Configure network architectures"],
            "tags": ["neural networks", "backpropagation", "deep learning"]
        },
        {
            "id": "dl_architectures_lesson", 
            "title": "Deep Learning Architectures",
            "content": "CNNs for computer vision, RNNs for sequences, Transformers for NLP.",
            "type": "lesson",
            "difficulty": "advanced",
            "topic": "deep_learning",
            "module": "deep_learning",
            "order": 2,
            "prerequisites": ["dl_intro_lesson"],
            "time_hours": 3,
            "learning_objectives": ["Master CNN architectures", "Understand RNN/LSTM", "Apply Transformers"],
            "tags": ["cnn", "rnn", "transformers", "architectures"]
        },
        {
            "id": "dl_project",
            "title": "Deep Learning Capstone Project", 
            "content": "Build and train a deep learning model for image classification or NLP task.",
            "type": "project",
            "difficulty": "advanced",
            "topic": "deep_learning",
            "module": "deep_learning",
            "prerequisites": ["dl_architectures_lesson", "ml_python_exercise"],
            "time_hours": 8,
            "interactive": True,
            "learning_objectives": ["Complete end-to-end project", "Apply best practices", "Present results"],
            "tags": ["capstone", "project", "implementation", "portfolio"]
        },
        
        # Programming for AI Module  
        {
            "id": "python_ai_lesson",
            "title": "Python for AI Development",
            "content": "Essential Python libraries: NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow/PyTorch.",
            "type": "lesson", 
            "difficulty": "beginner",
            "topic": "programming",
            "module": "programming",
            "order": 1,
            "time_hours": 2,
            "learning_objectives": ["Master AI libraries", "Handle data effectively", "Set up AI environment"],
            "tags": ["python", "numpy", "pandas", "scikit-learn", "tensorflow"]
        },
        {
            "id": "data_handling_exercise",
            "title": "Data Preprocessing and Visualization",
            "content": "Clean, transform, and visualize datasets for ML. Handle missing data and outliers.",
            "type": "exercise",
            "difficulty": "intermediate",
            "topic": "programming", 
            "module": "programming",
            "prerequisites": ["python_ai_lesson"],
            "time_hours": 3,
            "interactive": True,
            "learning_objectives": ["Clean datasets", "Create visualizations", "Prepare data for ML"],
            "tags": ["data preprocessing", "visualization", "pandas", "matplotlib"]
        }
    ]
    
    return learning_content

def main():
    """Demonstrate the AI Course Builder capabilities"""
    
    print("🎓 AI Course Builder Demo")
    print("=" * 50)
    
    # Create course builder
    course_builder = AI_Course_Builder("Comprehensive AI/ML Course")
    
    # Add sample course content
    learning_content = create_sample_ai_course()
    course_builder.add_learning_content(learning_content)
    
    # Build intelligent relationships
    relationships_count = course_builder.build_learning_relationships()
    
    # Generate learning paths for different scenarios
    print(f"\n🗺️ Generating Learning Paths:")
    
    # Beginner wanting to learn AI
    ai_path = course_builder.generate_learning_path(
        student_id="student_alice",
        current_topic="ai", 
        target_difficulty="intermediate",
        max_items=6
    )
    
    print(f"\n   👤 Alice's AI Learning Path:")
    for step in ai_path[:3]:
        print(f"      {step['step_number']}. {step['title']} ({step['difficulty']}) - {step['estimated_hours']}h")
    
    # Intermediate student wanting to learn ML
    ml_path = course_builder.generate_learning_path(
        student_id="student_bob",
        current_topic="ml",
        target_difficulty="advanced", 
        max_items=8
    )
    
    print(f"\n   👤 Bob's ML Learning Path:")
    for step in ml_path[:3]:
        print(f"      {step['step_number']}. {step['title']} ({step['difficulty']}) - {step['estimated_hours']}h")
    
    # Simulate student progress
    print(f"\n📈 Simulating Student Progress:")
    
    # Alice completes some AI content
    course_builder.track_student_progress("student_alice", "ai_intro_lesson", score=0.85, time_spent_hours=2.5)
    course_builder.track_student_progress("student_alice", "ai_ethics_lesson", score=0.78, time_spent_hours=1.8)
    course_builder.track_student_progress("student_alice", "ai_quiz_1", score=0.90, time_spent_hours=0.4)
    
    # Bob completes some ML content
    course_builder.track_student_progress("student_bob", "ml_intro_lesson", score=0.88, time_spent_hours=2.2)
    course_builder.track_student_progress("student_bob", "ml_algorithms_lesson", score=0.82, time_spent_hours=3.5)
    
    # Generate personalized recommendations
    alice_recommendations = course_builder.recommend_next_steps("student_alice", max_recommendations=3)
    bob_recommendations = course_builder.recommend_next_steps("student_bob", max_recommendations=3)
    
    print(f"\n🎯 Personalized Recommendations:")
    print(f"   👤 Alice's Next Steps:")
    for rec in alice_recommendations:
        print(f"      • {rec['title']} ({rec['difficulty']}) - {rec['recommendation_reason']}")
    
    print(f"\n   👤 Bob's Next Steps:")
    for rec in bob_recommendations:
        print(f"      • {rec['title']} ({rec['difficulty']}) - {rec['recommendation_reason']}")
    
    # Course analytics
    analytics = course_builder.get_course_analytics()
    
    print(f"\n📊 Course Analytics:")
    course_info = analytics['course_info']
    print(f"   📚 Course: {course_info['name']}")
    print(f"   📖 Content: {course_info['total_items']} items, {course_info['total_relationships']} relationships")
    print(f"   🎯 Embeddings: {course_info['dimension']}D")
    
    content_dist = analytics['content_distribution']
    print(f"   📋 Content Types: {dict(content_dist['by_type'])}")
    print(f"   📊 Difficulty: {dict(content_dist['by_difficulty'])}")
    print(f"   🏷️ Topics: {dict(content_dist['by_topic'])}")
    
    student_progress = analytics['student_progress']
    print(f"   👥 Students: {student_progress['total_students']} active")
    print(f"   ✅ Completions: {student_progress['total_completions']} items completed")
    
    capacity_usage = analytics['capacity_usage']
    print(f"   📈 Capacity: {capacity_usage['vector_usage_percent']:.1f}% vectors, {capacity_usage['relationship_usage_percent']:.1f}% relationships")
    
    print(f"\n✨ Key Features Demonstrated:")
    features = [
        f"Auto-dimension detection worked seamlessly",
        f"Built {relationships_count} intelligent learning relationships",
        f"Generated personalized learning paths using relationship traversal",
        f"Tracked student progress through relationship networks", 
        f"Provided smart recommendations based on learning graph",
        f"Delivered comprehensive course analytics and insights"
    ]
    
    for feature in features:
        print(f"   💡 {feature}")
    
    print(f"\n🎉 AI Course Builder Demo Complete!")
    print(f"   Perfect example of educational AI powered by relationship-aware search!")
    
    # Show upgrade path
    if capacity_usage['vector_usage_percent'] > 30 or capacity_usage['relationship_usage_percent'] > 30:
        print(f"\n🚀 Ready to Scale Your Educational Platform?")
        print(f"   Upgrade to full RudraDB for unlimited course content and students!")

if __name__ == "__main__":
    try:
        import rudradb
        import numpy as np
        
        print(f"🎯 Using RudraDB-Opin v{rudradb.__version__}")
        print(f"🏫 Perfect for educational AI applications!")
        
        main()
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install rudradb-opin numpy")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure RudraDB-Opin is properly installed")
