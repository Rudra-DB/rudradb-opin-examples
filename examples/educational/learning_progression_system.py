#!/usr/bin/env python3
"""
Educational Learning Progression with RudraDB-Opin
=================================================

This example demonstrates building intelligent learning paths and educational
progressions using RudraDB-Opin's relationship-aware search capabilities.

Features demonstrated:
- Auto-detection of learning prerequisites
- Intelligent learning path construction
- Difficulty progression modeling
- Personalized curriculum generation
- Learning outcome optimization

Perfect for:
- Educational platforms
- Course builders
- Learning management systems
- Skill development tracking

Requirements:
pip install numpy rudradb-opin sentence-transformers
"""

import rudradb
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
from dataclasses import dataclass
from enum import Enum

# Try importing sentence transformers for real embeddings
try:
    from sentence_transformers import SentenceTransformer
    REAL_EMBEDDINGS = True
except ImportError:
    REAL_EMBEDDINGS = False
    print("⚠️ sentence-transformers not available, using mock embeddings")

class DifficultyLevel(Enum):
    """Difficulty levels for learning content"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4

@dataclass
class LearningObjective:
    """Learning objective with prerequisites and outcomes"""
    id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    prerequisites: List[str]
    learning_outcomes: List[str]
    estimated_time_minutes: int
    topics: List[str]

class EducationalProgressionSystem:
    """Intelligent educational progression system using RudraDB-Opin"""
    
    def __init__(self, subject_area: str = "Computer Science"):
        self.subject_area = subject_area
        self.db = rudradb.RudraDB()  # 🎯 Auto-dimension detection
        self.learning_objectives: Dict[str, LearningObjective] = {}
        
        # Initialize embedding model
        if REAL_EMBEDDINGS:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("🤖 Using real sentence-transformers embeddings")
        else:
            self.embedding_model = None
            print("🤖 Using mock embeddings for demonstration")
        
        print(f"📚 Educational Progression System: {subject_area}")
        print("   🎯 Auto-dimension detection enabled")
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for text content"""
        if REAL_EMBEDDINGS and self.embedding_model:
            return self.embedding_model.encode([text])[0].astype(np.float32)
        else:
            # Mock embedding based on text characteristics
            text_features = [
                len(text) / 100.0,  # Length feature
                text.count(' ') / 50.0,  # Word count feature
                sum(1 for c in text if c.isupper()) / len(text),  # Uppercase ratio
                text.count('?') + text.count('!'),  # Question/exclamation count
            ]
            # Pad to 384 dimensions with controlled randomness
            mock_embedding = np.array(text_features + [np.random.rand() for _ in range(380)])
            return mock_embedding.astype(np.float32)
    
    def add_learning_objective(self, objective: LearningObjective):
        """Add a learning objective with auto-relationship detection"""
        
        # Store objective
        self.learning_objectives[objective.id] = objective
        
        # Create rich embedding from objective content
        content_text = f"{objective.title}. {objective.description}. Topics: {', '.join(objective.topics)}."
        embedding = self.create_embedding(content_text)
        
        # Enhanced metadata for intelligent relationship detection
        metadata = {
            "title": objective.title,
            "description": objective.description,
            "difficulty": objective.difficulty.name.lower(),
            "difficulty_level": objective.difficulty.value,
            "prerequisites": objective.prerequisites,
            "learning_outcomes": objective.learning_outcomes,
            "estimated_time": objective.estimated_time_minutes,
            "topics": objective.topics,
            "content_type": "learning_objective",
            "subject": self.subject_area
        }
        
        # Add to RudraDB with auto-dimension detection
        self.db.add_vector(objective.id, embedding, metadata)
        
        # 🧠 Auto-build educational relationships
        relationships_created = self._auto_detect_educational_relationships(objective.id, metadata)
        
        if self.db.vector_count() == 1:
            print(f"   🎯 Auto-detected dimension: {self.db.dimension()}D")
        
        print(f"📝 Added: {objective.title}")
        print(f"   📊 Difficulty: {objective.difficulty.name}")
        print(f"   🔗 Auto-relationships: {relationships_created}")
        
        return relationships_created
    
    def _auto_detect_educational_relationships(self, objective_id: str, metadata: Dict[str, Any]) -> int:
        """Auto-detect educational relationships based on learning science principles"""
        
        relationships_created = 0
        current_difficulty = metadata["difficulty_level"]
        current_topics = set(metadata["topics"])
        current_outcomes = set(metadata["learning_outcomes"])
        
        for existing_id in self.db.list_vectors():
            if existing_id == objective_id or relationships_created >= 5:
                continue
            
            existing = self.db.get_vector(existing_id)
            existing_meta = existing["metadata"]
            existing_difficulty = existing_meta["difficulty_level"]
            existing_topics = set(existing_meta["topics"])
            existing_outcomes = set(existing_meta["learning_outcomes"])
            
            # 1. 📈 HIERARCHICAL: Difficulty progression (prerequisite relationships)
            if existing_difficulty == current_difficulty - 1:  # One level easier
                shared_topics = current_topics & existing_topics
                if len(shared_topics) >= 1:  # Share at least one topic
                    self.db.add_relationship(existing_id, objective_id, "hierarchical", 0.9, {
                        "auto_detected": True,
                        "reason": "difficulty_progression",
                        "from_level": existing_meta["difficulty"],
                        "to_level": metadata["difficulty"],
                        "shared_topics": list(shared_topics)
                    })
                    relationships_created += 1
                    print(f"      📈 Prerequisite: {existing_id} → {objective_id}")
            
            # 2. ⏰ TEMPORAL: Learning sequence (same difficulty, complementary topics)
            elif existing_difficulty == current_difficulty:
                # Complementary topics (different but related)
                topic_overlap = len(current_topics & existing_topics)
                if 0 < topic_overlap < min(len(current_topics), len(existing_topics)):
                    strength = 0.7 + (topic_overlap * 0.1)
                    self.db.add_relationship(existing_id, objective_id, "temporal", strength, {
                        "auto_detected": True,
                        "reason": "complementary_topics",
                        "topic_overlap": topic_overlap,
                        "difficulty": metadata["difficulty"]
                    })
                    relationships_created += 1
                    print(f"      ⏰ Sequence: {existing_id} → {objective_id}")
            
            # 3. 🔗 SEMANTIC: Related content (same topics, any difficulty)
            topic_similarity = len(current_topics & existing_topics) / len(current_topics | existing_topics)
            if topic_similarity > 0.5:  # Significant topic overlap
                self.db.add_relationship(existing_id, objective_id, "semantic", topic_similarity, {
                    "auto_detected": True,
                    "reason": "topic_similarity",
                    "similarity_score": topic_similarity,
                    "shared_topics": list(current_topics & existing_topics)
                })
                relationships_created += 1
                print(f"      🔗 Related: {existing_id} ↔ {objective_id}")
            
            # 4. 🎯 CAUSAL: Learning outcome dependencies
            outcome_dependencies = existing_outcomes & {prereq for prereq in metadata["prerequisites"]}
            if outcome_dependencies:
                self.db.add_relationship(existing_id, objective_id, "causal", 0.95, {
                    "auto_detected": True,
                    "reason": "outcome_dependency",
                    "dependencies": list(outcome_dependencies)
                })
                relationships_created += 1
                print(f"      🎯 Dependency: {existing_id} → {objective_id}")
            
            # 5. 🔄 ASSOCIATIVE: Cross-topic connections
            elif len(current_topics & existing_topics) >= 1 and abs(existing_difficulty - current_difficulty) <= 1:
                shared = current_topics & existing_topics
                strength = min(0.6, len(shared) * 0.2)
                self.db.add_relationship(existing_id, objective_id, "associative", strength, {
                    "auto_detected": True,
                    "reason": "cross_topic_connection",
                    "shared_topics": list(shared)
                })
                relationships_created += 1
                print(f"      🔄 Associated: {existing_id} ↔ {objective_id}")
        
        return relationships_created
    
    def generate_learning_path(self, target_objective_id: str, learner_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized learning path to reach target objective"""
        
        print(f"\n🎓 Generating Learning Path")
        print(f"🎯 Target: {self.learning_objectives[target_objective_id].title}")
        print(f"👤 Learner: {learner_profile.get('name', 'Anonymous')}")
        
        # Get target objective
        target_objective = self.learning_objectives[target_objective_id]
        current_level = DifficultyLevel[learner_profile.get("current_level", "BEGINNER")]
        completed_objectives = set(learner_profile.get("completed", []))
        preferred_topics = set(learner_profile.get("interests", []))
        time_budget_minutes = learner_profile.get("time_budget", 300)  # 5 hours default
        
        # Find all prerequisites and related content using relationship traversal
        connected_objectives = self.db.get_connected_vectors(target_objective_id, max_hops=3)
        
        # Filter and rank potential learning path components
        path_candidates = []
        
        for vector_data, hop_count in connected_objectives:
            obj_id = vector_data["id"]
            
            # Skip already completed objectives
            if obj_id in completed_objectives:
                continue
            
            obj = self.learning_objectives.get(obj_id)
            if not obj:
                continue
            
            # Calculate suitability score
            suitability_score = self._calculate_objective_suitability(
                obj, learner_profile, current_level, preferred_topics, hop_count
            )
            
            if suitability_score > 0.3:  # Minimum threshold
                path_candidates.append({
                    "objective": obj,
                    "suitability": suitability_score,
                    "hop_count": hop_count,
                    "prerequisites_met": self._check_prerequisites_met(obj, completed_objectives)
                })
        
        # Sort by suitability and create optimal learning sequence
        path_candidates.sort(key=lambda x: (x["prerequisites_met"], x["suitability"]), reverse=True)
        
        # Build learning path within time budget
        learning_path = []
        total_time = 0
        current_completed = completed_objectives.copy()
        
        for candidate in path_candidates:
            obj = candidate["objective"]
            
            # Check if prerequisites are now met
            if not self._check_prerequisites_met(obj, current_completed):
                continue
            
            # Check time budget
            if total_time + obj.estimated_time_minutes > time_budget_minutes:
                continue
            
            # Add to path
            learning_path.append({
                "objective_id": obj.id,
                "title": obj.title,
                "difficulty": obj.difficulty.name,
                "estimated_time": obj.estimated_time_minutes,
                "topics": obj.topics,
                "suitability_score": candidate["suitability"],
                "connection_type": "direct" if candidate["hop_count"] == 0 else f"{candidate['hop_count']}-hop",
                "prerequisites": obj.prerequisites,
                "learning_outcomes": obj.learning_outcomes
            })
            
            total_time += obj.estimated_time_minutes
            current_completed.add(obj.id)
            
            # Stop if we've reached the target or optimal path length
            if obj.id == target_objective_id or len(learning_path) >= 10:
                break
        
        # Ensure target is included if possible
        if target_objective_id not in [item["objective_id"] for item in learning_path]:
            if total_time + target_objective.estimated_time_minutes <= time_budget_minutes:
                learning_path.append({
                    "objective_id": target_objective_id,
                    "title": target_objective.title,
                    "difficulty": target_objective.difficulty.name,
                    "estimated_time": target_objective.estimated_time_minutes,
                    "topics": target_objective.topics,
                    "suitability_score": 1.0,  # Target always has max suitability
                    "connection_type": "target",
                    "prerequisites": target_objective.prerequisites,
                    "learning_outcomes": target_objective.learning_outcomes
                })
                total_time += target_objective.estimated_time_minutes
        
        return {
            "learner_profile": learner_profile,
            "target_objective": target_objective_id,
            "learning_path": learning_path,
            "total_estimated_time": total_time,
            "objectives_count": len(learning_path),
            "difficulty_progression": self._analyze_difficulty_progression(learning_path),
            "topic_coverage": self._analyze_topic_coverage(learning_path, preferred_topics),
            "path_quality_score": self._calculate_path_quality(learning_path, learner_profile)
        }
    
    def _calculate_objective_suitability(self, obj: LearningObjective, learner_profile: Dict[str, Any],
                                       current_level: DifficultyLevel, preferred_topics: set, hop_count: int) -> float:
        """Calculate how suitable an objective is for the learner"""
        
        score = 0.0
        
        # Difficulty appropriateness (0.4 weight)
        level_diff = abs(obj.difficulty.value - current_level.value)
        if level_diff == 0:
            score += 0.4  # Perfect level match
        elif level_diff == 1:
            score += 0.3  # Slightly challenging
        elif level_diff == 2:
            score += 0.1  # Quite challenging
        # level_diff >= 3: too difficult, no points
        
        # Topic interest alignment (0.3 weight)
        topic_overlap = len(set(obj.topics) & preferred_topics) / len(set(obj.topics) | preferred_topics)
        score += topic_overlap * 0.3
        
        # Connection strength (0.2 weight)
        connection_bonus = max(0, (3 - hop_count) / 3 * 0.2)
        score += connection_bonus
        
        # Time efficiency (0.1 weight)
        if obj.estimated_time_minutes <= 60:  # Prefer shorter objectives
            score += 0.1
        elif obj.estimated_time_minutes <= 120:
            score += 0.05
        
        return min(1.0, score)
    
    def _check_prerequisites_met(self, obj: LearningObjective, completed_objectives: set) -> bool:
        """Check if all prerequisites for an objective are met"""
        return all(prereq in completed_objectives for prereq in obj.prerequisites)
    
    def _analyze_difficulty_progression(self, learning_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the difficulty progression in the learning path"""
        
        if not learning_path:
            return {"valid": False, "reason": "empty_path"}
        
        difficulties = [item["difficulty"] for item in learning_path]
        difficulty_values = [DifficultyLevel[diff].value for diff in difficulties]
        
        # Check for appropriate progression
        is_progressive = True
        max_jump = 0
        
        for i in range(1, len(difficulty_values)):
            jump = difficulty_values[i] - difficulty_values[i-1]
            max_jump = max(max_jump, jump)
            if jump > 1:  # Too big a jump
                is_progressive = False
        
        return {
            "valid": is_progressive,
            "max_difficulty_jump": max_jump,
            "difficulty_range": f"{min(difficulties)} to {max(difficulties)}",
            "progression_type": "gradual" if is_progressive else "steep"
        }
    
    def _analyze_topic_coverage(self, learning_path: List[Dict[str, Any]], preferred_topics: set) -> Dict[str, Any]:
        """Analyze topic coverage in the learning path"""
        
        all_topics = set()
        for item in learning_path:
            all_topics.update(item["topics"])
        
        preference_coverage = len(all_topics & preferred_topics) / len(preferred_topics) if preferred_topics else 0
        
        return {
            "total_topics": len(all_topics),
            "topics_covered": list(all_topics),
            "preference_coverage": preference_coverage,
            "breadth": len(all_topics) / max(1, len(learning_path))  # Topics per objective
        }
    
    def _calculate_path_quality(self, learning_path: List[Dict[str, Any]], learner_profile: Dict[str, Any]) -> float:
        """Calculate overall quality score for the learning path"""
        
        if not learning_path:
            return 0.0
        
        # Average suitability score (0.4 weight)
        avg_suitability = np.mean([item["suitability_score"] for item in learning_path])
        quality_score = avg_suitability * 0.4
        
        # Path coherence - prefer fewer hops (0.3 weight)
        connection_scores = []
        for item in learning_path:
            if "hop" in item["connection_type"]:
                hop_count = int(item["connection_type"][0])
                connection_scores.append(max(0, (3 - hop_count) / 3))
            else:
                connection_scores.append(1.0)  # Direct or target
        
        avg_connection = np.mean(connection_scores) if connection_scores else 1.0
        quality_score += avg_connection * 0.3
        
        # Time efficiency (0.2 weight)
        total_time = sum(item["estimated_time"] for item in learning_path)
        target_time = learner_profile.get("time_budget", 300)
        time_efficiency = min(1.0, target_time / max(1, total_time))
        quality_score += time_efficiency * 0.2
        
        # Completeness - includes target (0.1 weight)
        includes_target = any(item["connection_type"] == "target" for item in learning_path)
        if includes_target:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def recommend_next_objectives(self, learner_profile: Dict[str, Any], count: int = 3) -> List[Dict[str, Any]]:
        """Recommend next learning objectives based on current progress"""
        
        completed = set(learner_profile.get("completed", []))
        current_level = DifficultyLevel[learner_profile.get("current_level", "BEGINNER")]
        preferred_topics = set(learner_profile.get("interests", []))
        
        recommendations = []
        
        for obj_id, obj in self.learning_objectives.items():
            if obj_id in completed:
                continue
            
            # Check if prerequisites are met
            if not self._check_prerequisites_met(obj, completed):
                continue
            
            # Calculate recommendation score
            suitability = self._calculate_objective_suitability(obj, learner_profile, current_level, preferred_topics, 0)
            
            if suitability > 0.4:  # Good threshold for recommendations
                recommendations.append({
                    "objective_id": obj.id,
                    "title": obj.title,
                    "difficulty": obj.difficulty.name,
                    "estimated_time": obj.estimated_time_minutes,
                    "topics": obj.topics,
                    "suitability_score": suitability,
                    "learning_outcomes": obj.learning_outcomes[:3]  # Top 3 outcomes
                })
        
        # Sort by suitability and return top recommendations
        recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
        return recommendations[:count]

def create_sample_cs_curriculum():
    """Create a sample Computer Science curriculum with learning objectives"""
    
    print("📚 Creating Sample CS Curriculum")
    print("=" * 35)
    
    system = EducationalProgressionSystem("Computer Science")
    
    # Define learning objectives with realistic progression
    objectives = [
        LearningObjective(
            id="cs_intro",
            title="Introduction to Computer Science",
            description="Overview of computer science fundamentals, history, and applications",
            difficulty=DifficultyLevel.BEGINNER,
            prerequisites=[],
            learning_outcomes=["Understand CS history", "Identify CS applications", "Basic terminology"],
            estimated_time_minutes=60,
            topics=["computer_science", "fundamentals", "history"]
        ),
        LearningObjective(
            id="programming_basics",
            title="Programming Fundamentals",
            description="Basic programming concepts including variables, control structures, and functions",
            difficulty=DifficultyLevel.BEGINNER,
            prerequisites=["cs_intro"],
            learning_outcomes=["Write basic programs", "Use variables", "Control flow"],
            estimated_time_minutes=120,
            topics=["programming", "variables", "control_flow", "functions"]
        ),
        LearningObjective(
            id="python_syntax",
            title="Python Programming Syntax",
            description="Learn Python programming language syntax and basic constructs",
            difficulty=DifficultyLevel.BEGINNER,
            prerequisites=["programming_basics"],
            learning_outcomes=["Write Python code", "Use Python syntax", "Debug programs"],
            estimated_time_minutes=90,
            topics=["python", "syntax", "programming", "debugging"]
        ),
        LearningObjective(
            id="data_structures_intro",
            title="Introduction to Data Structures",
            description="Basic data structures: arrays, lists, stacks, queues",
            difficulty=DifficultyLevel.INTERMEDIATE,
            prerequisites=["python_syntax"],
            learning_outcomes=["Implement basic structures", "Choose appropriate structures", "Analyze complexity"],
            estimated_time_minutes=150,
            topics=["data_structures", "arrays", "lists", "complexity"]
        ),
        LearningObjective(
            id="algorithms_basics",
            title="Basic Algorithms",
            description="Fundamental algorithms: sorting, searching, basic problem-solving",
            difficulty=DifficultyLevel.INTERMEDIATE,
            prerequisites=["data_structures_intro"],
            learning_outcomes=["Implement sorting algorithms", "Understand search algorithms", "Analyze efficiency"],
            estimated_time_minutes=180,
            topics=["algorithms", "sorting", "searching", "problem_solving"]
        ),
        LearningObjective(
            id="object_oriented",
            title="Object-Oriented Programming",
            description="OOP concepts: classes, objects, inheritance, polymorphism",
            difficulty=DifficultyLevel.INTERMEDIATE,
            prerequisites=["python_syntax"],
            learning_outcomes=["Design classes", "Use inheritance", "Apply OOP principles"],
            estimated_time_minutes=120,
            topics=["oop", "classes", "inheritance", "design_patterns"]
        ),
        LearningObjective(
            id="advanced_algorithms",
            title="Advanced Algorithms",
            description="Complex algorithms: dynamic programming, graph algorithms, optimization",
            difficulty=DifficultyLevel.ADVANCED,
            prerequisites=["algorithms_basics"],
            learning_outcomes=["Implement DP solutions", "Graph traversal", "Optimization techniques"],
            estimated_time_minutes=240,
            topics=["advanced_algorithms", "dynamic_programming", "graphs", "optimization"]
        ),
        LearningObjective(
            id="machine_learning_intro",
            title="Introduction to Machine Learning",
            description="ML fundamentals: supervised learning, basic algorithms, evaluation",
            difficulty=DifficultyLevel.ADVANCED,
            prerequisites=["algorithms_basics", "python_syntax"],
            learning_outcomes=["Understand ML concepts", "Implement basic ML", "Evaluate models"],
            estimated_time_minutes=200,
            topics=["machine_learning", "supervised_learning", "algorithms", "evaluation"]
        )
    ]
    
    # Add all objectives to the system
    total_relationships = 0
    for objective in objectives:
        relationships = system.add_learning_objective(objective)
        total_relationships += relationships
    
    print(f"\n✅ Curriculum created successfully!")
    print(f"   📚 {len(objectives)} learning objectives")
    print(f"   🔗 {total_relationships} auto-detected relationships")
    print(f"   🎯 Auto-detected embedding dimension: {system.db.dimension()}D")
    
    return system

def demo_personalized_learning_paths():
    """Demonstrate personalized learning path generation"""
    
    print("\n🎓 Personalized Learning Path Generation")
    print("=" * 45)
    
    # Create curriculum
    system = create_sample_cs_curriculum()
    
    # Define different learner profiles
    learners = [
        {
            "name": "Alice (Complete Beginner)",
            "current_level": "BEGINNER",
            "completed": [],
            "interests": ["programming", "python"],
            "time_budget": 480  # 8 hours
        },
        {
            "name": "Bob (Has Some Programming)",
            "current_level": "INTERMEDIATE",
            "completed": ["cs_intro", "programming_basics", "python_syntax"],
            "interests": ["algorithms", "data_structures", "problem_solving"],
            "time_budget": 360  # 6 hours
        },
        {
            "name": "Carol (Advanced Student)",
            "current_level": "ADVANCED",
            "completed": ["cs_intro", "programming_basics", "python_syntax", "data_structures_intro", "algorithms_basics"],
            "interests": ["machine_learning", "advanced_algorithms", "optimization"],
            "time_budget": 300  # 5 hours
        }
    ]
    
    # Generate learning paths for each learner
    for learner in learners:
        print(f"\n👤 Learning Path for {learner['name']}")
        print("-" * 50)
        
        # Get recommendations
        recommendations = system.recommend_next_objectives(learner, count=3)
        print(f"🎯 Next Recommended Objectives:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec['title']} ({rec['difficulty']}) - {rec['estimated_time']}min")
            print(f"      Topics: {', '.join(rec['topics'])}")
            print(f"      Suitability: {rec['suitability_score']:.2f}")
        
        # Generate path to machine learning (common goal)
        if "machine_learning_intro" in system.learning_objectives:
            learning_path = system.generate_learning_path("machine_learning_intro", learner)
            
            print(f"\n📚 Path to Machine Learning:")
            print(f"   🎯 Target: {learning_path['target_objective']}")
            print(f"   ⏱️  Total time: {learning_path['total_estimated_time']} minutes")
            print(f"   📊 Path quality: {learning_path['path_quality_score']:.2f}/1.0")
            
            print(f"\n   Learning Sequence:")
            for i, step in enumerate(learning_path['learning_path'], 1):
                print(f"      {i}. {step['title']} ({step['difficulty']})")
                print(f"         Time: {step['estimated_time']}min | Connection: {step['connection_type']}")
                print(f"         Outcomes: {', '.join(step['learning_outcomes'][:2])}")
            
            # Analyze path characteristics
            progression = learning_path['difficulty_progression']
            coverage = learning_path['topic_coverage']
            
            print(f"\n   📈 Path Analysis:")
            print(f"      Difficulty progression: {progression['progression_type']}")
            print(f"      Topic coverage: {coverage['total_topics']} topics")
            print(f"      Interest alignment: {coverage['preference_coverage']:.1%}")

def demo_curriculum_analytics():
    """Demonstrate curriculum analytics and insights"""
    
    print("\n📊 Curriculum Analytics")
    print("=" * 25)
    
    system = create_sample_cs_curriculum()
    
    # Analyze curriculum structure
    stats = system.db.get_statistics()
    
    print(f"📚 Curriculum Overview:")
    print(f"   Learning objectives: {stats['vector_count']}")
    print(f"   Relationships: {stats['relationship_count']}")
    print(f"   Embedding dimension: {stats['dimension']}D")
    
    # Analyze relationship patterns
    relationship_types = {}
    total_relationships = 0
    
    for obj_id in system.learning_objectives.keys():
        relationships = system.db.get_relationships(obj_id)
        for rel in relationships:
            rel_type = rel["relationship_type"]
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            total_relationships += 1
    
    print(f"\n🔗 Relationship Analysis:")
    for rel_type, count in relationship_types.items():
        percentage = (count / total_relationships) * 100 if total_relationships > 0 else 0
        print(f"   {rel_type.title()}: {count} ({percentage:.1f}%)")
    
    # Find curriculum bottlenecks (objectives with many prerequisites)
    bottlenecks = []
    for obj_id, obj in system.learning_objectives.items():
        if len(obj.prerequisites) >= 2:
            bottlenecks.append((obj_id, obj.title, len(obj.prerequisites)))
    
    bottlenecks.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n🚧 Potential Bottlenecks:")
    for obj_id, title, prereq_count in bottlenecks[:3]:
        print(f"   {title}: {prereq_count} prerequisites")
    
    # Analyze difficulty distribution
    difficulty_counts = {}
    for obj in system.learning_objectives.values():
        diff = obj.difficulty.name
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
    
    print(f"\n📈 Difficulty Distribution:")
    for difficulty, count in difficulty_counts.items():
        print(f"   {difficulty}: {count} objectives")

if __name__ == "__main__":
    try:
        # Create and demonstrate the educational progression system
        demo_personalized_learning_paths()
        demo_curriculum_analytics()
        
        print(f"\n✅ Educational Progression System Demo Complete!")
        print(f"\n🎯 Key Features Demonstrated:")
        print(f"   • Auto-dimension detection for educational content")
        print(f"   • Intelligent relationship detection based on learning science")
        print(f"   • Personalized learning path generation")
        print(f"   • Curriculum analytics and optimization")
        print(f"   • Multi-hop discovery for learning dependencies")
        
        print(f"\n💡 Applications:")
        print(f"   • Educational platforms and LMS systems")
        print(f"   • Personalized curriculum generation")
        print(f"   • Learning analytics and optimization")
        print(f"   • Adaptive learning systems")
        print(f"   • Skill gap analysis and development")
        
    except Exception as e:
        print(f"❌ Error in educational progression demo: {e}")
        print(f"💡 Check troubleshooting/debug_guide.py for help")
        import traceback
        traceback.print_exc()
