#!/usr/bin/env python3
"""
📚 Knowledge Base Builder Tutorial

This tutorial demonstrates how to build an intelligent knowledge base using
RudraDB-Opin with auto-relationship detection and multi-hop discovery.

Features demonstrated:
- Knowledge base design patterns
- Auto-relationship building for educational content
- Learning path discovery through relationships
- Intelligent search and recommendation
- Content organization and categorization
- Prerequisites and dependencies tracking
"""

import rudradb
import numpy as np
from datetime import datetime
import time
import json

class KnowledgeBaseBuilder:
    """Intelligent knowledge base with relationship-aware search"""
    
    def __init__(self):
        """Initialize the knowledge base with auto-dimension detection"""
        self.db = rudradb.RudraDB()  # Auto-detects embedding dimensions
        self.categories = set()
        self.difficulty_levels = set()
        self.content_stats = {
            "documents_added": 0,
            "relationships_built": 0,
            "categories_created": 0
        }
        
        print("📚 Knowledge Base Builder initialized")
        print("   🎯 Auto-dimension detection enabled")
        print("   🧠 Auto-relationship building ready")
    
    def add_knowledge_item(self, item_id, title, content, category=None, 
                          difficulty="intermediate", item_type="article",
                          tags=None, prerequisites=None, **metadata):
        """Add a knowledge item with rich metadata for relationship building"""
        
        # Generate embedding (in real use, use actual embedding model)
        embedding = np.random.rand(384).astype(np.float32)
        
        # Build comprehensive metadata
        item_metadata = {
            "title": title,
            "content": content[:500] + ("..." if len(content) > 500 else ""),
            "full_content_length": len(content),
            "category": category or "general",
            "difficulty": difficulty,
            "type": item_type,
            "tags": tags or [],
            "prerequisites": prerequisites or [],
            "added_at": datetime.now().isoformat(),
            "content_hash": hash(content) % 10000,  # Simple content fingerprint
            **metadata
        }
        
        # Add to database - dimension auto-detected on first add
        self.db.add_vector(item_id, embedding, item_metadata)
        
        # Track categories and difficulty levels
        if category:
            self.categories.add(category)
        self.difficulty_levels.add(difficulty)
        
        # Auto-build relationships with existing content
        relationships_built = self._auto_build_relationships(item_id, item_metadata)
        
        # Update statistics
        self.content_stats["documents_added"] += 1
        self.content_stats["relationships_built"] += relationships_built
        self.content_stats["categories_created"] = len(self.categories)
        
        return relationships_built
    
    def _auto_build_relationships(self, new_item_id, new_metadata):
        """Automatically build relationships based on content analysis"""
        
        relationships_built = 0
        max_relationships_per_item = 5  # Limit for Opin capacity
        
        # Get all existing items for comparison
        existing_items = []
        for item_id in self.db.list_vectors():
            if item_id != new_item_id:
                vector = self.db.get_vector(item_id)
                if vector:
                    existing_items.append((item_id, vector['metadata']))
        
        # Analyze for different relationship types
        for existing_id, existing_metadata in existing_items:
            if relationships_built >= max_relationships_per_item:
                break
            
            # 1. Hierarchical: Category-based hierarchy
            if (new_metadata['category'] == existing_metadata.get('category') and
                new_metadata['type'] == 'overview' and existing_metadata.get('type') == 'detailed'):
                self._add_relationship(
                    new_item_id, existing_id, "hierarchical", 0.9,
                    f"Overview → Detail in {new_metadata['category']}"
                )
                relationships_built += 1
                continue
            
            # 2. Temporal: Learning progression
            difficulty_order = {"beginner": 1, "intermediate": 2, "advanced": 3}
            new_level = difficulty_order.get(new_metadata['difficulty'], 2)
            existing_level = difficulty_order.get(existing_metadata.get('difficulty'), 2)
            
            if (new_level == existing_level + 1 and 
                new_metadata['category'] == existing_metadata.get('category')):
                self._add_relationship(
                    existing_id, new_item_id, "temporal", 0.85,
                    f"Learning progression: {existing_metadata.get('difficulty')} → {new_metadata['difficulty']}"
                )
                relationships_built += 1
                continue
            
            # 3. Causal: Prerequisites
            if existing_id in new_metadata.get('prerequisites', []):
                self._add_relationship(
                    existing_id, new_item_id, "causal", 0.95,
                    f"Prerequisite relationship"
                )
                relationships_built += 1
                continue
            
            # 4. Semantic: Same category
            if (new_metadata['category'] == existing_metadata.get('category') and
                new_metadata['difficulty'] == existing_metadata.get('difficulty')):
                self._add_relationship(
                    new_item_id, existing_id, "semantic", 0.7,
                    f"Same category and difficulty level"
                )
                relationships_built += 1
                continue
            
            # 5. Associative: Tag overlap
            new_tags = set(new_metadata.get('tags', []))
            existing_tags = set(existing_metadata.get('tags', []))
            shared_tags = new_tags & existing_tags
            
            if len(shared_tags) >= 2:
                strength = min(0.8, len(shared_tags) * 0.2)
                self._add_relationship(
                    new_item_id, existing_id, "associative", strength,
                    f"Shared tags: {', '.join(shared_tags)}"
                )
                relationships_built += 1
        
        return relationships_built
    
    def _add_relationship(self, source_id, target_id, rel_type, strength, reason):
        """Add relationship with error handling for capacity limits"""
        try:
            self.db.add_relationship(source_id, target_id, rel_type, strength, {
                "reason": reason,
                "auto_generated": True,
                "confidence": strength,
                "created_at": datetime.now().isoformat()
            })
            return True
        except RuntimeError as e:
            if "capacity" in str(e).lower():
                print(f"   ⚠️  Relationship capacity reached: {e}")
                return False
            else:
                raise
    
    def find_learning_path(self, start_topic, max_steps=3):
        """Find optimal learning path from a starting topic"""
        
        if not self.db.vector_exists(start_topic):
            print(f"❌ Topic '{start_topic}' not found in knowledge base")
            return []
        
        print(f"🗺️  Finding learning path from '{start_topic}'...")
        
        # Get connected items through relationships
        connected_items = self.db.get_connected_vectors(start_topic, max_hops=max_steps)
        
        # Sort by hop count (learning progression)
        learning_path = sorted(connected_items, key=lambda x: x[1])
        
        formatted_path = []
        for vector_data, hop_count in learning_path:
            item_metadata = vector_data['metadata']
            formatted_path.append({
                "id": vector_data['id'],
                "title": item_metadata['title'],
                "category": item_metadata['category'],
                "difficulty": item_metadata['difficulty'],
                "type": item_metadata['type'],
                "hop_count": hop_count,
                "connection": "starting point" if hop_count == 0 else f"{hop_count} hop{'s' if hop_count > 1 else ''} away"
            })
        
        return formatted_path
    
    def get_recommendations(self, current_item, recommendation_type="next_steps"):
        """Get intelligent recommendations based on current item"""
        
        if not self.db.vector_exists(current_item):
            return []
        
        current_vector = self.db.get_vector(current_item)
        current_metadata = current_vector['metadata']
        current_embedding = current_vector['embedding']
        
        print(f"💡 Getting {recommendation_type} recommendations for '{current_item}'...")
        
        if recommendation_type == "next_steps":
            # Find items that build upon current item
            params = rudradb.SearchParams(
                top_k=5,
                include_relationships=True,
                max_hops=2,
                relationship_types=["temporal", "hierarchical", "causal"],
                relationship_weight=0.6
            )
        elif recommendation_type == "related":
            # Find related items in same domain
            params = rudradb.SearchParams(
                top_k=5,
                include_relationships=True,
                max_hops=1,
                relationship_types=["semantic", "associative"],
                relationship_weight=0.4
            )
        else:  # "comprehensive"
            # Use all relationship types
            params = rudradb.SearchParams(
                top_k=8,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.5
            )
        
        # Search for recommendations
        results = self.db.search(current_embedding, params)
        
        recommendations = []
        for result in results:
            if result.vector_id == current_item:  # Skip self
                continue
                
            vector = self.db.get_vector(result.vector_id)
            item_metadata = vector['metadata']
            
            recommendations.append({
                "id": result.vector_id,
                "title": item_metadata['title'],
                "category": item_metadata['category'],
                "difficulty": item_metadata['difficulty'],
                "relevance_score": result.combined_score,
                "connection": "direct similarity" if result.hop_count == 0 else f"via {result.hop_count}-hop relationship",
                "hop_count": result.hop_count
            })
        
        return recommendations[:5]  # Return top 5
    
    def search_knowledge_base(self, query, search_strategy="balanced"):
        """Search the knowledge base with different strategies"""
        
        # Generate query embedding (in real use, use same model as content)
        query_embedding = np.random.rand(self.db.dimension() or 384).astype(np.float32)
        
        print(f"🔍 Searching knowledge base: '{query}' (strategy: {search_strategy})")
        
        # Configure search based on strategy
        if search_strategy == "precise":
            params = rudradb.SearchParams(
                top_k=5,
                include_relationships=False,
                similarity_threshold=0.5
            )
        elif search_strategy == "discovery":
            params = rudradb.SearchParams(
                top_k=10,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.6,
                similarity_threshold=0.1
            )
        else:  # balanced
            params = rudradb.SearchParams(
                top_k=7,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.4,
                similarity_threshold=0.2
            )
        
        # Perform search
        results = self.db.search(query_embedding, params)
        
        formatted_results = []
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            item_metadata = vector['metadata']
            
            formatted_results.append({
                "id": result.vector_id,
                "title": item_metadata['title'],
                "category": item_metadata['category'],
                "difficulty": item_metadata['difficulty'],
                "content_preview": item_metadata['content'][:100] + "...",
                "similarity_score": result.similarity_score,
                "combined_score": result.combined_score,
                "connection": "direct match" if result.hop_count == 0 else f"{result.hop_count}-hop connection",
                "hop_count": result.hop_count
            })
        
        return formatted_results
    
    def get_knowledge_base_stats(self):
        """Get comprehensive statistics about the knowledge base"""
        
        stats = self.db.get_statistics()
        usage = stats['capacity_usage']
        
        # Additional content analysis
        difficulty_distribution = {}
        category_distribution = {}
        type_distribution = {}
        
        for item_id in self.db.list_vectors():
            vector = self.db.get_vector(item_id)
            if vector:
                metadata = vector['metadata']
                
                # Count by difficulty
                difficulty = metadata.get('difficulty', 'unknown')
                difficulty_distribution[difficulty] = difficulty_distribution.get(difficulty, 0) + 1
                
                # Count by category
                category = metadata.get('category', 'unknown')
                category_distribution[category] = category_distribution.get(category, 0) + 1
                
                # Count by type
                item_type = metadata.get('type', 'unknown')
                type_distribution[item_type] = type_distribution.get(item_type, 0) + 1
        
        return {
            "database_stats": stats,
            "content_distribution": {
                "by_difficulty": difficulty_distribution,
                "by_category": category_distribution,
                "by_type": type_distribution
            },
            "capacity_usage": {
                "vectors": f"{stats['vector_count']}/{rudradb.MAX_VECTORS} ({usage['vector_usage_percent']:.1f}%)",
                "relationships": f"{stats['relationship_count']}/{rudradb.MAX_RELATIONSHIPS} ({usage['relationship_usage_percent']:.1f}%)"
            },
            "knowledge_metrics": {
                "total_items": stats['vector_count'],
                "total_connections": stats['relationship_count'],
                "categories": len(category_distribution),
                "avg_relationships_per_item": stats['relationship_count'] / max(stats['vector_count'], 1)
            }
        }

def create_sample_knowledge_base():
    """Create a sample AI/ML knowledge base"""
    
    print("🏗️  Building Sample AI/ML Knowledge Base")
    print("=" * 50)
    
    kb = KnowledgeBaseBuilder()
    
    # Sample knowledge base content
    knowledge_items = [
        {
            "id": "ai_introduction",
            "title": "Introduction to Artificial Intelligence",
            "content": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. It encompasses various subfields including machine learning, natural language processing, computer vision, and robotics. AI systems can perform tasks that typically require human intelligence such as visual perception, speech recognition, decision-making, and language translation.",
            "category": "AI",
            "difficulty": "beginner",
            "type": "overview",
            "tags": ["ai", "introduction", "overview", "concepts"],
        },
        {
            "id": "machine_learning_basics",
            "title": "Machine Learning Fundamentals",
            "content": "Machine Learning is a subset of AI that focuses on algorithms that can learn from and make predictions or decisions based on data. Unlike traditional programming where explicit instructions are written, ML systems improve their performance through experience with data.",
            "category": "AI",
            "difficulty": "intermediate",
            "type": "detailed",
            "tags": ["ml", "algorithms", "data", "learning"],
            "prerequisites": ["ai_introduction"]
        },
        {
            "id": "supervised_learning",
            "title": "Supervised Learning Methods",
            "content": "Supervised learning is a type of machine learning where algorithms learn from labeled training data. The goal is to map inputs to correct outputs. Common algorithms include linear regression, decision trees, random forests, and support vector machines.",
            "category": "AI",
            "difficulty": "intermediate",
            "type": "detailed", 
            "tags": ["supervised", "regression", "classification", "algorithms"],
            "prerequisites": ["machine_learning_basics"]
        },
        {
            "id": "deep_learning",
            "title": "Deep Learning and Neural Networks",
            "content": "Deep learning is a specialized subset of machine learning that uses neural networks with multiple hidden layers. These deep neural networks can automatically learn hierarchical representations of data, making them particularly effective for tasks like image recognition, natural language processing, and speech recognition.",
            "category": "AI",
            "difficulty": "advanced",
            "type": "detailed",
            "tags": ["deep_learning", "neural_networks", "layers", "representation"],
            "prerequisites": ["machine_learning_basics", "supervised_learning"]
        },
        {
            "id": "python_for_ml",
            "title": "Python Programming for Machine Learning",
            "content": "Python is the most popular programming language for machine learning due to its extensive ecosystem of libraries like NumPy, Pandas, Scikit-learn, TensorFlow, and PyTorch. This guide covers essential Python skills and libraries needed for ML development.",
            "category": "Programming",
            "difficulty": "intermediate",
            "type": "tutorial",
            "tags": ["python", "programming", "libraries", "tools"],
            "prerequisites": ["machine_learning_basics"]
        },
        {
            "id": "data_preprocessing",
            "title": "Data Preprocessing and Feature Engineering",
            "content": "Data preprocessing is a crucial step in the machine learning pipeline. It involves cleaning data, handling missing values, encoding categorical variables, scaling features, and selecting relevant features. Quality preprocessing often determines the success of ML models.",
            "category": "Data Science",
            "difficulty": "intermediate",
            "type": "tutorial",
            "tags": ["data", "preprocessing", "features", "cleaning"],
            "prerequisites": ["python_for_ml"]
        },
        {
            "id": "model_evaluation",
            "title": "Model Evaluation and Validation",
            "content": "Proper model evaluation is essential for assessing ML model performance. Techniques include train-test splits, cross-validation, confusion matrices, ROC curves, precision, recall, F1-score, and other metrics specific to different types of problems.",
            "category": "Data Science", 
            "difficulty": "intermediate",
            "type": "tutorial",
            "tags": ["evaluation", "validation", "metrics", "performance"],
            "prerequisites": ["supervised_learning", "data_preprocessing"]
        },
        {
            "id": "nlp_introduction",
            "title": "Natural Language Processing Basics",
            "content": "Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. Key concepts include tokenization, part-of-speech tagging, named entity recognition, sentiment analysis, and language modeling.",
            "category": "NLP",
            "difficulty": "intermediate", 
            "type": "overview",
            "tags": ["nlp", "language", "text", "processing"],
            "prerequisites": ["machine_learning_basics"]
        },
        {
            "id": "computer_vision",
            "title": "Computer Vision Fundamentals",
            "content": "Computer Vision enables machines to interpret and analyze visual information from images and videos. Core topics include image preprocessing, feature extraction, object detection, image classification, and convolutional neural networks.",
            "category": "Computer Vision",
            "difficulty": "intermediate",
            "type": "overview", 
            "tags": ["cv", "vision", "images", "cnn"],
            "prerequisites": ["machine_learning_basics"]
        },
        {
            "id": "ml_ethics",
            "title": "Ethics in Machine Learning",
            "content": "As ML systems become more prevalent, ethical considerations become crucial. Topics include bias and fairness, privacy protection, transparency and explainability, accountability, and the social impact of AI systems.",
            "category": "Ethics",
            "difficulty": "advanced",
            "type": "overview",
            "tags": ["ethics", "fairness", "bias", "responsibility"],
            "prerequisites": ["machine_learning_basics"]
        }
    ]
    
    print(f"\n📝 Adding {len(knowledge_items)} knowledge items...")
    
    # Add all items
    total_relationships = 0
    for item in knowledge_items:
        relationships_built = kb.add_knowledge_item(**item)
        print(f"   ✅ Added '{item['title']}' ({relationships_built} relationships)")
        total_relationships += relationships_built
    
    # Show auto-dimension detection result
    print(f"\n🎯 Auto-dimension detection result: {kb.db.dimension()}D")
    print(f"🔗 Total auto-relationships built: {total_relationships}")
    
    return kb

def demonstrate_knowledge_base_features(kb):
    """Demonstrate various knowledge base features"""
    
    print(f"\n🔬 Demonstrating Knowledge Base Features")
    print("=" * 45)
    
    # 1. Learning path discovery
    print(f"\n1️⃣ Learning Path Discovery:")
    learning_paths = ["ai_introduction", "machine_learning_basics", "deep_learning"]
    
    for start_topic in learning_paths:
        if not kb.db.vector_exists(start_topic):
            continue
            
        path = kb.find_learning_path(start_topic, max_steps=2)
        print(f"\n   📚 Learning path from '{start_topic}':")
        
        for step in path[:5]:  # Show first 5 steps
            indent = "      " + "  " * step['hop_count']
            print(f"{indent}📄 {step['title']}")
            print(f"{indent}   Category: {step['category']}, Difficulty: {step['difficulty']}")
            print(f"{indent}   Connection: {step['connection']}")
    
    # 2. Recommendation system
    print(f"\n2️⃣ Recommendation System:")
    test_items = ["machine_learning_basics", "python_for_ml"]
    
    for item in test_items:
        if not kb.db.vector_exists(item):
            continue
            
        print(f"\n   💡 Recommendations for '{item}':")
        
        # Get next steps recommendations
        next_steps = kb.get_recommendations(item, "next_steps")
        if next_steps:
            print(f"      🚀 Next Steps:")
            for rec in next_steps[:3]:
                print(f"         • {rec['title']} ({rec['difficulty']})")
                print(f"           Connection: {rec['connection']}")
        
        # Get related content recommendations
        related = kb.get_recommendations(item, "related")
        if related:
            print(f"      🔗 Related Content:")
            for rec in related[:3]:
                print(f"         • {rec['title']} ({rec['category']})")
    
    # 3. Search functionality
    print(f"\n3️⃣ Search Functionality:")
    
    search_queries = [
        ("machine learning algorithms", "balanced"),
        ("python programming", "discovery"),
        ("neural networks", "precise")
    ]
    
    for query, strategy in search_queries:
        print(f"\n   🔍 Query: '{query}' (strategy: {strategy})")
        
        results = kb.search_knowledge_base(query, strategy)
        for i, result in enumerate(results[:3], 1):
            print(f"      {i}. {result['title']}")
            print(f"         Category: {result['category']}, Difficulty: {result['difficulty']}")
            print(f"         Connection: {result['connection']} (score: {result['combined_score']:.3f})")
    
    # 4. Knowledge base statistics
    print(f"\n4️⃣ Knowledge Base Analytics:")
    stats = kb.get_knowledge_base_stats()
    
    print(f"\n   📊 Content Distribution:")
    print(f"      By Category: {dict(stats['content_distribution']['by_category'])}")
    print(f"      By Difficulty: {dict(stats['content_distribution']['by_difficulty'])}")
    print(f"      By Type: {dict(stats['content_distribution']['by_type'])}")
    
    print(f"\n   📈 Knowledge Metrics:")
    metrics = stats['knowledge_metrics'] 
    print(f"      Total Items: {metrics['total_items']}")
    print(f"      Total Connections: {metrics['total_connections']}")
    print(f"      Categories: {metrics['categories']}")
    print(f"      Avg Relationships/Item: {metrics['avg_relationships_per_item']:.2f}")
    
    print(f"\n   💾 Capacity Usage:")
    usage = stats['capacity_usage']
    print(f"      Vectors: {usage['vectors']}")
    print(f"      Relationships: {usage['relationships']}")

def main():
    """Run the complete knowledge base tutorial"""
    
    print("📚 RudraDB-Opin Knowledge Base Tutorial")
    print("=" * 50)
    
    print("\n🎯 This tutorial demonstrates:")
    features = [
        "Building intelligent knowledge bases with auto-relationships",
        "Learning path discovery through relationship networks",
        "Intelligent recommendation systems",
        "Advanced search with relationship-aware discovery",
        "Content organization and analytics",
        "Scaling within RudraDB-Opin capacity limits"
    ]
    
    for feature in features:
        print(f"   • {feature}")
    
    try:
        # Create sample knowledge base
        kb = create_sample_knowledge_base()
        
        # Demonstrate features
        demonstrate_knowledge_base_features(kb)
        
        # Summary
        print(f"\n🎉 Knowledge Base Tutorial Complete!")
        print("=" * 45)
        
        final_stats = kb.get_knowledge_base_stats()
        capacity = final_stats['capacity_usage']
        
        key_achievements = [
            f"Built knowledge base with {final_stats['knowledge_metrics']['total_items']} items",
            f"Auto-generated {final_stats['knowledge_metrics']['total_connections']} relationships",
            f"Organized content across {final_stats['knowledge_metrics']['categories']} categories",
            f"Demonstrated learning paths and recommendations",
            f"Showcased relationship-aware search capabilities"
        ]
        
        print(f"\n🏆 Key Achievements:")
        for achievement in key_achievements:
            print(f"   ✅ {achievement}")
        
        print(f"\n📊 Final Capacity Usage:")
        print(f"   {capacity['vectors']}")
        print(f"   {capacity['relationships']}")
        
        if "80%" in capacity['vectors'] or "80%" in capacity['relationships']:
            print(f"\n🚀 Ready for Production Scale:")
            print(f"   You've built a substantial knowledge base!")
            print(f"   Ready for unlimited capacity? Upgrade to full RudraDB!")
        
    except Exception as e:
        print(f"\n❌ Tutorial error: {e}")
        print("💡 Make sure rudradb-opin is installed: pip install rudradb-opin")

if __name__ == "__main__":
    main()
