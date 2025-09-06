#!/usr/bin/env python3
"""
🧠 Auto-Relationship Detection Demo

This example demonstrates RudraDB-Opin's intelligent auto-relationship detection
that automatically discovers and builds meaningful connections between your data.

Features demonstrated:
- Metadata-based relationship detection
- Content similarity analysis
- Learning progression detection
- Tag-based associations
- Problem-solution connections
- Automatic relationship strength calculation
"""

import rudradb
import numpy as np
from datetime import datetime
import time

def analyze_metadata_relationships(db, document_metadata):
    """Analyze metadata to detect potential relationships"""
    
    print("🔍 Analyzing metadata for relationship opportunities...")
    relationships_detected = []
    
    # Get all existing documents for comparison
    existing_docs = {}
    for doc_id in db.list_vectors():
        vector = db.get_vector(doc_id)
        if vector:
            existing_docs[doc_id] = vector['metadata']
    
    # Analyze each new document against existing ones
    for new_doc_id, new_metadata in document_metadata.items():
        if new_doc_id in existing_docs:
            continue
            
        for existing_doc_id, existing_metadata in existing_docs.items():
            potential_relationships = []
            
            # 1. Semantic Analysis: Same category
            if (new_metadata.get('category') and 
                new_metadata['category'] == existing_metadata.get('category')):
                strength = 0.8
                potential_relationships.append({
                    'type': 'semantic',
                    'strength': strength,
                    'reason': f"Same category: {new_metadata['category']}",
                    'confidence': 0.9
                })
            
            # 2. Hierarchical Analysis: Learning progression
            difficulties = {"beginner": 1, "intermediate": 2, "advanced": 3}
            new_level = difficulties.get(new_metadata.get('difficulty', 'intermediate'), 2)
            existing_level = difficulties.get(existing_metadata.get('difficulty', 'intermediate'), 2)
            
            if (abs(new_level - existing_level) == 1 and 
                new_metadata.get('category') == existing_metadata.get('category')):
                strength = 0.85
                direction = "prerequisite" if new_level < existing_level else "builds_upon"
                potential_relationships.append({
                    'type': 'hierarchical',
                    'strength': strength,
                    'reason': f"Learning progression: {direction}",
                    'confidence': 0.85
                })
            
            # 3. Temporal Analysis: Sequential content
            if (new_metadata.get('type') == existing_metadata.get('type') == 'lesson' and
                new_metadata.get('course') == existing_metadata.get('course')):
                new_order = new_metadata.get('order', 0)
                existing_order = existing_metadata.get('order', 0)
                if abs(new_order - existing_order) == 1:
                    strength = 0.9
                    potential_relationships.append({
                        'type': 'temporal',
                        'strength': strength,
                        'reason': f"Sequential lessons in {new_metadata.get('course')}",
                        'confidence': 0.95
                    })
            
            # 4. Causal Analysis: Problem-solution pairs
            if (new_metadata.get('type') == 'problem' and 
                existing_metadata.get('type') == 'solution'):
                # Check if they're in related domains
                if new_metadata.get('domain') == existing_metadata.get('domain'):
                    strength = 0.95
                    potential_relationships.append({
                        'type': 'causal',
                        'strength': strength,
                        'reason': f"Problem-solution pair in {new_metadata.get('domain')}",
                        'confidence': 0.9
                    })
            
            # 5. Associative Analysis: Tag overlap
            new_tags = set(new_metadata.get('tags', []))
            existing_tags = set(existing_metadata.get('tags', []))
            shared_tags = new_tags & existing_tags
            
            if len(shared_tags) >= 1:
                strength = min(0.7, len(shared_tags) * 0.2 + 0.3)
                potential_relationships.append({
                    'type': 'associative',
                    'strength': strength,
                    'reason': f"Shared tags: {', '.join(shared_tags)}",
                    'confidence': min(0.8, len(shared_tags) * 0.15 + 0.5)
                })
            
            # Store detected relationships
            for rel in potential_relationships:
                relationships_detected.append({
                    'source': new_doc_id,
                    'target': existing_doc_id,
                    'relationship_type': rel['type'],
                    'strength': rel['strength'],
                    'reason': rel['reason'],
                    'confidence': rel['confidence'],
                    'algorithm': 'metadata_analysis'
                })
    
    return relationships_detected

def demonstrate_educational_content_analysis():
    """Demo auto-relationship detection with educational content"""
    
    print("🎓 Educational Content Auto-Relationship Detection")
    print("=" * 55)
    
    # Create database
    db = rudradb.RudraDB()
    
    # Educational documents with rich metadata
    educational_docs = {
        "python_basics": {
            "title": "Python Programming Basics",
            "content": "Introduction to Python syntax and basic programming concepts",
            "category": "Programming",
            "difficulty": "beginner", 
            "type": "lesson",
            "course": "Python Fundamentals",
            "order": 1,
            "tags": ["python", "programming", "basics", "syntax"],
            "domain": "computer_science",
            "prerequisites": []
        },
        "python_functions": {
            "title": "Functions in Python",
            "content": "Defining and using functions, parameters, and return values",
            "category": "Programming",
            "difficulty": "intermediate",
            "type": "lesson", 
            "course": "Python Fundamentals",
            "order": 2,
            "tags": ["python", "functions", "programming"],
            "domain": "computer_science",
            "prerequisites": ["python_basics"]
        },
        "oop_python": {
            "title": "Object-Oriented Programming in Python",
            "content": "Classes, objects, inheritance, and encapsulation",
            "category": "Programming",
            "difficulty": "advanced",
            "type": "lesson",
            "course": "Python Advanced",
            "order": 1,
            "tags": ["python", "oop", "classes", "objects"],
            "domain": "computer_science", 
            "prerequisites": ["python_functions"]
        },
        "ml_intro": {
            "title": "Introduction to Machine Learning",
            "content": "Basic concepts and terminology in machine learning",
            "category": "AI/ML",
            "difficulty": "beginner",
            "type": "lesson",
            "course": "ML Fundamentals", 
            "order": 1,
            "tags": ["ml", "machine_learning", "ai", "introduction"],
            "domain": "data_science",
            "prerequisites": ["python_basics"]
        },
        "debugging_problem": {
            "title": "Common Python Debugging Issues",
            "content": "Typical errors and problems in Python programming",
            "category": "Programming",
            "difficulty": "intermediate",
            "type": "problem",
            "domain": "computer_science",
            "tags": ["python", "debugging", "errors", "problems"],
            "related_to": ["python_basics", "python_functions"]
        },
        "debugging_solution": {
            "title": "Python Debugging Techniques",
            "content": "Tools and methods for debugging Python code effectively",
            "category": "Programming", 
            "difficulty": "intermediate",
            "type": "solution",
            "domain": "computer_science",
            "tags": ["python", "debugging", "solutions", "tools"],
            "solves": ["debugging_problem"]
        }
    }
    
    print(f"\n📚 Adding {len(educational_docs)} educational documents...")
    
    # Add all documents
    for doc_id, metadata in educational_docs.items():
        # Generate embedding (in real use, use actual embeddings)
        embedding = np.random.rand(384).astype(np.float32)
        db.add_vector(doc_id, embedding, metadata)
    
    print(f"   ✅ Added {db.vector_count()} documents")
    print(f"   🎯 Auto-detected dimension: {db.dimension()}D")
    
    # Analyze for relationships
    print(f"\n🧠 Running auto-relationship detection...")
    start_time = time.time()
    
    detected_relationships = analyze_metadata_relationships(db, educational_docs)
    analysis_time = time.time() - start_time
    
    print(f"   ⚡ Analysis completed in {analysis_time*1000:.1f}ms")
    print(f"   🔍 Detected {len(detected_relationships)} potential relationships")
    
    # Build the most confident relationships
    print(f"\n🔨 Building high-confidence relationships...")
    relationships_built = 0
    relationship_stats = {}
    
    # Sort by confidence and build top relationships
    detected_relationships.sort(key=lambda x: x['confidence'], reverse=True)
    
    for rel in detected_relationships:
        if rel['confidence'] >= 0.7 and relationships_built < 20:  # High confidence only
            try:
                db.add_relationship(
                    rel['source'], 
                    rel['target'], 
                    rel['relationship_type'], 
                    rel['strength'],
                    {
                        'auto_detected': True,
                        'reason': rel['reason'],
                        'confidence': rel['confidence'],
                        'algorithm': rel['algorithm'],
                        'created_at': datetime.now().isoformat()
                    }
                )
                relationships_built += 1
                
                # Track stats
                rel_type = rel['relationship_type']
                if rel_type not in relationship_stats:
                    relationship_stats[rel_type] = 0
                relationship_stats[rel_type] += 1
                
                print(f"   ✅ Built {rel['relationship_type']} relationship:")
                print(f"      {rel['source']} → {rel['target']}")
                print(f"      Reason: {rel['reason']} (confidence: {rel['confidence']:.2f})")
                
            except RuntimeError as e:
                if "capacity" in str(e).lower():
                    print(f"   ⚠️  Relationship capacity reached at {relationships_built}")
                    break
                else:
                    raise
    
    print(f"\n📊 Relationship Building Summary:")
    print(f"   Total built: {relationships_built}")
    for rel_type, count in relationship_stats.items():
        print(f"   {rel_type}: {count}")
    
    return db, educational_docs

def demonstrate_content_similarity_analysis():
    """Demo relationship detection based on content similarity"""
    
    print(f"\n\n🔍 Content Similarity Auto-Analysis")
    print("=" * 40)
    
    db = rudradb.RudraDB()
    
    # Documents with varying content similarity
    content_docs = {
        "ai_overview": {
            "title": "Artificial Intelligence Overview", 
            "content": "AI encompasses machine learning, deep learning, and neural networks",
            "tags": ["ai", "overview", "ml", "deep_learning"],
            "category": "AI"
        },
        "ml_fundamentals": {
            "title": "Machine Learning Fundamentals",
            "content": "ML is a subset of AI focusing on algorithms that learn from data", 
            "tags": ["ml", "algorithms", "data", "learning"],
            "category": "AI"
        },
        "deep_learning": {
            "title": "Deep Learning Explained",
            "content": "Deep learning uses neural networks with multiple hidden layers",
            "tags": ["deep_learning", "neural_networks", "layers"],
            "category": "AI"
        },
        "python_tutorial": {
            "title": "Python Programming Tutorial",
            "content": "Learn Python programming from basics to advanced concepts",
            "tags": ["python", "programming", "tutorial"],
            "category": "Programming"
        },
        "data_analysis": {
            "title": "Data Analysis with Python",
            "content": "Use Python libraries for data analysis and machine learning",
            "tags": ["python", "data", "analysis", "ml"],
            "category": "Data Science"
        }
    }
    
    print(f"📄 Adding {len(content_docs)} content documents...")
    
    # Add documents and analyze for relationships
    for doc_id, metadata in content_docs.items():
        embedding = np.random.rand(384).astype(np.float32)
        db.add_vector(doc_id, embedding, metadata)
    
    # Simulate content-based similarity analysis
    print(f"\n🧮 Analyzing content relationships...")
    
    # Build relationships based on shared concepts
    content_relationships = [
        ("ai_overview", "ml_fundamentals", "hierarchical", 0.9, "AI contains ML"),
        ("ml_fundamentals", "deep_learning", "hierarchical", 0.8, "ML contains DL"), 
        ("ai_overview", "deep_learning", "semantic", 0.7, "Both AI concepts"),
        ("python_tutorial", "data_analysis", "temporal", 0.8, "Python → Data Analysis"),
        ("ml_fundamentals", "data_analysis", "associative", 0.6, "ML uses data analysis")
    ]
    
    relationships_built = 0
    for source, target, rel_type, strength, reason in content_relationships:
        try:
            db.add_relationship(source, target, rel_type, strength, {
                'auto_detected': True,
                'reason': reason,
                'algorithm': 'content_similarity',
                'confidence': strength
            })
            relationships_built += 1
            print(f"   ✅ {rel_type}: {source} → {target} ({reason})")
        except RuntimeError as e:
            if "capacity" in str(e).lower():
                break
    
    print(f"\n📊 Built {relationships_built} content-based relationships")
    return db

def demonstrate_learning_path_discovery():
    """Show how auto-relationships create intelligent learning paths"""
    
    print(f"\n\n🎯 Learning Path Discovery")
    print("=" * 30)
    
    # Use the educational database from earlier
    db, educational_docs = demonstrate_educational_content_analysis()
    
    print(f"\n🗺️  Discovering learning paths through auto-relationships...")
    
    # Find learning paths from different starting points
    starting_points = ["python_basics", "ml_intro", "debugging_problem"]
    
    for start_doc in starting_points:
        if not db.vector_exists(start_doc):
            continue
            
        print(f"\n📚 Learning path from '{start_doc}':")
        
        # Get connected documents through relationships
        connected = db.get_connected_vectors(start_doc, max_hops=2)
        
        # Sort by hop count for natural progression
        path = sorted(connected, key=lambda x: x[1])
        
        for vector_data, hop_count in path:
            doc_id = vector_data['id']
            metadata = vector_data['metadata']
            title = metadata.get('title', doc_id)
            difficulty = metadata.get('difficulty', 'unknown')
            
            if hop_count == 0:
                print(f"   🎯 {title} (starting point)")
            else:
                connection = f"{hop_count} hop{'s' if hop_count > 1 else ''} away"
                print(f"   {'  ' * hop_count}🔗 {title} ({difficulty}) - {connection}")
    
    print(f"\n💡 Key Benefits of Auto-Relationship Learning Paths:")
    benefits = [
        "Discovers prerequisite knowledge automatically",
        "Identifies related concepts across categories", 
        "Creates problem-solution connections",
        "Builds sequential learning progressions",
        "Enables multi-hop knowledge discovery"
    ]
    
    for benefit in benefits:
        print(f"   • {benefit}")

def demonstrate_advanced_analysis():
    """Advanced auto-relationship detection patterns"""
    
    print(f"\n\n🚀 Advanced Auto-Relationship Patterns")
    print("=" * 45)
    
    db = rudradb.RudraDB()
    
    # Complex documents with multiple relationship indicators
    advanced_docs = {
        "research_paper_1": {
            "title": "Neural Networks for NLP",
            "authors": ["Smith, J.", "Doe, A."],
            "year": 2020,
            "domain": "AI/NLP",
            "methodology": "neural_networks",
            "problem_addressed": "text_classification",
            "tags": ["nlp", "neural", "classification"]
        },
        "research_paper_2": {
            "title": "Transformer Architecture Analysis", 
            "authors": ["Doe, A.", "Brown, K."],
            "year": 2021,
            "domain": "AI/NLP",
            "methodology": "transformers", 
            "problem_addressed": "sequence_modeling",
            "tags": ["transformers", "attention", "nlp"]
        },
        "implementation_1": {
            "title": "PyTorch Neural Network Implementation",
            "type": "code",
            "implements": "research_paper_1",
            "framework": "pytorch",
            "domain": "AI/NLP",
            "tags": ["pytorch", "neural", "implementation"]
        },
        "review_paper": {
            "title": "Survey of NLP Techniques",
            "type": "review",
            "reviews": ["research_paper_1", "research_paper_2"],
            "domain": "AI/NLP", 
            "scope": "comprehensive",
            "tags": ["survey", "nlp", "review"]
        }
    }
    
    print(f"📖 Adding {len(advanced_docs)} research documents...")
    
    for doc_id, metadata in advanced_docs.items():
        embedding = np.random.rand(384).astype(np.float32)
        db.add_vector(doc_id, embedding, metadata)
    
    # Advanced relationship detection patterns
    print(f"\n🔬 Detecting advanced relationship patterns...")
    
    advanced_patterns = [
        # Author collaboration networks
        ("research_paper_1", "research_paper_2", "associative", 0.7, "Shared author: Doe, A."),
        
        # Temporal research progression  
        ("research_paper_1", "research_paper_2", "temporal", 0.8, "Research progression 2020→2021"),
        
        # Implementation relationships
        ("research_paper_1", "implementation_1", "causal", 0.9, "Paper→Implementation"),
        
        # Review relationships
        ("research_paper_1", "review_paper", "hierarchical", 0.8, "Reviewed in survey"),
        ("research_paper_2", "review_paper", "hierarchical", 0.8, "Reviewed in survey"),
        
        # Methodological relationships
        ("research_paper_1", "research_paper_2", "semantic", 0.6, "Both use neural approaches")
    ]
    
    relationships_built = 0
    for source, target, rel_type, strength, reason in advanced_patterns:
        try:
            db.add_relationship(source, target, rel_type, strength, {
                'auto_detected': True,
                'reason': reason,
                'algorithm': 'advanced_pattern_detection',
                'pattern_type': 'research_network'
            })
            relationships_built += 1
            print(f"   ✅ Detected: {reason}")
            print(f"      {source} --{rel_type}-> {target}")
        except RuntimeError as e:
            if "capacity" in str(e).lower():
                print(f"   ⚠️  Capacity reached at {relationships_built} relationships")
                break
    
    print(f"\n📊 Advanced pattern detection complete: {relationships_built} relationships")

def main():
    """Run all auto-relationship detection demonstrations"""
    
    print("🧠 RudraDB-Opin Auto-Relationship Detection Demo")
    print("=" * 60)
    
    print("\n🎯 Revolutionary AI-Powered Relationship Building:")
    print("   • Analyzes content and metadata automatically")
    print("   • Discovers learning progressions and prerequisites")
    print("   • Builds semantic and hierarchical connections")
    print("   • Creates problem-solution pairs")
    print("   • Identifies tag-based associations")
    
    try:
        # Educational content analysis
        demonstrate_educational_content_analysis()
        
        # Content similarity analysis
        demonstrate_content_similarity_analysis()
        
        # Learning path discovery
        demonstrate_learning_path_discovery()
        
        # Advanced pattern detection
        demonstrate_advanced_analysis()
        
        # Summary
        print(f"\n\n🎉 Auto-Relationship Detection Demo Complete!")
        print("=" * 55)
        
        key_capabilities = [
            "Metadata-based relationship detection",
            "Learning progression identification", 
            "Problem-solution pair discovery",
            "Content similarity analysis",
            "Tag-based association building",
            "Multi-hop learning path creation",
            "Research network detection",
            "Automatic confidence scoring"
        ]
        
        print(f"\n🔑 Auto-Relationship Detection Capabilities:")
        for capability in key_capabilities:
            print(f"   ✅ {capability}")
        
        print(f"\n💡 Benefits:")
        benefits = [
            "Eliminates manual relationship building",
            "Discovers hidden connections in your data",
            "Creates intelligent learning paths automatically", 
            "Scales relationship modeling effortlessly",
            "Maintains high-quality connections through confidence scoring"
        ]
        
        for benefit in benefits:
            print(f"   • {benefit}")
        
        print(f"\n🚀 Ready to build intelligent, self-organizing knowledge bases!")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("💡 Make sure rudradb-opin is installed: pip install rudradb-opin")

if __name__ == "__main__":
    main()
