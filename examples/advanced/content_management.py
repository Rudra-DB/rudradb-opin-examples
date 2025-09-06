#!/usr/bin/env python3
"""
RudraDB-Opin: Content Management Intelligence System

This example demonstrates building an intelligent content management system using
relationship-aware vector search. Features include:

1. Automatic content relationship detection
2. Intelligent content organization
3. Author network mapping
4. Content series and progressions
5. Tag-based associations
6. Content recommendation engine
7. Publication workflow management

Perfect for learning how to build intelligent content systems!
"""

import rudradb
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json


class ContentManagementSystem:
    """Intelligent content management with relationship-aware search"""
    
    def __init__(self):
        self.db = rudradb.RudraDB()  # Auto-dimension detection
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.content_stats = {
            'total_content': 0,
            'authors': set(),
            'categories': set(),
            'tags': set(),
            'series': set()
        }
        print("📄 RudraDB-Opin Content Management System")
        print("=" * 50)
    
    def add_content(self, content_id: str, title: str, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add content with automatic relationship building"""
        
        # Create comprehensive content metadata
        content_metadata = {
            "title": title,
            "text": text[:1000],  # Store preview
            "content_type": metadata.get("content_type", "article"),
            "author": metadata.get("author", "unknown"),
            "publication_date": metadata.get("publication_date", datetime.now().isoformat()),
            "category": metadata.get("category", "general"),
            "tags": metadata.get("tags", []),
            "series": metadata.get("series"),
            "series_order": metadata.get("series_order"),
            "audience_level": metadata.get("audience_level", "intermediate"),
            "reading_time": len(text.split()) // 200,  # Estimated reading time
            "word_count": len(text.split()),
            "status": metadata.get("status", "published"),
            **metadata
        }
        
        # Generate embedding
        embedding_text = f"{title} {text}"
        embedding = self.model.encode([embedding_text])[0].astype(np.float32)
        
        # Add to database
        self.db.add_vector(content_id, embedding, content_metadata)
        
        # Update statistics
        self._update_stats(content_metadata)
        
        # Auto-build relationships
        relationships_created = self._auto_build_content_relationships(content_id, content_metadata)
        
        print(f"📝 Added '{title}' by {content_metadata['author']}")
        print(f"   Category: {content_metadata['category']}, Type: {content_metadata['content_type']}")
        print(f"   Auto-relationships: {relationships_created}")
        
        return {
            "content_id": content_id,
            "relationships_created": relationships_created,
            "metadata": content_metadata
        }
    
    def _update_stats(self, metadata: Dict[str, Any]):
        """Update content statistics"""
        self.content_stats['total_content'] += 1
        self.content_stats['authors'].add(metadata['author'])
        self.content_stats['categories'].add(metadata['category'])
        self.content_stats['tags'].update(metadata.get('tags', []))
        if metadata.get('series'):
            self.content_stats['series'].add(metadata['series'])
    
    def _auto_build_content_relationships(self, new_content_id: str, metadata: Dict[str, Any]) -> int:
        """Automatically build relationships based on content analysis"""
        relationships_created = 0
        
        category = metadata.get('category')
        tags = set(metadata.get('tags', []))
        author = metadata.get('author')
        series = metadata.get('series')
        series_order = metadata.get('series_order')
        audience_level = metadata.get('audience_level')
        content_type = metadata.get('content_type')
        
        for existing_id in self.db.list_vectors():
            if existing_id == new_content_id or relationships_created >= 8:
                continue
            
            existing_vector = self.db.get_vector(existing_id)
            existing_meta = existing_vector['metadata']
            
            existing_category = existing_meta.get('category')
            existing_tags = set(existing_meta.get('tags', []))
            existing_author = existing_meta.get('author')
            existing_series = existing_meta.get('series')
            existing_series_order = existing_meta.get('series_order')
            existing_level = existing_meta.get('audience_level')
            existing_type = existing_meta.get('content_type')
            
            # 1. Series relationships (hierarchical) - Highest priority
            if (series and series == existing_series and 
                series_order is not None and existing_series_order is not None):
                
                if abs(series_order - existing_series_order) == 1:
                    # Sequential content in series
                    if series_order > existing_series_order:
                        # existing → new (temporal progression)
                        self.db.add_relationship(existing_id, new_content_id, "temporal", 0.95,
                            {"reason": "series_progression", "series": series})
                    else:
                        # new → existing (temporal progression)
                        self.db.add_relationship(new_content_id, existing_id, "temporal", 0.95,
                            {"reason": "series_progression", "series": series})
                    relationships_created += 1
                    print(f"      ⏰ Series progression: {series} ({min(series_order, existing_series_order)} → {max(series_order, existing_series_order)})")
                
                elif series_order != existing_series_order:
                    # Same series, different parts (hierarchical)
                    self.db.add_relationship(new_content_id, existing_id, "hierarchical", 0.8,
                        {"reason": "same_series", "series": series})
                    relationships_created += 1
                    print(f"      📊 Series connection: {series}")
            
            # 2. Author relationships (associative)
            elif author and author == existing_author and author != "unknown":
                self.db.add_relationship(new_content_id, existing_id, "associative", 0.7,
                    {"reason": "same_author", "author": author})
                relationships_created += 1
                print(f"      🏷️ Author connection: {author}")
            
            # 3. Category relationships (semantic)
            elif category and category == existing_category and category != "general":
                self.db.add_relationship(new_content_id, existing_id, "semantic", 0.8,
                    {"reason": "same_category", "category": category})
                relationships_created += 1
                print(f"      🔗 Category connection: {category}")
            
            # 4. Tag overlap (associative)
            elif len(tags & existing_tags) >= 2:
                shared_tags = tags & existing_tags
                strength = min(0.75, len(shared_tags) * 0.2 + 0.4)
                self.db.add_relationship(new_content_id, existing_id, "associative", strength,
                    {"reason": "shared_tags", "tags": list(shared_tags)})
                relationships_created += 1
                print(f"      🏷️ Tag connection: {', '.join(list(shared_tags)[:3])}")
            
            # 5. Content type relationships (semantic)
            elif (content_type and content_type == existing_type and 
                  content_type not in ["article", "general"]):
                self.db.add_relationship(new_content_id, existing_id, "semantic", 0.6,
                    {"reason": "same_content_type", "type": content_type})
                relationships_created += 1
                print(f"      🔗 Content type: {content_type}")
            
            # 6. Learning progression (temporal)
            elif (audience_level and existing_level and category == existing_category):
                levels = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
                current_level = levels.get(audience_level, 2)
                existing_level_num = levels.get(existing_level, 2)
                
                if existing_level_num == current_level - 1:
                    # existing → new (learning progression)
                    self.db.add_relationship(existing_id, new_content_id, "temporal", 0.85,
                        {"reason": "learning_progression", "from": existing_level, "to": audience_level})
                    relationships_created += 1
                    print(f"      ⏰ Learning progression: {existing_level} → {audience_level}")
        
        return relationships_created
    
    def find_content(self, query: str, search_strategy: str = "balanced") -> List[Dict[str, Any]]:
        """Find content using different search strategies"""
        query_embedding = self.model.encode([query])[0].astype(np.float32)
        
        # Define search strategies
        strategies = {
            "precise": rudradb.SearchParams(
                top_k=5,
                include_relationships=False,
                similarity_threshold=0.4
            ),
            "balanced": rudradb.SearchParams(
                top_k=8,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.3,
                similarity_threshold=0.2
            ),
            "discovery": rudradb.SearchParams(
                top_k=12,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.5,
                similarity_threshold=0.1
            ),
            "series_focused": rudradb.SearchParams(
                top_k=10,
                include_relationships=True,
                max_hops=2,
                relationship_types=["temporal", "hierarchical"],
                relationship_weight=0.6
            )
        }
        
        params = strategies.get(search_strategy, strategies["balanced"])
        results = self.db.search(query_embedding, params)
        
        # Format results with content information
        formatted_results = []
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            metadata = vector['metadata']
            
            formatted_results.append({
                "content_id": result.vector_id,
                "title": metadata.get('title', 'Untitled'),
                "author": metadata.get('author'),
                "category": metadata.get('category'),
                "content_type": metadata.get('content_type'),
                "series": metadata.get('series'),
                "audience_level": metadata.get('audience_level'),
                "similarity_score": result.similarity_score,
                "combined_score": result.combined_score,
                "connection_type": "Direct" if result.hop_count == 0 else f"{result.hop_count}-hop",
                "tags": metadata.get('tags', []),
                "reading_time": metadata.get('reading_time', 0)
            })
        
        return formatted_results
    
    def get_author_network(self, author: str, max_connections: int = 10) -> Dict[str, Any]:
        """Get author's content network and collaborations"""
        author_content = []
        
        # Find all content by this author
        for content_id in self.db.list_vectors():
            vector = self.db.get_vector(content_id)
            if vector['metadata'].get('author') == author:
                author_content.append({
                    "content_id": content_id,
                    "title": vector['metadata'].get('title'),
                    "category": vector['metadata'].get('category'),
                    "content_type": vector['metadata'].get('content_type'),
                    "publication_date": vector['metadata'].get('publication_date'),
                    "tags": vector['metadata'].get('tags', [])
                })
        
        # Find related authors through content relationships
        related_authors = {}
        for content in author_content[:5]:  # Limit to avoid too many relationships
            connected = self.db.get_connected_vectors(content["content_id"], max_hops=2)
            
            for vector_data, hop_count in connected:
                other_author = vector_data['metadata'].get('author')
                if other_author and other_author != author and other_author != "unknown":
                    if other_author not in related_authors:
                        related_authors[other_author] = {
                            "connection_count": 0,
                            "shared_categories": set(),
                            "shared_tags": set(),
                            "min_hops": hop_count
                        }
                    
                    related_authors[other_author]["connection_count"] += 1
                    related_authors[other_author]["shared_categories"].add(
                        vector_data['metadata'].get('category', 'general')
                    )
                    related_authors[other_author]["shared_tags"].update(
                        vector_data['metadata'].get('tags', [])
                    )
                    related_authors[other_author]["min_hops"] = min(
                        related_authors[other_author]["min_hops"], hop_count
                    )
        
        # Convert sets to lists for JSON compatibility
        for author_info in related_authors.values():
            author_info["shared_categories"] = list(author_info["shared_categories"])
            author_info["shared_tags"] = list(author_info["shared_tags"])
        
        return {
            "author": author,
            "content_count": len(author_content),
            "author_content": author_content,
            "related_authors": dict(list(related_authors.items())[:max_connections]),
            "categories": list(set(c["category"] for c in author_content)),
            "total_tags": list(set(tag for c in author_content for tag in c["tags"]))
        }
    
    def get_content_series(self, series_name: str) -> Dict[str, Any]:
        """Get complete content series with reading order"""
        series_content = []
        
        for content_id in self.db.list_vectors():
            vector = self.db.get_vector(content_id)
            if vector['metadata'].get('series') == series_name:
                series_content.append({
                    "content_id": content_id,
                    "title": vector['metadata'].get('title'),
                    "series_order": vector['metadata'].get('series_order', 999),
                    "author": vector['metadata'].get('author'),
                    "audience_level": vector['metadata'].get('audience_level'),
                    "reading_time": vector['metadata'].get('reading_time'),
                    "publication_date": vector['metadata'].get('publication_date'),
                    "status": vector['metadata'].get('status', 'published')
                })
        
        # Sort by series order
        series_content.sort(key=lambda x: x['series_order'])
        
        # Find related series through relationships
        related_series = set()
        for content in series_content[:3]:  # Check first few items
            connected = self.db.get_connected_vectors(content["content_id"], max_hops=2)
            for vector_data, hop_count in connected:
                other_series = vector_data['metadata'].get('series')
                if other_series and other_series != series_name:
                    related_series.add(other_series)
        
        return {
            "series_name": series_name,
            "content_count": len(series_content),
            "series_content": series_content,
            "related_series": list(related_series),
            "total_reading_time": sum(c["reading_time"] for c in series_content),
            "authors": list(set(c["author"] for c in series_content)),
            "difficulty_progression": [c["audience_level"] for c in series_content]
        }
    
    def recommend_content(self, content_id: str, recommendation_type: str = "related") -> List[Dict[str, Any]]:
        """Recommend content based on different strategies"""
        base_vector = self.db.get_vector(content_id)
        if not base_vector:
            return []
        
        base_metadata = base_vector['metadata']
        
        # Define recommendation strategies
        if recommendation_type == "related":
            # Find semantically related content
            params = rudradb.SearchParams(
                top_k=8,
                include_relationships=True,
                relationship_types=["semantic", "associative"],
                max_hops=2,
                relationship_weight=0.4
            )
        elif recommendation_type == "progression":
            # Find next steps in learning/series
            params = rudradb.SearchParams(
                top_k=6,
                include_relationships=True,
                relationship_types=["temporal", "hierarchical"],
                max_hops=2,
                relationship_weight=0.6
            )
        elif recommendation_type == "author":
            # Find content by same author or collaborators
            params = rudradb.SearchParams(
                top_k=8,
                include_relationships=True,
                relationship_types=["associative"],
                max_hops=1,
                relationship_weight=0.5
            )
        else:  # comprehensive
            params = rudradb.SearchParams(
                top_k=10,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.4
            )
        
        # Search using base content as query
        results = self.db.search(base_vector['embedding'], params)
        
        # Filter out the original content and format recommendations
        recommendations = []
        for result in results:
            if result.vector_id != content_id:
                vector = self.db.get_vector(result.vector_id)
                metadata = vector['metadata']
                
                recommendations.append({
                    "content_id": result.vector_id,
                    "title": metadata.get('title'),
                    "author": metadata.get('author'),
                    "category": metadata.get('category'),
                    "audience_level": metadata.get('audience_level'),
                    "recommendation_score": result.combined_score,
                    "connection_reason": self._get_connection_reason(base_metadata, metadata, result.hop_count),
                    "reading_time": metadata.get('reading_time'),
                    "content_type": metadata.get('content_type')
                })
        
        return recommendations[:6]  # Top 6 recommendations
    
    def _get_connection_reason(self, base_meta: Dict, target_meta: Dict, hop_count: int) -> str:
        """Determine why content was recommended"""
        if hop_count == 0:
            return "Content similarity"
        
        # Check different connection types
        if base_meta.get('series') == target_meta.get('series'):
            return f"Same series: {base_meta.get('series')}"
        elif base_meta.get('author') == target_meta.get('author'):
            return f"Same author: {base_meta.get('author')}"
        elif base_meta.get('category') == target_meta.get('category'):
            return f"Same category: {base_meta.get('category')}"
        elif set(base_meta.get('tags', [])) & set(target_meta.get('tags', [])):
            shared = set(base_meta.get('tags', [])) & set(target_meta.get('tags', []))
            return f"Shared tags: {', '.join(list(shared)[:2])}"
        else:
            return f"Related through {hop_count}-hop connection"
    
    def get_content_analytics(self) -> Dict[str, Any]:
        """Get comprehensive content analytics"""
        stats = self.db.get_statistics()
        
        # Analyze content distribution
        category_distribution = {}
        author_productivity = {}
        content_type_distribution = {}
        level_distribution = {}
        series_stats = {}
        
        for content_id in self.db.list_vectors():
            vector = self.db.get_vector(content_id)
            metadata = vector['metadata']
            
            # Category distribution
            category = metadata.get('category', 'general')
            category_distribution[category] = category_distribution.get(category, 0) + 1
            
            # Author productivity
            author = metadata.get('author', 'unknown')
            author_productivity[author] = author_productivity.get(author, 0) + 1
            
            # Content type distribution
            content_type = metadata.get('content_type', 'article')
            content_type_distribution[content_type] = content_type_distribution.get(content_type, 0) + 1
            
            # Audience level distribution
            level = metadata.get('audience_level', 'intermediate')
            level_distribution[level] = level_distribution.get(level, 0) + 1
            
            # Series statistics
            series = metadata.get('series')
            if series:
                if series not in series_stats:
                    series_stats[series] = {"count": 0, "authors": set()}
                series_stats[series]["count"] += 1
                series_stats[series]["authors"].add(metadata.get('author', 'unknown'))
        
        # Convert author sets to counts
        for series_info in series_stats.values():
            series_info["authors"] = len(series_info["authors"])
        
        # Calculate relationship statistics
        relationship_types = {"semantic": 0, "hierarchical": 0, "temporal": 0, "causal": 0, "associative": 0}
        for content_id in self.db.list_vectors():
            relationships = self.db.get_relationships(content_id)
            for rel in relationships:
                rel_type = rel["relationship_type"]
                if rel_type in relationship_types:
                    relationship_types[rel_type] += 1
        
        return {
            "database_stats": {
                "total_content": stats['vector_count'],
                "total_relationships": stats['relationship_count'],
                "dimension": stats['dimension']
            },
            "content_distribution": {
                "by_category": category_distribution,
                "by_author": dict(sorted(author_productivity.items(), key=lambda x: x[1], reverse=True)[:10]),
                "by_content_type": content_type_distribution,
                "by_audience_level": level_distribution
            },
            "series_analysis": series_stats,
            "relationship_analysis": relationship_types,
            "top_authors": list(dict(sorted(author_productivity.items(), key=lambda x: x[1], reverse=True)[:5]).keys()),
            "most_connected_categories": list(dict(sorted(category_distribution.items(), key=lambda x: x[1], reverse=True)[:5]).keys())
        }
    
    def demonstrate_content_workflow(self):
        """Demonstrate complete content management workflow"""
        print(f"\n📋 CONTENT MANAGEMENT WORKFLOW DEMO")
        print(f"=" * 45)
        
        # Sample content for demonstration
        sample_content = [
            {
                "id": "python_basics_1",
                "title": "Python Programming Fundamentals",
                "text": "Learn Python programming from scratch with variables, data types, control structures, and functions. Perfect for beginners starting their programming journey.",
                "metadata": {
                    "author": "Dr. Sarah Johnson",
                    "category": "Programming",
                    "content_type": "tutorial",
                    "series": "Python Mastery",
                    "series_order": 1,
                    "audience_level": "beginner",
                    "tags": ["python", "programming", "fundamentals", "beginner"],
                    "publication_date": (datetime.now() - timedelta(days=30)).isoformat()
                }
            },
            {
                "id": "python_basics_2", 
                "title": "Python Data Structures and Algorithms",
                "text": "Master Python lists, dictionaries, sets, and tuples. Learn algorithmic thinking and problem-solving with Python's built-in data structures.",
                "metadata": {
                    "author": "Dr. Sarah Johnson",
                    "category": "Programming", 
                    "content_type": "tutorial",
                    "series": "Python Mastery",
                    "series_order": 2,
                    "audience_level": "beginner",
                    "tags": ["python", "data-structures", "algorithms", "lists"],
                    "publication_date": (datetime.now() - timedelta(days=25)).isoformat()
                }
            },
            {
                "id": "python_advanced",
                "title": "Advanced Python: Decorators and Metaclasses",
                "text": "Explore advanced Python concepts including decorators, context managers, metaclasses, and design patterns for professional development.",
                "metadata": {
                    "author": "Dr. Sarah Johnson", 
                    "category": "Programming",
                    "content_type": "tutorial",
                    "series": "Python Mastery", 
                    "series_order": 3,
                    "audience_level": "advanced",
                    "tags": ["python", "decorators", "metaclasses", "advanced"],
                    "publication_date": (datetime.now() - timedelta(days=20)).isoformat()
                }
            },
            {
                "id": "ml_intro_article",
                "title": "Introduction to Machine Learning Concepts",
                "text": "Understand machine learning fundamentals, supervised vs unsupervised learning, and common algorithms. Great starting point for ML journey.",
                "metadata": {
                    "author": "Prof. Michael Chen",
                    "category": "Machine Learning",
                    "content_type": "article",
                    "audience_level": "intermediate",
                    "tags": ["machine-learning", "ai", "algorithms", "introduction"],
                    "publication_date": (datetime.now() - timedelta(days=15)).isoformat()
                }
            },
            {
                "id": "ml_python_tutorial",
                "title": "Machine Learning with Python and Scikit-learn",
                "text": "Hands-on tutorial for implementing machine learning algorithms using Python and scikit-learn. Build your first ML models step by step.",
                "metadata": {
                    "author": "Prof. Michael Chen",
                    "category": "Machine Learning",
                    "content_type": "tutorial", 
                    "audience_level": "intermediate",
                    "tags": ["python", "machine-learning", "scikit-learn", "tutorial"],
                    "publication_date": (datetime.now() - timedelta(days=10)).isoformat()
                }
            },
            {
                "id": "data_viz_guide",
                "title": "Data Visualization Best Practices",
                "text": "Learn effective data visualization techniques using Python matplotlib and seaborn. Create compelling charts and graphs that tell stories.",
                "metadata": {
                    "author": "Dr. Sarah Johnson",
                    "category": "Data Science",
                    "content_type": "guide",
                    "audience_level": "intermediate", 
                    "tags": ["data-visualization", "python", "matplotlib", "charts"],
                    "publication_date": (datetime.now() - timedelta(days=5)).isoformat()
                }
            }
        ]
        
        # Add all content
        print(f"\n1️⃣ Adding Content to CMS...")
        for content in sample_content:
            result = self.add_content(
                content["id"], 
                content["title"],
                content["text"], 
                content["metadata"]
            )
        
        # Demonstrate search functionality
        print(f"\n2️⃣ Content Search Demonstrations...")
        
        search_queries = [
            ("python programming tutorials", "balanced"),
            ("machine learning for beginners", "discovery"), 
            ("advanced programming concepts", "series_focused")
        ]
        
        for query, strategy in search_queries:
            print(f"\n   Query: '{query}' (strategy: {strategy})")
            results = self.find_content(query, strategy)
            
            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results[:3], 1):
                print(f"      {i}. {result['title']}")
                print(f"         Author: {result['author']}, Level: {result['audience_level']}")
                print(f"         Connection: {result['connection_type']} (score: {result['combined_score']:.3f})")
        
        # Demonstrate author network
        print(f"\n3️⃣ Author Network Analysis...")
        author_network = self.get_author_network("Dr. Sarah Johnson")
        print(f"   Author: {author_network['author']}")
        print(f"   Content Count: {author_network['content_count']}")
        print(f"   Categories: {', '.join(author_network['categories'])}")
        print(f"   Related Authors: {len(author_network['related_authors'])}")
        
        for related_author, info in list(author_network['related_authors'].items())[:2]:
            print(f"      • {related_author}: {info['connection_count']} connections")
        
        # Demonstrate series functionality
        print(f"\n4️⃣ Content Series Analysis...")
        series_info = self.get_content_series("Python Mastery")
        print(f"   Series: {series_info['series_name']}")
        print(f"   Content Count: {series_info['content_count']}")
        print(f"   Total Reading Time: {series_info['total_reading_time']} minutes")
        print(f"   Difficulty Progression: {' → '.join(series_info['difficulty_progression'])}")
        
        print(f"   Reading Order:")
        for i, content in enumerate(series_info['series_content'], 1):
            print(f"      {i}. {content['title']} ({content['audience_level']})")
        
        # Demonstrate content recommendations
        print(f"\n5️⃣ Content Recommendation System...")
        recommendations = self.recommend_content("python_basics_1", "progression")
        print(f"   Recommendations for 'Python Programming Fundamentals':")
        
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"      {i}. {rec['title']}")
            print(f"         Reason: {rec['connection_reason']}")
            print(f"         Score: {rec['recommendation_score']:.3f}")
        
        # Show analytics
        print(f"\n6️⃣ Content Analytics Dashboard...")
        analytics = self.get_content_analytics()
        
        print(f"   📊 Database Overview:")
        db_stats = analytics['database_stats']
        print(f"      Total Content: {db_stats['total_content']}")
        print(f"      Total Relationships: {db_stats['total_relationships']}")
        
        print(f"   📈 Content Distribution:")
        for category, count in analytics['content_distribution']['by_category'].items():
            print(f"      {category}: {count} items")
        
        print(f"   🔗 Relationship Types:")
        for rel_type, count in analytics['relationship_analysis'].items():
            print(f"      {rel_type.capitalize()}: {count}")


def main():
    """Run the complete content management system demo"""
    cms = ContentManagementSystem()
    
    try:
        # Run the complete workflow demonstration
        cms.demonstrate_content_workflow()
        
        print(f"\n🎉 Content Management System Demo Complete!")
        print(f"   ✅ Intelligent content organization")
        print(f"   ✅ Automatic relationship detection")
        print(f"   ✅ Author network mapping")
        print(f"   ✅ Series and progression tracking")
        print(f"   ✅ Advanced search and recommendations")
        print(f"   ✅ Comprehensive analytics")
        print(f"\n💡 Ready to build intelligent content systems with RudraDB-Opin!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print(f"💡 Make sure you have sentence-transformers installed:")
        print(f"   pip install sentence-transformers")


if __name__ == "__main__":
    main()
