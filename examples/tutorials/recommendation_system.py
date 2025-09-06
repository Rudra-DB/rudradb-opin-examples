#!/usr/bin/env python3
"""
🎬 Smart Recommendation System Tutorial

This example demonstrates building an intelligent recommendation system using
RudraDB-Opin with relationship-aware collaborative and content-based filtering.

Features demonstrated:
- User-item interaction modeling
- Collaborative filtering with relationships
- Content-based recommendations
- Hybrid recommendation strategies
- Cold start problem handling
- Recommendation explanation through relationships
"""

import rudradb
import numpy as np
from datetime import datetime, timedelta
import random
import time

class SmartRecommendationSystem:
    """Intelligent recommendation system with relationship-aware filtering"""
    
    def __init__(self):
        """Initialize the recommendation system with auto-dimension detection"""
        self.db = rudradb.RudraDB()  # Auto-detects embedding dimensions
        self.users = {}
        self.items = {}
        self.interactions = {}
        self.stats = {
            "users_added": 0,
            "items_added": 0,
            "interactions_recorded": 0,
            "recommendations_generated": 0
        }
        
        print("🎬 Smart Recommendation System initialized")
        print("   🎯 Auto-dimension detection enabled")
        print("   🤝 Relationship-aware collaborative filtering ready")
    
    def add_item(self, item_id, title, features, category=None, tags=None, **metadata):
        """Add an item to the recommendation system"""
        
        # Create feature embedding (in real use, derive from actual item features)
        if isinstance(features, list):
            embedding = np.array(features, dtype=np.float32)
        else:
            # Generate embedding from features hash for consistency
            feature_hash = hash(str(features)) % (2**32)
            np.random.seed(feature_hash)
            embedding = np.random.rand(128).astype(np.float32)
            np.random.seed()  # Reset seed
        
        # Store item metadata
        item_metadata = {
            "title": title,
            "category": category or "general",
            "tags": tags or [],
            "features": features,
            "added_at": datetime.now().isoformat(),
            "popularity_score": 0.0,
            "avg_rating": 0.0,
            "interaction_count": 0,
            **metadata
        }
        
        # Add to database - dimension auto-detected on first add
        self.db.add_vector(item_id, embedding, item_metadata)
        self.items[item_id] = item_metadata
        
        # Build content-based relationships with existing items
        relationships_built = self._build_item_relationships(item_id, item_metadata)
        
        self.stats["items_added"] += 1
        
        return {
            "item_id": item_id,
            "relationships_built": relationships_built,
            "dimension_detected": self.db.dimension()
        }
    
    def add_user(self, user_id, preferences=None, demographics=None):
        """Add a user to the system"""
        
        # Create user profile embedding
        if preferences:
            # Generate embedding from preferences
            pref_hash = hash(str(preferences)) % (2**32)
            np.random.seed(pref_hash)
            user_embedding = np.random.rand(self.db.dimension() or 128).astype(np.float32)
            np.random.seed()
        else:
            # Random embedding for new user (cold start)
            user_embedding = np.random.rand(self.db.dimension() or 128).astype(np.float32) * 0.1
        
        user_metadata = {
            "preferences": preferences or {},
            "demographics": demographics or {},
            "added_at": datetime.now().isoformat(),
            "interaction_history": [],
            "is_user": True  # Flag to distinguish from items
        }
        
        # Add user to database
        self.db.add_vector(f"user_{user_id}", user_embedding, user_metadata)
        self.users[user_id] = user_metadata
        
        self.stats["users_added"] += 1
        
        return {"user_id": user_id, "profile_created": True}
    
    def record_interaction(self, user_id, item_id, interaction_type, rating=None, **metadata):
        """Record user-item interaction and build relationships"""
        
        user_vector_id = f"user_{user_id}"
        
        if not self.db.vector_exists(user_vector_id):
            print(f"⚠️  User {user_id} not found, creating profile...")
            self.add_user(user_id)
        
        if not self.db.vector_exists(item_id):
            print(f"⚠️  Item {item_id} not found")
            return False
        
        # Define interaction strength mapping
        interaction_strengths = {
            "view": 0.3,
            "like": 0.7,
            "purchase": 0.9,
            "bookmark": 0.6,
            "share": 0.8,
            "rate": 0.5 + (rating * 0.1 if rating else 0)  # 0.5-1.0 based on rating
        }
        
        strength = interaction_strengths.get(interaction_type, 0.5)
        if rating and interaction_type == "rate":
            strength = 0.2 + (rating / 5.0) * 0.8  # Scale 1-5 rating to 0.2-1.0
        
        # Create interaction relationship
        try:
            self.db.add_relationship(
                user_vector_id, item_id, "causal", strength,
                {
                    "interaction_type": interaction_type,
                    "rating": rating,
                    "timestamp": datetime.now().isoformat(),
                    **metadata
                }
            )
        except RuntimeError as e:
            if "capacity" in str(e).lower():
                print(f"   ⚠️  Relationship capacity reached")
                return False
        
        # Update item statistics
        if item_id in self.items:
            self.items[item_id]["interaction_count"] += 1
            self.items[item_id]["popularity_score"] += strength
            
            # Update item vector metadata
            self.db.update_vector_metadata(item_id, self.items[item_id])
        
        # Update user interaction history
        if user_id in self.users:
            self.users[user_id]["interaction_history"].append({
                "item_id": item_id,
                "interaction_type": interaction_type,
                "rating": rating,
                "timestamp": datetime.now().isoformat()
            })
        
        # Store interaction
        interaction_key = f"{user_id}_{item_id}"
        self.interactions[interaction_key] = {
            "user_id": user_id,
            "item_id": item_id,
            "interaction_type": interaction_type,
            "rating": rating,
            "strength": strength,
            "timestamp": datetime.now().isoformat()
        }
        
        self.stats["interactions_recorded"] += 1
        
        return True
    
    def _build_item_relationships(self, new_item_id, new_metadata):
        """Build content-based relationships between items"""
        
        relationships_built = 0
        max_relationships = 5  # Limit for Opin capacity
        
        # Get existing items for comparison
        for existing_id in self.db.list_vectors():
            if (existing_id == new_item_id or 
                existing_id.startswith("user_") or 
                relationships_built >= max_relationships):
                continue
            
            existing_vector = self.db.get_vector(existing_id)
            if not existing_vector:
                continue
                
            existing_metadata = existing_vector['metadata']
            
            # Skip if this is a user vector
            if existing_metadata.get('is_user'):
                continue
            
            # 1. Semantic: Same category
            if (new_metadata['category'] == existing_metadata.get('category') and
                new_metadata['category'] != "general"):
                try:
                    self.db.add_relationship(
                        new_item_id, existing_id, "semantic", 0.8,
                        {"reason": f"same_category_{new_metadata['category']}"}
                    )
                    relationships_built += 1
                    continue
                except RuntimeError:
                    break
            
            # 2. Associative: Shared tags
            new_tags = set(new_metadata.get('tags', []))
            existing_tags = set(existing_metadata.get('tags', []))
            shared_tags = new_tags & existing_tags
            
            if len(shared_tags) >= 2:
                strength = min(0.7, len(shared_tags) * 0.2)
                try:
                    self.db.add_relationship(
                        new_item_id, existing_id, "associative", strength,
                        {"reason": f"shared_tags", "tags": list(shared_tags)}
                    )
                    relationships_built += 1
                except RuntimeError:
                    break
        
        return relationships_built
    
    def get_recommendations(self, user_id, strategy="hybrid", top_k=5, explanation=True):
        """Get recommendations for a user using specified strategy"""
        
        print(f"🎯 Generating recommendations for user {user_id}")
        print(f"   Strategy: {strategy}, top_k: {top_k}")
        
        user_vector_id = f"user_{user_id}"
        
        if not self.db.vector_exists(user_vector_id):
            print(f"   ❌ User {user_id} not found")
            return []
        
        start_time = time.time()
        
        if strategy == "collaborative":
            recommendations = self._collaborative_filtering(user_vector_id, top_k * 2)
        elif strategy == "content_based":
            recommendations = self._content_based_filtering(user_vector_id, top_k * 2)
        elif strategy == "hybrid":
            collab_recs = self._collaborative_filtering(user_vector_id, top_k)
            content_recs = self._content_based_filtering(user_vector_id, top_k)
            recommendations = self._combine_recommendations(collab_recs, content_recs)
        else:  # popularity-based fallback
            recommendations = self._popularity_based_recommendations(top_k * 2)
        
        # Remove items user has already interacted with
        user_interactions = set()
        if user_id in self.users:
            user_interactions = {
                interaction['item_id'] 
                for interaction in self.users[user_id]['interaction_history']
            }
        
        filtered_recommendations = [
            rec for rec in recommendations 
            if rec['item_id'] not in user_interactions
        ]
        
        # Rank and select top recommendations
        final_recommendations = self._rank_recommendations(
            filtered_recommendations, user_id
        )[:top_k]
        
        # Add explanations if requested
        if explanation:
            for rec in final_recommendations:
                rec['explanation'] = self._explain_recommendation(user_vector_id, rec['item_id'])
        
        generation_time = time.time() - start_time
        self.stats["recommendations_generated"] += 1
        
        return {
            "user_id": user_id,
            "strategy": strategy,
            "recommendations": final_recommendations,
            "generation_time": generation_time,
            "total_candidates": len(recommendations),
            "filtered_candidates": len(filtered_recommendations)
        }
    
    def _collaborative_filtering(self, user_vector_id, top_k):
        """Collaborative filtering using relationship traversal"""
        
        # Find items through user-item relationships and item-item relationships
        connected_items = self.db.get_connected_vectors(user_vector_id, max_hops=2)
        
        recommendations = []
        for vector_data, hop_count in connected_items:
            item_id = vector_data['id']
            metadata = vector_data['metadata']
            
            # Skip user vectors and self
            if metadata.get('is_user') or item_id == user_vector_id:
                continue
            
            # Calculate recommendation score based on relationship path
            if hop_count == 1:
                # Direct user-item relationship
                score = 0.9
            elif hop_count == 2:
                # User -> Item -> Related Item (collaborative signal)
                score = 0.6
            else:
                score = 0.3
            
            # Boost based on item popularity
            popularity_boost = min(0.2, metadata.get('popularity_score', 0) * 0.1)
            score += popularity_boost
            
            recommendations.append({
                "item_id": item_id,
                "title": metadata.get('title', item_id),
                "category": metadata.get('category', 'unknown'),
                "score": score,
                "method": "collaborative",
                "hop_count": hop_count,
                "popularity": metadata.get('popularity_score', 0)
            })
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_k]
    
    def _content_based_filtering(self, user_vector_id, top_k):
        """Content-based filtering using vector similarity"""
        
        user_vector = self.db.get_vector(user_vector_id)
        if not user_vector:
            return []
        
        # Search for similar items based on user profile
        params = rudradb.SearchParams(
            top_k=top_k * 2,
            include_relationships=True,
            max_hops=1,
            relationship_weight=0.3,
            similarity_threshold=0.1
        )
        
        results = self.db.search(user_vector['embedding'], params)
        
        recommendations = []
        for result in results:
            # Skip user vectors
            vector = self.db.get_vector(result.vector_id)
            if not vector or vector['metadata'].get('is_user'):
                continue
            
            metadata = vector['metadata']
            
            recommendations.append({
                "item_id": result.vector_id,
                "title": metadata.get('title', result.vector_id),
                "category": metadata.get('category', 'unknown'),
                "score": result.combined_score,
                "method": "content_based",
                "similarity": result.similarity_score,
                "hop_count": result.hop_count
            })
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_k]
    
    def _popularity_based_recommendations(self, top_k):
        """Fallback popularity-based recommendations"""
        
        item_scores = []
        for item_id, metadata in self.items.items():
            item_scores.append({
                "item_id": item_id,
                "title": metadata.get('title', item_id),
                "category": metadata.get('category', 'unknown'),
                "score": metadata.get('popularity_score', 0),
                "method": "popularity",
                "interaction_count": metadata.get('interaction_count', 0)
            })
        
        return sorted(item_scores, key=lambda x: x['score'], reverse=True)[:top_k]
    
    def _combine_recommendations(self, collab_recs, content_recs):
        """Combine collaborative and content-based recommendations"""
        
        # Create weighted combination
        combined = {}
        
        # Add collaborative recommendations with higher weight
        for rec in collab_recs:
            item_id = rec['item_id']
            combined[item_id] = {
                **rec,
                "score": rec['score'] * 0.7,  # Weight collaborative filtering higher
                "method": "hybrid_collab"
            }
        
        # Add content-based recommendations
        for rec in content_recs:
            item_id = rec['item_id']
            if item_id in combined:
                # Boost score for items found by both methods
                combined[item_id]["score"] += rec['score'] * 0.5
                combined[item_id]["method"] = "hybrid_both"
            else:
                combined[item_id] = {
                    **rec,
                    "score": rec['score'] * 0.5,  # Weight content-based lower
                    "method": "hybrid_content"
                }
        
        return list(combined.values())
    
    def _rank_recommendations(self, recommendations, user_id):
        """Final ranking of recommendations with user-specific adjustments"""
        
        # Apply user-specific adjustments
        user_prefs = self.users.get(user_id, {}).get('preferences', {})
        
        for rec in recommendations:
            # Boost items in preferred categories
            preferred_categories = user_prefs.get('categories', [])
            if rec['category'] in preferred_categories:
                rec['score'] *= 1.2
            
            # Add diversity penalty for items in same category
            # (Simplified - in practice, you'd track category distribution)
            
            # Add recency boost for trending items
            # (Simplified - would use actual timestamp analysis)
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    def _explain_recommendation(self, user_vector_id, item_id):
        """Generate explanation for why an item was recommended"""
        
        # Check if there's a direct relationship
        if self.db.has_relationship(user_vector_id, item_id):
            return "Recommended because you interacted with similar items"
        
        # Check for relationship path
        connected = self.db.get_connected_vectors(user_vector_id, max_hops=2)
        for vector_data, hop_count in connected:
            if vector_data['id'] == item_id:
                if hop_count == 2:
                    return "Recommended based on items liked by similar users"
                else:
                    return f"Found through {hop_count}-hop relationship"
        
        # Fallback explanation
        item_vector = self.db.get_vector(item_id)
        if item_vector:
            category = item_vector['metadata'].get('category', 'unknown')
            return f"Recommended as popular item in {category} category"
        
        return "Recommended based on your profile"
    
    def get_system_stats(self):
        """Get comprehensive recommendation system statistics"""
        
        db_stats = self.db.get_statistics()
        usage = db_stats['capacity_usage']
        
        # Calculate additional metrics
        avg_interactions_per_user = 0
        if self.stats["users_added"] > 0:
            avg_interactions_per_user = self.stats["interactions_recorded"] / self.stats["users_added"]
        
        return {
            "system_stats": self.stats,
            "database_stats": {
                "vectors": f"{db_stats['vector_count']}/{rudradb.MAX_VECTORS}",
                "relationships": f"{db_stats['relationship_count']}/{rudradb.MAX_RELATIONSHIPS}",
                "dimension": db_stats['dimension']
            },
            "capacity_usage": {
                "vector_usage": f"{usage['vector_usage_percent']:.1f}%",
                "relationship_usage": f"{usage['relationship_usage_percent']:.1f}%"
            },
            "recommendation_metrics": {
                "users": len(self.users),
                "items": len(self.items),
                "interactions": len(self.interactions),
                "avg_interactions_per_user": avg_interactions_per_user,
                "avg_relationships_per_vector": db_stats['relationship_count'] / max(db_stats['vector_count'], 1)
            }
        }

def create_sample_movie_recommendation_system():
    """Create a sample movie recommendation system"""
    
    print("🎬 Building Sample Movie Recommendation System")
    print("=" * 55)
    
    rec_system = SmartRecommendationSystem()
    
    # Sample movies with features
    movies = [
        {
            "id": "movie_1",
            "title": "The Matrix",
            "features": [0.9, 0.8, 0.1, 0.7, 0.9],  # [action, sci-fi, romance, thriller, effects]
            "category": "Sci-Fi",
            "tags": ["action", "cyberpunk", "philosophy", "keanu reeves"],
            "year": 1999,
            "director": "Wachowski Sisters"
        },
        {
            "id": "movie_2", 
            "title": "The Godfather",
            "features": [0.6, 0.1, 0.2, 0.9, 0.3],
            "category": "Drama",
            "tags": ["crime", "family", "classic", "marlon brando"],
            "year": 1972,
            "director": "Francis Ford Coppola"
        },
        {
            "id": "movie_3",
            "title": "Blade Runner 2049",
            "features": [0.7, 0.9, 0.3, 0.6, 0.9],
            "category": "Sci-Fi", 
            "tags": ["cyberpunk", "dystopian", "ryan gosling", "sequel"],
            "year": 2017,
            "director": "Denis Villeneuve"
        },
        {
            "id": "movie_4",
            "title": "Casablanca",
            "features": [0.2, 0.1, 0.9, 0.4, 0.2],
            "category": "Romance",
            "tags": ["classic", "wartime", "humphrey bogart", "ingrid bergman"],
            "year": 1942,
            "director": "Michael Curtiz"
        },
        {
            "id": "movie_5",
            "title": "Inception",
            "features": [0.8, 0.8, 0.2, 0.9, 0.8],
            "category": "Sci-Fi",
            "tags": ["dreams", "heist", "christopher nolan", "leonardo dicaprio"],
            "year": 2010,
            "director": "Christopher Nolan"
        },
        {
            "id": "movie_6",
            "title": "Goodfellas", 
            "features": [0.7, 0.1, 0.1, 0.8, 0.4],
            "category": "Crime",
            "tags": ["mafia", "biography", "robert de niro", "martin scorsese"],
            "year": 1990,
            "director": "Martin Scorsese"
        },
        {
            "id": "movie_7",
            "title": "Interstellar",
            "features": [0.5, 0.9, 0.4, 0.7, 0.9],
            "category": "Sci-Fi",
            "tags": ["space", "time travel", "matthew mcconaughey", "christopher nolan"],
            "year": 2014,
            "director": "Christopher Nolan"
        },
        {
            "id": "movie_8",
            "title": "Pulp Fiction",
            "features": [0.8, 0.1, 0.2, 0.8, 0.5],
            "category": "Crime",
            "tags": ["nonlinear", "dialogue", "john travolta", "quentin tarantino"],
            "year": 1994,
            "director": "Quentin Tarantino"
        }
    ]
    
    print(f"\n🎭 Adding {len(movies)} movies...")
    
    # Add movies to the system
    total_relationships = 0
    for movie in movies:
        result = rec_system.add_item(**movie)
        print(f"   ✅ '{movie['title']}' ({result['relationships_built']} relationships)")
        total_relationships += result['relationships_built']
    
    print(f"   🎯 Auto-detected dimension: {rec_system.db.dimension()}D")
    print(f"   🔗 Total content relationships: {total_relationships}")
    
    # Sample users with preferences
    users = [
        {
            "id": "user_1",
            "preferences": {
                "categories": ["Sci-Fi", "Action"],
                "directors": ["Christopher Nolan"],
                "eras": ["Modern"]
            },
            "demographics": {"age": 28, "location": "US"}
        },
        {
            "id": "user_2", 
            "preferences": {
                "categories": ["Drama", "Crime"],
                "directors": ["Martin Scorsese"],
                "eras": ["Classic", "80s-90s"]
            },
            "demographics": {"age": 45, "location": "US"}
        },
        {
            "id": "user_3",
            "preferences": {
                "categories": ["Romance", "Classic"],
                "directors": ["Classic Hollywood"],
                "eras": ["Golden Age"]
            },
            "demographics": {"age": 35, "location": "UK"}
        }
    ]
    
    print(f"\n👥 Adding {len(users)} users...")
    for user in users:
        rec_system.add_user(**user)
        print(f"   ✅ User {user['id']} added")
    
    return rec_system, movies, users

def simulate_user_interactions(rec_system, movies, users):
    """Simulate realistic user interactions"""
    
    print(f"\n🎬 Simulating User Interactions...")
    
    # Define realistic interaction patterns
    interaction_patterns = [
        # User 1 (Sci-Fi fan)
        ("user_1", "movie_1", "like", 5),      # The Matrix
        ("user_1", "movie_3", "purchase", 4),  # Blade Runner 2049
        ("user_1", "movie_5", "like", 5),      # Inception
        ("user_1", "movie_7", "view", 4),      # Interstellar
        ("user_1", "movie_2", "view", 3),      # The Godfather (neutral)
        
        # User 2 (Crime/Drama fan)
        ("user_2", "movie_2", "purchase", 5),  # The Godfather
        ("user_2", "movie_6", "like", 5),      # Goodfellas
        ("user_2", "movie_8", "like", 4),      # Pulp Fiction
        ("user_2", "movie_1", "view", 3),      # The Matrix (neutral)
        ("user_2", "movie_4", "view", 2),      # Casablanca (not preferred)
        
        # User 3 (Classic fan)
        ("user_3", "movie_4", "purchase", 5),  # Casablanca
        ("user_3", "movie_2", "like", 4),      # The Godfather
        ("user_3", "movie_6", "view", 3),      # Goodfellas
        ("user_3", "movie_5", "view", 2),      # Inception (not preferred)
    ]
    
    total_interactions = 0
    for user_id, movie_id, interaction, rating in interaction_patterns:
        success = rec_system.record_interaction(user_id, movie_id, interaction, rating)
        if success:
            total_interactions += 1
            interaction_desc = f"{interaction}"
            if rating:
                interaction_desc += f" (★{rating})"
            print(f"   ✅ {user_id} → {movie_id}: {interaction_desc}")
        else:
            print(f"   ❌ Failed: {user_id} → {movie_id}")
    
    print(f"   📊 Total interactions recorded: {total_interactions}")
    
    return total_interactions

def demonstrate_recommendation_strategies(rec_system):
    """Demonstrate different recommendation strategies"""
    
    print(f"\n🎯 Demonstrating Recommendation Strategies")
    print("=" * 50)
    
    strategies = ["collaborative", "content_based", "hybrid", "popularity"]
    test_users = ["user_1", "user_2", "user_3"]
    
    for user_id in test_users:
        print(f"\n👤 Recommendations for {user_id}:")
        
        user_prefs = rec_system.users[user_id]['preferences']
        print(f"   Preferences: {user_prefs['categories']}")
        
        strategy_results = {}
        
        for strategy in strategies:
            result = rec_system.get_recommendations(
                user_id, strategy=strategy, top_k=3, explanation=True
            )
            
            strategy_results[strategy] = result
            
            print(f"\n   🔍 {strategy.title()} Strategy:")
            print(f"      Time: {result['generation_time']*1000:.2f}ms")
            print(f"      Candidates: {result['total_candidates']} → {result['filtered_candidates']} → {len(result['recommendations'])}")
            
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"      {i}. {rec['title']} ({rec['category']})")
                print(f"         Score: {rec['score']:.3f} | {rec['explanation']}")
        
        # Compare strategy effectiveness
        print(f"\n   📊 Strategy Comparison:")
        for strategy, result in strategy_results.items():
            avg_score = np.mean([r['score'] for r in result['recommendations']]) if result['recommendations'] else 0
            print(f"      {strategy}: {len(result['recommendations'])} recs, avg score: {avg_score:.3f}")

def analyze_recommendation_quality(rec_system):
    """Analyze recommendation quality and system performance"""
    
    print(f"\n📈 Recommendation Quality Analysis")
    print("=" * 40)
    
    # System statistics
    stats = rec_system.get_system_stats()
    
    print(f"\n📊 System Statistics:")
    print(f"   Users: {stats['recommendation_metrics']['users']}")
    print(f"   Items: {stats['recommendation_metrics']['items']}")
    print(f"   Interactions: {stats['recommendation_metrics']['interactions']}")
    print(f"   Avg interactions/user: {stats['recommendation_metrics']['avg_interactions_per_user']:.2f}")
    
    print(f"\n💾 Capacity Usage:")
    print(f"   Vectors: {stats['capacity_usage']['vector_usage']}")
    print(f"   Relationships: {stats['capacity_usage']['relationship_usage']}")
    
    # Cold start analysis
    print(f"\n🆕 Cold Start Analysis:")
    
    # Add a new user with no interactions
    rec_system.add_user("cold_start_user", 
                       preferences={"categories": ["Action", "Thriller"]})
    
    cold_start_recs = rec_system.get_recommendations(
        "cold_start_user", strategy="hybrid", top_k=3
    )
    
    print(f"   New user recommendations: {len(cold_start_recs['recommendations'])}")
    for rec in cold_start_recs['recommendations']:
        print(f"   • {rec['title']} (method: {rec.get('method', 'unknown')})")
    
    # Performance benchmarking
    print(f"\n⚡ Performance Benchmarking:")
    
    benchmark_user = "user_1"
    benchmark_iterations = 10
    
    strategy_times = {}
    for strategy in ["collaborative", "content_based", "hybrid"]:
        times = []
        for _ in range(benchmark_iterations):
            start_time = time.time()
            rec_system.get_recommendations(benchmark_user, strategy=strategy, top_k=5)
            times.append((time.time() - start_time) * 1000)
        
        avg_time = sum(times) / len(times)
        strategy_times[strategy] = avg_time
    
    for strategy, avg_time in strategy_times.items():
        print(f"   {strategy}: {avg_time:.2f}ms average")

def main():
    """Run the complete recommendation system tutorial"""
    
    print("🎬 RudraDB-Opin Smart Recommendation System Tutorial")
    print("=" * 65)
    
    print("\n🎯 This tutorial demonstrates:")
    features = [
        "Building intelligent recommendation systems with relationships",
        "Collaborative filtering through user-item relationships",
        "Content-based filtering with semantic similarity", 
        "Hybrid recommendation strategies",
        "Cold start problem handling",
        "Recommendation explanation through relationship paths",
        "Performance optimization and analysis"
    ]
    
    for feature in features:
        print(f"   • {feature}")
    
    try:
        # Create recommendation system
        rec_system, movies, users = create_sample_movie_recommendation_system()
        
        # Simulate interactions
        simulate_user_interactions(rec_system, movies, users)
        
        # Demonstrate strategies
        demonstrate_recommendation_strategies(rec_system)
        
        # Analyze quality
        analyze_recommendation_quality(rec_system)
        
        # Summary
        print(f"\n🎉 Recommendation System Tutorial Complete!")
        print("=" * 55)
        
        final_stats = rec_system.get_system_stats()
        
        key_achievements = [
            f"Built recommendation system with {final_stats['recommendation_metrics']['users']} users",
            f"Added {final_stats['recommendation_metrics']['items']} items with content relationships",
            f"Recorded {final_stats['recommendation_metrics']['interactions']} user interactions",
            f"Generated {final_stats['system_stats']['recommendations_generated']} recommendation sets",
            f"Demonstrated collaborative, content-based, and hybrid filtering"
        ]
        
        print(f"\n🏆 Key Achievements:")
        for achievement in key_achievements:
            print(f"   ✅ {achievement}")
        
        print(f"\n💡 Advantages of Relationship-Aware Recommendations:")
        advantages = [
            "Discovers items through user similarity networks",
            "Finds content connections beyond simple similarity",
            "Enables multi-hop collaborative filtering",
            "Provides explainable recommendations through relationship paths",
            "Handles cold start with content-based fallbacks",
            "Combines multiple signals for better recommendations"
        ]
        
        for advantage in advantages:
            print(f"   • {advantage}")
        
        print(f"\n📊 Final Capacity Usage:")
        print(f"   Vectors: {final_stats['capacity_usage']['vector_usage']}")
        print(f"   Relationships: {final_stats['capacity_usage']['relationship_usage']}")
        
        if "70%" in final_stats['capacity_usage']['vector_usage']:
            print(f"\n🚀 Ready for Production Scale:")
            print(f"   Your recommendation system shows great potential!")
            print(f"   Ready for millions of users and items? Upgrade to full RudraDB!")
        
    except Exception as e:
        print(f"\n❌ Tutorial error: {e}")
        print("💡 Make sure rudradb-opin is installed: pip install rudradb-opin")

if __name__ == "__main__":
    main()
