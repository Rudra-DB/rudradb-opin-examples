#!/usr/bin/env python3
"""
E-commerce Intelligence System with RudraDB-Opin

This example demonstrates how to build an intelligent e-commerce system using
RudraDB-Opin's relationship-aware search for product recommendations, customer
behavior analysis, and inventory management with auto-relationship detection.

Requirements:
    pip install rudradb-opin

Usage:
    python ecommerce_intelligence.py
"""

import rudradb
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import random
from datetime import datetime, timedelta


class Ecommerce_Intelligence_System:
    """E-commerce system with auto-relationship product discovery"""
    
    def __init__(self):
        self.db = rudradb.RudraDB()  # Auto-dimension detection for product embeddings
        self.user_profiles = {}
        self.interaction_history = []
        
        print("🛒 E-commerce Intelligence System initialized")
        print("   🎯 Auto-dimension detection for product embeddings")
        print("   🧠 Auto-relationship detection for product networks")
        
        # Product category hierarchy
        self.category_hierarchy = {
            "Electronics": ["Smartphones", "Laptops", "Headphones", "Smart_Home", "Gaming"],
            "Fashion": ["Clothing", "Shoes", "Accessories", "Jewelry", "Bags"],
            "Home": ["Furniture", "Kitchen", "Decor", "Garden", "Tools"],
            "Sports": ["Equipment", "Apparel", "Outdoor", "Fitness", "Team_Sports"],
            "Books": ["Fiction", "Non_Fiction", "Academic", "Children", "Reference"]
        }
        
        # Price ranges for relationship detection
        self.price_ranges = {
            "budget": (0, 50),
            "mid_range": (50, 200),
            "premium": (200, 500),
            "luxury": (500, float('inf'))
        }
    
    def create_product_embedding(self, product_info: Dict[str, Any]) -> np.ndarray:
        """Create product embedding based on features"""
        
        # Simulate product embedding based on features
        # In real implementation, you'd use product descriptions, images, etc.
        
        feature_vector = []
        
        # Category features (one-hot style)
        category = product_info.get("category", "")
        subcategory = product_info.get("subcategory", "")
        
        # Price feature (normalized)
        price = product_info.get("price", 0)
        price_normalized = min(price / 1000, 1.0)  # Normalize to 0-1
        feature_vector.extend([price_normalized] * 50)
        
        # Brand feature (simulated)
        brand = product_info.get("brand", "")
        brand_hash = hash(brand) % 100
        feature_vector.extend([brand_hash / 100] * 50)
        
        # Features and tags
        features = product_info.get("features", [])
        tags = product_info.get("tags", [])
        combined_text = " ".join([category, subcategory] + features + tags)
        
        # Text-based features (simulated)
        text_hash = hash(combined_text) % (2**16)
        np.random.seed(text_hash)
        text_features = np.random.rand(284).tolist()  # Make total 384D
        feature_vector.extend(text_features)
        
        return np.array(feature_vector, dtype=np.float32)
    
    def add_product(self, product_id: str, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add product with automatic relationship building"""
        
        # Create product embedding
        embedding = self.create_product_embedding(product_info)
        
        # Enhanced metadata for auto-relationship detection
        enhanced_metadata = {
            "name": product_info.get("name", ""),
            "description": product_info.get("description", ""),
            "category": product_info.get("category", ""),
            "subcategory": product_info.get("subcategory", ""),
            "brand": product_info.get("brand", ""),
            "price": product_info.get("price", 0),
            "price_range": self._get_price_range(product_info.get("price", 0)),
            "features": product_info.get("features", []),
            "tags": product_info.get("tags", []),
            "target_audience": product_info.get("target_audience", []),
            "use_cases": product_info.get("use_cases", []),
            "rating": product_info.get("rating", 0),
            "review_count": product_info.get("review_count", 0),
            "availability": product_info.get("availability", "in_stock"),
            "seasonal": product_info.get("seasonal", False),
            **product_info
        }
        
        # Add to database
        self.db.add_vector(product_id, embedding, enhanced_metadata)
        
        # 🧠 Auto-detect product relationships
        relationships_created = self._auto_build_product_relationships(product_id, enhanced_metadata)
        
        return {
            "product_id": product_id,
            "relationships_created": relationships_created,
            "total_products": self.db.vector_count(),
            "dimension": self.db.dimension()
        }
    
    def _get_price_range(self, price: float) -> str:
        """Categorize price into ranges"""
        for range_name, (min_price, max_price) in self.price_ranges.items():
            if min_price <= price < max_price:
                return range_name
        return "luxury"
    
    def _auto_build_product_relationships(self, product_id: str, metadata: Dict[str, Any]) -> int:
        """Auto-detect product relationships for e-commerce intelligence"""
        
        relationships_created = 0
        max_relationships = 6  # Limit per product
        
        category = metadata.get("category", "")
        subcategory = metadata.get("subcategory", "")
        brand = metadata.get("brand", "")
        price_range = metadata.get("price_range", "")
        price = metadata.get("price", 0)
        features = set(metadata.get("features", []))
        tags = set(metadata.get("tags", []))
        use_cases = set(metadata.get("use_cases", []))
        target_audience = set(metadata.get("target_audience", []))
        
        for existing_id in self.db.list_vectors():
            if existing_id == product_id or relationships_created >= max_relationships:
                continue
            
            existing_product = self.db.get_vector(existing_id)
            existing_meta = existing_product['metadata']
            
            existing_category = existing_meta.get("category", "")
            existing_subcategory = existing_meta.get("subcategory", "")
            existing_brand = existing_meta.get("brand", "")
            existing_price_range = existing_meta.get("price_range", "")
            existing_price = existing_meta.get("price", 0)
            existing_features = set(existing_meta.get("features", []))
            existing_tags = set(existing_meta.get("tags", []))
            existing_use_cases = set(existing_meta.get("use_cases", []))
            existing_target_audience = set(existing_meta.get("target_audience", []))
            
            # 📊 Hierarchical: Same brand, different categories (brand portfolio)
            if (brand == existing_brand and brand != "" and
                category != existing_category):
                
                self.db.add_relationship(product_id, existing_id, "hierarchical", 0.7,
                    {"reason": "same_brand_portfolio", "brand": brand})
                relationships_created += 1
                print(f"      📊 {product_id} ↔ {existing_id} (brand portfolio: {brand})")
            
            # 🔗 Semantic: Same subcategory, complementary features
            elif (subcategory == existing_subcategory and subcategory != "" and
                  len(features & existing_features) >= 1):
                
                shared_features = features & existing_features
                strength = min(0.85, len(shared_features) * 0.2 + 0.5)
                
                self.db.add_relationship(product_id, existing_id, "semantic", strength,
                    {"reason": "subcategory_with_shared_features", "subcategory": subcategory,
                     "shared_features": list(shared_features)})
                relationships_created += 1
                print(f"      🔗 {product_id} ↔ {existing_id} (subcategory: {subcategory})")
            
            # 🎯 Causal: Accessory/complementary products (problem-solution pattern)
            elif (len(use_cases & existing_use_cases) >= 1 and
                  category != existing_category and
                  abs(price - existing_price) < price * 0.5):  # Similar price range
                
                shared_use_cases = use_cases & existing_use_cases
                self.db.add_relationship(product_id, existing_id, "causal", 0.8,
                    {"reason": "complementary_use_cases", "use_cases": list(shared_use_cases)})
                relationships_created += 1
                print(f"      🎯 {product_id} → {existing_id} (complementary: {shared_use_cases})")
            
            # 🏷️ Associative: Same target audience, price range compatibility
            elif (len(target_audience & existing_target_audience) >= 1 and
                  (price_range == existing_price_range or 
                   abs(self._price_range_to_num(price_range) - self._price_range_to_num(existing_price_range)) <= 1)):
                
                shared_audience = target_audience & existing_target_audience
                strength = min(0.75, len(shared_audience) * 0.25 + 0.25)
                
                self.db.add_relationship(product_id, existing_id, "associative", strength,
                    {"reason": "target_audience_and_price", "audience": list(shared_audience),
                     "price_compatibility": True})
                relationships_created += 1
                print(f"      🏷️ {product_id} ↔ {existing_id} (audience: {shared_audience})")
            
            # ⏰ Temporal: Seasonal/trending products
            elif (metadata.get("seasonal") and existing_meta.get("seasonal") and
                  category == existing_category):
                
                self.db.add_relationship(product_id, existing_id, "temporal", 0.6,
                    {"reason": "seasonal_products", "category": category})
                relationships_created += 1
                print(f"      ⏰ {product_id} ↔ {existing_id} (seasonal: {category})")
            
            # 🔍 Associative: Tag overlap (browsing patterns)
            elif len(tags & existing_tags) >= 2:
                shared_tags = tags & existing_tags
                strength = min(0.6, len(shared_tags) * 0.15 + 0.2)
                
                self.db.add_relationship(product_id, existing_id, "associative", strength,
                    {"reason": "shared_tags", "tags": list(shared_tags)})
                relationships_created += 1
                print(f"      🔍 {product_id} ↔ {existing_id} (tags: {shared_tags})")
        
        return relationships_created
    
    def _price_range_to_num(self, price_range: str) -> int:
        """Convert price range to numeric for comparison"""
        range_map = {"budget": 0, "mid_range": 1, "premium": 2, "luxury": 3}
        return range_map.get(price_range, 1)
    
    def add_user_interaction(self, user_id: str, product_id: str, interaction_type: str, 
                           rating: Optional[float] = None, context: Optional[Dict] = None):
        """Record user interaction with products"""
        
        if not self.db.vector_exists(product_id):
            print(f"⚠️ Product {product_id} not found")
            return
        
        # Store interaction
        interaction = {
            "user_id": user_id,
            "product_id": product_id,
            "interaction_type": interaction_type,  # viewed, purchased, liked, cart_added, reviewed
            "rating": rating,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        self.interaction_history.append(interaction)
        
        # Update user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "interactions": [],
                "preferences": {},
                "purchase_history": [],
                "avg_price_range": None
            }
        
        self.user_profiles[user_id]["interactions"].append(interaction)
        
        if interaction_type == "purchased":
            self.user_profiles[user_id]["purchase_history"].append(product_id)
        
        print(f"   📝 Recorded {interaction_type} interaction: {user_id} → {product_id}")
    
    def get_intelligent_recommendations(self, user_id: str, recommendation_type: str = "comprehensive",
                                     top_k: int = 5) -> Dict[str, Any]:
        """Get personalized recommendations using relationship-aware search"""
        
        user_profile = self.user_profiles.get(user_id, {})
        user_interactions = user_profile.get("interactions", [])
        
        if not user_interactions:
            # Cold start: return popular/trending products
            return self._get_cold_start_recommendations(top_k)
        
        # Get user's recent interactions for context
        recent_interactions = sorted(user_interactions, 
                                   key=lambda x: x["timestamp"], reverse=True)[:5]
        
        # Create composite query from user's interaction history
        query_embeddings = []
        interaction_weights = {
            "purchased": 1.0,
            "liked": 0.8,
            "cart_added": 0.6,
            "reviewed": 0.7,
            "viewed": 0.3
        }
        
        for interaction in recent_interactions:
            if self.db.vector_exists(interaction["product_id"]):
                product_vector = self.db.get_vector(interaction["product_id"])
                embedding = product_vector["embedding"]
                weight = interaction_weights.get(interaction["interaction_type"], 0.5)
                
                # Weight the embedding
                weighted_embedding = embedding * weight
                query_embeddings.append(weighted_embedding)
        
        if not query_embeddings:
            return self._get_cold_start_recommendations(top_k)
        
        # Combine embeddings (weighted average)
        composite_query = np.mean(query_embeddings, axis=0)
        
        # Configure search based on recommendation type
        if recommendation_type == "similar":
            # Similar to what user has interacted with
            search_params = rudradb.SearchParams(
                top_k=top_k * 2,
                include_relationships=False,
                similarity_threshold=0.3
            )
        elif recommendation_type == "discovery":
            # Discover new products through relationships
            search_params = rudradb.SearchParams(
                top_k=top_k * 2,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.7,
                similarity_threshold=0.1,
                relationship_types=["associative", "causal", "semantic"]
            )
        elif recommendation_type == "complementary":
            # Find complementary products
            search_params = rudradb.SearchParams(
                top_k=top_k * 2,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.6,
                relationship_types=["causal", "associative"]
            )
        else:  # comprehensive
            search_params = rudradb.SearchParams(
                top_k=top_k * 2,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.4
            )
        
        # Perform relationship-aware search
        results = self.db.search(composite_query, search_params)
        
        # Filter out products user has already interacted with
        user_product_ids = {interaction["product_id"] for interaction in user_interactions}
        filtered_results = [r for r in results if r.vector_id not in user_product_ids]
        
        # Process and rank recommendations
        recommendations = []
        for result in filtered_results[:top_k]:
            product = self.db.get_vector(result.vector_id)
            
            recommendation = {
                "product_id": result.vector_id,
                "name": product["metadata"].get("name", ""),
                "category": product["metadata"].get("category", ""),
                "brand": product["metadata"].get("brand", ""),
                "price": product["metadata"].get("price", 0),
                "rating": product["metadata"].get("rating", 0),
                "similarity_score": result.similarity_score,
                "combined_score": result.combined_score,
                "discovery_method": "direct_similarity" if result.hop_count == 0 else f"{result.hop_count}-hop_relationship",
                "recommendation_reason": self._get_recommendation_reason(result, user_profile)
            }
            recommendations.append(recommendation)
        
        return {
            "user_id": user_id,
            "recommendation_type": recommendation_type,
            "recommendations": recommendations,
            "user_interaction_count": len(user_interactions),
            "discovery_stats": {
                "total_found": len(results),
                "after_filtering": len(filtered_results),
                "relationship_discoveries": len([r for r in results if r.hop_count > 0])
            }
        }
    
    def _get_cold_start_recommendations(self, top_k: int) -> Dict[str, Any]:
        """Get recommendations for new users (cold start)"""
        
        # Get popular products (simulate with random selection for demo)
        all_products = self.db.list_vectors()
        if not all_products:
            return {"recommendations": [], "reason": "No products available"}
        
        popular_products = random.sample(all_products, min(top_k, len(all_products)))
        
        recommendations = []
        for product_id in popular_products:
            product = self.db.get_vector(product_id)
            recommendations.append({
                "product_id": product_id,
                "name": product["metadata"].get("name", ""),
                "category": product["metadata"].get("category", ""),
                "price": product["metadata"].get("price", 0),
                "rating": product["metadata"].get("rating", 0),
                "recommendation_reason": "Popular product (cold start)"
            })
        
        return {
            "recommendation_type": "cold_start",
            "recommendations": recommendations,
            "reason": "New user - showing popular products"
        }
    
    def _get_recommendation_reason(self, result, user_profile: Dict[str, Any]) -> str:
        """Generate explanation for why product was recommended"""
        
        if result.hop_count == 0:
            return "Similar to products you've liked"
        elif result.hop_count == 1:
            return "Customers also bought (1-hop connection)"
        else:
            return f"Discovered through {result.hop_count}-step product relationships"
    
    def analyze_product_network(self) -> Dict[str, Any]:
        """Analyze the product relationship network"""
        
        stats = self.db.get_statistics()
        
        # Analyze relationship types
        relationship_analysis = {}
        product_centrality = {}
        
        for product_id in self.db.list_vectors():
            relationships = self.db.get_relationships(product_id)
            product_centrality[product_id] = len(relationships)
            
            for rel in relationships:
                rel_type = rel["relationship_type"]
                relationship_analysis[rel_type] = relationship_analysis.get(rel_type, 0) + 1
        
        # Find most connected products (hubs)
        top_connected = sorted(product_centrality.items(), 
                             key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "network_stats": {
                "total_products": stats["vector_count"],
                "total_relationships": stats["relationship_count"],
                "avg_relationships_per_product": stats["relationship_count"] / max(stats["vector_count"], 1),
                "dimension": stats["dimension"]
            },
            "relationship_breakdown": relationship_analysis,
            "top_connected_products": [
                {"product_id": pid, "connection_count": count, 
                 "product_name": self.db.get_vector(pid)["metadata"].get("name", "")}
                for pid, count in top_connected
            ],
            "capacity_usage": stats["capacity_usage"]
        }


def demo_ecommerce_intelligence():
    """Demo e-commerce intelligence system"""
    
    print("🛒 E-commerce Intelligence System Demo")
    print("=" * 45)
    
    # Initialize system
    ecommerce = Ecommerce_Intelligence_System()
    
    # Sample products
    products = [
        {
            "id": "smartphone_pro",
            "info": {
                "name": "SmartPhone Pro X1",
                "description": "High-end smartphone with advanced camera and AI features",
                "category": "Electronics",
                "subcategory": "Smartphones",
                "brand": "TechCorp",
                "price": 899,
                "features": ["5G", "AI_Camera", "Wireless_Charging", "Face_ID"],
                "tags": ["premium", "photography", "connectivity"],
                "target_audience": ["tech_enthusiasts", "professionals"],
                "use_cases": ["photography", "business", "entertainment"],
                "rating": 4.5,
                "review_count": 1250
            }
        },
        {
            "id": "laptop_gaming",
            "info": {
                "name": "Gaming Laptop Ultra",
                "description": "High-performance gaming laptop with RTX graphics",
                "category": "Electronics",
                "subcategory": "Laptops",
                "brand": "GameTech",
                "price": 1599,
                "features": ["RTX_Graphics", "High_Refresh_Display", "RGB_Keyboard"],
                "tags": ["gaming", "performance", "RGB"],
                "target_audience": ["gamers", "content_creators"],
                "use_cases": ["gaming", "video_editing", "streaming"],
                "rating": 4.7,
                "review_count": 890
            }
        },
        {
            "id": "headphones_pro",
            "info": {
                "name": "Pro Audio Headphones",
                "description": "Professional noise-cancelling headphones",
                "category": "Electronics",
                "subcategory": "Headphones",
                "brand": "AudioPro",
                "price": 299,
                "features": ["Noise_Cancelling", "Wireless", "Premium_Audio"],
                "tags": ["audio", "professional", "wireless"],
                "target_audience": ["professionals", "audiophiles"],
                "use_cases": ["music", "work", "travel"],
                "rating": 4.6,
                "review_count": 567
            }
        },
        {
            "id": "smartwatch_fitness",
            "info": {
                "name": "Fitness SmartWatch",
                "description": "Advanced fitness tracking smartwatch",
                "category": "Electronics",
                "subcategory": "Smart_Home",
                "brand": "FitTech",
                "price": 249,
                "features": ["Heart_Rate", "GPS", "Waterproof", "Sleep_Tracking"],
                "tags": ["fitness", "health", "tracking"],
                "target_audience": ["fitness_enthusiasts", "health_conscious"],
                "use_cases": ["fitness", "health_monitoring", "notifications"],
                "rating": 4.3,
                "review_count": 2100
            }
        },
        {
            "id": "phone_case_pro",
            "info": {
                "name": "Pro Protection Phone Case",
                "description": "Premium protective case for smartphones",
                "category": "Electronics",
                "subcategory": "Accessories",
                "brand": "ProtectTech",
                "price": 45,
                "features": ["Drop_Protection", "Wireless_Compatible", "Premium_Material"],
                "tags": ["protection", "accessories", "premium"],
                "target_audience": ["smartphone_users", "professionals"],
                "use_cases": ["protection", "style", "functionality"],
                "rating": 4.4,
                "review_count": 890
            }
        },
        {
            "id": "wireless_charger",
            "info": {
                "name": "Fast Wireless Charger",
                "description": "High-speed wireless charging pad",
                "category": "Electronics",
                "subcategory": "Accessories",
                "brand": "ChargeTech",
                "price": 39,
                "features": ["Fast_Charging", "Universal_Compatible", "LED_Indicator"],
                "tags": ["charging", "wireless", "convenience"],
                "target_audience": ["smartphone_users", "tech_enthusiasts"],
                "use_cases": ["charging", "desk_setup", "bedside"],
                "rating": 4.2,
                "review_count": 1456
            }
        }
    ]
    
    # Add products to the system
    print("\n🏪 Building product catalog...")
    for product in products:
        result = ecommerce.add_product(product["id"], product["info"])
        print(f"   📦 {product['id']}: {result['relationships_created']} auto-relationships")
    
    print(f"\n✅ Product catalog built: {ecommerce.db.vector_count()} products, {ecommerce.db.relationship_count()} relationships")
    
    # Simulate user interactions
    print(f"\n👥 Simulating user interactions...")
    
    interactions = [
        ("user_1", "smartphone_pro", "purchased", 5.0),
        ("user_1", "headphones_pro", "viewed", None),
        ("user_1", "phone_case_pro", "cart_added", None),
        ("user_2", "laptop_gaming", "purchased", 4.8),
        ("user_2", "headphones_pro", "liked", None),
        ("user_2", "smartwatch_fitness", "viewed", None),
        ("user_3", "smartphone_pro", "viewed", None),
        ("user_3", "wireless_charger", "purchased", 4.0),
        ("user_3", "phone_case_pro", "cart_added", None)
    ]
    
    for user_id, product_id, interaction, rating in interactions:
        ecommerce.add_user_interaction(user_id, product_id, interaction, rating)
    
    # Test different recommendation types
    print(f"\n🎯 Testing Intelligent Recommendations:")
    
    recommendation_types = ["comprehensive", "discovery", "complementary"]
    test_users = ["user_1", "user_2", "user_3"]
    
    for user_id in test_users:
        print(f"\n👤 Recommendations for {user_id}:")
        
        for rec_type in recommendation_types:
            recommendations = ecommerce.get_intelligent_recommendations(
                user_id, rec_type, top_k=3
            )
            
            print(f"\n   🔍 {rec_type.title()} Recommendations:")
            
            if recommendations.get("recommendations"):
                discovery_stats = recommendations.get("discovery_stats", {})
                print(f"      📊 Found {discovery_stats.get('total_found', 0)} products, {discovery_stats.get('relationship_discoveries', 0)} through relationships")
                
                for i, rec in enumerate(recommendations["recommendations"], 1):
                    print(f"         {i}. {rec['name']} (${rec['price']})")
                    print(f"            Method: {rec['discovery_method']} | Score: {rec['combined_score']:.3f}")
                    print(f"            Reason: {rec['recommendation_reason']}")
            else:
                print(f"      📭 No recommendations available")
    
    # Analyze product network
    print(f"\n📊 Product Network Analysis:")
    network_analysis = ecommerce.analyze_product_network()
    
    stats = network_analysis["network_stats"]
    print(f"   🏪 Products: {stats['total_products']}")
    print(f"   🔗 Relationships: {stats['total_relationships']}")
    print(f"   📈 Avg connections per product: {stats['avg_relationships_per_product']:.1f}")
    print(f"   🎯 Embedding dimension: {stats['dimension']}D")
    
    if network_analysis["relationship_breakdown"]:
        print(f"   🧠 Relationship types:")
        for rel_type, count in network_analysis["relationship_breakdown"].items():
            print(f"      • {rel_type}: {count}")
    
    if network_analysis["top_connected_products"]:
        print(f"   🌟 Most connected products:")
        for product in network_analysis["top_connected_products"][:3]:
            print(f"      • {product['product_name']}: {product['connection_count']} connections")
    
    # Show capacity usage
    capacity = network_analysis["capacity_usage"]
    print(f"   💾 Capacity: {capacity['vector_usage_percent']:.1f}% products, {capacity['relationship_usage_percent']:.1f}% relationships")
    
    print(f"\n🎉 E-commerce Intelligence System demo complete!")
    print("    🛒 Auto-detected product relationships for intelligent recommendations")
    print("    🎯 Relationship-aware search discovered complementary and similar products")
    print("    👥 Personalized recommendations based on user interaction patterns")
    print("    📊 Network analysis revealed product connection patterns")
    print("    🔗 Multi-hop discovery enabled 'customers also bought' functionality")


if __name__ == "__main__":
    demo_ecommerce_intelligence()
