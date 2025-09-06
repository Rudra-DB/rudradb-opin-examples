#!/usr/bin/env python3
"""
Relationship Quality Analysis and Optimization
==============================================

This example provides comprehensive tools for analyzing and optimizing 
relationship quality in RudraDB-Opin. Essential for research, content
curation, and system optimization.

Features demonstrated:
- Relationship quality metrics and scoring
- Auto-relationship validation and refinement
- Network analysis and visualization (text-based)
- Quality-based relationship pruning
- Optimization recommendations
- Research-grade analytics

Use cases:
- Content quality assessment
- Knowledge graph optimization
- Research validation
- Relationship network analysis

Requirements:
pip install numpy rudradb-opin scipy
"""

import rudradb
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum

# Try importing scipy for advanced analytics
try:
    from scipy import stats
    from scipy.spatial.distance import cosine
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available, using numpy fallbacks for analytics")

class RelationshipQuality(Enum):
    """Relationship quality classifications"""
    EXCELLENT = (0.8, 1.0, "Excellent")
    GOOD = (0.6, 0.8, "Good")
    FAIR = (0.4, 0.6, "Fair")
    POOR = (0.2, 0.4, "Poor")
    VERY_POOR = (0.0, 0.2, "Very Poor")
    
    def __init__(self, min_score: float, max_score: float, label: str):
        self.min_score = min_score
        self.max_score = max_score
        self.label = label
    
    @classmethod
    def from_score(cls, score: float):
        """Get quality level from score"""
        for quality in cls:
            if quality.min_score <= score < quality.max_score:
                return quality
        return cls.VERY_POOR  # Default for edge cases

@dataclass
class RelationshipMetrics:
    """Comprehensive relationship metrics"""
    relationship_id: str
    source_id: str
    target_id: str
    relationship_type: str
    strength: float
    
    # Quality metrics
    semantic_coherence: float
    strength_appropriateness: float
    reciprocity_score: float
    network_centrality: float
    content_relevance: float
    
    # Derived metrics
    overall_quality: float
    quality_level: RelationshipQuality
    confidence_interval: Tuple[float, float]
    
    # Metadata
    auto_detected: bool
    validation_passed: bool
    recommendations: List[str]

class RelationshipQualityAnalyzer:
    """Advanced relationship quality analysis and optimization"""
    
    def __init__(self, db: rudradb.RudraDB):
        self.db = db
        self.metrics_cache: Dict[str, RelationshipMetrics] = {}
        self.network_metrics: Dict[str, Dict[str, float]] = {}
        
        print("🔬 Relationship Quality Analyzer")
        print("=" * 35)
        print(f"📊 Database: {db.vector_count()} vectors, {db.relationship_count()} relationships")
    
    def analyze_all_relationships(self, force_refresh: bool = False) -> Dict[str, RelationshipMetrics]:
        """Comprehensive analysis of all relationships in the database"""
        
        if not force_refresh and self.metrics_cache:
            return self.metrics_cache
        
        print(f"🔍 Analyzing {self.db.relationship_count()} relationships...")
        
        # First pass: collect all relationships and compute network metrics
        all_relationships = []
        for vector_id in self.db.list_vectors():
            relationships = self.db.get_relationships(vector_id)
            all_relationships.extend(relationships)
        
        # Compute network-wide metrics
        self._compute_network_metrics(all_relationships)
        
        # Second pass: analyze each relationship
        analyzed_relationships = {}
        
        for i, relationship in enumerate(all_relationships):
            rel_id = f"{relationship['source_id']}-{relationship['target_id']}-{relationship['relationship_type']}"
            
            if rel_id in analyzed_relationships:
                continue  # Skip duplicates
            
            metrics = self._analyze_single_relationship(relationship)
            analyzed_relationships[rel_id] = metrics
            
            if (i + 1) % 10 == 0:
                print(f"   Analyzed {i + 1}/{len(all_relationships)} relationships...")
        
        self.metrics_cache = analyzed_relationships
        
        # Generate summary statistics
        self._print_analysis_summary(analyzed_relationships)
        
        return analyzed_relationships
    
    def _compute_network_metrics(self, relationships: List[Dict[str, Any]]):
        """Compute network-wide metrics for centrality analysis"""
        
        # Build adjacency information
        connections = defaultdict(set)
        relationship_counts = Counter()
        
        for rel in relationships:
            source = rel['source_id']
            target = rel['target_id']
            
            connections[source].add(target)
            connections[target].add(source)  # Treat as undirected for centrality
            
            relationship_counts[source] += 1
            relationship_counts[target] += 1
        
        # Compute centrality measures
        total_nodes = len(connections)
        
        for node in connections:
            degree = len(connections[node])
            
            # Degree centrality (normalized)
            degree_centrality = degree / max(1, total_nodes - 1) if total_nodes > 1 else 0
            
            # Closeness approximation (simplified)
            # In a full implementation, this would use shortest paths
            closeness_approx = degree / max(1, total_nodes) if total_nodes > 0 else 0
            
            self.network_metrics[node] = {
                'degree_centrality': degree_centrality,
                'closeness_centrality': closeness_approx,
                'relationship_count': relationship_counts[node]
            }
    
    def _analyze_single_relationship(self, relationship: Dict[str, Any]) -> RelationshipMetrics:
        """Analyze a single relationship comprehensively"""
        
        source_id = relationship['source_id']
        target_id = relationship['target_id']
        rel_type = relationship['relationship_type']
        strength = relationship['strength']
        
        # Get vector data
        source_vector = self.db.get_vector(source_id)
        target_vector = self.db.get_vector(target_id)
        
        if not source_vector or not target_vector:
            # Handle missing vectors
            return self._create_invalid_metrics(relationship)
        
        # 1. Semantic Coherence Analysis
        semantic_coherence = self._compute_semantic_coherence(source_vector, target_vector, rel_type)
        
        # 2. Strength Appropriateness
        strength_appropriateness = self._compute_strength_appropriateness(
            source_vector, target_vector, rel_type, strength
        )
        
        # 3. Reciprocity Score
        reciprocity_score = self._compute_reciprocity_score(source_id, target_id, rel_type)
        
        # 4. Network Centrality Impact
        network_centrality = self._compute_network_centrality_impact(source_id, target_id)
        
        # 5. Content Relevance
        content_relevance = self._compute_content_relevance(source_vector, target_vector, rel_type)
        
        # Overall quality score (weighted combination)
        overall_quality = self._compute_overall_quality(
            semantic_coherence, strength_appropriateness, reciprocity_score,
            network_centrality, content_relevance
        )
        
        # Confidence interval estimation
        confidence_interval = self._estimate_confidence_interval(overall_quality, [
            semantic_coherence, strength_appropriateness, content_relevance
        ])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            semantic_coherence, strength_appropriateness, reciprocity_score,
            network_centrality, content_relevance, overall_quality
        )
        
        return RelationshipMetrics(
            relationship_id=f"{source_id}-{target_id}-{rel_type}",
            source_id=source_id,
            target_id=target_id,
            relationship_type=rel_type,
            strength=strength,
            semantic_coherence=semantic_coherence,
            strength_appropriateness=strength_appropriateness,
            reciprocity_score=reciprocity_score,
            network_centrality=network_centrality,
            content_relevance=content_relevance,
            overall_quality=overall_quality,
            quality_level=RelationshipQuality.from_score(overall_quality),
            confidence_interval=confidence_interval,
            auto_detected=relationship.get('metadata', {}).get('auto_detected', False),
            validation_passed=overall_quality >= 0.4,
            recommendations=recommendations
        )
    
    def _compute_semantic_coherence(self, source_vector: Dict, target_vector: Dict, rel_type: str) -> float:
        """Compute semantic coherence between two vectors"""
        
        # Embedding similarity
        source_emb = source_vector['embedding']
        target_emb = target_vector['embedding']
        
        if SCIPY_AVAILABLE:
            embedding_similarity = 1.0 - cosine(source_emb, target_emb)
        else:
            # Cosine similarity using numpy
            dot_product = np.dot(source_emb, target_emb)
            norm_product = np.linalg.norm(source_emb) * np.linalg.norm(target_emb)
            embedding_similarity = dot_product / max(norm_product, 1e-8)
        
        # Metadata coherence
        source_meta = source_vector.get('metadata', {})
        target_meta = target_vector.get('metadata', {})
        
        metadata_coherence = self._compute_metadata_coherence(source_meta, target_meta, rel_type)
        
        # Combine embedding and metadata coherence
        return 0.7 * embedding_similarity + 0.3 * metadata_coherence
    
    def _compute_metadata_coherence(self, source_meta: Dict, target_meta: Dict, rel_type: str) -> float:
        """Compute coherence based on metadata"""
        
        coherence_score = 0.0
        checks = 0
        
        # Category coherence
        source_cat = source_meta.get('category')
        target_cat = target_meta.get('category')
        
        if source_cat and target_cat:
            checks += 1
            if rel_type in ['semantic', 'associative']:
                # Same category expected for semantic relationships
                coherence_score += 1.0 if source_cat == target_cat else 0.3
            elif rel_type in ['hierarchical', 'temporal', 'causal']:
                # Related but not necessarily same category
                coherence_score += 0.8 if source_cat == target_cat else 0.6
        
        # Topic coherence
        source_topics = set(source_meta.get('topics', []))
        target_topics = set(target_meta.get('topics', []))
        
        if source_topics and target_topics:
            checks += 1
            overlap = len(source_topics & target_topics)
            total = len(source_topics | target_topics)
            topic_similarity = overlap / max(total, 1)
            
            if rel_type in ['semantic', 'associative']:
                coherence_score += topic_similarity
            else:
                coherence_score += min(1.0, topic_similarity + 0.3)  # Bonus for other types
        
        # Difficulty coherence (for educational content)
        source_diff = source_meta.get('difficulty')
        target_diff = target_meta.get('difficulty')
        
        if source_diff and target_diff:
            checks += 1
            diff_levels = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
            source_level = diff_levels.get(source_diff, 2)
            target_level = diff_levels.get(target_diff, 2)
            level_diff = abs(source_level - target_level)
            
            if rel_type == 'hierarchical' and level_diff == 1:
                coherence_score += 1.0  # Perfect for hierarchical
            elif rel_type == 'temporal' and level_diff <= 1:
                coherence_score += 0.9  # Good for temporal
            elif level_diff <= 2:
                coherence_score += 0.6  # Acceptable
            else:
                coherence_score += 0.2  # Poor match
        
        return coherence_score / max(checks, 1)
    
    def _compute_strength_appropriateness(self, source_vector: Dict, target_vector: Dict, 
                                        rel_type: str, strength: float) -> float:
        """Assess if the relationship strength is appropriate"""
        
        # Compute expected strength based on various factors
        expected_strength = 0.5  # Default
        
        # Embedding similarity influence
        source_emb = source_vector['embedding']
        target_emb = target_vector['embedding']
        
        if SCIPY_AVAILABLE:
            similarity = 1.0 - cosine(source_emb, target_emb)
        else:
            dot_product = np.dot(source_emb, target_emb)
            norm_product = np.linalg.norm(source_emb) * np.linalg.norm(target_emb)
            similarity = dot_product / max(norm_product, 1e-8)
        
        # Adjust expected strength based on relationship type
        type_multipliers = {
            'hierarchical': 0.9,  # Should be strong
            'causal': 0.8,       # Should be fairly strong
            'temporal': 0.7,     # Moderate strength
            'semantic': 0.6,     # Based on similarity
            'associative': 0.5   # Can be weaker
        }
        
        expected_strength = similarity * type_multipliers.get(rel_type, 0.6)
        
        # Compute appropriateness score
        strength_diff = abs(strength - expected_strength)
        appropriateness = max(0.0, 1.0 - (strength_diff / 0.5))  # Normalize
        
        return appropriateness
    
    def _compute_reciprocity_score(self, source_id: str, target_id: str, rel_type: str) -> float:
        """Compute reciprocity score for the relationship"""
        
        # Check for reverse relationship
        target_relationships = self.db.get_relationships(target_id)
        
        has_reverse = False
        reverse_strength = 0.0
        
        for rel in target_relationships:
            if (rel['target_id'] == source_id and 
                (rel['relationship_type'] == rel_type or 
                 self._are_complementary_types(rel['relationship_type'], rel_type))):
                has_reverse = True
                reverse_strength = rel['strength']
                break
        
        if not has_reverse:
            # No reciprocal relationship
            if rel_type in ['semantic', 'associative']:
                return 0.3  # Expected to be reciprocal
            else:
                return 0.7  # Not expected to be reciprocal
        else:
            # Has reciprocal relationship - score based on strength balance
            return 1.0 - abs(reverse_strength - 0.6) / 0.6  # Normalize around 0.6
    
    def _are_complementary_types(self, type1: str, type2: str) -> bool:
        """Check if two relationship types are complementary"""
        complementary_pairs = {
            ('hierarchical', 'hierarchical'),
            ('semantic', 'semantic'),
            ('associative', 'associative'),
            ('temporal', 'causal'),  # Sequential relationships
            ('causal', 'temporal')
        }
        
        return (type1, type2) in complementary_pairs or (type2, type1) in complementary_pairs
    
    def _compute_network_centrality_impact(self, source_id: str, target_id: str) -> float:
        """Compute the network centrality impact of this relationship"""
        
        source_metrics = self.network_metrics.get(source_id, {})
        target_metrics = self.network_metrics.get(target_id, {})
        
        # Higher centrality nodes contribute more to network quality
        source_centrality = source_metrics.get('degree_centrality', 0)
        target_centrality = target_metrics.get('degree_centrality', 0)
        
        # Average centrality as impact score
        return (source_centrality + target_centrality) / 2.0
    
    def _compute_content_relevance(self, source_vector: Dict, target_vector: Dict, rel_type: str) -> float:
        """Compute content relevance score"""
        
        source_meta = source_vector.get('metadata', {})
        target_meta = target_vector.get('metadata', {})
        
        relevance_score = 0.0
        
        # Text content relevance (if available)
        source_text = source_meta.get('text', '')
        target_text = target_meta.get('text', '')
        
        if source_text and target_text:
            # Simple text relevance based on shared words
            source_words = set(source_text.lower().split())
            target_words = set(target_text.lower().split())
            
            if source_words and target_words:
                overlap = len(source_words & target_words)
                total_unique = len(source_words | target_words)
                text_relevance = overlap / max(total_unique, 1)
                relevance_score += text_relevance * 0.6
        
        # Metadata relevance
        metadata_relevance = self._compute_metadata_coherence(source_meta, target_meta, rel_type)
        relevance_score += metadata_relevance * 0.4
        
        return min(1.0, relevance_score)
    
    def _compute_overall_quality(self, semantic_coherence: float, strength_appropriateness: float,
                               reciprocity_score: float, network_centrality: float, 
                               content_relevance: float) -> float:
        """Compute overall relationship quality score"""
        
        # Weighted combination of all factors
        weights = {
            'semantic_coherence': 0.3,
            'strength_appropriateness': 0.25,
            'content_relevance': 0.25,
            'reciprocity_score': 0.1,
            'network_centrality': 0.1
        }
        
        overall = (
            semantic_coherence * weights['semantic_coherence'] +
            strength_appropriateness * weights['strength_appropriateness'] +
            content_relevance * weights['content_relevance'] +
            reciprocity_score * weights['reciprocity_score'] +
            network_centrality * weights['network_centrality']
        )
        
        return min(1.0, max(0.0, overall))
    
    def _estimate_confidence_interval(self, quality_score: float, 
                                    component_scores: List[float]) -> Tuple[float, float]:
        """Estimate confidence interval for quality score"""
        
        if SCIPY_AVAILABLE and len(component_scores) > 2:
            # Use statistical methods
            std_dev = np.std(component_scores)
            margin = 1.96 * std_dev / np.sqrt(len(component_scores))  # 95% confidence
            
            lower = max(0.0, quality_score - margin)
            upper = min(1.0, quality_score + margin)
        else:
            # Simple heuristic approach
            variance = np.var(component_scores) if component_scores else 0.1
            margin = min(0.2, variance * 2)  # Conservative margin
            
            lower = max(0.0, quality_score - margin)
            upper = min(1.0, quality_score + margin)
        
        return (lower, upper)
    
    def _generate_recommendations(self, semantic_coherence: float, strength_appropriateness: float,
                                reciprocity_score: float, network_centrality: float,
                                content_relevance: float, overall_quality: float) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        if semantic_coherence < 0.5:
            recommendations.append("Low semantic coherence - verify relationship type appropriateness")
        
        if strength_appropriateness < 0.5:
            recommendations.append("Strength may be inappropriate - consider adjusting based on content similarity")
        
        if content_relevance < 0.4:
            recommendations.append("Low content relevance - review if relationship is meaningful")
        
        if reciprocity_score < 0.3 and semantic_coherence > 0.7:
            recommendations.append("Consider adding reciprocal relationship")
        
        if network_centrality < 0.2:
            recommendations.append("Low network impact - relationship between peripheral nodes")
        
        if overall_quality < 0.3:
            recommendations.append("Overall quality is poor - consider removing this relationship")
        elif overall_quality > 0.8:
            recommendations.append("High quality relationship - good for system performance")
        
        return recommendations if recommendations else ["Relationship quality is acceptable"]
    
    def _create_invalid_metrics(self, relationship: Dict[str, Any]) -> RelationshipMetrics:
        """Create metrics for invalid relationships (missing vectors)"""
        
        return RelationshipMetrics(
            relationship_id=f"{relationship['source_id']}-{relationship['target_id']}-{relationship['relationship_type']}",
            source_id=relationship['source_id'],
            target_id=relationship['target_id'],
            relationship_type=relationship['relationship_type'],
            strength=relationship['strength'],
            semantic_coherence=0.0,
            strength_appropriateness=0.0,
            reciprocity_score=0.0,
            network_centrality=0.0,
            content_relevance=0.0,
            overall_quality=0.0,
            quality_level=RelationshipQuality.VERY_POOR,
            confidence_interval=(0.0, 0.0),
            auto_detected=relationship.get('metadata', {}).get('auto_detected', False),
            validation_passed=False,
            recommendations=["Invalid relationship - source or target vector missing"]
        )
    
    def _print_analysis_summary(self, analyzed_relationships: Dict[str, RelationshipMetrics]):
        """Print comprehensive analysis summary"""
        
        if not analyzed_relationships:
            print("No relationships to analyze")
            return
        
        total_count = len(analyzed_relationships)
        
        # Quality distribution
        quality_distribution = Counter()
        auto_detected_count = 0
        validation_passed_count = 0
        
        # Component score statistics
        semantic_scores = []
        strength_scores = []
        content_scores = []
        overall_scores = []
        
        for metrics in analyzed_relationships.values():
            quality_distribution[metrics.quality_level.label] += 1
            
            if metrics.auto_detected:
                auto_detected_count += 1
            
            if metrics.validation_passed:
                validation_passed_count += 1
            
            semantic_scores.append(metrics.semantic_coherence)
            strength_scores.append(metrics.strength_appropriateness)
            content_scores.append(metrics.content_relevance)
            overall_scores.append(metrics.overall_quality)
        
        print(f"\n📊 Relationship Quality Analysis Summary")
        print("=" * 45)
        
        print(f"📈 Quality Distribution:")
        for quality_label, count in quality_distribution.items():
            percentage = (count / total_count) * 100
            print(f"   {quality_label}: {count} ({percentage:.1f}%)")
        
        print(f"\n🔍 Detection & Validation:")
        print(f"   Auto-detected: {auto_detected_count}/{total_count} ({auto_detected_count/total_count*100:.1f}%)")
        print(f"   Validation passed: {validation_passed_count}/{total_count} ({validation_passed_count/total_count*100:.1f}%)")
        
        print(f"\n📊 Component Score Statistics:")
        print(f"   Semantic Coherence: μ={np.mean(semantic_scores):.3f}, σ={np.std(semantic_scores):.3f}")
        print(f"   Strength Appropriateness: μ={np.mean(strength_scores):.3f}, σ={np.std(strength_scores):.3f}")
        print(f"   Content Relevance: μ={np.mean(content_scores):.3f}, σ={np.std(content_scores):.3f}")
        print(f"   Overall Quality: μ={np.mean(overall_scores):.3f}, σ={np.std(overall_scores):.3f}")
    
    def get_quality_report(self, min_quality: float = 0.0) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        
        analyzed_relationships = self.analyze_all_relationships()
        
        # Filter by minimum quality
        filtered_relationships = {
            rel_id: metrics for rel_id, metrics in analyzed_relationships.items()
            if metrics.overall_quality >= min_quality
        }
        
        # Group by relationship type
        by_type = defaultdict(list)
        for metrics in filtered_relationships.values():
            by_type[metrics.relationship_type].append(metrics)
        
        # Generate type-specific statistics
        type_stats = {}
        for rel_type, metrics_list in by_type.items():
            quality_scores = [m.overall_quality for m in metrics_list]
            auto_detected_pct = sum(1 for m in metrics_list if m.auto_detected) / len(metrics_list) * 100
            
            type_stats[rel_type] = {
                'count': len(metrics_list),
                'avg_quality': np.mean(quality_scores),
                'std_quality': np.std(quality_scores),
                'min_quality': np.min(quality_scores),
                'max_quality': np.max(quality_scores),
                'auto_detected_percentage': auto_detected_pct
            }
        
        # Find best and worst relationships
        all_metrics = list(filtered_relationships.values())
        all_metrics.sort(key=lambda m: m.overall_quality, reverse=True)
        
        best_relationships = all_metrics[:5]
        worst_relationships = all_metrics[-5:] if len(all_metrics) > 5 else []
        
        return {
            'total_relationships': len(analyzed_relationships),
            'filtered_relationships': len(filtered_relationships),
            'type_statistics': type_stats,
            'best_relationships': [
                {
                    'id': m.relationship_id,
                    'type': m.relationship_type,
                    'quality': m.overall_quality,
                    'quality_level': m.quality_level.label
                } for m in best_relationships
            ],
            'worst_relationships': [
                {
                    'id': m.relationship_id,
                    'type': m.relationship_type,
                    'quality': m.overall_quality,
                    'quality_level': m.quality_level.label,
                    'recommendations': m.recommendations
                } for m in worst_relationships
            ]
        }
    
    def optimize_relationships(self, quality_threshold: float = 0.3) -> Dict[str, Any]:
        """Optimize relationships by removing low-quality ones"""
        
        analyzed_relationships = self.analyze_all_relationships()
        
        to_remove = []
        to_adjust = []
        high_quality = []
        
        for rel_id, metrics in analyzed_relationships.items():
            if metrics.overall_quality < quality_threshold:
                to_remove.append(metrics)
            elif metrics.overall_quality < 0.6:
                to_adjust.append(metrics)
            else:
                high_quality.append(metrics)
        
        print(f"\n🔧 Relationship Optimization Recommendations")
        print("=" * 45)
        print(f"📊 Analysis: {len(analyzed_relationships)} total relationships")
        print(f"   🗑️  Remove: {len(to_remove)} (quality < {quality_threshold})")
        print(f"   ⚙️  Adjust: {len(to_adjust)} (quality 0.3-0.6)")
        print(f"   ✅ Keep: {len(high_quality)} (quality > 0.6)")
        
        # Show examples of what would be removed
        if to_remove:
            print(f"\n🗑️  Relationships to Remove (lowest quality):")
            to_remove.sort(key=lambda m: m.overall_quality)
            for metrics in to_remove[:3]:
                print(f"   {metrics.source_id} → {metrics.target_id} ({metrics.relationship_type})")
                print(f"      Quality: {metrics.overall_quality:.3f} - {metrics.recommendations[0]}")
        
        # Show adjustment recommendations
        if to_adjust:
            print(f"\n⚙️  Relationships to Adjust:")
            to_adjust.sort(key=lambda m: m.overall_quality)
            for metrics in to_adjust[:3]:
                print(f"   {metrics.source_id} → {metrics.target_id} ({metrics.relationship_type})")
                print(f"      Quality: {metrics.overall_quality:.3f}")
                print(f"      Recommendations: {'; '.join(metrics.recommendations[:2])}")
        
        return {
            'total_analyzed': len(analyzed_relationships),
            'to_remove': len(to_remove),
            'to_adjust': len(to_adjust),
            'high_quality': len(high_quality),
            'removal_candidates': [m.relationship_id for m in to_remove],
            'adjustment_candidates': [m.relationship_id for m in to_adjust],
            'optimization_score': len(high_quality) / max(len(analyzed_relationships), 1)
        }

def demo_relationship_quality_analysis():
    """Comprehensive demonstration of relationship quality analysis"""
    
    print("🔬 Relationship Quality Analysis Demo")
    print("=" * 42)
    
    # Create a database with varying quality relationships
    db = rudradb.RudraDB()
    
    # Add sample documents
    documents = [
        {
            "id": "ai_intro",
            "text": "Introduction to Artificial Intelligence and its applications",
            "metadata": {"category": "AI", "difficulty": "beginner", "topics": ["ai", "introduction"]}
        },
        {
            "id": "ml_basics", 
            "text": "Machine Learning fundamentals and basic algorithms",
            "metadata": {"category": "AI", "difficulty": "intermediate", "topics": ["ml", "algorithms"]}
        },
        {
            "id": "deep_learning",
            "text": "Deep learning with neural networks and backpropagation",
            "metadata": {"category": "AI", "difficulty": "advanced", "topics": ["dl", "neural", "backprop"]}
        },
        {
            "id": "python_basics",
            "text": "Python programming fundamentals and syntax",
            "metadata": {"category": "Programming", "difficulty": "beginner", "topics": ["python", "programming"]}
        },
        {
            "id": "data_science",
            "text": "Data science workflow and statistical analysis",
            "metadata": {"category": "Data", "difficulty": "intermediate", "topics": ["data", "statistics"]}
        }
    ]
    
    # Add documents to database
    for doc in documents:
        embedding = np.random.rand(384).astype(np.float32)  # Mock embeddings
        db.add_vector(doc["id"], embedding, doc["metadata"])
    
    print(f"✅ Added {len(documents)} documents to database")
    
    # Add relationships with varying quality
    relationships = [
        # High quality relationships
        ("ai_intro", "ml_basics", "hierarchical", 0.9),  # Good progression
        ("ml_basics", "deep_learning", "hierarchical", 0.8),  # Logical sequence
        ("ai_intro", "ml_basics", "semantic", 0.7),  # Related content
        
        # Medium quality relationships
        ("python_basics", "data_science", "associative", 0.6),  # Somewhat related
        ("ml_basics", "data_science", "associative", 0.5),  # Cross-domain
        
        # Lower quality relationships
        ("ai_intro", "python_basics", "causal", 0.9),  # Questionable causal link
        ("deep_learning", "python_basics", "temporal", 0.4),  # Weak temporal link
    ]
    
    for source, target, rel_type, strength in relationships:
        db.add_relationship(source, target, rel_type, strength, {
            "created_for": "quality_analysis_demo"
        })
    
    print(f"✅ Added {len(relationships)} relationships")
    print(f"📊 Database: {db.vector_count()} vectors, {db.relationship_count()} relationships")
    
    # Initialize analyzer
    analyzer = RelationshipQualityAnalyzer(db)
    
    # Perform comprehensive analysis
    print(f"\n🔍 Starting comprehensive analysis...")
    analysis_results = analyzer.analyze_all_relationships()
    
    # Generate quality report
    print(f"\n📋 Generating Quality Report...")
    quality_report = analyzer.get_quality_report(min_quality=0.0)
    
    print(f"\n📊 Quality Report Summary:")
    print(f"   Total relationships: {quality_report['total_relationships']}")
    
    print(f"\n📈 By Relationship Type:")
    for rel_type, stats in quality_report['type_statistics'].items():
        print(f"   {rel_type.title()}: {stats['count']} relationships")
        print(f"      Avg Quality: {stats['avg_quality']:.3f} (±{stats['std_quality']:.3f})")
        print(f"      Range: {stats['min_quality']:.3f} - {stats['max_quality']:.3f}")
        print(f"      Auto-detected: {stats['auto_detected_percentage']:.1f}%")
    
    print(f"\n🏆 Best Relationships:")
    for rel in quality_report['best_relationships'][:3]:
        print(f"   {rel['id']}: {rel['quality']:.3f} ({rel['quality_level']})")
    
    print(f"\n⚠️ Worst Relationships:")
    for rel in quality_report['worst_relationships'][:3]:
        print(f"   {rel['id']}: {rel['quality']:.3f} ({rel['quality_level']})")
        print(f"      Recommendation: {rel['recommendations'][0]}")
    
    # Perform optimization analysis
    optimization_results = analyzer.optimize_relationships(quality_threshold=0.4)
    
    print(f"\n🎯 Optimization Results:")
    print(f"   Optimization Score: {optimization_results['optimization_score']:.3f}")
    print(f"   Relationships in good condition: {optimization_results['optimization_score']*100:.1f}%")
    
    return analyzer, quality_report, optimization_results

def demo_research_analytics():
    """Demonstrate advanced analytics for research purposes"""
    
    print(f"\n🔬 Research-Grade Analytics")
    print("=" * 30)
    
    # Create a larger, more complex dataset for research
    db = rudradb.RudraDB()
    analyzer = RelationshipQualityAnalyzer(db)
    
    # Simulate a research dataset
    categories = ["AI", "ML", "NLP", "CV", "Robotics"]
    difficulties = ["beginner", "intermediate", "advanced"]
    
    documents_created = 0
    for i in range(25):  # Within Opin limits
        category = categories[i % len(categories)]
        difficulty = difficulties[i % len(difficulties)]
        
        doc_id = f"research_doc_{i}"
        embedding = np.random.rand(384).astype(np.float32)
        
        metadata = {
            "category": category,
            "difficulty": difficulty,
            "topics": [category.lower(), difficulty, f"topic_{i%7}"],
            "research_area": f"area_{i%5}",
            "publication_year": 2020 + (i % 4)
        }
        
        db.add_vector(doc_id, embedding, metadata)
        documents_created += 1
    
    print(f"📚 Created research dataset: {documents_created} documents")
    
    # Create diverse relationships with metadata
    relationship_patterns = [
        ("semantic", 0.7, "content_similarity"),
        ("hierarchical", 0.8, "knowledge_hierarchy"),
        ("temporal", 0.6, "chronological_order"),
        ("associative", 0.5, "cross_reference"),
        ("causal", 0.7, "prerequisite_knowledge")
    ]
    
    relationships_created = 0
    for i in range(40):  # Create substantial relationship network
        if i >= 40:  # Stay within Opin limits
            break
        
        source_idx = i % documents_created
        target_idx = (i + 1) % documents_created
        
        if source_idx != target_idx:
            pattern = relationship_patterns[i % len(relationship_patterns)]
            rel_type, base_strength, reason = pattern
            
            # Add some variance to strength
            strength = min(1.0, max(0.1, base_strength + np.random.normal(0, 0.15)))
            
            try:
                db.add_relationship(
                    f"research_doc_{source_idx}",
                    f"research_doc_{target_idx}",
                    rel_type,
                    strength,
                    {
                        "auto_detected": i % 3 == 0,  # Mix of auto and manual
                        "reason": reason,
                        "research_context": True,
                        "confidence": np.random.rand()
                    }
                )
                relationships_created += 1
            except Exception as e:
                if "capacity" in str(e).lower():
                    print(f"   Reached relationship capacity at {relationships_created}")
                    break
                else:
                    raise
    
    print(f"🔗 Created {relationships_created} research relationships")
    
    # Perform advanced analysis
    print(f"\n📊 Advanced Research Analytics:")
    
    # Full relationship analysis
    analysis_results = analyzer.analyze_all_relationships()
    
    # Research-specific metrics
    auto_detected_metrics = [m for m in analysis_results.values() if m.auto_detected]
    manual_metrics = [m for m in analysis_results.values() if not m.auto_detected]
    
    if auto_detected_metrics and manual_metrics:
        auto_quality = np.mean([m.overall_quality for m in auto_detected_metrics])
        manual_quality = np.mean([m.overall_quality for m in manual_metrics])
        
        print(f"   🤖 Auto-detected relationships: {len(auto_detected_metrics)} (avg quality: {auto_quality:.3f})")
        print(f"   👤 Manual relationships: {len(manual_metrics)} (avg quality: {manual_quality:.3f})")
        
        quality_difference = manual_quality - auto_quality
        print(f"   📈 Quality difference: {quality_difference:+.3f} (manual vs auto)")
        
        if abs(quality_difference) < 0.1:
            print("   ✅ Auto-detection quality is comparable to manual relationships")
        elif quality_difference > 0.1:
            print("   ⚠️ Manual relationships show higher quality")
        else:
            print("   ⚠️ Auto-detected relationships show higher quality")
    
    # Component analysis
    semantic_scores = [m.semantic_coherence for m in analysis_results.values()]
    strength_scores = [m.strength_appropriateness for m in analysis_results.values()]
    content_scores = [m.content_relevance for m in analysis_results.values()]
    
    print(f"\n📊 Component Correlation Analysis:")
    if SCIPY_AVAILABLE and len(semantic_scores) > 3:
        sem_str_corr = stats.pearsonr(semantic_scores, strength_scores)[0]
        sem_cont_corr = stats.pearsonr(semantic_scores, content_scores)[0]
        str_cont_corr = stats.pearsonr(strength_scores, content_scores)[0]
        
        print(f"   Semantic ↔ Strength: r = {sem_str_corr:.3f}")
        print(f"   Semantic ↔ Content: r = {sem_cont_corr:.3f}")
        print(f"   Strength ↔ Content: r = {str_cont_corr:.3f}")
    else:
        print("   📝 Statistical correlation requires scipy and larger dataset")
    
    # Network analysis
    print(f"\n🌐 Network Structure Analysis:")
    total_nodes = db.vector_count()
    total_relationships = db.relationship_count()
    
    if total_nodes > 1:
        network_density = total_relationships / (total_nodes * (total_nodes - 1))
        print(f"   Network density: {network_density:.4f}")
        
        if network_density > 0.1:
            print("   📊 Dense network - high connectivity")
        elif network_density > 0.05:
            print("   📊 Moderate network connectivity")
        else:
            print("   📊 Sparse network - low connectivity")
    
    # Quality distribution research
    quality_levels = [m.quality_level.label for m in analysis_results.values()]
    quality_dist = Counter(quality_levels)
    
    print(f"\n🔬 Research Quality Distribution:")
    for level, count in quality_dist.items():
        percentage = (count / len(analysis_results)) * 100
        print(f"   {level}: {count} ({percentage:.1f}%)")
    
    # Export research data (simulation)
    research_export = {
        "total_relationships": len(analysis_results),
        "quality_metrics": {
            "mean_quality": np.mean([m.overall_quality for m in analysis_results.values()]),
            "std_quality": np.std([m.overall_quality for m in analysis_results.values()]),
            "quality_distribution": dict(quality_dist)
        },
        "network_metrics": {
            "density": network_density if total_nodes > 1 else 0,
            "node_count": total_nodes,
            "edge_count": total_relationships
        }
    }
    
    print(f"\n💾 Research data export simulation complete")
    print(f"   Export would contain {len(research_export)} metric categories")

if __name__ == "__main__":
    try:
        # Run comprehensive relationship quality analysis demo
        analyzer, report, optimization = demo_relationship_quality_analysis()
        
        # Run research analytics
        demo_research_analytics()
        
        print(f"\n✅ Relationship Quality Analysis Demo Complete!")
        
        print(f"\n🎯 Key Capabilities Demonstrated:")
        print(f"   • Comprehensive relationship quality metrics")
        print(f"   • Multi-dimensional quality scoring")
        print(f"   • Network centrality analysis")
        print(f"   • Quality-based optimization recommendations")
        print(f"   • Research-grade analytics and statistics")
        print(f"   • Auto-detected vs manual relationship comparison")
        
        print(f"\n💡 Applications:")
        print(f"   • Content quality assessment and curation")
        print(f"   • Knowledge graph optimization")
        print(f"   • Research validation and analysis")
        print(f"   • Automated relationship refinement")
        print(f"   • System performance optimization")
        
        print(f"\n📚 This example provides production-ready tools for:")
        print(f"   • Relationship quality monitoring")
        print(f"   • Database optimization")
        print(f"   • Research and analytics")
        print(f"   • Quality assurance workflows")
        
    except Exception as e:
        print(f"❌ Error in relationship quality analysis: {e}")
        print(f"💡 Check troubleshooting/debug_guide.py for help")
        import traceback
        traceback.print_exc()
