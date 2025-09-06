#!/usr/bin/env python3
"""
Research Discovery System with RudraDB-Opin

This example demonstrates how to build a research paper discovery system using
RudraDB-Opin's relationship-aware search to find related papers through citation
networks, methodological connections, and research area relationships.

Requirements:
    pip install rudradb-opin sentence-transformers

Usage:
    python research_discovery.py
"""

import rudradb
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ sentence-transformers not available - using simulated embeddings")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class Research_Discovery_System:
    """Research paper discovery with auto-relationship intelligence"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.db = rudradb.RudraDB()  # Auto-detects dimension
        self.embedding_model_name = embedding_model
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(embedding_model)
                self.use_real_embeddings = True
            except Exception as e:
                print(f"⚠️ Error loading model: {e}, using simulated embeddings")
                self.use_real_embeddings = False
        else:
            self.use_real_embeddings = False
        
        print("🔬 Research Discovery System initialized")
        print(f"   🎯 Embedding model: {embedding_model}")
        print(f"   🤖 Real embeddings: {self.use_real_embeddings}")
        
        # Research field taxonomy
        self.field_hierarchy = {
            "AI": ["machine_learning", "deep_learning", "nlp", "computer_vision", "robotics"],
            "ML": ["supervised_learning", "unsupervised_learning", "reinforcement_learning", "neural_networks"],
            "NLP": ["language_models", "text_classification", "information_extraction", "machine_translation"],
            "CV": ["image_recognition", "object_detection", "segmentation", "generative_models"]
        }
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using real model or simulation"""
        if self.use_real_embeddings:
            return self.model.encode([text])[0].astype(np.float32)
        else:
            # Simulated embedding based on text hash for consistency
            np.random.seed(hash(text) % (2**32))
            return np.random.rand(384).astype(np.float32)
    
    def add_paper(self, paper_id: str, abstract: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add research paper with automatic relationship detection"""
        
        # Generate embedding for abstract
        embedding = self.get_embedding(abstract)
        
        # Enhanced metadata for relationship detection
        enhanced_metadata = {
            "abstract": abstract[:1000],  # Store truncated abstract
            "title": metadata.get("title", ""),
            "year": metadata.get("year", 2024),
            "field": metadata.get("field", "Unknown"),
            "subfields": metadata.get("subfields", []),
            "methodology": metadata.get("methodology", []),
            "problem_type": metadata.get("problem_type", []),
            "authors": metadata.get("authors", []),
            "venue": metadata.get("venue", ""),
            "citations_count": metadata.get("citations_count", 0),
            "keywords": metadata.get("keywords", []),
            "contribution_type": metadata.get("contribution_type", "empirical"),  # theoretical, empirical, survey
            **metadata
        }
        
        # Add to database
        self.db.add_vector(paper_id, embedding, enhanced_metadata)
        
        # Auto-detect research relationships
        relationships_created = self._auto_detect_research_relationships(paper_id, enhanced_metadata)
        
        return {
            "paper_id": paper_id,
            "relationships_created": relationships_created,
            "total_papers": self.db.vector_count(),
            "dimension": self.db.dimension()
        }
    
    def _auto_detect_research_relationships(self, new_paper_id: str, metadata: Dict[str, Any]) -> int:
        """Auto-detect research relationships based on academic criteria"""
        
        relationships_created = 0
        max_relationships = 5  # Limit for each paper
        
        new_year = metadata.get("year", 2024)
        new_field = metadata.get("field", "")
        new_subfields = set(metadata.get("subfields", []))
        new_methodology = set(metadata.get("methodology", []))
        new_problem_types = set(metadata.get("problem_type", []))
        new_authors = set(metadata.get("authors", []))
        new_keywords = set(metadata.get("keywords", []))
        new_contribution_type = metadata.get("contribution_type", "")
        
        for existing_id in self.db.list_vectors():
            if existing_id == new_paper_id or relationships_created >= max_relationships:
                continue
            
            existing_paper = self.db.get_vector(existing_id)
            existing_meta = existing_paper['metadata']
            
            existing_year = existing_meta.get("year", 2024)
            existing_field = existing_meta.get("field", "")
            existing_subfields = set(existing_meta.get("subfields", []))
            existing_methodology = set(existing_meta.get("methodology", []))
            existing_problem_types = set(existing_meta.get("problem_type", []))
            existing_authors = set(existing_meta.get("authors", []))
            existing_keywords = set(existing_meta.get("keywords", []))
            existing_contribution_type = existing_meta.get("contribution_type", "")
            
            # 📊 Hierarchical: Same field, different years (temporal academic progression)
            if (new_field == existing_field and 
                abs(new_year - existing_year) <= 3 and  # Within 3 years
                new_year != existing_year):
                
                strength = 0.9 if abs(new_year - existing_year) == 1 else 0.7
                self.db.add_relationship(new_paper_id, existing_id, "hierarchical", strength,
                    {"reason": "field_temporal_progression", "field": new_field, 
                     "year_gap": abs(new_year - existing_year)})
                relationships_created += 1
                print(f"      📊 {new_paper_id} ↔ {existing_id} (field progression: {new_field})")
            
            # 🔗 Semantic: Shared subfields and problem types
            elif len(new_subfields & existing_subfields) >= 1 and len(new_problem_types & existing_problem_types) >= 1:
                shared_subfields = new_subfields & existing_subfields
                shared_problems = new_problem_types & existing_problem_types
                strength = min(0.85, (len(shared_subfields) + len(shared_problems)) * 0.2)
                
                self.db.add_relationship(new_paper_id, existing_id, "semantic", strength,
                    {"reason": "shared_research_focus", "shared_subfields": list(shared_subfields),
                     "shared_problems": list(shared_problems)})
                relationships_created += 1
                print(f"      🔗 {new_paper_id} ↔ {existing_id} (research focus: {shared_subfields})")
            
            # ⏰ Temporal: Methodological evolution (newer methods building on older)
            elif (len(new_methodology & existing_methodology) >= 1 and
                  new_year > existing_year and
                  new_year - existing_year <= 5):
                
                shared_methods = new_methodology & existing_methodology
                strength = 0.8
                self.db.add_relationship(new_paper_id, existing_id, "temporal", strength,
                    {"reason": "methodological_evolution", "shared_methods": list(shared_methods),
                     "years_apart": new_year - existing_year})
                relationships_created += 1
                print(f"      ⏰ {new_paper_id} → {existing_id} (method evolution: {shared_methods})")
            
            # 🎯 Causal: Survey/theoretical papers citing empirical work
            elif (new_contribution_type in ["survey", "theoretical"] and
                  existing_contribution_type == "empirical" and
                  new_field == existing_field):
                
                self.db.add_relationship(new_paper_id, existing_id, "causal", 0.85,
                    {"reason": "survey_cites_empirical", "survey_type": new_contribution_type})
                relationships_created += 1
                print(f"      🎯 {new_paper_id} → {existing_id} (survey → empirical)")
            
            # 🏷️ Associative: Author collaboration networks
            elif len(new_authors & existing_authors) >= 1:
                shared_authors = new_authors & existing_authors
                strength = min(0.7, len(shared_authors) * 0.3)
                
                self.db.add_relationship(new_paper_id, existing_id, "associative", strength,
                    {"reason": "author_collaboration", "shared_authors": list(shared_authors)})
                relationships_created += 1
                print(f"      🏷️ {new_paper_id} ↔ {existing_id} (authors: {shared_authors})")
            
            # 🔍 Associative: Keyword overlap (weaker signal)
            elif len(new_keywords & existing_keywords) >= 2:
                shared_keywords = new_keywords & existing_keywords
                strength = min(0.6, len(shared_keywords) * 0.15)
                
                self.db.add_relationship(new_paper_id, existing_id, "associative", strength,
                    {"reason": "keyword_overlap", "shared_keywords": list(shared_keywords)})
                relationships_created += 1
                print(f"      🔍 {new_paper_id} ↔ {existing_id} (keywords: {shared_keywords})")
        
        return relationships_created
    
    def discover_research_connections(self, query_paper: str, discovery_type: str = "comprehensive") -> Dict[str, Any]:
        """Discover research connections through auto-relationship networks"""
        
        if not self.db.vector_exists(query_paper):
            return {"error": f"Paper {query_paper} not found"}
        
        paper_vector = self.db.get_vector(query_paper)
        query_embedding = paper_vector["embedding"]
        
        # Configure search based on discovery type
        if discovery_type == "comprehensive":
            # Broad discovery across all relationship types
            params = rudradb.SearchParams(
                top_k=15,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.4,
                similarity_threshold=0.1
            )
        elif discovery_type == "methodological":
            # Focus on methodological connections
            params = rudradb.SearchParams(
                top_k=10,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.6,
                relationship_types=["temporal", "semantic"]
            )
        elif discovery_type == "citation_network":
            # Focus on citation-like patterns
            params = rudradb.SearchParams(
                top_k=12,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.5,
                relationship_types=["hierarchical", "causal", "temporal"]
            )
        elif discovery_type == "collaboration":
            # Focus on author and institutional connections
            params = rudradb.SearchParams(
                top_k=10,
                include_relationships=True,
                max_hops=1,
                relationship_weight=0.7,
                relationship_types=["associative"]
            )
        else:
            # Default comprehensive search
            params = rudradb.SearchParams(
                top_k=10,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.4
            )
        
        # Perform relationship-aware search
        results = self.db.search(query_embedding, params)
        
        # Analyze and categorize results
        discoveries = {
            "query_paper": query_paper,
            "query_metadata": paper_vector["metadata"],
            "discovery_type": discovery_type,
            "total_discovered": len(results),
            "direct_similar": [],
            "relationship_discoveries": [],
            "connection_analysis": self._analyze_connection_patterns(results),
            "research_insights": self._extract_research_insights(query_paper, results)
        }
        
        for result in results:
            result_vector = self.db.get_vector(result.vector_id)
            
            connection_info = {
                "paper_id": result.vector_id,
                "title": result_vector["metadata"].get("title", ""),
                "year": result_vector["metadata"].get("year"),
                "field": result_vector["metadata"].get("field"),
                "similarity_score": result.similarity_score,
                "combined_score": result.combined_score,
                "connection_hops": result.hop_count,
                "connection_type": "direct_similarity" if result.hop_count == 0 else f"{result.hop_count}-hop_relationship"
            }
            
            if result.hop_count == 0:
                discoveries["direct_similar"].append(connection_info)
            else:
                discoveries["relationship_discoveries"].append(connection_info)
        
        return discoveries
    
    def _analyze_connection_patterns(self, results: List) -> Dict[str, Any]:
        """Analyze patterns in the discovered connections"""
        
        if not results:
            return {}
        
        # Analyze by connection type
        direct_connections = [r for r in results if r.hop_count == 0]
        one_hop = [r for r in results if r.hop_count == 1]
        multi_hop = [r for r in results if r.hop_count > 1]
        
        # Analyze by relationship strength
        strengths = [r.combined_score for r in results]
        
        # Analyze temporal patterns
        years = []
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            year = vector["metadata"].get("year")
            if year:
                years.append(year)
        
        analysis = {
            "connection_distribution": {
                "direct_similarity": len(direct_connections),
                "one_hop_relationships": len(one_hop),
                "multi_hop_relationships": len(multi_hop)
            },
            "strength_analysis": {
                "average_score": np.mean(strengths) if strengths else 0,
                "max_score": max(strengths) if strengths else 0,
                "min_score": min(strengths) if strengths else 0
            },
            "temporal_span": {
                "years_covered": list(set(years)) if years else [],
                "year_range": max(years) - min(years) if len(years) > 1 else 0,
                "most_recent": max(years) if years else None,
                "oldest": min(years) if years else None
            }
        }
        
        return analysis
    
    def _extract_research_insights(self, query_paper: str, results: List) -> Dict[str, Any]:
        """Extract research insights from discovered connections"""
        
        # Collect metadata from all discovered papers
        fields = []
        methodologies = []
        problem_types = []
        venues = []
        contribution_types = []
        
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            metadata = vector["metadata"]
            
            if metadata.get("field"):
                fields.append(metadata["field"])
            methodologies.extend(metadata.get("methodology", []))
            problem_types.extend(metadata.get("problem_type", []))
            if metadata.get("venue"):
                venues.append(metadata["venue"])
            if metadata.get("contribution_type"):
                contribution_types.append(metadata["contribution_type"])
        
        # Generate insights
        insights = {
            "research_areas": {
                "primary_fields": list(set(fields)),
                "field_distribution": {field: fields.count(field) for field in set(fields)}
            },
            "methodological_trends": {
                "common_methods": list(set(methodologies)),
                "method_frequency": {method: methodologies.count(method) for method in set(methodologies)}
            },
            "problem_landscape": {
                "problem_types": list(set(problem_types)),
                "problem_frequency": {prob: problem_types.count(prob) for prob in set(problem_types)}
            },
            "publication_venues": list(set(venues)),
            "contribution_mix": {
                "types": list(set(contribution_types)),
                "type_distribution": {ctype: contribution_types.count(ctype) for ctype in set(contribution_types)}
            },
            "network_density": len(results) / max(1, self.db.vector_count()) * 100  # % of corpus connected
        }
        
        return insights


def demo_research_discovery():
    """Demo research discovery system"""
    
    print("🔬 Research Discovery System Demo")
    print("=" * 40)
    
    # Initialize system
    discovery_system = Research_Discovery_System()
    
    # Sample research papers database
    research_papers = [
        {
            "id": "attention_paper",
            "abstract": "We propose the Transformer, a novel neural network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            "metadata": {
                "title": "Attention Is All You Need",
                "year": 2017,
                "field": "NLP",
                "subfields": ["language_models", "neural_architectures"],
                "methodology": ["attention_mechanisms", "transformer_architecture"],
                "problem_type": ["sequence_modeling", "machine_translation"],
                "authors": ["Vaswani", "Shazeer", "Parmar"],
                "venue": "NIPS",
                "citations_count": 50000,
                "keywords": ["attention", "transformer", "neural_machine_translation"],
                "contribution_type": "empirical"
            }
        },
        {
            "id": "bert_paper",
            "abstract": "We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
            "metadata": {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "year": 2018,
                "field": "NLP",
                "subfields": ["language_models", "pretraining"],
                "methodology": ["bidirectional_training", "transformer_architecture", "masked_language_modeling"],
                "problem_type": ["language_understanding", "representation_learning"],
                "authors": ["Devlin", "Chang", "Lee", "Toutanova"],
                "venue": "NAACL",
                "citations_count": 40000,
                "keywords": ["bert", "bidirectional", "pretraining", "transformer"],
                "contribution_type": "empirical"
            }
        },
        {
            "id": "gpt_paper",
            "abstract": "We demonstrate that large gains on language modeling tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text.",
            "metadata": {
                "title": "Improving Language Understanding by Generative Pre-Training",
                "year": 2018,
                "field": "NLP",
                "subfields": ["language_models", "pretraining"],
                "methodology": ["generative_pretraining", "transformer_architecture", "unsupervised_learning"],
                "problem_type": ["language_understanding", "few_shot_learning"],
                "authors": ["Radford", "Narasimhan", "Salimans", "Sutskever"],
                "venue": "OpenAI",
                "citations_count": 8000,
                "keywords": ["gpt", "generative", "pretraining", "language_modeling"],
                "contribution_type": "empirical"
            }
        },
        {
            "id": "vision_transformer",
            "abstract": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited.",
            "metadata": {
                "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
                "year": 2020,
                "field": "CV",
                "subfields": ["image_recognition", "neural_architectures"],
                "methodology": ["transformer_architecture", "patch_embeddings", "self_attention"],
                "problem_type": ["image_classification", "representation_learning"],
                "authors": ["Dosovitskiy", "Beyer", "Kolesnikov"],
                "venue": "ICLR",
                "citations_count": 15000,
                "keywords": ["vision_transformer", "vit", "image_recognition", "attention"],
                "contribution_type": "empirical"
            }
        },
        {
            "id": "transformer_survey",
            "abstract": "This survey provides a comprehensive overview of Transformer models, their variants, and applications across different domains including NLP and computer vision.",
            "metadata": {
                "title": "A Survey of Transformers",
                "year": 2021,
                "field": "AI",
                "subfields": ["neural_architectures", "survey"],
                "methodology": ["literature_review", "taxonomy", "comparative_analysis"],
                "problem_type": ["architecture_analysis", "performance_comparison"],
                "authors": ["Lin", "Wang", "Yang"],
                "venue": "AI Review",
                "citations_count": 2000,
                "keywords": ["transformer", "survey", "deep_learning", "attention"],
                "contribution_type": "survey"
            }
        },
        {
            "id": "efficient_transformers",
            "abstract": "We present efficient variants of Transformer models that reduce computational complexity while maintaining performance on various NLP tasks.",
            "metadata": {
                "title": "Efficient Transformers: A Survey",
                "year": 2022,
                "field": "NLP",
                "subfields": ["efficient_architectures", "optimization"],
                "methodology": ["sparse_attention", "linear_attention", "pruning"],
                "problem_type": ["computational_efficiency", "model_optimization"],
                "authors": ["Tay", "Dehghani", "Rao"],
                "venue": "ACL",
                "citations_count": 1500,
                "keywords": ["efficient", "transformers", "sparse_attention", "optimization"],
                "contribution_type": "survey"
            }
        }
    ]
    
    # Add papers to the system
    print("\n📚 Building research paper database...")
    for paper in research_papers:
        result = discovery_system.add_paper(paper["id"], paper["abstract"], paper["metadata"])
        print(f"   📄 {paper['id']}: {result['relationships_created']} auto-relationships")
    
    print(f"\n✅ Research database built: {discovery_system.db.vector_count()} papers, {discovery_system.db.relationship_count()} relationships")
    
    # Demonstrate different types of research discovery
    discovery_types = ["comprehensive", "methodological", "citation_network", "collaboration"]
    query_papers = ["attention_paper", "bert_paper"]
    
    print(f"\n🔍 Research Discovery Demonstrations:")
    
    for query_paper in query_papers:
        print(f"\n📄 Analyzing connections for: {query_paper}")
        
        for discovery_type in discovery_types[:2]:  # Limit for demo
            print(f"\n   🔬 {discovery_type.title()} Discovery:")
            
            discoveries = discovery_system.discover_research_connections(query_paper, discovery_type)
            
            if "error" in discoveries:
                print(f"      ❌ {discoveries['error']}")
                continue
            
            print(f"      📊 Found {discoveries['total_discovered']} connected papers")
            print(f"      🎯 Direct similar: {len(discoveries['direct_similar'])}")
            print(f"      🕸️ Through relationships: {len(discoveries['relationship_discoveries'])}")
            
            # Show some discoveries
            if discoveries['relationship_discoveries']:
                print(f"      🔗 Key relationship discoveries:")
                for disco in discoveries['relationship_discoveries'][:3]:
                    print(f"         • {disco['paper_id']} ({disco['year']}) - {disco['connection_type']}")
                    print(f"           Score: {disco['combined_score']:.3f} | Field: {disco['field']}")
            
            # Show connection analysis
            if discoveries['connection_analysis']:
                analysis = discoveries['connection_analysis']
                print(f"      📈 Connection Analysis:")
                if analysis.get('temporal_span', {}).get('year_range'):
                    print(f"         Temporal span: {analysis['temporal_span']['year_range']} years")
                if analysis.get('connection_distribution'):
                    dist = analysis['connection_distribution']
                    print(f"         Connections: {dist['direct_similarity']} direct, {dist['one_hop_relationships']} 1-hop")
            
            # Show research insights
            if discoveries['research_insights']:
                insights = discoveries['research_insights']
                if insights.get('research_areas', {}).get('primary_fields'):
                    fields = insights['research_areas']['primary_fields']
                    print(f"         Research areas: {', '.join(fields)}")
                
                if insights.get('methodological_trends', {}).get('common_methods'):
                    methods = insights['methodological_trends']['common_methods'][:3]
                    print(f"         Common methods: {', '.join(methods)}")
    
    # Show database statistics
    stats = discovery_system.db.get_statistics()
    print(f"\n📊 Research Discovery System Statistics:")
    print(f"   📚 Papers indexed: {stats['vector_count']}")
    print(f"   🔗 Research relationships: {stats['relationship_count']}")
    print(f"   🎯 Embedding dimension: {stats['dimension']}D")
    print(f"   💾 Capacity usage: {stats['capacity_usage']['vector_usage_percent']:.1f}% vectors")
    
    # Show relationship type analysis
    relationship_types = {}
    for paper_id in discovery_system.db.list_vectors()[:5]:  # Sample analysis
        relationships = discovery_system.db.get_relationships(paper_id)
        for rel in relationships:
            rel_type = rel["relationship_type"]
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
    
    if relationship_types:
        print(f"   🧠 Relationship types used: {', '.join(relationship_types.keys())}")
        for rel_type, count in relationship_types.items():
            print(f"      • {rel_type}: {count} connections")
    
    print(f"\n🎉 Research Discovery System demo complete!")
    print("    🔬 Auto-detected research connections based on academic patterns")
    print("    📊 Analyzed citation networks, methodological relationships, and collaborations")
    print("    🕸️ Multi-hop discovery revealed indirect research connections")
    print("    📈 Generated insights about research trends and patterns")


if __name__ == "__main__":
    demo_research_discovery()
