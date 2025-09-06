#!/usr/bin/env python3
"""
Academic Research Assistant with RudraDB-Opin
============================================

Demonstrates an academic research assistant that can manage papers, citations,
and research relationships using RudraDB-Opin's relationship-aware search.
"""

import rudradb
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import re

class Academic_Research_Assistant:
    """Research assistant with relationship-aware paper management"""
    
    def __init__(self):
        self.db = rudradb.RudraDB()  # Auto-dimension detection
        self.citation_graph = {}
        self.research_domains = set()
        
        print("📚 Academic Research Assistant with RudraDB-Opin")
        print("=" * 55)
        print("   🎓 Research paper management and discovery")
        print("   🔗 Citation network analysis")
        print("   🧠 Relationship-aware literature search")
    
    def add_paper(self, paper_id: str, title: str, abstract: str, authors: List[str], 
                  year: int, venue: str, keywords: List[str], 
                  metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add research paper with comprehensive metadata"""
        
        # Create research-focused embedding
        research_content = f"{title} {abstract} {' '.join(keywords)}"
        embedding = self._create_research_embedding(research_content, keywords)
        
        # Determine research domain from keywords and abstract
        domain = self._classify_research_domain(abstract, keywords)
        self.research_domains.add(domain)
        
        # Enhanced academic metadata
        enhanced_metadata = {
            "title": title,
            "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
            "authors": authors,
            "year": year,
            "venue": venue,
            "keywords": keywords,
            "domain": domain,
            "citation_count": 0,  # Will be updated as we add citations
            "influence_score": self._calculate_initial_influence(venue, year),
            "research_type": self._classify_research_type(abstract),
            "methodology": self._extract_methodology(abstract),
            "added_to_system": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        self.db.add_vector(paper_id, embedding, enhanced_metadata)
        
        # Auto-build research relationships
        relationships_created = self._auto_build_research_relationships(paper_id, enhanced_metadata)
        
        return {
            "paper_id": paper_id,
            "dimension": self.db.dimension(),
            "relationships_created": relationships_created,
            "research_domain": domain,
            "total_papers": self.db.vector_count()
        }
    
    def _create_research_embedding(self, content: str, keywords: List[str]) -> np.ndarray:
        """Create research-specific embedding"""
        # Mock research embedding that emphasizes academic content
        
        # Base embedding influenced by content type
        base_values = []
        
        # Technical vs theoretical emphasis
        technical_keywords = ["algorithm", "implementation", "system", "method", "approach"]
        theoretical_keywords = ["theory", "model", "framework", "analysis", "study"]
        
        technical_score = sum(1 for kw in keywords if any(tech in kw.lower() for tech in technical_keywords))
        theoretical_score = sum(1 for kw in keywords if any(theo in kw.lower() for theo in theoretical_keywords))
        
        base_values.extend([
            0.7 + 0.2 * (technical_score / max(len(keywords), 1)),  # Technical dimension
            0.6 + 0.3 * (theoretical_score / max(len(keywords), 1)),  # Theoretical dimension
            0.8 if "deep learning" in content.lower() or "neural" in content.lower() else 0.4,  # AI/ML dimension
            0.8 if "data" in content.lower() or "dataset" in content.lower() else 0.3,  # Data dimension
            0.7 if len(keywords) > 5 else 0.5,  # Breadth dimension
            min(1.0, len(content) / 1000)  # Content richness
        ])
        
        # Add some controlled randomness for uniqueness
        noise = np.random.normal(0, 0.1, len(base_values))
        embedding = np.array(base_values, dtype=np.float32) + noise.astype(np.float32)
        
        return np.clip(embedding, 0, 1)  # Keep values in [0,1] range
    
    def _classify_research_domain(self, abstract: str, keywords: List[str]) -> str:
        """Classify research domain based on content"""
        
        text = (abstract + " " + " ".join(keywords)).lower()
        
        domain_indicators = {
            "machine_learning": ["machine learning", "neural network", "deep learning", "classification", "regression"],
            "computer_vision": ["computer vision", "image", "visual", "object detection", "segmentation"],
            "natural_language_processing": ["natural language", "nlp", "text", "language model", "sentiment"],
            "robotics": ["robot", "robotics", "autonomous", "control", "manipulation"],
            "human_computer_interaction": ["hci", "user interface", "interaction", "usability", "user experience"],
            "software_engineering": ["software", "programming", "development", "testing", "maintenance"],
            "systems": ["system", "distributed", "network", "performance", "scalability"],
            "theory": ["algorithm", "complexity", "proof", "mathematical", "theoretical"]
        }
        
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        else:
            return "general_computer_science"
    
    def _classify_research_type(self, abstract: str) -> str:
        """Classify type of research"""
        
        abstract_lower = abstract.lower()
        
        if any(word in abstract_lower for word in ["survey", "review", "systematic"]):
            return "survey"
        elif any(word in abstract_lower for word in ["propose", "novel", "new method", "introduce"]):
            return "novel_method"
        elif any(word in abstract_lower for word in ["empirical", "experiment", "evaluation", "comparison"]):
            return "empirical_study"
        elif any(word in abstract_lower for word in ["theoretical", "analysis", "proof", "framework"]):
            return "theoretical"
        else:
            return "application"
    
    def _extract_methodology(self, abstract: str) -> List[str]:
        """Extract methodology keywords from abstract"""
        
        methodology_patterns = [
            r"using (\w+(?:\s+\w+)*)",
            r"based on (\w+(?:\s+\w+)*)",
            r"(\w+) approach",
            r"(\w+) method",
            r"(\w+) algorithm"
        ]
        
        methodologies = []
        for pattern in methodology_patterns:
            matches = re.findall(pattern, abstract.lower())
            methodologies.extend(matches)
        
        return list(set(methodologies))[:5]  # Return top 5 unique methodologies
    
    def _calculate_initial_influence(self, venue: str, year: int) -> float:
        """Calculate initial influence score based on venue and recency"""
        
        # Mock venue rankings (in practice, would use real venue impact factors)
        venue_scores = {
            "nature": 0.95,
            "science": 0.95,
            "neurips": 0.9,
            "icml": 0.9,
            "iclr": 0.85,
            "cvpr": 0.85,
            "iccv": 0.85,
            "acl": 0.8,
            "emnlp": 0.8,
            "icse": 0.8,
            "sigmod": 0.75,
            "kdd": 0.75
        }
        
        venue_score = venue_scores.get(venue.lower(), 0.5)
        
        # Recency bonus (more recent papers get slight boost)
        current_year = datetime.now().year
        recency_bonus = max(0, min(0.1, (year - (current_year - 10)) / 10 * 0.1))
        
        return min(1.0, venue_score + recency_bonus)
    
    def _auto_build_research_relationships(self, paper_id: str, metadata: Dict[str, Any]) -> int:
        """Auto-build research relationships based on academic connections"""
        
        relationships_created = 0
        domain = metadata.get("domain")
        authors = set(metadata.get("authors", []))
        keywords = set(metadata.get("keywords", []))
        year = metadata.get("year")
        research_type = metadata.get("research_type")
        
        for existing_id in self.db.list_vectors():
            if existing_id == paper_id or relationships_created >= 5:
                continue
            
            existing = self.db.get_vector(existing_id)
            existing_meta = existing["metadata"]
            
            existing_domain = existing_meta.get("domain")
            existing_authors = set(existing_meta.get("authors", []))
            existing_keywords = set(existing_meta.get("keywords", []))
            existing_year = existing_meta.get("year")
            existing_type = existing_meta.get("research_type")
            
            # Same domain semantic relationship
            if domain == existing_domain:
                self.db.add_relationship(paper_id, existing_id, "semantic", 0.8,
                    {"reason": "same_research_domain", "domain": domain, "auto_detected": True})
                relationships_created += 1
                print(f"      🔬 Research domain: {paper_id} ↔ {existing_id} ({domain})")
            
            # Author collaboration relationship
            elif len(authors & existing_authors) > 0:
                shared_authors = authors & existing_authors
                strength = min(0.9, len(shared_authors) / min(len(authors), len(existing_authors)))
                self.db.add_relationship(paper_id, existing_id, "associative", strength,
                    {"reason": "shared_authors", "authors": list(shared_authors), "auto_detected": True})
                relationships_created += 1
                print(f"      👥 Author collaboration: {paper_id} ↔ {existing_id} ({shared_authors})")
            
            # Keyword overlap (research topic similarity)
            elif len(keywords & existing_keywords) >= 2:
                shared_keywords = keywords & existing_keywords
                strength = min(0.7, len(shared_keywords) / max(len(keywords), len(existing_keywords), 1))
                self.db.add_relationship(paper_id, existing_id, "semantic", strength,
                    {"reason": "shared_keywords", "keywords": list(shared_keywords), "auto_detected": True})
                relationships_created += 1
                print(f"      🏷️ Topic similarity: {paper_id} ↔ {existing_id} ({shared_keywords})")
            
            # Temporal relationship (research timeline)
            elif (abs(year - existing_year) <= 2 and domain == existing_domain and 
                  research_type == "novel_method" and existing_type in ["empirical_study", "application"]):
                self.db.add_relationship(paper_id, existing_id, "temporal", 0.7,
                    {"reason": "research_timeline", "method_to_application": True, "auto_detected": True})
                relationships_created += 1
                print(f"      ⏰ Research timeline: {paper_id} → {existing_id}")
            
            # Citation-like relationship (survey to specific papers)
            elif research_type == "survey" and existing_domain == domain:
                self.db.add_relationship(paper_id, existing_id, "hierarchical", 0.8,
                    {"reason": "survey_relationship", "auto_detected": True})
                relationships_created += 1
                print(f"      📖 Survey relationship: {paper_id} → {existing_id}")
        
        return relationships_created
    
    def find_related_papers(self, query: str, search_type: str = "comprehensive") -> Dict[str, Any]:
        """Find papers related to a research query"""
        
        print(f"\n🔍 Research Query: '{query}' (type: {search_type})")
        
        # Create query embedding
        query_keywords = query.split()  # Simple keyword extraction
        query_embedding = self._create_research_embedding(query, query_keywords)
        
        # Configure search based on type
        if search_type == "similar_papers":
            params = rudradb.SearchParams(
                top_k=8,
                include_relationships=False,  # Only direct similarity
                similarity_threshold=0.3
            )
        elif search_type == "comprehensive":
            params = rudradb.SearchParams(
                top_k=10,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.4,
                relationship_types=["semantic", "associative"]
            )
        elif search_type == "literature_survey":
            params = rudradb.SearchParams(
                top_k=15,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.6,  # Heavy relationship emphasis
                similarity_threshold=0.1
            )
        else:
            params = rudradb.SearchParams(top_k=10, include_relationships=True)
        
        results = self.db.search(query_embedding, params)
        
        # Process results with academic context
        papers = []
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            metadata = vector["metadata"]
            
            paper_info = {
                "paper_id": result.vector_id,
                "title": metadata["title"],
                "authors": metadata["authors"],
                "year": metadata["year"],
                "venue": metadata["venue"],
                "domain": metadata["domain"],
                "research_type": metadata["research_type"],
                "relevance_score": result.combined_score,
                "connection_type": "direct" if result.hop_count == 0 else f"{result.hop_count}-hop",
                "influence_score": metadata.get("influence_score", 0.5),
                "keywords": metadata["keywords"][:5],  # Top 5 keywords
                "abstract_preview": metadata["abstract"][:200] + "..."
            }
            
            papers.append(paper_info)
        
        # Analyze results
        domains = list(set(p["domain"] for p in papers))
        research_types = list(set(p["research_type"] for p in papers))
        venues = list(set(p["venue"] for p in papers))
        
        return {
            "query": query,
            "search_type": search_type,
            "total_papers_found": len(papers),
            "relationship_enhanced": sum(1 for p in papers if p["connection_type"] != "direct"),
            "papers": papers,
            "analysis": {
                "domains_covered": domains,
                "research_types": research_types,
                "venues_represented": venues,
                "avg_relevance": np.mean([p["relevance_score"] for p in papers]),
                "avg_influence": np.mean([p["influence_score"] for p in papers])
            }
        }
    
    def get_research_network_analysis(self) -> Dict[str, Any]:
        """Analyze the research paper network"""
        
        stats = self.db.get_statistics()
        
        # Analyze paper distribution
        domain_counts = {}
        type_counts = {}
        venue_counts = {}
        yearly_distribution = {}
        
        for paper_id in self.db.list_vectors():
            paper = self.db.get_vector(paper_id)
            metadata = paper["metadata"]
            
            domain = metadata.get("domain", "unknown")
            research_type = metadata.get("research_type", "unknown")
            venue = metadata.get("venue", "unknown")
            year = metadata.get("year", 2020)
            
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            type_counts[research_type] = type_counts.get(research_type, 0) + 1
            venue_counts[venue] = venue_counts.get(venue, 0) + 1
            yearly_distribution[year] = yearly_distribution.get(year, 0) + 1
        
        # Analyze relationship network
        relationship_types = {}
        for paper_id in self.db.list_vectors():
            relationships = self.db.get_relationships(paper_id)
            for rel in relationships:
                rel_type = rel["relationship_type"]
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            "network_overview": {
                "total_papers": stats["vector_count"],
                "total_relationships": stats["relationship_count"],
                "research_domains": len(self.research_domains),
                "embedding_dimension": stats["dimension"]
            },
            "paper_distribution": {
                "by_domain": domain_counts,
                "by_research_type": type_counts,
                "by_venue": venue_counts,
                "by_year": yearly_distribution
            },
            "relationship_analysis": {
                "relationship_types": relationship_types,
                "avg_relationships_per_paper": stats["relationship_count"] / max(stats["vector_count"], 1),
                "most_connected_domains": max(domain_counts.items(), key=lambda x: x[1])[0] if domain_counts else "none"
            },
            "capacity_usage": {
                "papers_capacity": f"{stats['capacity_usage']['vector_usage_percent']:.1f}%",
                "relationships_capacity": f"{stats['capacity_usage']['relationship_usage_percent']:.1f}%"
            }
        }

def create_sample_research_database():
    """Create sample research papers for demonstration"""
    
    sample_papers = [
        {
            "paper_id": "attention_paper_2017",
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            "year": 2017,
            "venue": "NeurIPS",
            "keywords": ["transformer", "attention", "neural networks", "sequence modeling", "machine translation"]
        },
        {
            "paper_id": "bert_paper_2018",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
            "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
            "year": 2018,
            "venue": "NAACL",
            "keywords": ["bert", "transformers", "language modeling", "pre-training", "bidirectional"]
        },
        {
            "paper_id": "resnet_paper_2015",
            "title": "Deep Residual Learning for Image Recognition",
            "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
            "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren"],
            "year": 2015,
            "venue": "CVPR",
            "keywords": ["resnet", "residual learning", "deep learning", "image recognition", "computer vision"]
        },
        {
            "paper_id": "gan_paper_2014",
            "title": "Generative Adversarial Networks",
            "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
            "authors": ["Ian Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza"],
            "year": 2014,
            "venue": "NeurIPS",
            "keywords": ["gan", "generative models", "adversarial training", "deep learning", "unsupervised learning"]
        },
        {
            "paper_id": "dropout_paper_2014",
            "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
            "abstract": "Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time. Dropout is a technique for addressing this problem.",
            "authors": ["Nitish Srivastava", "Geoffrey Hinton", "Alex Krizhevsky"],
            "year": 2014,
            "venue": "JMLR",
            "keywords": ["dropout", "regularization", "overfitting", "neural networks", "machine learning"]
        },
        {
            "paper_id": "dl_survey_2015",
            "title": "Deep Learning Survey: Recent Advances and New Frontiers",
            "abstract": "Deep learning has emerged as a powerful machine learning technique that has been successfully applied to various domains including computer vision, natural language processing, and speech recognition. This survey provides a comprehensive overview of recent advances in deep learning, covering architectural innovations, training techniques, and applications across different domains.",
            "authors": ["Survey Author", "Review Author"],
            "year": 2015,
            "venue": "Survey Journal",
            "keywords": ["deep learning", "survey", "neural networks", "machine learning", "review"]
        }
    ]
    
    return sample_papers

def main():
    """Demonstrate Academic Research Assistant"""
    
    print("🚀 Academic Research Assistant Demo")
    print("=" * 40)
    
    # Initialize research assistant
    assistant = Academic_Research_Assistant()
    
    # Add sample papers
    print("\n📚 Building Research Database:")
    sample_papers = create_sample_research_database()
    
    for paper in sample_papers:
        result = assistant.add_paper(
            paper["paper_id"],
            paper["title"], 
            paper["abstract"],
            paper["authors"],
            paper["year"],
            paper["venue"],
            paper["keywords"]
        )
        print(f"   ✅ Added '{paper['title'][:50]}...' ({result['relationships_created']} relationships)")
    
    # Network analysis
    network_analysis = assistant.get_research_network_analysis()
    print(f"\n📊 Research Network Analysis:")
    print(f"   Papers: {network_analysis['network_overview']['total_papers']}")
    print(f"   Relationships: {network_analysis['network_overview']['total_relationships']}")
    print(f"   Domains: {network_analysis['network_overview']['research_domains']}")
    print(f"   Dimension: {network_analysis['network_overview']['embedding_dimension']}D")
    
    print(f"   Domain distribution: {dict(list(network_analysis['paper_distribution']['by_domain'].items())[:3])}")
    print(f"   Relationship types: {network_analysis['relationship_analysis']['relationship_types']}")
    
    # Demonstrate different search types
    print(f"\n🔍 Research Search Demonstrations:")
    
    search_queries = [
        ("transformer attention mechanisms", "similar_papers"),
        ("deep learning computer vision", "comprehensive"),
        ("neural network training techniques", "literature_survey")
    ]
    
    for query, search_type in search_queries:
        print(f"\n📖 {search_type.replace('_', ' ').title()}: '{query}'")
        
        results = assistant.find_related_papers(query, search_type)
        
        print(f"   📊 Found {results['total_papers_found']} papers")
        print(f"   🧠 Relationship-enhanced: {results['relationship_enhanced']} additional discoveries")
        print(f"   🏷️ Domains: {', '.join(results['analysis']['domains_covered'][:3])}")
        print(f"   📈 Avg relevance: {results['analysis']['avg_relevance']:.3f}")
        
        print("   Top Results:")
        for i, paper in enumerate(results['papers'][:3], 1):
            print(f"      {i}. {paper['title'][:60]}...")
            print(f"         Authors: {', '.join(paper['authors'][:2])}{'...' if len(paper['authors']) > 2 else ''}")
            print(f"         {paper['year']} • {paper['venue']} • {paper['connection_type']}")
    
    # Final statistics
    final_analysis = assistant.get_research_network_analysis()
    
    print(f"\n🎓 Research Assistant Capabilities:")
    capabilities = [
        "Auto-dimension detection for research content",
        "Research domain classification",
        "Author collaboration networks",
        "Citation-style relationship modeling", 
        "Multi-hop literature discovery",
        "Venue and influence scoring",
        "Research type classification",
        "Temporal research progression tracking"
    ]
    
    for capability in capabilities:
        print(f"   ✅ {capability}")
    
    print(f"\n🚀 Perfect for:")
    print(f"   • Literature review automation")
    print(f"   • Research gap identification") 
    print(f"   • Citation network analysis")
    print(f"   • Academic collaboration discovery")
    print(f"   • Research trend analysis")
    
    return final_analysis

if __name__ == "__main__":
    main()
