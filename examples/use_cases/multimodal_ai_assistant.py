#!/usr/bin/env python3
"""
🌐 Multi-Modal AI Assistant with RudraDB-Opin

This example demonstrates how to build a multi-modal AI assistant that can handle
text, images, and documents using relationship-aware vector search. It shows:

1. Multi-modal content processing (text, images, documents)
2. Cross-modal relationship building (text-image, document-summary, etc.)
3. Unified search across different content types
4. Intelligent content recommendations across modalities
5. Context-aware multi-modal responses

Perfect example of how relationship-aware search enables sophisticated AI assistants!
"""

import rudradb
import numpy as np
import json
import base64
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
import hashlib
import io

class MultiModal_AI_Assistant:
    """Multi-modal AI Assistant using RudraDB-Opin relationship-aware search"""
    
    def __init__(self, assistant_name: str = "RudraBot"):
        self.db = rudradb.RudraDB()  # Auto-dimension detection for any content
        self.assistant_name = assistant_name
        self.conversation_history = []
        self.user_preferences = {}
        
        print(f"🌐 Initializing Multi-Modal AI Assistant: {assistant_name}")
        print(f"   🤖 Auto-dimension detection for all content types")
        print(f"   🔗 Relationship-aware cross-modal search enabled")
        
    def add_text_content(self, content_id: str, text: str, metadata: Dict = None):
        """Add text content with rich metadata for cross-modal relationships"""
        
        # Simulate text embedding (in production, use actual text embedding model)
        text_embedding = self._generate_text_embedding(text)
        
        enhanced_metadata = {
            "content_type": "text",
            "text": text[:500],  # Store preview
            "text_length": len(text),
            "word_count": len(text.split()),
            "language": metadata.get("language", "en"),
            "topic": metadata.get("topic", "general"),
            "sentiment": metadata.get("sentiment", "neutral"),
            "keywords": self._extract_keywords(text),
            "created_at": datetime.now().isoformat(),
            "source": metadata.get("source", "user_input"),
            **(metadata or {})
        }
        
        self.db.add_vector(content_id, text_embedding, enhanced_metadata)
        print(f"📝 Added text content: {content_id}")
        
        return content_id
    
    def add_image_content(self, content_id: str, image_description: str, 
                         image_data: bytes = None, metadata: Dict = None):
        """Add image content with description-based embedding"""
        
        # Simulate image embedding (in production, use vision model like CLIP)
        image_embedding = self._generate_image_embedding(image_description)
        
        # Store image data as base64 if provided
        image_b64 = base64.b64encode(image_data).decode() if image_data else None
        
        enhanced_metadata = {
            "content_type": "image",
            "description": image_description,
            "image_data": image_b64[:100] if image_b64 else None,  # Store preview
            "image_size": len(image_data) if image_data else 0,
            "detected_objects": metadata.get("detected_objects", []),
            "scene_description": metadata.get("scene_description", ""),
            "colors": metadata.get("colors", []),
            "style": metadata.get("style", "photograph"),
            "created_at": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        self.db.add_vector(content_id, image_embedding, enhanced_metadata)
        print(f"🖼️ Added image content: {content_id}")
        
        return content_id
    
    def add_document_content(self, content_id: str, document_text: str, 
                            doc_type: str = "pdf", metadata: Dict = None):
        """Add document content with structured metadata"""
        
        # Simulate document embedding
        doc_embedding = self._generate_text_embedding(document_text)
        
        enhanced_metadata = {
            "content_type": "document",
            "document_type": doc_type,  # pdf, word, powerpoint, etc.
            "content": document_text[:1000],  # Store longer preview for documents
            "page_count": metadata.get("page_count", 1),
            "author": metadata.get("author", "unknown"),
            "title": metadata.get("title", f"Document {content_id}"),
            "summary": metadata.get("summary", ""),
            "key_points": metadata.get("key_points", []),
            "topics_covered": metadata.get("topics_covered", []),
            "complexity_level": metadata.get("complexity_level", "medium"),
            "created_at": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        self.db.add_vector(content_id, doc_embedding, enhanced_metadata)
        print(f"📄 Added document content: {content_id}")
        
        return content_id
    
    def add_conversation_turn(self, turn_id: str, user_query: str, 
                             assistant_response: str, context_ids: List[str] = None):
        """Add conversation turn for context-aware responses"""
        
        # Create combined embedding for the conversation turn
        combined_text = f"User: {user_query}\nAssistant: {assistant_response}"
        conversation_embedding = self._generate_text_embedding(combined_text)
        
        enhanced_metadata = {
            "content_type": "conversation",
            "user_query": user_query,
            "assistant_response": assistant_response[:300],  # Preview
            "context_used": context_ids or [],
            "query_length": len(user_query),
            "response_length": len(assistant_response),
            "conversation_timestamp": datetime.now().isoformat(),
            "query_intent": self._classify_intent(user_query),
            "response_type": self._classify_response_type(assistant_response)
        }
        
        self.db.add_vector(turn_id, conversation_embedding, enhanced_metadata)
        
        # Add to conversation history
        self.conversation_history.append({
            "turn_id": turn_id,
            "user_query": user_query,
            "assistant_response": assistant_response,
            "timestamp": datetime.now()
        })
        
        print(f"💬 Added conversation turn: {turn_id}")
        
        return turn_id
    
    def build_multimodal_relationships(self):
        """Build intelligent relationships across different content types"""
        
        print(f"\n🔗 Building multi-modal relationships...")
        
        all_content = self.db.list_vectors()
        relationships_created = 0
        
        for content_id in all_content:
            content = self.db.get_vector(content_id)
            if not content:
                continue
                
            metadata = content['metadata']
            content_type = metadata.get('content_type')
            
            # Build relationships based on content types
            for other_id in all_content:
                if other_id == content_id or relationships_created >= 300:  # Opin limits
                    continue
                    
                other_content = self.db.get_vector(other_id)
                if not other_content:
                    continue
                    
                other_metadata = other_content['metadata']
                other_type = other_metadata.get('content_type')
                
                # 1. Cross-modal semantic relationships
                if self._are_semantically_related(metadata, other_metadata):
                    strength = self._calculate_semantic_strength(metadata, other_metadata)
                    self.db.add_relationship(content_id, other_id, "semantic", strength, {
                        "relationship_type": "cross_modal_semantic",
                        "modalities": f"{content_type}-{other_type}",
                        "reason": "semantic_similarity"
                    })
                    relationships_created += 1
                    continue
                
                # 2. Document-summary relationships (hierarchical)
                if (content_type == "document" and other_type == "text" and
                    "summary" in other_metadata.get("source", "")):
                    self.db.add_relationship(content_id, other_id, "hierarchical", 0.9, {
                        "relationship_type": "document_summary",
                        "reason": "document_has_summary"
                    })
                    relationships_created += 1
                    continue
                
                # 3. Image-description relationships (causal)
                if (content_type == "image" and other_type == "text" and
                    any(keyword in other_metadata.get("keywords", []) 
                        for keyword in ["image", "picture", "visual", "photo"])):
                    self.db.add_relationship(content_id, other_id, "causal", 0.8, {
                        "relationship_type": "image_description",
                        "reason": "text_describes_image"
                    })
                    relationships_created += 1
                    continue
                
                # 4. Conversation context relationships (temporal)
                if content_type == "conversation" and other_type in ["text", "document", "image"]:
                    if other_id in metadata.get("context_used", []):
                        self.db.add_relationship(other_id, content_id, "temporal", 0.7, {
                            "relationship_type": "conversation_context",
                            "reason": "content_used_in_conversation"
                        })
                        relationships_created += 1
                        continue
                
                # 5. Topic-based associations (associative)
                shared_topics = self._find_shared_topics(metadata, other_metadata)
                if shared_topics and len(shared_topics) >= 1:
                    strength = min(0.6, len(shared_topics) * 0.2)
                    self.db.add_relationship(content_id, other_id, "associative", strength, {
                        "relationship_type": "topic_association",
                        "shared_topics": shared_topics,
                        "modalities": f"{content_type}-{other_type}"
                    })
                    relationships_created += 1
        
        print(f"   ✅ Created {relationships_created} multi-modal relationships")
        return relationships_created
    
    def multimodal_search(self, query: str, content_types: List[str] = None, 
                         max_results: int = 10, search_strategy: str = "comprehensive"):
        """Search across all content types with relationship awareness"""
        
        print(f"\n🔍 Multi-modal search: '{query}'")
        print(f"   🎯 Content types: {content_types or 'all'}")
        print(f"   📊 Strategy: {search_strategy}")
        
        # Generate query embedding
        query_embedding = self._generate_text_embedding(query)
        
        # Configure search parameters based on strategy
        if search_strategy == "comprehensive":
            params = rudradb.SearchParams(
                top_k=max_results * 2,  # Get more to filter
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.4,
                similarity_threshold=0.1
            )
        elif search_strategy == "precise":
            params = rudradb.SearchParams(
                top_k=max_results,
                include_relationships=False,
                similarity_threshold=0.4
            )
        elif search_strategy == "discovery":
            params = rudradb.SearchParams(
                top_k=max_results * 3,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.6,
                similarity_threshold=0.05
            )
        else:
            params = rudradb.SearchParams(top_k=max_results, include_relationships=True)
        
        # Perform search
        results = self.db.search(query_embedding, params)
        
        # Filter by content types if specified
        if content_types:
            filtered_results = []
            for result in results:
                vector = self.db.get_vector(result.vector_id)
                if vector and vector['metadata'].get('content_type') in content_types:
                    filtered_results.append(result)
            results = filtered_results[:max_results]
        else:
            results = results[:max_results]
        
        # Format results with multi-modal information
        formatted_results = []
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            if not vector:
                continue
                
            metadata = vector['metadata']
            content_type = metadata.get('content_type', 'unknown')
            
            # Create type-specific preview
            preview = self._create_content_preview(content_type, metadata)
            
            formatted_results.append({
                "content_id": result.vector_id,
                "content_type": content_type,
                "preview": preview,
                "similarity_score": result.similarity_score,
                "combined_score": result.combined_score,
                "connection_type": "direct" if result.hop_count == 0 else f"{result.hop_count}-hop",
                "hop_count": result.hop_count,
                "metadata_summary": self._create_metadata_summary(metadata)
            })
        
        print(f"   ✅ Found {len(formatted_results)} multi-modal results")
        
        # Group results by content type
        by_type = {}
        for result in formatted_results:
            content_type = result['content_type']
            if content_type not in by_type:
                by_type[content_type] = []
            by_type[content_type].append(result)
        
        print(f"   📊 Results by type: {dict((k, len(v)) for k, v in by_type.items())}")
        
        return formatted_results, by_type
    
    def generate_multimodal_response(self, query: str, context_limit: int = 5):
        """Generate intelligent response using multi-modal context"""
        
        print(f"\n🤖 Generating multi-modal response for: '{query}'")
        
        # Search for relevant context across all modalities
        context_results, context_by_type = self.multimodal_search(
            query, 
            max_results=context_limit,
            search_strategy="comprehensive"
        )
        
        # Build response based on available context types
        response_parts = []
        
        # Add text context
        if "text" in context_by_type:
            text_contexts = context_by_type["text"][:2]
            response_parts.append(f"Based on text content, {self._synthesize_text_context(text_contexts)}")
        
        # Add document context
        if "document" in context_by_type:
            doc_contexts = context_by_type["document"][:2]
            response_parts.append(f"From documents, {self._synthesize_document_context(doc_contexts)}")
        
        # Add image context
        if "image" in context_by_type:
            image_contexts = context_by_type["image"][:2]
            response_parts.append(f"Regarding visual content, {self._synthesize_image_context(image_contexts)}")
        
        # Add conversation context
        if "conversation" in context_by_type:
            conv_contexts = context_by_type["conversation"][:1]
            response_parts.append(f"From our previous conversations, {self._synthesize_conversation_context(conv_contexts)}")
        
        # Generate comprehensive response
        if response_parts:
            response = f"I can help you with that! " + " ".join(response_parts)
            response += f"\n\nThis response draws from {len(context_results)} relevant sources across different content types."
        else:
            response = "I'd be happy to help! However, I don't have specific context for this query yet. Could you provide more details or share relevant content?"
        
        # Record this conversation turn
        turn_id = f"turn_{len(self.conversation_history) + 1}"
        context_ids = [result['content_id'] for result in context_results]
        
        self.add_conversation_turn(turn_id, query, response, context_ids)
        
        return {
            "response": response,
            "context_used": context_results,
            "context_types": list(context_by_type.keys()),
            "turn_id": turn_id
        }
    
    def recommend_content(self, content_id: str, max_recommendations: int = 5):
        """Recommend related content across all modalities"""
        
        print(f"\n💡 Generating multi-modal recommendations for: {content_id}")
        
        # Get connected content through relationships
        connected_vectors = self.db.get_connected_vectors(content_id, max_hops=2)
        
        recommendations = []
        for vector_data, hop_count in connected_vectors:
            if vector_data['id'] == content_id:
                continue
                
            metadata = vector_data['metadata']
            content_type = metadata.get('content_type', 'unknown')
            
            recommendations.append({
                "content_id": vector_data['id'],
                "content_type": content_type,
                "preview": self._create_content_preview(content_type, metadata),
                "connection_distance": hop_count,
                "recommendation_reason": self._get_recommendation_reason(content_id, vector_data['id'], hop_count)
            })
        
        # Sort by connection distance and limit
        recommendations.sort(key=lambda x: x['connection_distance'])
        recommendations = recommendations[:max_recommendations]
        
        print(f"   ✅ Generated {len(recommendations)} cross-modal recommendations")
        
        return recommendations
    
    def get_assistant_analytics(self):
        """Get comprehensive analytics about the multi-modal assistant"""
        
        stats = self.db.get_statistics()
        
        # Analyze content distribution
        content_by_type = {}
        total_relationships = 0
        relationship_types = {}
        
        for content_id in self.db.list_vectors():
            content = self.db.get_vector(content_id)
            if content:
                content_type = content['metadata'].get('content_type', 'unknown')
                content_by_type[content_type] = content_by_type.get(content_type, 0) + 1
                
                # Count relationships
                relationships = self.db.get_relationships(content_id)
                total_relationships += len(relationships)
                
                for rel in relationships:
                    rel_type = rel.get('relationship_type', 'unknown')
                    relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        # Conversation analytics
        conversation_stats = {
            "total_turns": len(self.conversation_history),
            "avg_query_length": np.mean([len(turn["user_query"]) for turn in self.conversation_history]) if self.conversation_history else 0,
            "avg_response_length": np.mean([len(turn["assistant_response"]) for turn in self.conversation_history]) if self.conversation_history else 0
        }
        
        return {
            "assistant_info": {
                "name": self.assistant_name,
                "total_content": stats['vector_count'],
                "total_relationships": stats['relationship_count'],
                "dimension": stats['dimension']
            },
            "content_distribution": content_by_type,
            "relationship_analytics": {
                "total_relationships": total_relationships,
                "types": relationship_types
            },
            "conversation_analytics": conversation_stats,
            "capacity_usage": stats['capacity_usage']
        }
    
    # Helper methods for embeddings and processing
    def _generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding (simulated)"""
        # In production, use actual embedding model like sentence-transformers
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        return np.random.rand(384).astype(np.float32)
    
    def _generate_image_embedding(self, description: str) -> np.ndarray:
        """Generate image embedding based on description (simulated)"""
        # In production, use vision model like CLIP
        # For demo, use text embedding with slight variation
        text_emb = self._generate_text_embedding(description)
        # Add some "visual" variation
        visual_noise = np.random.rand(384).astype(np.float32) * 0.1
        return (text_emb + visual_noise) / 2
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simplified)"""
        # In production, use NLP library or model
        words = text.lower().split()
        # Simple keyword extraction - in production use TF-IDF or similar
        keywords = [word for word in words if len(word) > 4][:5]
        return keywords
    
    def _classify_intent(self, query: str) -> str:
        """Classify user query intent (simplified)"""
        query_lower = query.lower()
        if any(word in query_lower for word in ["what", "how", "why", "when", "where"]):
            return "question"
        elif any(word in query_lower for word in ["find", "search", "show", "get"]):
            return "search"
        elif any(word in query_lower for word in ["recommend", "suggest"]):
            return "recommendation"
        else:
            return "general"
    
    def _classify_response_type(self, response: str) -> str:
        """Classify assistant response type (simplified)"""
        if len(response) > 200:
            return "detailed"
        elif "?" in response:
            return "clarifying"
        else:
            return "informative"
    
    def _are_semantically_related(self, metadata1: Dict, metadata2: Dict) -> bool:
        """Check if two pieces of content are semantically related"""
        # Check for topic overlap
        topics1 = set([metadata1.get("topic", ""), metadata1.get("subject", "")])
        topics2 = set([metadata2.get("topic", ""), metadata2.get("subject", "")])
        
        if topics1 & topics2:
            return True
        
        # Check for keyword overlap
        keywords1 = set(metadata1.get("keywords", []))
        keywords2 = set(metadata2.get("keywords", []))
        
        return len(keywords1 & keywords2) >= 1
    
    def _calculate_semantic_strength(self, metadata1: Dict, metadata2: Dict) -> float:
        """Calculate semantic relationship strength"""
        strength = 0.3  # Base strength
        
        # Same topic boost
        if metadata1.get("topic") == metadata2.get("topic"):
            strength += 0.3
        
        # Keyword overlap boost
        keywords1 = set(metadata1.get("keywords", []))
        keywords2 = set(metadata2.get("keywords", []))
        overlap = len(keywords1 & keywords2)
        strength += min(0.3, overlap * 0.1)
        
        return min(0.9, strength)
    
    def _find_shared_topics(self, metadata1: Dict, metadata2: Dict) -> List[str]:
        """Find shared topics between content"""
        topics1 = [metadata1.get("topic", ""), metadata1.get("subject", "")]
        topics2 = [metadata2.get("topic", ""), metadata2.get("subject", "")]
        
        shared = []
        for topic in topics1:
            if topic and topic in topics2:
                shared.append(topic)
        
        return shared
    
    def _create_content_preview(self, content_type: str, metadata: Dict) -> str:
        """Create preview text for different content types"""
        if content_type == "text":
            return metadata.get("text", "")[:100] + "..."
        elif content_type == "image":
            return f"Image: {metadata.get('description', 'No description')}"
        elif content_type == "document":
            return f"Document: {metadata.get('title', 'Untitled')} ({metadata.get('page_count', 1)} pages)"
        elif content_type == "conversation":
            return f"Conversation: {metadata.get('user_query', '')[:50]}..."
        else:
            return "Unknown content type"
    
    def _create_metadata_summary(self, metadata: Dict) -> Dict:
        """Create summary of metadata for display"""
        return {
            "type": metadata.get("content_type", "unknown"),
            "created": metadata.get("created_at", "unknown"),
            "topic": metadata.get("topic", "general"),
            "source": metadata.get("source", "unknown")
        }
    
    def _synthesize_text_context(self, contexts: List[Dict]) -> str:
        """Synthesize text context into response"""
        if not contexts:
            return ""
        
        topics = [ctx.get('metadata_summary', {}).get('topic', 'general') for ctx in contexts]
        return f"I found information about {', '.join(set(topics))} in the text content"
    
    def _synthesize_document_context(self, contexts: List[Dict]) -> str:
        """Synthesize document context into response"""
        if not contexts:
            return ""
        
        doc_count = len(contexts)
        return f"I referenced {doc_count} document{'s' if doc_count > 1 else ''} with relevant information"
    
    def _synthesize_image_context(self, contexts: List[Dict]) -> str:
        """Synthesize image context into response"""
        if not contexts:
            return ""
        
        return f"I found {len(contexts)} relevant visual content that relates to your query"
    
    def _synthesize_conversation_context(self, contexts: List[Dict]) -> str:
        """Synthesize conversation context into response"""
        if not contexts:
            return ""
        
        return "I can see we've discussed related topics before"
    
    def _get_recommendation_reason(self, source_id: str, target_id: str, hop_count: int) -> str:
        """Get human-readable recommendation reason"""
        if hop_count == 0:
            return "Directly similar content"
        elif hop_count == 1:
            return "Directly connected content"
        else:
            return f"Related through {hop_count}-step connection"

def create_sample_multimodal_content():
    """Create sample multi-modal content for demonstration"""
    
    sample_content = {
        "text_content": [
            {
                "id": "ai_overview_text",
                "text": "Artificial Intelligence is revolutionizing industries by enabling machines to perform tasks that typically require human intelligence. From healthcare to finance, AI applications are becoming increasingly sophisticated and impactful.",
                "metadata": {
                    "topic": "artificial_intelligence",
                    "source": "educational_article",
                    "sentiment": "positive"
                }
            },
            {
                "id": "ml_algorithms_text",
                "text": "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning. Each category serves different purposes and is suitable for various types of problems in data science.",
                "metadata": {
                    "topic": "machine_learning",
                    "source": "technical_guide",
                    "sentiment": "neutral"
                }
            }
        ],
        
        "image_content": [
            {
                "id": "ai_brain_visualization",
                "description": "A colorful digital brain illustration showing neural network connections, representing artificial intelligence and machine learning concepts",
                "metadata": {
                    "detected_objects": ["brain", "neurons", "network"],
                    "colors": ["blue", "purple", "cyan"],
                    "style": "digital_art",
                    "scene_description": "Technology visualization"
                }
            },
            {
                "id": "ml_workflow_diagram",
                "description": "A flowchart diagram showing the machine learning workflow: data collection, preprocessing, model training, evaluation, and deployment",
                "metadata": {
                    "detected_objects": ["flowchart", "arrows", "boxes"],
                    "colors": ["white", "blue", "gray"],
                    "style": "diagram",
                    "scene_description": "Technical workflow"
                }
            }
        ],
        
        "document_content": [
            {
                "id": "ai_research_paper",
                "text": "This comprehensive research paper explores the latest advances in artificial intelligence, focusing on deep learning architectures and their applications in computer vision and natural language processing. The paper presents novel approaches to improving model efficiency and accuracy.",
                "metadata": {
                    "title": "Advances in Deep Learning for AI Applications",
                    "author": "Dr. Sarah Chen",
                    "page_count": 15,
                    "summary": "Latest AI advances in deep learning",
                    "topics_covered": ["deep_learning", "computer_vision", "nlp"],
                    "complexity_level": "advanced"
                }
            },
            {
                "id": "ml_tutorial_doc",
                "text": "This beginner-friendly tutorial introduces machine learning concepts through practical examples. It covers data preparation, algorithm selection, model training, and evaluation metrics, with hands-on Python code examples using scikit-learn.",
                "metadata": {
                    "title": "Machine Learning Tutorial for Beginners",
                    "author": "ML Academy",
                    "page_count": 8,
                    "summary": "Beginner ML tutorial with Python examples",
                    "topics_covered": ["machine_learning", "python", "scikit_learn"],
                    "complexity_level": "beginner"
                }
            }
        ]
    }
    
    return sample_content

def main():
    """Demonstrate multi-modal AI assistant capabilities"""
    
    print("🌐 Multi-Modal AI Assistant Demo")
    print("=" * 50)
    
    # Create multi-modal AI assistant
    assistant = MultiModal_AI_Assistant("RudraBot Multi-Modal")
    
    # Add sample multi-modal content
    print(f"\n📚 Adding multi-modal content...")
    sample_content = create_sample_multimodal_content()
    
    # Add text content
    for text_item in sample_content["text_content"]:
        assistant.add_text_content(text_item["id"], text_item["text"], text_item["metadata"])
    
    # Add image content
    for image_item in sample_content["image_content"]:
        assistant.add_image_content(image_item["id"], image_item["description"], None, image_item["metadata"])
    
    # Add document content
    for doc_item in sample_content["document_content"]:
        assistant.add_document_content(doc_item["id"], doc_item["text"], "pdf", doc_item["metadata"])
    
    # Build multi-modal relationships
    relationships_count = assistant.build_multimodal_relationships()
    
    # Demonstrate multi-modal search
    print(f"\n🔍 Multi-Modal Search Demonstrations:")
    
    # Search 1: Comprehensive search across all modalities
    results1, by_type1 = assistant.multimodal_search(
        "machine learning algorithms and techniques",
        search_strategy="comprehensive",
        max_results=6
    )
    
    print(f"\n   📊 Comprehensive Search Results:")
    for result in results1[:3]:
        print(f"      🔸 {result['content_type']}: {result['preview'][:60]}...")
        print(f"         Connection: {result['connection_type']} (score: {result['combined_score']:.3f})")
    
    # Search 2: Image-specific search
    results2, by_type2 = assistant.multimodal_search(
        "neural networks and brain visualization",
        content_types=["image"],
        search_strategy="precise",
        max_results=3
    )
    
    print(f"\n   🖼️ Image-Specific Search Results:")
    for result in results2:
        print(f"      🔸 {result['content_type']}: {result['preview']}")
        print(f"         Score: {result['similarity_score']:.3f}")
    
    # Generate multi-modal responses
    print(f"\n🤖 Multi-Modal Response Generation:")
    
    queries = [
        "What is artificial intelligence?",
        "Show me information about machine learning workflows",
        "How do neural networks work?"
    ]
    
    for query in queries:
        response_data = assistant.generate_multimodal_response(query, context_limit=4)
        
        print(f"\n   ❓ Query: {query}")
        print(f"   🤖 Response: {response_data['response'][:150]}...")
        print(f"   📊 Context types used: {response_data['context_types']}")
        print(f"   🔗 Sources: {len(response_data['context_used'])} pieces of content")
    
    # Generate content recommendations
    print(f"\n💡 Multi-Modal Content Recommendations:")
    
    recommendations = assistant.recommend_content("ai_overview_text", max_recommendations=4)
    
    print(f"   🎯 Recommendations for 'AI Overview Text':")
    for rec in recommendations:
        print(f"      🔸 {rec['content_type']}: {rec['preview'][:50]}...")
        print(f"         Connection: {rec['connection_distance']} hops - {rec['recommendation_reason']}")
    
    # Assistant analytics
    analytics = assistant.get_assistant_analytics()
    
    print(f"\n📊 Multi-Modal Assistant Analytics:")
    assistant_info = analytics['assistant_info']
    print(f"   🤖 Assistant: {assistant_info['name']}")
    print(f"   📚 Content: {assistant_info['total_content']} items, {assistant_info['total_relationships']} relationships")
    print(f"   🎯 Dimensions: {assistant_info['dimension']}D embeddings")
    
    content_dist = analytics['content_distribution']
    print(f"   📊 Content Distribution: {dict(content_dist)}")
    
    relationship_analytics = analytics['relationship_analytics']
    print(f"   🔗 Relationship Types: {dict(relationship_analytics['types'])}")
    
    conversation_analytics = analytics['conversation_analytics']
    print(f"   💬 Conversations: {conversation_analytics['total_turns']} turns")
    
    capacity_usage = analytics['capacity_usage']
    print(f"   📈 Capacity: {capacity_usage['vector_usage_percent']:.1f}% content, {capacity_usage['relationship_usage_percent']:.1f}% relationships")
    
    print(f"\n✨ Key Multi-Modal Features Demonstrated:")
    features = [
        f"Added {len(sample_content['text_content'])} text, {len(sample_content['image_content'])} images, {len(sample_content['document_content'])} documents",
        f"Built {relationships_count} cross-modal relationships automatically",
        f"Performed unified search across all content types",
        f"Generated context-aware responses using multi-modal information",
        f"Provided intelligent recommendations across different modalities",
        f"Delivered comprehensive analytics for multi-modal content system"
    ]
    
    for feature in features:
        print(f"   💡 {feature}")
    
    print(f"\n🎉 Multi-Modal AI Assistant Demo Complete!")
    print(f"   Perfect example of relationship-aware search enabling sophisticated AI!")
    
    # Show upgrade path for scaling
    if capacity_usage['vector_usage_percent'] > 20 or capacity_usage['relationship_usage_percent'] > 20:
        print(f"\n🚀 Ready to Scale Your Multi-Modal AI?")
        print(f"   Upgrade to full RudraDB for unlimited multi-modal content capacity!")

if __name__ == "__main__":
    try:
        import rudradb
        import numpy as np
        
        print(f"🎯 Using RudraDB-Opin v{rudradb.__version__}")
        print(f"🌐 Perfect for multi-modal AI applications!")
        
        main()
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install rudradb-opin numpy")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure RudraDB-Opin is properly installed")
