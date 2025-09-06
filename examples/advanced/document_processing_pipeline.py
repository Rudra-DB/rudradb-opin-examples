#!/usr/bin/env python3
"""
📄 Complete Document Processing Pipeline with RudraDB-Opin

This example demonstrates a comprehensive document processing system that:

1. Ingests various document types (PDF, Word, text files, web pages)
2. Extracts and processes text content with intelligent chunking
3. Builds hierarchical document relationships (document -> sections -> chunks)
4. Creates semantic relationships between related content
5. Enables intelligent document search and discovery
6. Provides document analytics and insights

Perfect example of enterprise document management powered by relationship-aware search!
"""

import rudradb
import numpy as np
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import hashlib
import urllib.parse

class Document_Processing_Pipeline:
    """Complete document processing pipeline using RudraDB-Opin"""
    
    def __init__(self, pipeline_name: str = "DocuMind"):
        self.db = rudradb.RudraDB()  # Auto-dimension detection
        self.pipeline_name = pipeline_name
        self.processing_stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "relationships_built": 0,
            "processing_errors": 0
        }
        
        print(f"📄 Initializing Document Processing Pipeline: {pipeline_name}")
        print(f"   🤖 Auto-dimension detection for all document types")
        print(f"   🔗 Intelligent document relationship building")
        
    def ingest_document(self, doc_id: str, content: str, doc_type: str = "text", 
                       source_info: Dict = None, processing_options: Dict = None):
        """Ingest and process a document with intelligent chunking"""
        
        print(f"\n📥 Ingesting document: {doc_id} ({doc_type})")
        
        try:
            # Extract document metadata
            doc_metadata = self._extract_document_metadata(content, doc_type, source_info or {})
            
            # Create main document vector
            doc_embedding = self._generate_document_embedding(content)
            
            enhanced_doc_metadata = {
                "content_type": "document",
                "document_type": doc_type,
                "title": doc_metadata.get("title", doc_id),
                "author": doc_metadata.get("author", "unknown"),
                "creation_date": doc_metadata.get("creation_date"),
                "word_count": doc_metadata["word_count"],
                "char_count": doc_metadata["char_count"],
                "language": doc_metadata.get("language", "en"),
                "summary": doc_metadata.get("summary", ""),
                "key_topics": doc_metadata.get("key_topics", []),
                "complexity_score": doc_metadata.get("complexity_score", 0.5),
                "ingestion_timestamp": datetime.now().isoformat(),
                "processing_version": "1.0",
                **(source_info or {})
            }
            
            self.db.add_vector(doc_id, doc_embedding, enhanced_doc_metadata)
            
            # Process document sections if it's structured
            sections = self._extract_sections(content, doc_type)
            section_ids = []
            
            if len(sections) > 1:  # Multi-section document
                for i, section in enumerate(sections):
                    section_id = f"{doc_id}_section_{i+1}"
                    self._process_document_section(section_id, section, doc_id, i+1)
                    section_ids.append(section_id)
            
            # Create intelligent chunks
            chunk_options = processing_options.get("chunking", {}) if processing_options else {}
            chunk_ids = self._create_intelligent_chunks(doc_id, content, chunk_options)
            
            # Build document structure relationships
            self._build_document_structure_relationships(doc_id, section_ids, chunk_ids)
            
            self.processing_stats["documents_processed"] += 1
            
            print(f"   ✅ Processed: {len(section_ids)} sections, {len(chunk_ids)} chunks")
            
            return {
                "document_id": doc_id,
                "sections": section_ids,
                "chunks": chunk_ids,
                "metadata": enhanced_doc_metadata
            }
            
        except Exception as e:
            self.processing_stats["processing_errors"] += 1
            print(f"   ❌ Error processing {doc_id}: {e}")
            raise
    
    def _process_document_section(self, section_id: str, section_content: Dict, 
                                 parent_doc_id: str, section_number: int):
        """Process individual document section"""
        
        section_embedding = self._generate_document_embedding(section_content["content"])
        
        section_metadata = {
            "content_type": "document_section",
            "parent_document": parent_doc_id,
            "section_number": section_number,
            "section_title": section_content.get("title", f"Section {section_number}"),
            "section_content": section_content["content"][:300],  # Preview
            "word_count": len(section_content["content"].split()),
            "char_count": len(section_content["content"]),
            "section_type": section_content.get("type", "content"),  # intro, body, conclusion, etc.
            "key_topics": self._extract_topics(section_content["content"]),
            "created_at": datetime.now().isoformat()
        }
        
        self.db.add_vector(section_id, section_embedding, section_metadata)
        
        return section_id
    
    def _create_intelligent_chunks(self, doc_id: str, content: str, 
                                  chunk_options: Dict = None) -> List[str]:
        """Create intelligent document chunks with overlap and context preservation"""
        
        options = {
            "chunk_size": 500,  # words
            "overlap_size": 50,  # words
            "respect_paragraphs": True,
            "respect_sentences": True,
            **chunk_options or {}
        }
        
        # Split content into chunks intelligently
        chunks = self._intelligent_text_chunking(content, options)
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i+1:03d}"
            
            chunk_embedding = self._generate_document_embedding(chunk["content"])
            
            chunk_metadata = {
                "content_type": "document_chunk",
                "parent_document": doc_id,
                "chunk_number": i + 1,
                "chunk_content": chunk["content"],
                "word_count": len(chunk["content"].split()),
                "char_count": len(chunk["content"]),
                "start_position": chunk["start_pos"],
                "end_position": chunk["end_pos"],
                "chunk_type": chunk.get("type", "content"),  # paragraph, section, etc.
                "has_overlap": chunk.get("has_overlap", False),
                "key_phrases": self._extract_key_phrases(chunk["content"]),
                "sentence_count": len(self._split_into_sentences(chunk["content"])),
                "created_at": datetime.now().isoformat()
            }
            
            self.db.add_vector(chunk_id, chunk_embedding, chunk_metadata)
            chunk_ids.append(chunk_id)
            self.processing_stats["chunks_created"] += 1
        
        return chunk_ids
    
    def _build_document_structure_relationships(self, doc_id: str, 
                                              section_ids: List[str], 
                                              chunk_ids: List[str]):
        """Build hierarchical relationships within document structure"""
        
        relationships_built = 0
        
        # Document -> Sections (hierarchical)
        for section_id in section_ids:
            self.db.add_relationship(doc_id, section_id, "hierarchical", 0.95, {
                "relationship_type": "document_section",
                "structure_level": "document_to_section"
            })
            relationships_built += 1
        
        # Sections -> Sequential (temporal)
        for i in range(len(section_ids) - 1):
            self.db.add_relationship(section_ids[i], section_ids[i+1], "temporal", 0.9, {
                "relationship_type": "section_sequence",
                "sequence_order": i + 1
            })
            relationships_built += 1
        
        # Document -> Chunks (hierarchical)
        for chunk_id in chunk_ids:
            self.db.add_relationship(doc_id, chunk_id, "hierarchical", 0.85, {
                "relationship_type": "document_chunk",
                "structure_level": "document_to_chunk"
            })
            relationships_built += 1
        
        # Sequential chunks (temporal)
        for i in range(len(chunk_ids) - 1):
            self.db.add_relationship(chunk_ids[i], chunk_ids[i+1], "temporal", 0.8, {
                "relationship_type": "chunk_sequence",
                "sequence_order": i + 1
            })
            relationships_built += 1
        
        # Overlapping chunks (associative)
        for i in range(len(chunk_ids) - 1):
            chunk = self.db.get_vector(chunk_ids[i])
            next_chunk = self.db.get_vector(chunk_ids[i+1])
            
            if (chunk and next_chunk and 
                chunk['metadata'].get('has_overlap') and 
                next_chunk['metadata'].get('has_overlap')):
                
                self.db.add_relationship(chunk_ids[i], chunk_ids[i+1], "associative", 0.7, {
                    "relationship_type": "chunk_overlap",
                    "overlap_reason": "content_overlap"
                })
                relationships_built += 1
        
        self.processing_stats["relationships_built"] += relationships_built
        
        return relationships_built
    
    def build_cross_document_relationships(self, similarity_threshold: float = 0.7,
                                         max_relationships_per_doc: int = 10):
        """Build semantic relationships between different documents"""
        
        print(f"\n🔗 Building cross-document relationships...")
        print(f"   🎯 Similarity threshold: {similarity_threshold}")
        
        documents = [vid for vid in self.db.list_vectors() 
                    if self.db.get_vector(vid)['metadata'].get('content_type') == 'document']
        
        relationships_built = 0
        
        for i, doc_id1 in enumerate(documents):
            doc1 = self.db.get_vector(doc_id1)
            if not doc1:
                continue
                
            doc1_metadata = doc1['metadata']
            doc_relationships = 0
            
            for doc_id2 in documents[i+1:]:
                if relationships_built >= 200 or doc_relationships >= max_relationships_per_doc:
                    break
                    
                doc2 = self.db.get_vector(doc_id2)
                if not doc2:
                    continue
                
                doc2_metadata = doc2['metadata']
                
                # Calculate document similarity
                similarity_score = self._calculate_document_similarity(doc1_metadata, doc2_metadata)
                
                if similarity_score >= similarity_threshold:
                    # Determine relationship type based on similarity characteristics
                    rel_type, strength = self._determine_relationship_type(
                        doc1_metadata, doc2_metadata, similarity_score
                    )
                    
                    self.db.add_relationship(doc_id1, doc_id2, rel_type, strength, {
                        "relationship_type": "cross_document",
                        "similarity_score": similarity_score,
                        "similarity_basis": self._get_similarity_basis(doc1_metadata, doc2_metadata)
                    })
                    
                    relationships_built += 1
                    doc_relationships += 1
        
        print(f"   ✅ Built {relationships_built} cross-document relationships")
        
        # Also build chunk-level cross-document relationships
        chunk_relationships = self._build_cross_chunk_relationships(similarity_threshold * 0.8)
        
        print(f"   ✅ Built {chunk_relationships} cross-chunk relationships")
        
        self.processing_stats["relationships_built"] += relationships_built + chunk_relationships
        
        return relationships_built + chunk_relationships
    
    def _build_cross_chunk_relationships(self, similarity_threshold: float) -> int:
        """Build relationships between chunks from different documents"""
        
        chunks = [vid for vid in self.db.list_vectors() 
                 if self.db.get_vector(vid)['metadata'].get('content_type') == 'document_chunk']
        
        relationships_built = 0
        
        # Sample chunks to avoid too many relationships in Opin
        chunk_sample = chunks[:50] if len(chunks) > 50 else chunks
        
        for i, chunk_id1 in enumerate(chunk_sample):
            chunk1 = self.db.get_vector(chunk_id1)
            if not chunk1:
                continue
                
            chunk1_doc = chunk1['metadata']['parent_document']
            
            for chunk_id2 in chunk_sample[i+1:]:
                if relationships_built >= 100:  # Limit for Opin
                    break
                    
                chunk2 = self.db.get_vector(chunk_id2)
                if not chunk2:
                    continue
                
                chunk2_doc = chunk2['metadata']['parent_document']
                
                # Only relate chunks from different documents
                if chunk1_doc != chunk2_doc:
                    # Calculate chunk similarity
                    chunk_similarity = self._calculate_chunk_similarity(chunk1['metadata'], chunk2['metadata'])
                    
                    if chunk_similarity >= similarity_threshold:
                        self.db.add_relationship(chunk_id1, chunk_id2, "semantic", chunk_similarity, {
                            "relationship_type": "cross_document_chunk",
                            "chunk_similarity": chunk_similarity,
                            "different_documents": True
                        })
                        relationships_built += 1
        
        return relationships_built
    
    def search_documents(self, query: str, search_scope: str = "all", 
                        max_results: int = 10, search_strategy: str = "comprehensive"):
        """Search documents with various scopes and strategies"""
        
        print(f"\n🔍 Document search: '{query}'")
        print(f"   📊 Scope: {search_scope}, Strategy: {search_strategy}")
        
        # Generate query embedding
        query_embedding = self._generate_document_embedding(query)
        
        # Configure search based on strategy
        if search_strategy == "comprehensive":
            params = rudradb.SearchParams(
                top_k=max_results * 2,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.4,
                similarity_threshold=0.1
            )
        elif search_strategy == "precise":
            params = rudradb.SearchParams(
                top_k=max_results,
                include_relationships=False,
                similarity_threshold=0.5
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
        
        # Filter results by search scope
        filtered_results = []
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            if not vector:
                continue
                
            content_type = vector['metadata'].get('content_type')
            
            if search_scope == "all":
                filtered_results.append(result)
            elif search_scope == "documents" and content_type == "document":
                filtered_results.append(result)
            elif search_scope == "sections" and content_type == "document_section":
                filtered_results.append(result)
            elif search_scope == "chunks" and content_type == "document_chunk":
                filtered_results.append(result)
        
        # Limit results
        filtered_results = filtered_results[:max_results]
        
        # Format results with document context
        formatted_results = []
        for result in filtered_results:
            vector = self.db.get_vector(result.vector_id)
            metadata = vector['metadata']
            content_type = metadata.get('content_type')
            
            # Get parent document info if this is a chunk or section
            parent_doc_info = None
            if content_type in ['document_section', 'document_chunk']:
                parent_doc_id = metadata.get('parent_document')
                if parent_doc_id:
                    parent_doc = self.db.get_vector(parent_doc_id)
                    if parent_doc:
                        parent_doc_info = {
                            "title": parent_doc['metadata'].get('title', parent_doc_id),
                            "author": parent_doc['metadata'].get('author', 'unknown'),
                            "document_type": parent_doc['metadata'].get('document_type', 'unknown')
                        }
            
            formatted_results.append({
                "content_id": result.vector_id,
                "content_type": content_type,
                "title": self._get_content_title(metadata),
                "preview": self._get_content_preview(metadata),
                "similarity_score": result.similarity_score,
                "combined_score": result.combined_score,
                "connection_type": "direct" if result.hop_count == 0 else f"{result.hop_count}-hop",
                "hop_count": result.hop_count,
                "parent_document": parent_doc_info,
                "metadata_summary": self._create_search_metadata_summary(metadata)
            })
        
        print(f"   ✅ Found {len(formatted_results)} results")
        
        # Group results by content type
        by_type = {}
        for result in formatted_results:
            content_type = result['content_type']
            if content_type not in by_type:
                by_type[content_type] = []
            by_type[content_type].append(result)
        
        return formatted_results, by_type
    
    def get_document_insights(self, doc_id: str) -> Dict:
        """Get comprehensive insights about a specific document"""
        
        doc = self.db.get_vector(doc_id)
        if not doc:
            return {"error": "Document not found"}
        
        print(f"\n📊 Generating insights for: {doc_id}")
        
        doc_metadata = doc['metadata']
        
        # Find all related content (sections, chunks)
        related_content = self.db.get_connected_vectors(doc_id, max_hops=1)
        
        sections = []
        chunks = []
        related_docs = []
        
        for vector_data, hop_count in related_content:
            content_type = vector_data['metadata'].get('content_type')
            if content_type == 'document_section':
                sections.append(vector_data)
            elif content_type == 'document_chunk':
                chunks.append(vector_data)
            elif content_type == 'document' and vector_data['id'] != doc_id:
                related_docs.append(vector_data)
        
        # Calculate document statistics
        total_words = doc_metadata.get('word_count', 0)
        chunk_words = sum(chunk['metadata'].get('word_count', 0) for chunk in chunks)
        
        # Analyze relationships
        relationships = self.db.get_relationships(doc_id)
        relationship_analysis = self._analyze_relationships(relationships)
        
        insights = {
            "document_info": {
                "id": doc_id,
                "title": doc_metadata.get('title', doc_id),
                "author": doc_metadata.get('author', 'unknown'),
                "document_type": doc_metadata.get('document_type', 'unknown'),
                "word_count": total_words,
                "complexity_score": doc_metadata.get('complexity_score', 0.5)
            },
            "content_structure": {
                "total_sections": len(sections),
                "total_chunks": len(chunks),
                "words_in_chunks": chunk_words,
                "coverage_percentage": (chunk_words / total_words * 100) if total_words > 0 else 0
            },
            "relationship_analysis": relationship_analysis,
            "related_documents": len(related_docs),
            "key_topics": doc_metadata.get('key_topics', []),
            "processing_info": {
                "ingestion_date": doc_metadata.get('ingestion_timestamp'),
                "processing_version": doc_metadata.get('processing_version', '1.0')
            }
        }
        
        return insights
    
    def get_pipeline_analytics(self) -> Dict:
        """Get comprehensive analytics about the document processing pipeline"""
        
        stats = self.db.get_statistics()
        
        # Analyze content distribution
        content_distribution = {
            "document": 0,
            "document_section": 0,
            "document_chunk": 0
        }
        
        document_types = {}
        total_word_count = 0
        complexity_scores = []
        
        for content_id in self.db.list_vectors():
            content = self.db.get_vector(content_id)
            if content:
                metadata = content['metadata']
                content_type = metadata.get('content_type', 'unknown')
                
                if content_type in content_distribution:
                    content_distribution[content_type] += 1
                
                if content_type == 'document':
                    doc_type = metadata.get('document_type', 'unknown')
                    document_types[doc_type] = document_types.get(doc_type, 0) + 1
                    
                    word_count = metadata.get('word_count', 0)
                    total_word_count += word_count
                    
                    complexity = metadata.get('complexity_score')
                    if complexity:
                        complexity_scores.append(complexity)
        
        # Relationship analytics
        relationship_types = {}
        for content_id in self.db.list_vectors():
            relationships = self.db.get_relationships(content_id)
            for rel in relationships:
                rel_type = rel.get('relationship_type', 'unknown')
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            "pipeline_info": {
                "name": self.pipeline_name,
                "total_content_items": stats['vector_count'],
                "total_relationships": stats['relationship_count'],
                "dimension": stats['dimension']
            },
            "processing_stats": self.processing_stats,
            "content_distribution": content_distribution,
            "document_types": document_types,
            "content_analytics": {
                "total_words_processed": total_word_count,
                "average_document_complexity": np.mean(complexity_scores) if complexity_scores else 0,
                "complexity_std": np.std(complexity_scores) if complexity_scores else 0
            },
            "relationship_analytics": relationship_types,
            "capacity_usage": stats['capacity_usage']
        }
    
    # Helper methods for document processing
    def _extract_document_metadata(self, content: str, doc_type: str, source_info: Dict) -> Dict:
        """Extract metadata from document content"""
        
        word_count = len(content.split())
        char_count = len(content)
        
        # Extract title (first line or from source info)
        title = source_info.get('title')
        if not title:
            lines = content.split('\n')
            title = lines[0].strip()[:100] if lines else "Untitled"
        
        # Simple complexity scoring based on sentence length and vocabulary
        sentences = self._split_into_sentences(content)
        avg_sentence_length = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        complexity_score = min(1.0, avg_sentence_length / 20)  # Simple heuristic
        
        return {
            "title": title,
            "author": source_info.get('author', 'unknown'),
            "creation_date": source_info.get('creation_date'),
            "word_count": word_count,
            "char_count": char_count,
            "language": source_info.get('language', 'en'),
            "summary": self._generate_summary(content),
            "key_topics": self._extract_topics(content),
            "complexity_score": complexity_score
        }
    
    def _extract_sections(self, content: str, doc_type: str) -> List[Dict]:
        """Extract sections from structured documents"""
        
        # Simple section detection - in production use more sophisticated parsing
        if doc_type in ['pdf', 'word', 'markdown']:
            # Look for headers (lines that start with numbers, caps, or markdown headers)
            lines = content.split('\n')
            sections = []
            current_section = {"title": "Introduction", "content": "", "type": "intro"}
            
            for line in lines:
                line = line.strip()
                if not line:
                    current_section["content"] += "\n"
                    continue
                
                # Simple header detection
                if (line.isupper() or 
                    re.match(r'^\d+\.', line) or 
                    line.startswith('#')):
                    
                    # Save current section if it has content
                    if current_section["content"].strip():
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        "title": line[:50],
                        "content": "",
                        "type": "body"
                    }
                else:
                    current_section["content"] += line + "\n"
            
            # Add final section
            if current_section["content"].strip():
                sections.append(current_section)
            
            return sections if len(sections) > 1 else [{"title": "Content", "content": content, "type": "body"}]
        
        return [{"title": "Content", "content": content, "type": "body"}]
    
    def _intelligent_text_chunking(self, content: str, options: Dict) -> List[Dict]:
        """Intelligent text chunking that respects paragraphs and sentences"""
        
        chunk_size = options.get("chunk_size", 500)  # words
        overlap_size = options.get("overlap_size", 50)  # words
        
        # Split into paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_word_count = 0
        start_pos = 0
        
        for paragraph in paragraphs:
            paragraph_words = len(paragraph.split())
            
            # If adding this paragraph would exceed chunk size
            if current_word_count + paragraph_words > chunk_size and current_chunk:
                # Create chunk with current content
                chunks.append({
                    "content": current_chunk.strip(),
                    "start_pos": start_pos,
                    "end_pos": start_pos + len(current_chunk),
                    "type": "paragraph",
                    "has_overlap": False
                })
                
                # Start new chunk with overlap
                if overlap_size > 0 and current_word_count >= overlap_size:
                    words = current_chunk.split()
                    overlap_content = " ".join(words[-overlap_size:])
                    current_chunk = overlap_content + " " + paragraph
                    current_word_count = overlap_size + paragraph_words
                else:
                    start_pos += len(current_chunk)
                    current_chunk = paragraph
                    current_word_count = paragraph_words
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    start_pos = content.find(paragraph)
                
                current_word_count += paragraph_words
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "start_pos": start_pos,
                "end_pos": start_pos + len(current_chunk),
                "type": "paragraph",
                "has_overlap": False
            })
        
        # Mark chunks with overlap
        for i in range(1, len(chunks)):
            if chunks[i]["content"].startswith(chunks[i-1]["content"].split()[-overlap_size:][0] if overlap_size > 0 else ""):
                chunks[i]["has_overlap"] = True
        
        return chunks
    
    def _generate_document_embedding(self, text: str) -> np.ndarray:
        """Generate document embedding (simulated)"""
        # In production, use actual embedding model
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        return np.random.rand(384).astype(np.float32)
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text (simplified)"""
        # In production, use NLP library or topic modeling
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most frequent words as topics
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [topic[0] for topic in topics]
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text chunk"""
        # Simple phrase extraction - in production use NLP
        sentences = self._split_into_sentences(text)
        phrases = []
        
        for sentence in sentences[:3]:  # First 3 sentences
            words = sentence.split()
            if len(words) >= 3:
                # Extract noun phrases (simplified)
                for i in range(len(words) - 2):
                    phrase = " ".join(words[i:i+3])
                    if len(phrase) > 10:
                        phrases.append(phrase)
        
        return phrases[:5]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate document summary (simplified)"""
        # In production, use summarization model
        sentences = self._split_into_sentences(content)
        if len(sentences) <= 3:
            return content[:max_length]
        
        # Take first and last sentences plus a middle one
        summary_sentences = [sentences[0]]
        if len(sentences) > 2:
            summary_sentences.append(sentences[len(sentences)//2])
        summary_sentences.append(sentences[-1])
        
        summary = " ".join(summary_sentences)
        return summary[:max_length] + "..." if len(summary) > max_length else summary
    
    def _calculate_document_similarity(self, doc1_metadata: Dict, doc2_metadata: Dict) -> float:
        """Calculate similarity between two documents"""
        similarity = 0.0
        
        # Topic overlap
        topics1 = set(doc1_metadata.get('key_topics', []))
        topics2 = set(doc2_metadata.get('key_topics', []))
        topic_overlap = len(topics1 & topics2)
        if topic_overlap > 0:
            similarity += topic_overlap * 0.3
        
        # Same author
        if (doc1_metadata.get('author', '') == doc2_metadata.get('author', '') and 
            doc1_metadata.get('author') != 'unknown'):
            similarity += 0.2
        
        # Same document type
        if doc1_metadata.get('document_type') == doc2_metadata.get('document_type'):
            similarity += 0.1
        
        # Similar complexity
        complexity1 = doc1_metadata.get('complexity_score', 0.5)
        complexity2 = doc2_metadata.get('complexity_score', 0.5)
        complexity_diff = abs(complexity1 - complexity2)
        if complexity_diff < 0.2:
            similarity += 0.1
        
        return min(1.0, similarity)
    
    def _calculate_chunk_similarity(self, chunk1_metadata: Dict, chunk2_metadata: Dict) -> float:
        """Calculate similarity between chunks from different documents"""
        similarity = 0.0
        
        # Key phrase overlap
        phrases1 = set(chunk1_metadata.get('key_phrases', []))
        phrases2 = set(chunk2_metadata.get('key_phrases', []))
        phrase_overlap = len(phrases1 & phrases2)
        if phrase_overlap > 0:
            similarity += phrase_overlap * 0.2
        
        # Similar word count
        words1 = chunk1_metadata.get('word_count', 0)
        words2 = chunk2_metadata.get('word_count', 0)
        if words1 > 0 and words2 > 0:
            word_ratio = min(words1, words2) / max(words1, words2)
            if word_ratio > 0.8:
                similarity += 0.1
        
        return min(1.0, similarity)
    
    def _determine_relationship_type(self, doc1_metadata: Dict, doc2_metadata: Dict, 
                                   similarity_score: float) -> Tuple[str, float]:
        """Determine the most appropriate relationship type and strength"""
        
        # Same author - hierarchical relationship (series/collection)
        if (doc1_metadata.get('author', '') == doc2_metadata.get('author', '') and 
            doc1_metadata.get('author') != 'unknown'):
            return "hierarchical", min(0.9, similarity_score + 0.1)
        
        # High topic overlap - semantic relationship
        topics1 = set(doc1_metadata.get('key_topics', []))
        topics2 = set(doc2_metadata.get('key_topics', []))
        if len(topics1 & topics2) >= 2:
            return "semantic", similarity_score
        
        # Default to associative for general similarity
        return "associative", similarity_score * 0.8
    
    def _get_similarity_basis(self, doc1_metadata: Dict, doc2_metadata: Dict) -> List[str]:
        """Get the basis for document similarity"""
        basis = []
        
        if doc1_metadata.get('author') == doc2_metadata.get('author'):
            basis.append("same_author")
        
        topics1 = set(doc1_metadata.get('key_topics', []))
        topics2 = set(doc2_metadata.get('key_topics', []))
        if topics1 & topics2:
            basis.append("shared_topics")
        
        if doc1_metadata.get('document_type') == doc2_metadata.get('document_type'):
            basis.append("same_document_type")
        
        return basis
    
    def _get_content_title(self, metadata: Dict) -> str:
        """Get appropriate title for content"""
        content_type = metadata.get('content_type')
        
        if content_type == 'document':
            return metadata.get('title', metadata.get('id', 'Untitled'))
        elif content_type == 'document_section':
            return metadata.get('section_title', f"Section {metadata.get('section_number', 1)}")
        elif content_type == 'document_chunk':
            return f"Chunk {metadata.get('chunk_number', 1)}"
        else:
            return metadata.get('id', 'Unknown')
    
    def _get_content_preview(self, metadata: Dict) -> str:
        """Get content preview"""
        content_type = metadata.get('content_type')
        
        if content_type == 'document':
            return metadata.get('summary', '')[:100] + "..."
        elif content_type == 'document_section':
            return metadata.get('section_content', '')[:100] + "..."
        elif content_type == 'document_chunk':
            return metadata.get('chunk_content', '')[:100] + "..."
        else:
            return "No preview available"
    
    def _create_search_metadata_summary(self, metadata: Dict) -> Dict:
        """Create metadata summary for search results"""
        return {
            "type": metadata.get('content_type', 'unknown'),
            "word_count": metadata.get('word_count', 0),
            "created": metadata.get('created_at', 'unknown'),
            "parent": metadata.get('parent_document', None)
        }
    
    def _analyze_relationships(self, relationships: List[Dict]) -> Dict:
        """Analyze document relationships"""
        
        by_type = {}
        by_target_type = {}
        
        for rel in relationships:
            rel_type = rel.get('relationship_type', 'unknown')
            by_type[rel_type] = by_type.get(rel_type, 0) + 1
            
            # Get target content type
            target = self.db.get_vector(rel['target_id'])
            if target:
                target_type = target['metadata'].get('content_type', 'unknown')
                by_target_type[target_type] = by_target_type.get(target_type, 0) + 1
        
        return {
            "total_relationships": len(relationships),
            "by_relationship_type": by_type,
            "by_target_content_type": by_target_type
        }

def create_sample_documents():
    """Create sample documents for demonstration"""
    
    return [
        {
            "id": "ai_research_paper",
            "content": """Artificial Intelligence in Healthcare: A Comprehensive Review

Introduction
Artificial intelligence (AI) has emerged as a transformative technology in healthcare, offering unprecedented opportunities to improve patient care, reduce costs, and enhance medical research. This comprehensive review examines the current state of AI applications in healthcare and their potential impact.

Machine Learning in Medical Diagnosis
Machine learning algorithms have shown remarkable success in medical diagnosis. Convolutional neural networks excel at medical image analysis, while natural language processing helps extract insights from clinical notes. Deep learning models can detect patterns in medical data that might be missed by human experts.

Clinical Decision Support Systems
AI-powered clinical decision support systems assist healthcare providers in making informed decisions. These systems analyze patient data, medical literature, and treatment guidelines to provide evidence-based recommendations. The integration of AI in electronic health records streamlines workflows and reduces diagnostic errors.

Challenges and Future Directions
Despite the promising applications, AI in healthcare faces several challenges including data privacy, regulatory approval, and integration with existing systems. Future research should focus on developing more interpretable models and ensuring equitable access to AI-powered healthcare solutions.

Conclusion
AI represents a paradigm shift in healthcare delivery. As the technology continues to evolve, we can expect more sophisticated applications that will revolutionize patient care and medical research.""",
            "doc_type": "pdf",
            "source_info": {
                "title": "Artificial Intelligence in Healthcare: A Comprehensive Review",
                "author": "Dr. Sarah Johnson",
                "creation_date": "2024-01-15",
                "language": "en"
            }
        },
        {
            "id": "ml_tutorial_guide",
            "content": """Machine Learning Tutorial: From Basics to Implementation

Getting Started with Machine Learning
Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn from and make decisions based on data. This tutorial covers the essential concepts and practical implementation techniques.

Types of Machine Learning
Supervised Learning: Algorithms learn from labeled data to make predictions on new, unseen data. Examples include linear regression, decision trees, and support vector machines.

Unsupervised Learning: These algorithms find patterns in data without labeled examples. Common techniques include clustering, dimensionality reduction, and association rules.

Reinforcement Learning: Agents learn to make decisions by interacting with an environment and receiving rewards or penalties for their actions.

Data Preprocessing
Data quality is crucial for machine learning success. Key preprocessing steps include:
- Data cleaning to handle missing values and outliers
- Feature scaling and normalization
- Feature selection and engineering
- Data splitting for training and testing

Model Selection and Evaluation
Choosing the right algorithm depends on your data and problem type. Evaluation metrics vary by task - accuracy for classification, RMSE for regression, and silhouette score for clustering. Cross-validation helps ensure model generalizability.

Implementation with Python
Popular Python libraries for machine learning include scikit-learn, TensorFlow, and PyTorch. Start with scikit-learn for traditional ML algorithms, then explore deep learning frameworks for complex neural networks.""",
            "doc_type": "markdown",
            "source_info": {
                "title": "Machine Learning Tutorial: From Basics to Implementation",
                "author": "Tech Academy",
                "creation_date": "2024-02-01",
                "language": "en"
            }
        },
        {
            "id": "data_science_methodology",
            "content": """Data Science Methodology: A Systematic Approach

Understanding the Data Science Process
Data science is an interdisciplinary field that combines statistics, computer science, and domain expertise to extract insights from data. A systematic methodology ensures reproducible and reliable results.

Problem Definition
Every data science project begins with a clear problem definition. Stakeholders must articulate business objectives, success criteria, and constraints. This phase involves understanding the domain, identifying data sources, and defining measurable outcomes.

Data Collection and Exploration
Data collection involves gathering relevant information from various sources including databases, APIs, web scraping, and surveys. Exploratory data analysis helps understand data distributions, correlations, and potential issues.

Data Preparation and Feature Engineering
Raw data rarely comes in analysis-ready format. Data preparation includes cleaning, transformation, and integration of multiple data sources. Feature engineering creates new variables that better represent the underlying problem.

Modeling and Analysis
The modeling phase involves selecting appropriate algorithms, training models, and tuning hyperparameters. Statistical analysis and machine learning techniques are applied based on the problem type and data characteristics.

Model Evaluation and Validation
Rigorous evaluation ensures model reliability. Techniques include cross-validation, holdout testing, and A/B testing. Models must be validated on unseen data to assess real-world performance.

Deployment and Monitoring
Successful models must be deployed in production environments. This involves creating pipelines, monitoring performance, and maintaining models as new data becomes available. Continuous monitoring ensures models remain accurate over time.""",
            "doc_type": "word",
            "source_info": {
                "title": "Data Science Methodology: A Systematic Approach",
                "author": "Dr. Michael Chen",
                "creation_date": "2024-01-28",
                "language": "en"
            }
        }
    ]

def main():
    """Demonstrate complete document processing pipeline capabilities"""
    
    print("📄 Document Processing Pipeline Demo")
    print("=" * 50)
    
    # Create document processing pipeline
    pipeline = Document_Processing_Pipeline("DocuMind Enterprise")
    
    # Ingest sample documents
    print(f"\n📥 Document Ingestion Phase:")
    sample_documents = create_sample_documents()
    
    ingested_docs = []
    for doc in sample_documents:
        result = pipeline.ingest_document(
            doc["id"], 
            doc["content"], 
            doc["doc_type"], 
            doc["source_info"],
            {"chunking": {"chunk_size": 300, "overlap_size": 30}}
        )
        ingested_docs.append(result)
    
    # Build cross-document relationships
    cross_relationships = pipeline.build_cross_document_relationships(
        similarity_threshold=0.6,
        max_relationships_per_doc=8
    )
    
    # Demonstrate document search
    print(f"\n🔍 Document Search Demonstrations:")
    
    search_queries = [
        ("machine learning algorithms", "all", "comprehensive"),
        ("healthcare applications", "documents", "precise"),
        ("data preprocessing steps", "chunks", "discovery")
    ]
    
    for query, scope, strategy in search_queries:
        results, by_type = pipeline.search_documents(query, scope, 5, strategy)
        
        print(f"\n   🔍 Query: '{query}' (scope: {scope}, strategy: {strategy})")
        print(f"      Results: {len(results)} total")
        print(f"      By type: {dict((k, len(v)) for k, v in by_type.items())}")
        
        for result in results[:2]:
            print(f"         📄 {result['content_type']}: {result['title']}")
            print(f"            Connection: {result['connection_type']} (score: {result['combined_score']:.3f})")
            if result['parent_document']:
                print(f"            From: {result['parent_document']['title']}")
    
    # Generate document insights
    print(f"\n📊 Document Insights:")
    
    for doc in ingested_docs[:2]:
        insights = pipeline.get_document_insights(doc["document_id"])
        
        print(f"\n   📄 Document: {insights['document_info']['title']}")
        print(f"      Author: {insights['document_info']['author']}")
        print(f"      Structure: {insights['content_structure']['total_sections']} sections, {insights['content_structure']['total_chunks']} chunks")
        print(f"      Relationships: {len(insights['relationship_analysis']['by_relationship_type'])} types")
        print(f"      Related docs: {insights['related_documents']}")
        print(f"      Key topics: {', '.join(insights['key_topics'][:3])}")
    
    # Pipeline analytics
    analytics = pipeline.get_pipeline_analytics()
    
    print(f"\n📈 Pipeline Analytics:")
    pipeline_info = analytics['pipeline_info']
    print(f"   🔧 Pipeline: {pipeline_info['name']}")
    print(f"   📚 Content: {pipeline_info['total_content_items']} items, {pipeline_info['total_relationships']} relationships")
    print(f"   🎯 Dimensions: {pipeline_info['dimension']}D embeddings")
    
    processing_stats = analytics['processing_stats']
    print(f"   📊 Processing: {processing_stats['documents_processed']} docs, {processing_stats['chunks_created']} chunks")
    
    content_dist = analytics['content_distribution']
    print(f"   📋 Distribution: {dict(content_dist)}")
    
    content_analytics = analytics['content_analytics']
    print(f"   📖 Content: {content_analytics['total_words_processed']} words processed")
    print(f"   🧠 Complexity: {content_analytics['average_document_complexity']:.2f} average")
    
    relationship_analytics = analytics['relationship_analytics']
    print(f"   🔗 Relationships: {dict(relationship_analytics)}")
    
    capacity_usage = analytics['capacity_usage']
    print(f"   📈 Capacity: {capacity_usage['vector_usage_percent']:.1f}% content, {capacity_usage['relationship_usage_percent']:.1f}% relationships")
    
    print(f"\n✨ Key Pipeline Features Demonstrated:")
    features = [
        f"Ingested {len(sample_documents)} documents with intelligent chunking",
        f"Created hierarchical document structure (docs → sections → chunks)",
        f"Built {cross_relationships} cross-document semantic relationships",
        f"Demonstrated multi-scope search (documents, sections, chunks)",
        f"Generated comprehensive document insights and analytics",
        f"Processed {content_analytics['total_words_processed']} words with auto-dimension detection"
    ]
    
    for feature in features:
        print(f"   💡 {feature}")
    
    print(f"\n🎉 Document Processing Pipeline Demo Complete!")
    print(f"   Perfect example of enterprise document management with relationship intelligence!")
    
    # Show upgrade path for enterprise use
    if capacity_usage['vector_usage_percent'] > 25 or capacity_usage['relationship_usage_percent'] > 25:
        print(f"\n🚀 Ready to Scale Your Document Processing?")
        print(f"   Upgrade to full RudraDB for unlimited document capacity!")

if __name__ == "__main__":
    try:
        import rudradb
        import numpy as np
        
        print(f"🎯 Using RudraDB-Opin v{rudradb.__version__}")
        print(f"📄 Perfect for document processing applications!")
        
        main()
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install rudradb-opin numpy")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure RudraDB-Opin is properly installed")
