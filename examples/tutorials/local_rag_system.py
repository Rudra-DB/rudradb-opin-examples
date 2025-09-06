#!/usr/bin/env python3
"""
🤖 Local RAG (Retrieval-Augmented Generation) System

This example demonstrates building a complete RAG system using RudraDB-Opin
with relationship-aware retrieval for enhanced context discovery.

Features demonstrated:
- Document ingestion and chunking
- Relationship-aware context retrieval
- Multi-hop document discovery
- Context ranking and filtering
- RAG pipeline optimization
- Performance monitoring
"""

import rudradb
import numpy as np
import time
from datetime import datetime
import hashlib
import re

class LocalRAGSystem:
    """Complete RAG system with relationship-aware retrieval"""
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        """Initialize the RAG system with auto-dimension detection"""
        self.db = rudradb.RudraDB()  # Auto-detects embedding dimensions
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = {}  # Store original documents
        self.chunks = {}     # Store chunk metadata
        self.stats = {
            "documents_ingested": 0,
            "chunks_created": 0,
            "relationships_built": 0,
            "queries_processed": 0
        }
        
        print("🤖 Local RAG System initialized")
        print(f"   🎯 Auto-dimension detection enabled")
        print(f"   📄 Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    
    def ingest_document(self, doc_id, content, title=None, metadata=None):
        """Ingest a document with intelligent chunking and relationship building"""
        
        print(f"📄 Ingesting document: {doc_id}")
        
        # Store original document
        doc_metadata = {
            "title": title or doc_id,
            "content": content,
            "length": len(content),
            "ingested_at": datetime.now().isoformat(),
            "content_hash": hashlib.md5(content.encode()).hexdigest()[:8],
            **(metadata or {})
        }
        
        self.documents[doc_id] = doc_metadata
        
        # Create chunks
        chunks = self._create_chunks(content, doc_id)
        print(f"   📝 Created {len(chunks)} chunks")
        
        # Process each chunk
        chunk_ids = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            # Generate embedding (in real use, use actual embedding model)
            embedding = self._generate_embedding(chunk_text)
            
            # Create chunk metadata
            chunk_metadata = {
                "chunk_id": chunk_id,
                "source_document": doc_id,
                "chunk_index": i,
                "chunk_text": chunk_text,
                "chunk_length": len(chunk_text),
                "overlap_with_previous": i > 0,
                "overlap_with_next": i < len(chunks) - 1,
                "document_title": doc_metadata["title"],
                **doc_metadata,  # Inherit document metadata
                "is_chunk": True
            }
            
            # Add chunk to vector database
            self.db.add_vector(chunk_id, embedding, chunk_metadata)
            self.chunks[chunk_id] = chunk_metadata
        
        # Build relationships between chunks and with existing content
        relationships_built = self._build_chunk_relationships(chunk_ids, doc_id)
        
        # Update statistics
        self.stats["documents_ingested"] += 1
        self.stats["chunks_created"] += len(chunks)
        self.stats["relationships_built"] += relationships_built
        
        return {
            "chunk_ids": chunk_ids,
            "relationships_built": relationships_built,
            "total_chunks": len(chunks)
        }
    
    def _create_chunks(self, content, doc_id):
        """Create overlapping chunks from document content"""
        
        # Simple sentence-aware chunking
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, create new chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_words = current_chunk.split()[-self.chunk_overlap:]
                    current_chunk = " ".join(overlap_words) + " " + sentence
                    current_length = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _generate_embedding(self, text):
        """Generate embedding for text (simulated - use real model in production)"""
        # In real implementation, use sentence-transformers, OpenAI, etc.
        # For demo, create deterministic embedding based on text hash
        text_hash = hashlib.md5(text.encode()).digest()
        np.random.seed(int.from_bytes(text_hash[:4], 'big') % (2**32))
        embedding = np.random.rand(384).astype(np.float32)
        np.random.seed()  # Reset seed
        return embedding
    
    def _build_chunk_relationships(self, chunk_ids, doc_id):
        """Build relationships between chunks and with existing content"""
        
        relationships_built = 0
        
        # 1. Sequential relationships within document
        for i in range(len(chunk_ids) - 1):
            try:
                self.db.add_relationship(
                    chunk_ids[i], chunk_ids[i+1], "temporal", 0.9,
                    {"reason": "sequential_chunks", "document": doc_id}
                )
                relationships_built += 1
            except RuntimeError as e:
                if "capacity" in str(e).lower():
                    print(f"   ⚠️  Relationship capacity reached")
                    break
        
        # 2. Hierarchical relationships: document -> chunks
        for chunk_id in chunk_ids[:5]:  # Limit to avoid capacity issues
            if relationships_built >= 20:  # Conservative limit
                break
            try:
                self.db.add_relationship(
                    doc_id + "_doc", chunk_id, "hierarchical", 0.8,
                    {"reason": "document_chunk_relationship"}
                )
                relationships_built += 1
            except RuntimeError:
                break
        
        # 3. Semantic relationships with other documents
        doc_metadata = self.documents[doc_id]
        
        for other_doc_id, other_metadata in self.documents.items():
            if other_doc_id == doc_id or relationships_built >= 30:
                break
                
            # Find semantic similarity indicators
            shared_topics = self._find_shared_topics(doc_metadata, other_metadata)
            
            if shared_topics:
                # Create relationships between representative chunks
                try:
                    self.db.add_relationship(
                        chunk_ids[0], f"{other_doc_id}_chunk_0", "semantic", 0.6,
                        {"reason": f"shared_topics: {', '.join(shared_topics)}"}
                    )
                    relationships_built += 1
                except (RuntimeError, Exception):
                    break
        
        return relationships_built
    
    def _find_shared_topics(self, metadata1, metadata2):
        """Find shared topics between documents"""
        shared = []
        
        # Check categories
        if metadata1.get("category") == metadata2.get("category"):
            shared.append(metadata1.get("category"))
        
        # Check tags
        tags1 = set(metadata1.get("tags", []))
        tags2 = set(metadata2.get("tags", []))
        shared.extend(list(tags1 & tags2))
        
        # Check keywords in titles
        title1_words = set(metadata1.get("title", "").lower().split())
        title2_words = set(metadata2.get("title", "").lower().split())
        shared.extend(list(title1_words & title2_words))
        
        return list(set(shared))  # Remove duplicates
    
    def retrieve_context(self, query, strategy="relationship_aware", top_k=5):
        """Retrieve context for a query using different strategies"""
        
        print(f"🔍 Retrieving context for query: '{query}'")
        print(f"   Strategy: {strategy}, top_k: {top_k}")
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        start_time = time.time()
        
        if strategy == "similarity_only":
            # Traditional similarity search
            params = rudradb.SearchParams(
                top_k=top_k,
                include_relationships=False,
                similarity_threshold=0.1
            )
            
        elif strategy == "relationship_aware":
            # Relationship-enhanced retrieval
            params = rudradb.SearchParams(
                top_k=top_k * 2,  # Get more candidates
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.4,
                similarity_threshold=0.05
            )
            
        elif strategy == "multi_hop_discovery":
            # Emphasize relationship discovery
            params = rudradb.SearchParams(
                top_k=top_k * 3,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.6,
                similarity_threshold=0.0
            )
            
        else:  # balanced
            params = rudradb.SearchParams(
                top_k=top_k,
                include_relationships=True,
                max_hops=1,
                relationship_weight=0.3
            )
        
        # Perform search
        results = self.db.search(query_embedding, params)
        search_time = time.time() - start_time
        
        # Process and rank results
        context_chunks = []
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            if vector and vector['metadata'].get('is_chunk'):
                context_chunks.append({
                    "chunk_id": result.vector_id,
                    "text": vector['metadata']['chunk_text'],
                    "source_document": vector['metadata']['source_document'],
                    "document_title": vector['metadata'].get('document_title', 'Unknown'),
                    "chunk_index": vector['metadata']['chunk_index'],
                    "similarity_score": result.similarity_score,
                    "combined_score": result.combined_score,
                    "hop_count": result.hop_count,
                    "connection_type": "direct" if result.hop_count == 0 else f"{result.hop_count}-hop"
                })
        
        # Rank and filter
        context_chunks = self._rank_and_filter_context(context_chunks, query)
        
        # Update statistics
        self.stats["queries_processed"] += 1
        
        return {
            "query": query,
            "strategy": strategy,
            "context_chunks": context_chunks[:top_k],
            "total_candidates": len(results),
            "search_time": search_time,
            "direct_matches": len([c for c in context_chunks if c["hop_count"] == 0]),
            "relationship_discoveries": len([c for c in context_chunks if c["hop_count"] > 0])
        }
    
    def _rank_and_filter_context(self, chunks, query):
        """Rank and filter context chunks for relevance"""
        
        # Simple ranking based on combined score and document diversity
        chunk_scores = {}
        
        for chunk in chunks:
            # Base score from search
            score = chunk["combined_score"]
            
            # Boost for direct matches
            if chunk["hop_count"] == 0:
                score *= 1.2
            
            # Slight penalty for very long chunks (may be less focused)
            if len(chunk["text"]) > 800:
                score *= 0.9
            
            # Boost for chunks from different documents (diversity)
            doc_id = chunk["source_document"]
            doc_count = sum(1 for c in chunks if c["source_document"] == doc_id)
            if doc_count == 1:  # Unique document
                score *= 1.1
            
            chunk_scores[chunk["chunk_id"]] = score
        
        # Sort by computed score
        ranked_chunks = sorted(chunks, key=lambda x: chunk_scores[x["chunk_id"]], reverse=True)
        
        return ranked_chunks
    
    def generate_answer(self, query, context_chunks, max_context_length=2000):
        """Generate answer using retrieved context (simulated - use real LLM in production)"""
        
        print(f"🤖 Generating answer for: '{query}'")
        
        # Combine context from top chunks
        context_text = ""
        context_sources = set()
        
        for chunk in context_chunks:
            if len(context_text) + len(chunk["text"]) <= max_context_length:
                context_text += f"\n{chunk['text']}"
                context_sources.add(chunk['source_document'])
        
        # Simulated answer generation (in real use, call OpenAI, Claude, etc.)
        simulated_answer = self._simulate_answer_generation(query, context_text, context_sources)
        
        return {
            "query": query,
            "answer": simulated_answer,
            "context_used": context_text,
            "sources": list(context_sources),
            "context_chunks_used": len([c for c in context_chunks if c["text"] in context_text]),
            "total_context_length": len(context_text)
        }
    
    def _simulate_answer_generation(self, query, context, sources):
        """Simulate answer generation (replace with real LLM)"""
        
        # Simple template-based answer simulation
        answer_templates = [
            f"Based on the provided context from {len(sources)} source(s), here's what I found about '{query}':",
            f"According to the documents, regarding '{query}':",
            f"The context provides the following information about '{query}':"
        ]
        
        # Pick template based on query hash for consistency
        template_idx = hash(query) % len(answer_templates)
        answer_start = answer_templates[template_idx]
        
        # Extract key points from context (very basic simulation)
        sentences = context.split('.')[:3]  # Take first few sentences
        key_points = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        
        if key_points:
            answer = f"{answer_start}\n\n" + "\n".join(f"• {point}." for point in key_points[:2])
            answer += f"\n\nThis information comes from {len(sources)} document(s) in the knowledge base."
        else:
            answer = f"I found some relevant context for '{query}', but would need a language model to generate a comprehensive answer."
        
        return answer
    
    def run_complete_rag_pipeline(self, query, retrieval_strategy="relationship_aware", top_k=5):
        """Run the complete RAG pipeline from query to answer"""
        
        print(f"\n🔄 Running Complete RAG Pipeline")
        print(f"   Query: '{query}'")
        print(f"   Strategy: {retrieval_strategy}")
        
        pipeline_start = time.time()
        
        # Step 1: Retrieve context
        retrieval_result = self.retrieve_context(query, retrieval_strategy, top_k)
        
        # Step 2: Generate answer
        if retrieval_result["context_chunks"]:
            answer_result = self.generate_answer(query, retrieval_result["context_chunks"])
        else:
            answer_result = {
                "query": query,
                "answer": "No relevant context found in the knowledge base.",
                "context_used": "",
                "sources": [],
                "context_chunks_used": 0,
                "total_context_length": 0
            }
        
        pipeline_time = time.time() - pipeline_start
        
        return {
            "query": query,
            "retrieval": retrieval_result,
            "generation": answer_result,
            "pipeline_time": pipeline_time,
            "performance": {
                "retrieval_time": retrieval_result["search_time"],
                "generation_time": pipeline_time - retrieval_result["search_time"],
                "total_time": pipeline_time
            }
        }
    
    def get_system_stats(self):
        """Get comprehensive system statistics"""
        
        db_stats = self.db.get_statistics()
        usage = db_stats['capacity_usage']
        
        return {
            "ingestion_stats": self.stats,
            "database_stats": {
                "vectors": f"{db_stats['vector_count']}/{rudradb.MAX_VECTORS}",
                "relationships": f"{db_stats['relationship_count']}/{rudradb.MAX_RELATIONSHIPS}",
                "dimension": db_stats['dimension']
            },
            "capacity_usage": {
                "vector_usage": f"{usage['vector_usage_percent']:.1f}%",
                "relationship_usage": f"{usage['relationship_usage_percent']:.1f}%"
            },
            "content_metrics": {
                "documents": len(self.documents),
                "chunks": len(self.chunks),
                "avg_chunks_per_doc": len(self.chunks) / max(len(self.documents), 1),
                "avg_relationships_per_vector": db_stats['relationship_count'] / max(db_stats['vector_count'], 1)
            }
        }

def create_sample_rag_system():
    """Create a sample RAG system with AI/ML documents"""
    
    print("🏗️  Building Sample RAG System")
    print("=" * 40)
    
    rag = LocalRAGSystem(chunk_size=600, chunk_overlap=50)
    
    # Sample documents for the RAG system
    documents = [
        {
            "id": "ai_overview",
            "title": "Artificial Intelligence Overview",
            "content": """Artificial Intelligence (AI) represents one of the most significant technological developments of the 21st century. At its core, AI refers to the development of computer systems that can perform tasks typically requiring human intelligence. These tasks include learning from experience, recognizing patterns, making decisions, and solving problems.
            
The field of AI encompasses several key subdomains. Machine Learning (ML) focuses on algorithms that improve their performance through exposure to data. Deep Learning, a subset of ML, uses neural networks with multiple layers to process complex patterns. Natural Language Processing enables computers to understand and generate human language. Computer Vision allows machines to interpret visual information from images and videos.

Modern AI applications are transforming industries worldwide. In healthcare, AI assists in medical diagnosis and drug discovery. In finance, it powers algorithmic trading and fraud detection. Transportation benefits from autonomous vehicles and traffic optimization. Entertainment services use AI for content recommendation and generation.

The development of AI involves several crucial components. Data serves as the foundation, providing the information from which AI systems learn. Algorithms define the mathematical approaches used to process this data. Computing power, particularly GPUs and specialized AI chips, enables the training and deployment of complex models. Human expertise remains essential for designing systems, interpreting results, and ensuring ethical implementation.""",
            "category": "AI",
            "tags": ["ai", "overview", "machine learning", "applications"]
        },
        {
            "id": "ml_fundamentals", 
            "title": "Machine Learning Fundamentals",
            "content": """Machine Learning represents a paradigm shift in how we approach problem-solving with computers. Instead of explicitly programming solutions, ML enables systems to learn patterns from data and make predictions or decisions based on that learning.

The field is broadly divided into three main categories. Supervised Learning uses labeled datasets to train models, enabling them to make predictions on new, unseen data. Common supervised learning tasks include classification (predicting categories) and regression (predicting continuous values). Popular algorithms include linear regression, decision trees, random forests, and support vector machines.

Unsupervised Learning works with unlabeled data to discover hidden patterns or structures. Clustering algorithms group similar data points together, while dimensionality reduction techniques simplify complex datasets while preserving important information. Principal Component Analysis (PCA) and k-means clustering are widely used unsupervised methods.

Reinforcement Learning takes a different approach, where agents learn to make decisions by interacting with an environment and receiving rewards or penalties for their actions. This approach has shown remarkable success in game playing, robotics, and autonomous systems.

The machine learning workflow typically follows several steps. Data collection and preparation form the foundation. Feature engineering involves selecting and transforming relevant variables. Model selection requires choosing appropriate algorithms for the task. Training involves fitting the model to data. Validation ensures the model generalizes well to new data. Finally, deployment makes the model available for real-world use.

Evaluation metrics play a crucial role in assessing model performance. For classification tasks, accuracy, precision, recall, and F1-score provide different perspectives on model effectiveness. Regression tasks commonly use metrics like mean squared error and R-squared. Cross-validation techniques help ensure robust performance estimates.""",
            "category": "AI", 
            "tags": ["machine learning", "supervised", "unsupervised", "algorithms"]
        },
        {
            "id": "python_ml_tools",
            "title": "Python Tools for Machine Learning",
            "content": """Python has emerged as the dominant programming language for machine learning, offering a rich ecosystem of libraries and tools that streamline the development process. The combination of Python's simplicity, readability, and extensive library support makes it ideal for both beginners and experienced practitioners.

NumPy forms the foundation of scientific computing in Python, providing efficient array operations and mathematical functions. Its ndarray object offers high-performance operations on homogeneous data, essential for numerical computations in machine learning. Broadcasting capabilities allow operations between arrays of different shapes, while vectorization enables fast operations without explicit loops.

Pandas revolutionizes data manipulation and analysis, offering DataFrame and Series objects that simplify working with structured data. It provides intuitive methods for data cleaning, transformation, aggregation, and visualization. Features like handling missing data, merging datasets, and time series analysis make pandas indispensable for data preprocessing.

Scikit-learn serves as the go-to library for machine learning algorithms and utilities. It provides consistent APIs for classification, regression, clustering, and dimensionality reduction. The library includes tools for model selection, evaluation metrics, and preprocessing. Its design philosophy emphasizes ease of use while maintaining flexibility for advanced users.

For deep learning, TensorFlow and PyTorch dominate the landscape. TensorFlow, developed by Google, offers production-ready deployment capabilities and extensive tooling. PyTorch, preferred by many researchers, provides dynamic computation graphs and intuitive debugging. Both frameworks support GPU acceleration and distributed training.

Matplotlib and Seaborn enable comprehensive data visualization. Matplotlib provides low-level control over plot elements, while Seaborn offers high-level statistical visualizations. These tools are essential for exploratory data analysis and communicating results.

Jupyter Notebooks revolutionize the development workflow, combining code, documentation, and visualizations in a single interactive environment. They facilitate experimentation, collaboration, and reproducible research.""",
            "category": "Programming",
            "tags": ["python", "tools", "libraries", "numpy", "pandas", "scikit-learn"]
        },
        {
            "id": "deep_learning_intro",
            "title": "Introduction to Deep Learning",
            "content": """Deep Learning represents a revolutionary approach to machine learning that has transformed artificial intelligence applications across numerous domains. By utilizing neural networks with multiple hidden layers, deep learning systems can automatically learn hierarchical representations of data, eliminating the need for manual feature engineering.

The fundamental building block of deep learning is the artificial neuron, inspired by biological neural networks. These neurons receive inputs, apply weights and biases, and produce outputs through activation functions. When organized into layers, these neurons form networks capable of learning complex patterns.

Feedforward neural networks, also called multilayer perceptrons, represent the simplest deep learning architecture. Information flows from input to output through hidden layers, with each layer learning increasingly abstract representations. Despite their simplicity, these networks can approximate complex functions given sufficient depth and width.

Convolutional Neural Networks (CNNs) revolutionized computer vision by incorporating spatial structure into the learning process. Convolutional layers apply filters across input images, detecting features like edges, textures, and shapes. Pooling layers reduce spatial dimensions while preserving important information. This architecture enables CNNs to achieve state-of-the-art performance in image classification, object detection, and segmentation tasks.

Recurrent Neural Networks (RNNs) address sequential data by maintaining hidden states that capture temporal dependencies. Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) variants solve the vanishing gradient problem, enabling learning of long-term dependencies. These architectures excel at natural language processing, time series prediction, and sequence generation tasks.

The training process involves backpropagation, where gradients are calculated and propagated backward through the network to update weights. Optimization algorithms like Adam and RMSprop improve upon basic gradient descent by adapting learning rates and incorporating momentum. Regularization techniques such as dropout and batch normalization prevent overfitting and stabilize training.

Transfer learning leverages pre-trained models to accelerate learning on new tasks. By starting with weights learned on large datasets, models can achieve good performance with limited data. Fine-tuning adjusts pre-trained models for specific applications, while feature extraction uses pre-trained models as fixed feature extractors.""",
            "category": "AI",
            "tags": ["deep learning", "neural networks", "cnn", "rnn", "backpropagation"]
        }
    ]
    
    print(f"\n📚 Ingesting {len(documents)} documents...")
    
    # Ingest all documents
    total_chunks = 0
    total_relationships = 0
    
    for doc in documents:
        result = rag.ingest_document(
            doc["id"], 
            doc["content"],
            title=doc["title"],
            metadata={
                "category": doc["category"],
                "tags": doc["tags"]
            }
        )
        
        print(f"   ✅ '{doc['title']}': {result['total_chunks']} chunks, {result['relationships_built']} relationships")
        total_chunks += result['total_chunks']
        total_relationships += result['relationships_built']
    
    print(f"\n📊 Ingestion Summary:")
    print(f"   Documents: {len(documents)}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Total relationships: {total_relationships}")
    print(f"   Auto-detected dimension: {rag.db.dimension()}D")
    
    return rag

def demonstrate_rag_capabilities(rag):
    """Demonstrate various RAG system capabilities"""
    
    print(f"\n🔬 Demonstrating RAG System Capabilities")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "What is machine learning and how does it work?",
        "What Python libraries are best for machine learning?", 
        "How do neural networks learn from data?",
        "What are the differences between supervised and unsupervised learning?"
    ]
    
    retrieval_strategies = ["similarity_only", "relationship_aware", "multi_hop_discovery"]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}️⃣ Query: '{query}'")
        
        for strategy in retrieval_strategies:
            print(f"\n   🔍 Strategy: {strategy}")
            
            # Run complete pipeline
            result = rag.run_complete_rag_pipeline(query, strategy, top_k=3)
            
            # Show results
            retrieval = result["retrieval"]
            generation = result["generation"]
            performance = result["performance"]
            
            print(f"      ⏱️  Performance: {performance['total_time']*1000:.1f}ms total")
            print(f"         Retrieval: {performance['retrieval_time']*1000:.1f}ms")
            print(f"         Generation: {performance['generation_time']*1000:.1f}ms")
            
            print(f"      📊 Context: {retrieval['total_candidates']} candidates → {len(retrieval['context_chunks'])} used")
            print(f"         Direct matches: {retrieval['direct_matches']}")
            print(f"         Relationship discoveries: {retrieval['relationship_discoveries']}")
            
            print(f"      🎯 Answer preview: {generation['answer'][:150]}...")
            
            if len(generation['sources']) > 0:
                print(f"      📚 Sources: {', '.join(generation['sources'])}")
        
        print(f"\n   💡 Strategy Comparison for this query:")
        strategies_tested = []
        for strategy in retrieval_strategies:
            result = rag.run_complete_rag_pipeline(query, strategy, top_k=3)
            strategies_tested.append({
                "strategy": strategy,
                "discoveries": result["retrieval"]["relationship_discoveries"],
                "total_time": result["performance"]["total_time"] * 1000
            })
        
        for test in strategies_tested:
            print(f"      {test['strategy']}: {test['discoveries']} relationship discoveries, {test['total_time']:.1f}ms")

def analyze_rag_performance(rag):
    """Analyze RAG system performance and characteristics"""
    
    print(f"\n📈 RAG System Performance Analysis")
    print("=" * 45)
    
    # System statistics
    stats = rag.get_system_stats()
    
    print(f"\n📊 System Statistics:")
    print(f"   Ingestion: {stats['ingestion_stats']}")
    print(f"   Database: {stats['database_stats']}")
    print(f"   Capacity: {stats['capacity_usage']}")
    print(f"   Content Metrics: {stats['content_metrics']}")
    
    # Performance benchmarking
    print(f"\n⚡ Performance Benchmarking:")
    
    benchmark_query = "How do machine learning algorithms work?"
    benchmark_iterations = 5
    
    strategy_performance = {}
    
    for strategy in ["similarity_only", "relationship_aware", "multi_hop_discovery"]:
        times = []
        
        for _ in range(benchmark_iterations):
            start_time = time.time()
            result = rag.retrieve_context(benchmark_query, strategy, top_k=5)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        strategy_performance[strategy] = avg_time
    
    print(f"   Query: '{benchmark_query}'")
    for strategy, avg_time in strategy_performance.items():
        print(f"   {strategy}: {avg_time:.2f}ms average")
    
    # Relationship impact analysis
    print(f"\n🔗 Relationship Impact Analysis:")
    
    test_query = "What are Python tools for data science?"
    
    # Compare with and without relationships
    similarity_result = rag.retrieve_context(test_query, "similarity_only", top_k=5)
    relationship_result = rag.retrieve_context(test_query, "relationship_aware", top_k=5)
    
    print(f"   Test query: '{test_query}'")
    print(f"   Similarity only: {len(similarity_result['context_chunks'])} results")
    print(f"   Relationship-aware: {len(relationship_result['context_chunks'])} results")
    print(f"   Additional discoveries: {relationship_result['relationship_discoveries']}")
    print(f"   Performance impact: {(relationship_result['search_time'] / similarity_result['search_time'] - 1) * 100:.1f}% slower")

def main():
    """Run the complete Local RAG System tutorial"""
    
    print("🤖 RudraDB-Opin Local RAG System Tutorial")
    print("=" * 55)
    
    print("\n🎯 This tutorial demonstrates:")
    features = [
        "Document ingestion with intelligent chunking",
        "Auto-relationship building between content",
        "Relationship-aware context retrieval",
        "Multi-hop document discovery",
        "Complete RAG pipeline with answer generation",
        "Performance optimization and monitoring"
    ]
    
    for feature in features:
        print(f"   • {feature}")
    
    try:
        # Create RAG system
        rag = create_sample_rag_system()
        
        # Demonstrate capabilities
        demonstrate_rag_capabilities(rag)
        
        # Performance analysis
        analyze_rag_performance(rag)
        
        # Summary
        print(f"\n🎉 Local RAG System Tutorial Complete!")
        print("=" * 50)
        
        final_stats = rag.get_system_stats()
        
        key_achievements = [
            f"Built RAG system with {final_stats['content_metrics']['documents']} documents",
            f"Created {final_stats['content_metrics']['chunks']} searchable chunks",
            f"Auto-generated {final_stats['database_stats']['relationships'].split('/')[0]} relationships",
            f"Demonstrated relationship-aware retrieval advantages",
            f"Processed {final_stats['ingestion_stats']['queries_processed']} queries"
        ]
        
        print(f"\n🏆 Key Achievements:")
        for achievement in key_achievements:
            print(f"   ✅ {achievement}")
        
        print(f"\n💡 Key Benefits of Relationship-Aware RAG:")
        benefits = [
            "Discovers relevant context through relationship chains",
            "Finds connections between related documents",
            "Improves context quality for answer generation",
            "Enables multi-hop reasoning across knowledge",
            "Reduces reliance on perfect similarity matches"
        ]
        
        for benefit in benefits:
            print(f"   • {benefit}")
        
        print(f"\n📊 Capacity Usage:")
        print(f"   Vectors: {final_stats['capacity_usage']['vector_usage']}")
        print(f"   Relationships: {final_stats['capacity_usage']['relationship_usage']}")
        
        if "80%" in final_stats['capacity_usage']['vector_usage']:
            print(f"\n🚀 Ready for Production Scale:")
            print(f"   Your RAG system is using substantial capacity!")
            print(f"   Ready for unlimited documents? Upgrade to full RudraDB!")
        
    except Exception as e:
        print(f"\n❌ Tutorial error: {e}")
        print("💡 Make sure rudradb-opin is installed: pip install rudradb-opin")

if __name__ == "__main__":
    main()
