# 🧬 RudraDB-Opin: Complete Developer Guide

> **The World's First Free Relationship-Aware Vector Database**
> 
> Perfect for learning, tutorials, and proof-of-concepts with 100 vectors and 500 relationships. Experience the power of relationship-aware search with zero restrictions on features!

## 🎯 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Core Concepts](#-core-concepts)
- [API Reference](#-api-reference)
- [Relationship Types](#-relationship-types)
- [Search Patterns](#-search-patterns)
- [Tutorial Examples](#-tutorial-examples)
- [ML Framework Integration](#-ml-framework-integration)
- [Best Practices](#-best-practices)
- [Performance Guide](#-performance-guide)
- [Capacity Management](#-capacity-management)
- [Upgrade Path](#-upgrade-path)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start

### Installation
```bash
pip install rudradb-opin
```

### 30-Second Example
```python
import rudradb
import numpy as np

# Create database (auto-detects embedding dimensions)
db = rudradb.RudraDB()

# Add vectors
embedding1 = np.random.rand(384).astype(np.float32)
embedding2 = np.random.rand(384).astype(np.float32)

db.add_vector("doc1", embedding1, {"title": "Machine Learning Basics"})
db.add_vector("doc2", embedding2, {"title": "Deep Learning Guide"})

# Add relationship
db.add_relationship("doc1", "doc2", "hierarchical", 0.8)

# Search with relationship awareness
query = np.random.rand(384).astype(np.float32)
params = rudradb.SearchParams(top_k=5, include_relationships=True, max_hops=2)
results = db.search(query, params)

print(f"Found {len(results)} results with relationship-aware search!")
```

### What You Get (100% Free)
- ✅ **100 vectors** - Perfect for tutorials and learning
- ✅ **500 relationships** - Rich relationship modeling
- ✅ **5 relationship types** - Complete feature set
- ✅ **2-hop traversal** - Multi-degree relationship discovery
- ✅ **Auto-dimension detection** - Works with any embedding model
- ✅ **Production code quality** - Same algorithms as full RudraDB

---

## 📦 Installation

### Method 1: PyPI Installation (Recommended)
```bash
# Install RudraDB-Opin
pip install rudradb-opin

# Verify installation
python -c "import rudradb; print(f'RudraDB-Opin v{rudradb.__version__} ready!')"
```

### Method 2: Virtual Environment
```bash
# Create isolated environment
python -m venv rudradb_env
source rudradb_env/bin/activate  # Windows: rudradb_env\Scripts\activate

# Install
pip install rudradb-opin
```

### Requirements
- Python 3.8+
- NumPy >= 1.20.0
- Works on Windows, macOS, Linux

### Verify Installation
```python
import rudradb
import numpy as np

print(f"🧬 RudraDB-Opin {rudradb.__version__}")
print(f"📊 Max vectors: {rudradb.MAX_VECTORS}")
print(f"📊 Max relationships: {rudradb.MAX_RELATIONSHIPS}")
print(f"🎯 Edition: {rudradb.EDITION}")

# Quick functionality test
db = rudradb.RudraDB()
db.add_vector("test", np.random.rand(384).astype(np.float32))
print("✅ Installation successful!")
```

---

## 🧠 Core Concepts

### What Makes RudraDB-Opin Special?

**Traditional Vector Databases:**
- Store vectors and find similar ones
- Limited to similarity-based search
- Miss important connections

**RudraDB-Opin (Relationship-Aware):**
- Stores vectors AND relationships between them
- Discovers indirect connections through relationships
- Combines similarity + relationship strength for better results

### Key Components

#### 1. Vectors
```python
# Vectors store embeddings with rich metadata
db.add_vector(
    id="document_1",
    embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    metadata={
        "title": "Introduction to AI",
        "author": "Jane Smith",
        "category": "education",
        "tags": ["ai", "basics", "tutorial"]
    }
)
```

#### 2. Relationships
```python
# Relationships connect vectors with semantic meaning
db.add_relationship(
    source_id="intro_ai",
    target_id="deep_learning",
    relationship_type="hierarchical",  # Parent -> Child
    strength=0.9,  # How strong the connection is (0.0-1.0)
    metadata={"connection_reason": "prerequisite"}
)
```

#### 3. Multi-Hop Discovery
```python
# Find connections through relationship chains
# A -> B -> C (2 hops)
connected = db.get_connected_vectors("starting_doc", max_hops=2)
for vector, hop_count in connected:
    print(f"Found {vector['id']} at {hop_count} hops away")
```

---

## 📚 API Reference

### RudraDB Class

#### Constructor
```python
rudradb.RudraDB(dimension=None, config=None)
```
- `dimension`: Optional[int] - Embedding dimension (auto-detected if None)
- `config`: Optional[dict] - Configuration options

#### Properties
```python
db.dimension()          # Current embedding dimension
db.vector_count()       # Number of vectors stored
db.relationship_count() # Number of relationships stored
db.is_empty()          # True if no vectors or relationships
```

#### Vector Operations
```python
# Add vector
db.add_vector(id: str, embedding: np.ndarray, metadata: dict = None)

# Get vector
vector = db.get_vector(id: str)  # Returns dict or None

# Check existence
exists = db.vector_exists(id: str)  # Returns bool
exists = db.has_vector(id: str)     # Alias

# List all vectors
vector_ids = db.list_vectors()      # Returns List[str]

# Remove vector
db.remove_vector(id: str)

# Update metadata
db.update_vector_metadata(id: str, metadata: dict)
```

#### Relationship Operations
```python
# Add relationship
db.add_relationship(
    source_id: str,
    target_id: str, 
    relationship_type: str,  # "semantic", "hierarchical", etc.
    strength: float = 0.8,   # 0.0 to 1.0
    metadata: dict = None
)

# Get relationships
relationships = db.get_relationships(
    vector_id: str,
    relationship_type: str = None  # Filter by type
)

# Check relationship existence
exists = db.has_relationship(from_id: str, to_id: str)

# Remove relationship
db.remove_relationship(from_id: str, to_id: str)
```

#### Search Operations
```python
# Main search method
results = db.search(
    query: np.ndarray,           # Query embedding
    params: SearchParams = None  # Search configuration
)

# Get connected vectors
connected = db.get_connected_vectors(
    vector_id: str,
    max_hops: int = 2,
    relationship_types: List[str] = None
)
```

#### Database Management
```python
# Statistics
stats = db.get_statistics()  # Returns detailed stats dict

# Export/Import
data = db.export_data()      # Export to JSON-compatible dict
db.import_data(data: dict)   # Import from dict

# Limits info
limits = rudradb.get_opin_limits()  # Get capacity information
```

### SearchParams Class
```python
params = rudradb.SearchParams(
    top_k=10,                          # Number of results
    include_relationships=True,         # Enable relationship search
    max_hops=2,                        # Maximum relationship hops
    similarity_threshold=0.1,          # Minimum similarity score
    relationship_weight=0.3,           # Relationship influence (0.0-1.0)
    relationship_types=["semantic"]    # Filter relationship types
)
```

### SearchResult Structure
```python
for result in results:
    print(f"Vector ID: {result.vector_id}")
    print(f"Similarity Score: {result.similarity_score}")
    print(f"Combined Score: {result.combined_score}")
    print(f"Source: {result.source}")  # "direct" or "relationship" 
    print(f"Hop Count: {result.hop_count}")
```

---

## 🔗 Relationship Types

RudraDB-Opin supports 5 relationship types, each optimized for different connection patterns:

### 1. Semantic Relationships
**Use Case**: Content similarity, topical connections

```python
# Related articles or documents
db.add_relationship("ai_intro", "ml_tutorial", "semantic", 0.8)
db.add_relationship("python_basics", "python_advanced", "semantic", 0.7)

# Product recommendations
db.add_relationship("laptop_gaming", "laptop_business", "semantic", 0.6)
```

**Best For**: Similar content, related topics, recommendation systems

### 2. Hierarchical Relationships  
**Use Case**: Parent-child structures, categorization

```python
# Knowledge hierarchies
db.add_relationship("machine_learning", "supervised_learning", "hierarchical", 0.9)
db.add_relationship("supervised_learning", "linear_regression", "hierarchical", 0.8)

# Organizational structures
db.add_relationship("company", "department", "hierarchical", 0.9)
db.add_relationship("department", "employee", "hierarchical", 0.8)
```

**Best For**: Taxonomies, org charts, knowledge trees, category structures

### 3. Temporal Relationships
**Use Case**: Sequential content, time-based flow

```python
# Course progression
db.add_relationship("lesson_1", "lesson_2", "temporal", 0.9)
db.add_relationship("lesson_2", "lesson_3", "temporal", 0.9)

# Process steps
db.add_relationship("data_collection", "data_cleaning", "temporal", 0.8)
db.add_relationship("data_cleaning", "model_training", "temporal", 0.8)
```

**Best For**: Workflows, course sequences, process chains, timelines

### 4. Causal Relationships
**Use Case**: Cause-effect, problem-solution pairs

```python
# Problem-solution mapping
db.add_relationship("error_403", "auth_solution", "causal", 0.85)
db.add_relationship("slow_query", "index_optimization", "causal", 0.9)

# Research methodology
db.add_relationship("research_question", "methodology", "causal", 0.8)
db.add_relationship("hypothesis", "experiment_design", "causal", 0.7)
```

**Best For**: Troubleshooting, Q&A systems, cause-effect analysis

### 5. Associative Relationships
**Use Case**: General associations, loose connections

```python
# General associations
db.add_relationship("coffee", "productivity", "associative", 0.6)
db.add_relationship("exercise", "health", "associative", 0.7)

# Cross-references
db.add_relationship("python_tutorial", "data_science", "associative", 0.5)
```

**Best For**: General recommendations, cross-references, loose connections

### Choosing Relationship Types

```python
def choose_relationship_type(connection_nature):
    """Guide for selecting appropriate relationship types"""
    if connection_nature == "parent_child":
        return "hierarchical"
    elif connection_nature == "happens_before":
        return "temporal" 
    elif connection_nature == "causes_solves":
        return "causal"
    elif connection_nature == "similar_content":
        return "semantic"
    else:
        return "associative"  # Default for general connections
```

---

## 🔍 Search Patterns

### Basic Similarity Search
```python
# Pure vector similarity (no relationships)
params = rudradb.SearchParams(
    top_k=5,
    include_relationships=False,
    similarity_threshold=0.3
)

results = db.search(query_embedding, params)
print(f"Found {len(results)} similar vectors")
```

### Relationship-Enhanced Search
```python
# Combine similarity + relationships
params = rudradb.SearchParams(
    top_k=10,
    include_relationships=True,
    max_hops=2,
    relationship_weight=0.3  # 30% relationship influence
)

results = db.search(query_embedding, params)
for result in results:
    if result.source == "relationship":
        print(f"Found via relationship: {result.vector_id}")
    else:
        print(f"Found via similarity: {result.vector_id}")
```

### Discovery-Focused Search
```python
# Emphasize discovering new connections
params = rudradb.SearchParams(
    top_k=15,
    include_relationships=True,
    max_hops=2,
    relationship_weight=0.7,  # High relationship influence
    similarity_threshold=0.1   # Lower similarity requirement
)

results = db.search(query_embedding, params)
print(f"Discovered {len([r for r in results if r.hop_count > 0])} indirect connections")
```

### Filtered Search
```python
# Search only through specific relationship types
params = rudradb.SearchParams(
    top_k=10,
    include_relationships=True,
    relationship_types=["hierarchical", "semantic"]
)

results = db.search(query_embedding, params)
```

### Progressive Search Strategy
```python
def progressive_search(db, query, target_results=10):
    """Search with increasingly broad parameters"""
    
    # Try 1: Strict similarity
    params = rudradb.SearchParams(
        top_k=target_results,
        include_relationships=False,
        similarity_threshold=0.5
    )
    results = db.search(query, params)
    
    if len(results) >= target_results:
        return results
    
    # Try 2: Add relationships
    params.include_relationships = True
    params.similarity_threshold = 0.3
    results = db.search(query, params)
    
    if len(results) >= target_results:
        return results
    
    # Try 3: Broader discovery
    params.relationship_weight = 0.6
    params.similarity_threshold = 0.1
    params.max_hops = 2
    
    return db.search(query, params)
```

---

## 🎓 Tutorial Examples

### Tutorial 1: Building a Knowledge Base

```python
import rudradb
import numpy as np
from sentence_transformers import SentenceTransformer

# Setup
model = SentenceTransformer('all-MiniLM-L6-v2')
db = rudradb.RudraDB()  # Auto-detects dimension

# Knowledge base content
topics = {
    "python_basics": "Python programming fundamentals and syntax",
    "data_structures": "Lists, dictionaries, sets in Python", 
    "pandas_intro": "Data manipulation with pandas library",
    "numpy_arrays": "Numerical computing with NumPy",
    "matplotlib_viz": "Data visualization using matplotlib",
    "machine_learning": "Introduction to ML concepts",
    "scikit_learn": "ML with scikit-learn library",
    "deep_learning": "Neural networks and deep learning"
}

# Add vectors with embeddings
for topic_id, description in topics.items():
    embedding = model.encode([description])[0].astype(np.float32)
    metadata = {
        "description": description,
        "category": "programming" if "python" in topic_id else "data_science",
        "level": "beginner" if "basics" in topic_id else "intermediate"
    }
    db.add_vector(topic_id, embedding, metadata)

# Build knowledge relationships
relationships = [
    ("python_basics", "data_structures", "hierarchical", 0.9),
    ("data_structures", "pandas_intro", "temporal", 0.8),
    ("pandas_intro", "numpy_arrays", "semantic", 0.7),
    ("numpy_arrays", "matplotlib_viz", "temporal", 0.8),
    ("python_basics", "machine_learning", "causal", 0.6),
    ("machine_learning", "scikit_learn", "hierarchical", 0.8),
    ("machine_learning", "deep_learning", "hierarchical", 0.9),
    ("scikit_learn", "deep_learning", "temporal", 0.7)
]

for source, target, rel_type, strength in relationships:
    db.add_relationship(source, target, rel_type, strength)

print(f"✅ Knowledge base: {db.vector_count()} topics, {db.relationship_count()} connections")

# Smart learning path discovery
def get_learning_path(start_topic, max_steps=3):
    connected = db.get_connected_vectors(start_topic, max_hops=max_steps)
    
    # Sort by hop count for natural progression
    path = sorted(connected, key=lambda x: x[1])
    
    print(f"\n📚 Learning path from '{start_topic}':")
    for vector, hops in path:
        level = vector["metadata"]["level"]
        print(f"  Step {hops}: {vector['id']} ({level})")
    
    return path

# Example usage
learning_path = get_learning_path("python_basics")
```

### Tutorial 2: Local RAG System

```python
import rudradb
import numpy as np
from sentence_transformers import SentenceTransformer

class LocalRAG:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.db = rudradb.RudraDB()  # Auto-detects dimension
        
    def add_document(self, doc_id, text, metadata=None):
        """Add document with automatic relationship building"""
        embedding = self.model.encode([text])[0].astype(np.float32)
        
        # Enhance metadata
        doc_metadata = {
            "text": text[:500],  # Store preview
            "word_count": len(text.split()),
            "char_count": len(text),
            **(metadata or {})
        }
        
        self.db.add_vector(doc_id, embedding, doc_metadata)
        
        # Auto-build relationships with existing documents
        self._build_relationships(doc_id, doc_metadata)
        
    def _build_relationships(self, new_doc_id, metadata):
        """Automatically build relationships based on metadata"""
        category = metadata.get("category")
        tags = metadata.get("tags", [])
        
        # Find similar documents by category
        if category:
            for doc_id in self.db.list_vectors():
                if doc_id == new_doc_id:
                    continue
                    
                existing = self.db.get_vector(doc_id)
                if existing and existing["metadata"].get("category") == category:
                    self.db.add_relationship(new_doc_id, doc_id, "semantic", 0.6)
        
        # Connect by shared tags
        for tag in tags:
            for doc_id in self.db.list_vectors():
                if doc_id == new_doc_id:
                    continue
                    
                existing = self.db.get_vector(doc_id)
                if existing and tag in existing["metadata"].get("tags", []):
                    self.db.add_relationship(new_doc_id, doc_id, "associative", 0.5)
    
    def query(self, question, top_k=5, strategy="balanced"):
        """Query with different search strategies"""
        query_embedding = self.model.encode([question])[0].astype(np.float32)
        
        if strategy == "precise":
            params = rudradb.SearchParams(
                top_k=top_k,
                include_relationships=False,
                similarity_threshold=0.4
            )
        elif strategy == "discovery":
            params = rudradb.SearchParams(
                top_k=top_k * 2,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.6,
                similarity_threshold=0.1
            )
        else:  # balanced
            params = rudradb.SearchParams(
                top_k=top_k,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.3
            )
        
        results = self.db.search(query_embedding, params)
        
        # Format results
        formatted = []
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            formatted.append({
                "doc_id": result.vector_id,
                "text": vector["metadata"]["text"],
                "similarity": result.similarity_score,
                "combined_score": result.combined_score,
                "source": result.source,
                "hops": result.hop_count
            })
        
        return formatted

# Example usage
rag = LocalRAG()

# Add documents
documents = [
    ("doc1", "Python is a programming language", {"category": "programming", "tags": ["python"]}),
    ("doc2", "Machine learning uses algorithms to find patterns", {"category": "ai", "tags": ["ml", "algorithms"]}),
    ("doc3", "Pandas is a Python library for data analysis", {"category": "programming", "tags": ["python", "pandas"]}),
    ("doc4", "Scikit-learn provides ML algorithms in Python", {"category": "ai", "tags": ["ml", "python", "scikit-learn"]}),
    ("doc5", "Data visualization helps understand patterns", {"category": "analysis", "tags": ["visualization", "patterns"]})
]

for doc_id, text, metadata in documents:
    rag.add_document(doc_id, text, metadata)

print(f"📚 RAG system: {rag.db.vector_count()} documents, {rag.db.relationship_count()} connections")

# Query examples
results = rag.query("How to analyze data with Python?", strategy="discovery")
print(f"\n🔍 Found {len(results)} relevant documents:")
for result in results:
    source_info = f"(via {result['source']}, {result['hops']} hops)" if result['hops'] > 0 else "(direct)"
    print(f"  {result['doc_id']}: {result['text']} {source_info}")
```

### Tutorial 3: Recommendation System

```python
import rudradb
import numpy as np

class SmartRecommender:
    def __init__(self):
        self.db = rudradb.RudraDB()
        self.user_profiles = {}
    
    def add_item(self, item_id, features, category=None, tags=None):
        """Add item with feature embedding"""
        embedding = np.array(features, dtype=np.float32)
        metadata = {
            "category": category or "general",
            "tags": tags or [],
            "popularity": 0,
            "ratings": []
        }
        self.db.add_vector(item_id, embedding, metadata)
    
    def add_user_interaction(self, user_id, item_id, interaction_type, rating=None):
        """Record user-item interaction"""
        # Create relationship based on interaction type
        interaction_map = {
            "purchased": ("causal", 0.9),
            "viewed": ("associative", 0.3),
            "liked": ("semantic", 0.8),
            "bookmarked": ("semantic", 0.7)
        }
        
        rel_type, strength = interaction_map.get(interaction_type, ("associative", 0.5))
        
        # Add user if not exists (using zero vector as placeholder)
        if not self.db.vector_exists(user_id):
            user_embedding = np.zeros(self.db.dimension() or 384, dtype=np.float32)
            self.db.add_vector(user_id, user_embedding, {"type": "user", "interactions": 0})
        
        # Record interaction
        try:
            self.db.add_relationship(user_id, item_id, rel_type, strength, 
                                   {"interaction": interaction_type, "rating": rating})
        except RuntimeError as e:
            if "capacity" in str(e).lower():
                print(f"💡 Relationship capacity reached - showing upgrade path")
                print(f"   Current interactions: {self.db.relationship_count()}")
                print(f"   Upgrade for unlimited interactions!")
            else:
                raise
        
        # Update item popularity
        item = self.db.get_vector(item_id)
        if item:
            item["metadata"]["popularity"] += 1
            if rating:
                item["metadata"]["ratings"].append(rating)
            self.db.update_vector_metadata(item_id, item["metadata"])
    
    def get_recommendations(self, user_id, top_k=5, strategy="collaborative"):
        """Get personalized recommendations"""
        if not self.db.vector_exists(user_id):
            return self._get_popular_items(top_k)
        
        if strategy == "collaborative":
            # Find items through user relationships
            connected = self.db.get_connected_vectors(user_id, max_hops=2)
            
            # Filter for items only and score them
            item_scores = {}
            for vector, hops in connected:
                if vector["metadata"].get("type") != "user":
                    # Score based on relationship strength and popularity
                    base_score = 1.0 / (hops + 1)  # Closer items score higher
                    popularity_boost = vector["metadata"].get("popularity", 0) * 0.1
                    rating_boost = np.mean(vector["metadata"].get("ratings", [0])) * 0.1
                    
                    final_score = base_score + popularity_boost + rating_boost
                    item_scores[vector["id"]] = final_score
            
            # Sort by score
            recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            return [(item_id, score) for item_id, score in recommendations[:top_k]]
        
        else:  # content-based
            user_vector = self.db.get_vector(user_id)
            if user_vector:
                params = rudradb.SearchParams(
                    top_k=top_k + 1,  # +1 to exclude user itself
                    include_relationships=True,
                    similarity_threshold=0.2
                )
                results = self.db.search(user_vector["embedding"], params)
                
                # Filter out the user and return items
                items = []
                for result in results:
                    if result.vector_id != user_id:
                        vector = self.db.get_vector(result.vector_id)
                        if vector["metadata"].get("type") != "user":
                            items.append((result.vector_id, result.combined_score))
                
                return items[:top_k]
        
        return []
    
    def _get_popular_items(self, top_k):
        """Fallback: popular items for new users"""
        items = []
        for item_id in self.db.list_vectors():
            vector = self.db.get_vector(item_id)
            if vector and vector["metadata"].get("type") != "user":
                popularity = vector["metadata"].get("popularity", 0)
                items.append((item_id, popularity))
        
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]

# Example usage
recommender = SmartRecommender()

# Add items (movies)
movies = [
    ("movie1", [0.8, 0.2, 0.9], "action", ["adventure", "sci-fi"]),
    ("movie2", [0.2, 0.9, 0.1], "drama", ["romance", "emotional"]),
    ("movie3", [0.7, 0.3, 0.8], "action", ["adventure", "thriller"]),
    ("movie4", [0.1, 0.8, 0.2], "drama", ["indie", "emotional"]),
    ("movie5", [0.9, 0.1, 0.7], "action", ["sci-fi", "future"])
]

for movie_id, features, category, tags in movies:
    recommender.add_item(movie_id, features, category, tags)

# Simulate user interactions
interactions = [
    ("user1", "movie1", "liked", 4.5),
    ("user1", "movie3", "viewed", None),
    ("user1", "movie5", "bookmarked", None),
    ("user2", "movie2", "liked", 4.0),
    ("user2", "movie4", "purchased", 5.0),
    ("user3", "movie1", "liked", 4.0),
    ("user3", "movie2", "viewed", None)
]

for user_id, movie_id, interaction, rating in interactions:
    recommender.add_user_interaction(user_id, movie_id, interaction, rating)

# Get recommendations
print("🎬 Movie Recommendations:")
for user_id in ["user1", "user2", "user3"]:
    recs = recommender.get_recommendations(user_id, top_k=3)
    print(f"\n{user_id}: {recs}")
```

---

## 🔌 ML Framework Integration

### OpenAI Embeddings
```python
import openai
import rudradb
import numpy as np

class OpenAI_RudraDB:
    def __init__(self, api_key):
        openai.api_key = api_key
        # Use auto-detection for OpenAI embeddings (1536 dimensions)
        self.db = rudradb.RudraDB()
    
    def add_document(self, doc_id, text, metadata=None):
        """Add document with OpenAI embedding"""
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
        
        doc_metadata = {
            "text": text[:500],
            "model": "ada-002",
            "tokens": len(text.split()),
            **(metadata or {})
        }
        
        self.db.add_vector(doc_id, embedding, doc_metadata)
    
    def semantic_search(self, query, top_k=5, include_relationships=True):
        """Search with OpenAI query embedding"""
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query
        )
        
        query_embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
        
        params = rudradb.SearchParams(
            top_k=top_k,
            include_relationships=include_relationships,
            max_hops=2
        )
        
        return self.db.search(query_embedding, params)

# Usage
# openai_db = OpenAI_RudraDB("your-api-key")
# openai_db.add_document("doc1", "Your document text here")
# results = openai_db.semantic_search("query text")
```

### Sentence Transformers
```python
from sentence_transformers import SentenceTransformer
import rudradb
import numpy as np

class SentenceTransformer_RudraDB:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        # Auto-detect dimension from first embedding
        self.db = rudradb.RudraDB()
        print(f"Using model: {model_name}")
    
    def add_texts(self, texts, ids=None, metadatas=None):
        """Add multiple texts efficiently"""
        if ids is None:
            ids = [f"text_{i}" for i in range(len(texts))]
        
        # Batch encode for efficiency
        embeddings = self.model.encode(texts)
        
        for i, (text_id, embedding) in enumerate(zip(ids, embeddings)):
            metadata = metadatas[i] if metadatas else {}
            metadata.update({
                "text": texts[i][:500],
                "model": self.model.get_sentence_embedding_dimension(),
                "length": len(texts[i])
            })
            
            self.db.add_vector(text_id, embedding.astype(np.float32), metadata)
        
        # Auto-detected dimension info
        if self.db.dimension():
            print(f"Auto-detected dimension: {self.db.dimension()}")
    
    def semantic_search(self, query, top_k=5, search_strategy="balanced"):
        """Search with different strategies"""
        query_embedding = self.model.encode([query])[0].astype(np.float32)
        
        strategies = {
            "similarity_only": rudradb.SearchParams(
                top_k=top_k,
                include_relationships=False,
                similarity_threshold=0.3
            ),
            "balanced": rudradb.SearchParams(
                top_k=top_k,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.3
            ),
            "discovery": rudradb.SearchParams(
                top_k=top_k,
                include_relationships=True,
                max_hops=2,
                relationship_weight=0.6,
                similarity_threshold=0.1
            )
        }
        
        params = strategies.get(search_strategy, strategies["balanced"])
        return self.db.search(query_embedding, params)
    
    def build_topic_relationships(self, similarity_threshold=0.7):
        """Automatically build relationships based on similarity"""
        vectors = self.db.list_vectors()
        
        print(f"Building relationships for {len(vectors)} vectors...")
        relationships_added = 0
        
        for i, vec_id1 in enumerate(vectors):
            vector1 = self.db.get_vector(vec_id1)
            if not vector1:
                continue
                
            for vec_id2 in vectors[i+1:]:
                vector2 = self.db.get_vector(vec_id2)
                if not vector2:
                    continue
                
                # Calculate similarity
                similarity = np.dot(vector1["embedding"], vector2["embedding"]) / (
                    np.linalg.norm(vector1["embedding"]) * np.linalg.norm(vector2["embedding"])
                )
                
                if similarity > similarity_threshold:
                    try:
                        self.db.add_relationship(vec_id1, vec_id2, "semantic", similarity)
                        relationships_added += 1
                    except RuntimeError as e:
                        if "capacity" in str(e).lower():
                            print(f"Relationship capacity reached at {relationships_added} relationships")
                            break
                        else:
                            raise
        
        print(f"Added {relationships_added} automatic relationships")

# Usage example
st_db = SentenceTransformer_RudraDB('all-MiniLM-L6-v2')

# Add documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language for data science",
    "Deep learning uses neural networks with multiple layers",
    "Data preprocessing is crucial for ML model performance",
    "Natural language processing helps computers understand text"
]

st_db.add_texts(documents)

# Build automatic relationships
st_db.build_topic_relationships(similarity_threshold=0.6)

# Search
results = st_db.semantic_search("How to use Python for AI?", search_strategy="discovery")
print(f"Found {len(results)} results")
```

### HuggingFace Transformers
```python
from transformers import AutoTokenizer, AutoModel
import torch
import rudradb
import numpy as np

class HuggingFace_RudraDB:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Auto-detect dimension on first use
        self.db = rudradb.RudraDB()
        self.model_name = model_name
    
    def get_embedding(self, text):
        """Generate embedding for text"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                               padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Mean pooling
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            mean_pooled = sum_embeddings / sum_mask
            
        return mean_pooled.squeeze().numpy().astype(np.float32)
    
    def add_document(self, doc_id, text, metadata=None):
        """Add document with HuggingFace embedding"""
        embedding = self.get_embedding(text)
        
        doc_metadata = {
            "text": text[:500],
            "model": self.model_name,
            "length": len(text),
            **(metadata or {})
        }
        
        self.db.add_vector(doc_id, embedding, doc_metadata)
        
        # Show dimension info on first add
        if self.db.vector_count() == 1:
            print(f"Auto-detected dimension: {self.db.dimension()}")
    
    def search_documents(self, query, top_k=5, filter_by=None):
        """Search with optional metadata filtering"""
        query_embedding = self.get_embedding(query)
        
        # Adjust search based on filters
        if filter_by:
            # Use broader search when filtering
            params = rudradb.SearchParams(
                top_k=top_k * 2,  # Get more results to filter
                include_relationships=True,
                max_hops=2
            )
        else:
            params = rudradb.SearchParams(
                top_k=top_k,
                include_relationships=True,
                max_hops=2
            )
        
        results = self.db.search(query_embedding, params)
        
        # Apply metadata filters
        if filter_by:
            filtered_results = []
            for result in results:
                vector = self.db.get_vector(result.vector_id)
                if vector:
                    metadata = vector["metadata"]
                    # Check if all filter criteria match
                    matches = all(
                        metadata.get(key) == value 
                        for key, value in filter_by.items()
                    )
                    if matches:
                        filtered_results.append(result)
            results = filtered_results[:top_k]
        
        return results

# Usage example
hf_db = HuggingFace_RudraDB('sentence-transformers/all-MiniLM-L6-v2')

# Add documents with metadata
documents = [
    ("doc1", "Python machine learning tutorial", {"category": "tutorial", "level": "beginner"}),
    ("doc2", "Advanced deep learning concepts", {"category": "tutorial", "level": "advanced"}),
    ("doc3", "Python data science introduction", {"category": "tutorial", "level": "beginner"}),
    ("doc4", "Machine learning research paper", {"category": "research", "level": "expert"}),
    ("doc5", "Beginner's guide to programming", {"category": "tutorial", "level": "beginner"})
]

for doc_id, text, metadata in documents:
    hf_db.add_document(doc_id, text, metadata)

# Search with filtering
results = hf_db.search_documents(
    "learn Python programming", 
    top_k=3, 
    filter_by={"category": "tutorial", "level": "beginner"}
)

print(f"Found {len(results)} beginner tutorials")
```

---

## 💡 Best Practices

### 1. Dimension Management
```python
# ✅ DO: Use auto-detection for flexibility
db = rudradb.RudraDB()  # Will auto-detect from first embedding

# ✅ DO: Use explicit dimension when you know it
db = rudradb.RudraDB(dimension=384)  # For sentence-transformers/all-MiniLM-L6-v2

# ❌ DON'T: Mix different embedding dimensions
# This will cause dimension mismatch errors
```

### 2. Metadata Design
```python
# ✅ DO: Use consistent metadata structure
def create_metadata(title, category, tags, **kwargs):
    return {
        "title": title,
        "category": category,
        "tags": tags or [],
        "created_at": datetime.now().isoformat(),
        "version": "1.0",
        **kwargs
    }

# ✅ DO: Keep metadata searchable
metadata = {
    "title": "Document title",
    "category": "education",  # Use for filtering
    "tags": ["python", "tutorial"],  # Use for associations
    "level": "beginner",  # Use for recommendations
    "author": "Jane Doe"
}

# ❌ DON'T: Store large text in metadata
# Store summaries or previews instead of full content
```

### 3. Relationship Strategy
```python
def build_smart_relationships(db, doc_id, metadata, max_connections=5):
    """Build relationships strategically"""
    category = metadata.get("category")
    tags = metadata.get("tags", [])
    
    relationships_added = 0
    
    # Connect to same category (hierarchical)
    if category and relationships_added < max_connections:
        similar_docs = [
            vid for vid in db.list_vectors()
            if db.get_vector(vid)["metadata"].get("category") == category
        ]
        
        # Connect to most recent in category
        for other_doc in similar_docs[-3:]:
            if other_doc != doc_id and relationships_added < max_connections:
                db.add_relationship(doc_id, other_doc, "hierarchical", 0.7)
                relationships_added += 1
    
    # Connect by shared tags (associative)
    for tag in tags[:3]:  # Limit tag connections
        if relationships_added >= max_connections:
            break
            
        tagged_docs = [
            vid for vid in db.list_vectors()
            if tag in db.get_vector(vid)["metadata"].get("tags", [])
        ]
        
        for other_doc in tagged_docs[-2:]:
            if other_doc != doc_id and relationships_added < max_connections:
                db.add_relationship(doc_id, other_doc, "associative", 0.5)
                relationships_added += 1
    
    return relationships_added

# Usage
doc_id = "new_document"
metadata = {"category": "tutorial", "tags": ["python", "ml", "beginner"]}
connections = build_smart_relationships(db, doc_id, metadata)
print(f"Built {connections} strategic relationships")
```

### 4. Search Optimization
```python
class SearchOptimizer:
    def __init__(self, db):
        self.db = db
    
    def adaptive_search(self, query_embedding, context="general"):
        """Adapt search parameters based on context"""
        
        base_params = {
            "top_k": 10,
            "include_relationships": True,
            "max_hops": 2
        }
        
        if context == "precise":
            # High precision, low recall
            base_params.update({
                "top_k": 5,
                "similarity_threshold": 0.5,
                "include_relationships": False
            })
            
        elif context == "discovery":
            # High recall, discovery-focused
            base_params.update({
                "top_k": 15,
                "similarity_threshold": 0.1,
                "relationship_weight": 0.6
            })
            
        elif context == "recommendation":
            # Balanced with relationship emphasis
            base_params.update({
                "top_k": 10,
                "similarity_threshold": 0.2,
                "relationship_weight": 0.5,
                "relationship_types": ["semantic", "associative"]
            })
        
        params = rudradb.SearchParams(**base_params)
        return self.db.search(query_embedding, params)
    
    def progressive_search(self, query_embedding, target_results=5):
        """Try different search strategies until target met"""
        
        strategies = [
            ("precise", 0.5, False, 0.2),
            ("balanced", 0.3, True, 0.3), 
            ("broad", 0.1, True, 0.5)
        ]
        
        for strategy_name, threshold, use_rels, rel_weight in strategies:
            params = rudradb.SearchParams(
                top_k=target_results,
                similarity_threshold=threshold,
                include_relationships=use_rels,
                relationship_weight=rel_weight,
                max_hops=2
            )
            
            results = self.db.search(query_embedding, params)
            
            if len(results) >= target_results:
                print(f"Found {len(results)} results using {strategy_name} strategy")
                return results
        
        return results  # Return whatever we got

# Usage
optimizer = SearchOptimizer(db)

# Context-aware search
results = optimizer.adaptive_search(query_embedding, context="discovery")

# Progressive search
results = optimizer.progressive_search(query_embedding, target_results=8)
```

### 5. Capacity Management
```python
def monitor_capacity(db):
    """Monitor and report capacity usage"""
    stats = db.get_statistics()
    usage = stats['capacity_usage']
    
    print(f"📊 RudraDB-Opin Capacity Usage:")
    print(f"   Vectors: {stats['vector_count']}/{rudradb.MAX_VECTORS} ({usage['vector_usage_percent']:.1f}%)")
    print(f"   Relationships: {stats['relationship_count']}/{rudradb.MAX_RELATIONSHIPS} ({usage['relationship_usage_percent']:.1f}%)")
    
    # Warnings
    if usage['vector_usage_percent'] > 90:
        print("⚠️  Vector capacity critical - consider upgrade")
    elif usage['vector_usage_percent'] > 80:
        print("⚠️  Vector capacity warning")
    
    if usage['relationship_usage_percent'] > 90:
        print("⚠️  Relationship capacity critical - consider upgrade")
    elif usage['relationship_usage_percent'] > 80:
        print("⚠️  Relationship capacity warning")
    
    return usage

def capacity_aware_add_vector(db, vec_id, embedding, metadata):
    """Add vector with capacity awareness"""
    try:
        db.add_vector(vec_id, embedding, metadata)
        return True
    except RuntimeError as e:
        if "RudraDB-Opin Vector Limit Reached" in str(e):
            print("💡 Vector capacity reached!")
            print("   This is perfect for learning - you've explored 100 vectors!")
            print("   Ready for production? Upgrade to full RudraDB for 100,000+ vectors")
            return False
        else:
            raise

def capacity_aware_add_relationship(db, source_id, target_id, rel_type, strength):
    """Add relationship with capacity awareness"""
    try:
        db.add_relationship(source_id, target_id, rel_type, strength)
        return True
    except RuntimeError as e:
        if "RudraDB-Opin Relationship Limit Reached" in str(e):
            print("💡 Relationship capacity reached!")
            print("   Amazing! You've built 500 relationships - you understand relationship modeling!")
            print("   Ready for production? Upgrade to full RudraDB for 250,000+ relationships")
            return False
        else:
            raise

# Usage
usage = monitor_capacity(db)

# Capacity-aware operations
if capacity_aware_add_vector(db, "new_doc", embedding, metadata):
    print("✅ Vector added successfully")
    
    if capacity_aware_add_relationship(db, "new_doc", "related_doc", "semantic", 0.8):
        print("✅ Relationship added successfully")
```

---

## ⚡ Performance Guide

### Understanding Performance Characteristics

RudraDB-Opin is optimized for learning and tutorial scenarios (100 vectors, 500 relationships). Here's what to expect:

```python
import time
import rudradb
import numpy as np

def benchmark_opin_performance():
    """Benchmark RudraDB-Opin performance within its limits"""
    db = rudradb.RudraDB()
    
    print("⚡ RudraDB-Opin Performance Benchmark")
    print("="*50)
    
    # Vector addition performance
    print("\n1️⃣ Vector Addition Performance:")
    vectors_to_add = 50  # Well within limit
    
    start_time = time.time()
    for i in range(vectors_to_add):
        embedding = np.random.rand(384).astype(np.float32)
        metadata = {"index": i, "category": f"cat_{i % 5}"}
        db.add_vector(f"vec_{i}", embedding, metadata)
    
    vector_time = time.time() - start_time
    print(f"   Added {vectors_to_add} vectors in {vector_time:.3f}s")
    print(f"   Rate: {vectors_to_add/vector_time:.0f} vectors/second")
    
    # Relationship addition performance
    print("\n2️⃣ Relationship Addition Performance:")
    relationships_to_add = 100  # Well within limit
    
    start_time = time.time()
    relationships_added = 0
    for i in range(relationships_to_add):
        source = i % vectors_to_add
        target = (i + 1) % vectors_to_add
        if source != target:
            db.add_relationship(f"vec_{source}", f"vec_{target}", "semantic", 0.8)
            relationships_added += 1
    
    rel_time = time.time() - start_time
    print(f"   Added {relationships_added} relationships in {rel_time:.3f}s")
    print(f"   Rate: {relationships_added/rel_time:.0f} relationships/second")
    
    # Search performance
    print("\n3️⃣ Search Performance:")
    query = np.random.rand(384).astype(np.float32)
    search_iterations = 50
    
    # Similarity-only search
    start_time = time.time()
    for _ in range(search_iterations):
        params = rudradb.SearchParams(top_k=10, include_relationships=False)
        results = db.search(query, params)
    
    similarity_time = time.time() - start_time
    print(f"   {search_iterations} similarity searches in {similarity_time:.3f}s")
    print(f"   Rate: {search_iterations/similarity_time:.0f} searches/second")
    
    # Relationship-aware search
    start_time = time.time()
    for _ in range(search_iterations):
        params = rudradb.SearchParams(top_k=10, include_relationships=True, max_hops=2)
        results = db.search(query, params)
    
    relationship_time = time.time() - start_time
    print(f"   {search_iterations} relationship-aware searches in {relationship_time:.3f}s")
    print(f"   Rate: {search_iterations/relationship_time:.0f} searches/second")
    
    # Multi-hop traversal performance
    print("\n4️⃣ Multi-Hop Traversal Performance:")
    start_time = time.time()
    
    traversal_iterations = 20
    for i in range(traversal_iterations):
        connected = db.get_connected_vectors(f"vec_{i % vectors_to_add}", max_hops=2)
    
    traversal_time = time.time() - start_time
    print(f"   {traversal_iterations} multi-hop traversals in {traversal_time:.3f}s")
    print(f"   Rate: {traversal_iterations/traversal_time:.0f} traversals/second")
    
    # Final statistics
    stats = db.get_statistics()
    print(f"\n📊 Final Database State:")
    print(f"   Vectors: {stats['vector_count']}/{rudradb.MAX_VECTORS}")
    print(f"   Relationships: {stats['relationship_count']}/{rudradb.MAX_RELATIONSHIPS}")
    print(f"   Dimension: {stats['dimension']}")
    
    print(f"\n✅ RudraDB-Opin Performance Summary:")
    print(f"   Perfect for learning and tutorials!")
    print(f"   All operations complete in milliseconds")
    print(f"   Ready to scale? Upgrade to full RudraDB!")

# Run benchmark
benchmark_opin_performance()
```

### Optimizing for Opin Limits

```python
class OpinOptimizer:
    """Optimization strategies specifically for Opin limits"""
    
    @staticmethod
    def optimize_vector_selection(candidates, limit=100):
        """Select most representative vectors within Opin limit"""
        if len(candidates) <= limit:
            return candidates
        
        # Strategy 1: Diverse sampling
        # Select vectors that maximize coverage
        selected = []
        categories = set()
        
        # First pass: one from each category
        for candidate in candidates:
            metadata = candidate.get("metadata", {})
            category = metadata.get("category")
            if category and category not in categories:
                selected.append(candidate)
                categories.add(category)
                if len(selected) >= limit:
                    break
        
        # Second pass: fill remaining slots with highest quality
        remaining_slots = limit - len(selected)
        if remaining_slots > 0:
            remaining_candidates = [c for c in candidates if c not in selected]
            # Sort by some quality metric (e.g., relevance score)
            remaining_candidates.sort(
                key=lambda x: x.get("quality_score", 0), 
                reverse=True
            )
            selected.extend(remaining_candidates[:remaining_slots])
        
        return selected[:limit]
    
    @staticmethod
    def optimize_relationship_selection(relationships, limit=500):
        """Select most important relationships within Opin limit"""
        if len(relationships) <= limit:
            return relationships
        
        # Sort by strength and importance
        sorted_rels = sorted(
            relationships,
            key=lambda r: (r["strength"], r.get("importance", 0)),
            reverse=True
        )
        
        return sorted_rels[:limit]
    
    @staticmethod
    def efficient_batch_loading(db, vectors, relationships):
        """Efficiently load data within Opin constraints"""
        print(f"Loading {len(vectors)} vectors and {len(relationships)} relationships...")
        
        # Optimize selections
        selected_vectors = OpinOptimizer.optimize_vector_selection(
            vectors, rudradb.MAX_VECTORS
        )
        selected_relationships = OpinOptimizer.optimize_relationship_selection(
            relationships, rudradb.MAX_RELATIONSHIPS
        )
        
        print(f"Optimized to {len(selected_vectors)} vectors and {len(selected_relationships)} relationships")
        
        # Load vectors first
        loaded_vectors = set()
        for vector in selected_vectors:
            try:
                db.add_vector(vector["id"], vector["embedding"], vector["metadata"])
                loaded_vectors.add(vector["id"])
            except RuntimeError as e:
                if "capacity" in str(e).lower():
                    print(f"Vector capacity reached at {len(loaded_vectors)} vectors")
                    break
                raise
        
        # Load relationships (only for loaded vectors)
        loaded_relationships = 0
        for rel in selected_relationships:
            if rel["source_id"] in loaded_vectors and rel["target_id"] in loaded_vectors:
                try:
                    db.add_relationship(
                        rel["source_id"], 
                        rel["target_id"],
                        rel["relationship_type"], 
                        rel["strength"]
                    )
                    loaded_relationships += 1
                except RuntimeError as e:
                    if "capacity" in str(e).lower():
                        print(f"Relationship capacity reached at {loaded_relationships} relationships")
                        break
                    raise
        
        print(f"✅ Loaded {len(loaded_vectors)} vectors and {loaded_relationships} relationships")
        return len(loaded_vectors), loaded_relationships

# Usage example
optimizer = OpinOptimizer()

# Simulate large dataset
large_vectors = [
    {
        "id": f"doc_{i}",
        "embedding": np.random.rand(384).astype(np.float32),
        "metadata": {"category": f"cat_{i % 10}", "quality_score": np.random.rand()},
    }
    for i in range(200)  # More than Opin limit
]

large_relationships = [
    {
        "source_id": f"doc_{i}",
        "target_id": f"doc_{j}",
        "relationship_type": "semantic",
        "strength": 0.5 + np.random.rand() * 0.5,
        "importance": np.random.rand()
    }
    for i in range(50) for j in range(i+1, min(i+20, 200))  # More than Opin limit
]

# Efficiently load optimized subset
db = rudradb.RudraDB()
vectors_loaded, relationships_loaded = optimizer.efficient_batch_loading(
    db, large_vectors, large_relationships
)
```

---

## 📏 Capacity Management

### Understanding Opin Limits

RudraDB-Opin is designed with specific limits that are perfect for learning:

```python
import rudradb

def show_opin_specifications():
    """Display complete Opin specifications"""
    print("🧬 RudraDB-Opin Specifications")
    print("="*40)
    
    print(f"📊 Capacity Limits:")
    print(f"   Max Vectors: {rudradb.MAX_VECTORS:,}")
    print(f"   Max Relationships: {rudradb.MAX_RELATIONSHIPS:,}")
    print(f"   Max Hops: {rudradb.MAX_HOPS}")
    print(f"   Relationship Types: {rudradb.RELATIONSHIP_TYPES}")
    
    print(f"\n🎯 Edition Info:")
    print(f"   Edition: {rudradb.EDITION}")
    print(f"   Version: {rudradb.__version__}")
    print(f"   Is Free: {rudradb.IS_FREE_VERSION}")
    
    print(f"\n🚀 Upgrade Path:")
    print(f"   Message: {rudradb.UPGRADE_MESSAGE}")
    print(f"   URL: {rudradb.UPGRADE_URL}")
    
    # Get detailed limits
    limits = rudradb.get_opin_limits()
    print(f"\n📋 Detailed Limits:")
    for key, value in limits.items():
        print(f"   {key}: {value}")

show_opin_specifications()
```

### Capacity Monitoring

```python
class CapacityMonitor:
    """Monitor and manage RudraDB-Opin capacity"""
    
    def __init__(self, db):
        self.db = db
        
    def get_usage_report(self):
        """Get detailed capacity usage report"""
        stats = self.db.get_statistics()
        usage = stats['capacity_usage']
        
        report = {
            "vectors": {
                "used": stats['vector_count'],
                "max": rudradb.MAX_VECTORS,
                "percentage": usage['vector_usage_percent'],
                "remaining": usage['vector_capacity_remaining'],
                "status": self._get_status(usage['vector_usage_percent'])
            },
            "relationships": {
                "used": stats['relationship_count'],
                "max": rudradb.MAX_RELATIONSHIPS,
                "percentage": usage['relationship_usage_percent'],
                "remaining": usage['relationship_capacity_remaining'],
                "status": self._get_status(usage['relationship_usage_percent'])
            },
            "overall_health": self._get_overall_health(stats)
        }
        
        return report
    
    def _get_status(self, percentage):
        """Get status based on usage percentage"""
        if percentage >= 95:
            return "CRITICAL"
        elif percentage >= 85:
            return "WARNING"
        elif percentage >= 70:
            return "MODERATE"
        else:
            return "HEALTHY"
    
    def _get_overall_health(self, stats):
        """Get overall database health"""
        usage = stats['capacity_usage']
        max_usage = max(usage['vector_usage_percent'], usage['relationship_usage_percent'])
        
        if max_usage >= 95:
            return "Time to upgrade to full RudraDB!"
        elif max_usage >= 85:
            return "Approaching capacity - consider upgrade soon"
        elif max_usage >= 70:
            return "Good usage - learning well!"
        else:
            return "Plenty of capacity for learning"
    
    def print_usage_report(self):
        """Print formatted usage report"""
        report = self.get_usage_report()
        
        print("📊 RudraDB-Opin Capacity Report")
        print("="*45)
        
        # Vector usage
        vec = report["vectors"]
        print(f"\n🔢 Vectors: {vec['used']:,}/{vec['max']:,} ({vec['percentage']:.1f}%)")
        print(f"   Status: {vec['status']}")
        print(f"   Remaining: {vec['remaining']:,}")
        self._print_progress_bar(vec['percentage'])
        
        # Relationship usage  
        rel = report["relationships"]
        print(f"\n🔗 Relationships: {rel['used']:,}/{rel['max']:,} ({rel['percentage']:.1f}%)")
        print(f"   Status: {rel['status']}")
        print(f"   Remaining: {rel['remaining']:,}")
        self._print_progress_bar(rel['percentage'])
        
        # Overall health
        print(f"\n🏥 Overall Health: {report['overall_health']}")
        
        # Upgrade suggestion if needed
        max_usage = max(vec['percentage'], rel['percentage'])
        if max_usage >= 85:
            print(f"\n💡 Upgrade Suggestion:")
            print(f"   You've learned relationship-aware vector search!")
            print(f"   Ready for production? {rudradb.UPGRADE_MESSAGE}")
    
    def _print_progress_bar(self, percentage, width=30):
        """Print a visual progress bar"""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        print(f"   [{bar}] {percentage:.1f}%")
    
    def predict_capacity(self, vectors_per_hour=0, relationships_per_hour=0, hours=1):
        """Predict when capacity will be reached"""
        current = self.get_usage_report()
        
        vec_remaining = current["vectors"]["remaining"]
        rel_remaining = current["relationships"]["remaining"]
        
        if vectors_per_hour > 0:
            vec_hours = vec_remaining / vectors_per_hour
            print(f"📈 Vector capacity will be reached in {vec_hours:.1f} hours")
        
        if relationships_per_hour > 0:
            rel_hours = rel_remaining / relationships_per_hour
            print(f"📈 Relationship capacity will be reached in {rel_hours:.1f} hours")
    
    def suggest_cleanup(self):
        """Suggest cleanup strategies when approaching limits"""
        report = self.get_usage_report()
        
        suggestions = []
        
        if report["vectors"]["percentage"] > 85:
            suggestions.append("Consider removing test/temporary vectors")
            suggestions.append("Archive older vectors if doing time-series learning")
        
        if report["relationships"]["percentage"] > 85:
            suggestions.append("Remove low-strength relationships (< 0.3)")
            suggestions.append("Focus on most important relationship types")
        
        if suggestions:
            print("💡 Cleanup Suggestions:")
            for suggestion in suggestions:
                print(f"   • {suggestion}")
        else:
            print("✅ No cleanup needed - capacity is healthy")
        
        return suggestions

# Usage
monitor = CapacityMonitor(db)

# Regular monitoring
monitor.print_usage_report()

# Capacity planning
monitor.predict_capacity(vectors_per_hour=10, relationships_per_hour=50, hours=2)

# Cleanup suggestions
monitor.suggest_cleanup()
```

### Graceful Capacity Handling

```python
class GracefulCapacityHandler:
    """Handle capacity limits gracefully with user guidance"""
    
    def __init__(self, db):
        self.db = db
        self.monitor = CapacityMonitor(db)
    
    def smart_add_vector(self, vec_id, embedding, metadata=None, auto_cleanup=False):
        """Add vector with intelligent capacity handling"""
        try:
            self.db.add_vector(vec_id, embedding, metadata)
            return {"success": True, "message": f"Vector '{vec_id}' added successfully"}
            
        except RuntimeError as e:
            if "RudraDB-Opin Vector Limit Reached" in str(e):
                return self._handle_vector_capacity_reached(vec_id, embedding, metadata, auto_cleanup)
            else:
                raise
    
    def smart_add_relationship(self, source_id, target_id, rel_type, strength, metadata=None, auto_cleanup=False):
        """Add relationship with intelligent capacity handling"""
        try:
            self.db.add_relationship(source_id, target_id, rel_type, strength, metadata)
            return {"success": True, "message": f"Relationship '{source_id}' -> '{target_id}' added"}
            
        except RuntimeError as e:
            if "RudraDB-Opin Relationship Limit Reached" in str(e):
                return self._handle_relationship_capacity_reached(
                    source_id, target_id, rel_type, strength, metadata, auto_cleanup
                )
            else:
                raise
    
    def _handle_vector_capacity_reached(self, vec_id, embedding, metadata, auto_cleanup):
        """Handle vector capacity reached"""
        current_count = self.db.vector_count()
        
        result = {
            "success": False,
            "capacity_reached": True,
            "resource": "vectors",
            "current_count": current_count,
            "max_count": rudradb.MAX_VECTORS,
            "message": f"🎓 Congratulations! You've learned with {current_count} vectors!"
        }
        
        if auto_cleanup:
            cleaned = self._auto_cleanup_vectors()
            if cleaned > 0:
                # Try again after cleanup
                try:
                    self.db.add_vector(vec_id, embedding, metadata)
                    result.update({
                        "success": True,
                        "cleaned": cleaned,
                        "message": f"Added '{vec_id}' after cleaning up {cleaned} vectors"
                    })
                except RuntimeError:
                    result["message"] = "Even after cleanup, capacity is still full"
        else:
            result.update({
                "suggestions": [
                    "Remove test/temporary vectors",
                    "Export current database and start fresh",
                    f"Upgrade to full RudraDB: {rudradb.UPGRADE_MESSAGE}"
                ]
            })
        
        return result
    
    def _handle_relationship_capacity_reached(self, source_id, target_id, rel_type, strength, metadata, auto_cleanup):
        """Handle relationship capacity reached"""
        current_count = self.db.relationship_count()
        
        result = {
            "success": False,
            "capacity_reached": True,
            "resource": "relationships", 
            "current_count": current_count,
            "max_count": rudradb.MAX_RELATIONSHIPS,
            "message": f"🎓 Amazing! You've built {current_count} relationships!"
        }
        
        if auto_cleanup:
            cleaned = self._auto_cleanup_relationships()
            if cleaned > 0:
                try:
                    self.db.add_relationship(source_id, target_id, rel_type, strength, metadata)
                    result.update({
                        "success": True,
                        "cleaned": cleaned,
                        "message": f"Added relationship after cleaning up {cleaned} weak connections"
                    })
                except RuntimeError:
                    result["message"] = "Even after cleanup, relationship capacity is full"
        else:
            result.update({
                "suggestions": [
                    "Remove weak relationships (strength < 0.3)",
                    "Focus on most important relationship types",
                    f"Upgrade to full RudraDB: {rudradb.UPGRADE_MESSAGE}"
                ]
            })
        
        return result
    
    def _auto_cleanup_vectors(self, target_removal=5):
        """Automatically clean up low-priority vectors"""
        # This is a simple example - in practice, you might have more sophisticated cleanup logic
        vectors = self.db.list_vectors()
        removed = 0
        
        # Remove vectors with minimal relationships (lowest priority)
        for vec_id in vectors[:target_removal]:
            relationships = self.db.get_relationships(vec_id)
            if len(relationships) == 0:  # No relationships = lower priority
                self.db.remove_vector(vec_id)
                removed += 1
                if removed >= target_removal:
                    break
        
        return removed
    
    def _auto_cleanup_relationships(self, target_removal=10):
        """Automatically clean up weak relationships"""
        # Remove weakest relationships
        removed = 0
        
        for vec_id in self.db.list_vectors():
            relationships = self.db.get_relationships(vec_id)
            
            # Sort by strength, remove weakest
            weak_relationships = [r for r in relationships if r["strength"] < 0.3]
            
            for rel in weak_relationships[:target_removal - removed]:
                self.db.remove_relationship(rel["source_id"], rel["target_id"])
                removed += 1
                
                if removed >= target_removal:
                    break
            
            if removed >= target_removal:
                break
        
        return removed

# Usage example
handler = GracefulCapacityHandler(db)

# Smart vector addition
result = handler.smart_add_vector(
    "new_doc", 
    np.random.rand(384).astype(np.float32),
    {"title": "New Document"},
    auto_cleanup=True
)

print(f"Result: {result}")

if not result["success"] and result.get("capacity_reached"):
    print("💡 Capacity reached - this is perfect for learning!")
    print("   You've successfully explored the full capacity of relationship-aware vector search")
    
    if "suggestions" in result:
        print("   Suggestions:")
        for suggestion in result["suggestions"]:
            print(f"     • {suggestion}")
```

---

## 🚀 Upgrade Path

### When to Upgrade

RudraDB-Opin is perfect for learning, but you'll know it's time to upgrade when:

```python
def should_upgrade_assessment(db):
    """Assess if it's time to upgrade from Opin to full RudraDB"""
    
    stats = db.get_statistics()
    usage = stats['capacity_usage']
    
    print("🔍 Upgrade Assessment")
    print("="*30)
    
    upgrade_indicators = []
    
    # Capacity indicators
    if usage['vector_usage_percent'] > 90:
        upgrade_indicators.append("Vector capacity nearly full")
    
    if usage['relationship_usage_percent'] > 90:
        upgrade_indicators.append("Relationship capacity nearly full")
    
    # Usage pattern indicators
    vector_count = stats['vector_count']
    relationship_count = stats['relationship_count']
    
    if vector_count > 50:
        upgrade_indicators.append("You understand vector operations well")
    
    if relationship_count > 200:
        upgrade_indicators.append("You've mastered relationship modeling")
    
    # Relationship complexity
    avg_relationships_per_vector = relationship_count / max(vector_count, 1)
    if avg_relationships_per_vector > 3:
        upgrade_indicators.append("You're building complex relationship networks")
    
    # Learning completion indicators
    relationship_types_used = set()
    for vec_id in db.list_vectors():
        relationships = db.get_relationships(vec_id)
        for rel in relationships:
            relationship_types_used.add(rel["relationship_type"])
    
    if len(relationship_types_used) >= 4:
        upgrade_indicators.append("You've explored most relationship types")
    
    print(f"📊 Current Usage:")
    print(f"   Vectors: {vector_count}/{rudradb.MAX_VECTORS}")
    print(f"   Relationships: {relationship_count}/{rudradb.MAX_RELATIONSHIPS}")
    print(f"   Avg relationships per vector: {avg_relationships_per_vector:.1f}")
    print(f"   Relationship types used: {len(relationship_types_used)}/5")
    
    print(f"\n🎯 Upgrade Indicators:")
    if upgrade_indicators:
        for indicator in upgrade_indicators:
            print(f"   ✅ {indicator}")
    else:
        print(f"   📚 Keep learning - you're doing great!")
    
    # Recommendation
    if len(upgrade_indicators) >= 3:
        print(f"\n🚀 RECOMMENDATION: Time to upgrade!")
        print(f"   You've mastered relationship-aware vector search with RudraDB-Opin")
        print(f"   Ready for production scale: {rudradb.UPGRADE_MESSAGE}")
        return True
    elif len(upgrade_indicators) >= 1:
        print(f"\n💡 RECOMMENDATION: Consider upgrading soon")
        print(f"   You're making great progress with relationship-aware search")
        print(f"   When ready for more scale: {rudradb.UPGRADE_MESSAGE}")
        return False
    else:
        print(f"\n📖 RECOMMENDATION: Keep learning!")
        print(f"   RudraDB-Opin is perfect for your current learning phase")
        return False

# Usage
should_upgrade = should_upgrade_assessment(db)
```

### Migration Guide

```python
def migration_guide():
    """Complete guide for upgrading from Opin to full RudraDB"""
    
    print("🚀 RudraDB-Opin to RudraDB Migration Guide")
    print("="*50)
    
    print("\n📋 Step-by-Step Migration:")
    
    steps = [
        ("1️⃣ Export Your Data", "Export your current RudraDB-Opin data"),
        ("2️⃣ Install Full RudraDB", "Upgrade to the full version"),
        ("3️⃣ Import Your Data", "Bring your learning project forward"),
        ("4️⃣ Scale Up", "Take advantage of full capacity"),
        ("5️⃣ Production Ready", "Deploy with confidence")
    ]
    
    for step, description in steps:
        print(f"\n{step}: {description}")
    
    print(f"\n💻 Code Examples:")
    
    # Step 1: Export
    print(f"\n# Step 1: Export your RudraDB-Opin data")
    print(f"import rudradb")
    print(f"import json")
    print(f"")
    print(f"# Export from Opin")
    print(f"db_opin = rudradb.RudraDB()  # Your current Opin database")
    print(f"exported_data = db_opin.export_data()")
    print(f"")
    print(f"# Save to file") 
    print(f"with open('my_rudradb_data.json', 'w') as f:")
    print(f"    json.dump(exported_data, f, indent=2)")
    print(f"")
    print(f"print(f'Exported {{db_opin.vector_count()}} vectors and {{db_opin.relationship_count()}} relationships')")
    
    # Step 2: Install
    print(f"\n# Step 2: Install full RudraDB")
    print(f"# Run in terminal:")
    print(f"# pip uninstall rudradb-opin")
    print(f"# pip install rudradb")
    
    # Step 3: Import
    print(f"\n# Step 3: Import to full RudraDB")
    print(f"import rudradb  # Now the full version!")
    print(f"import json")
    print(f"")
    print(f"# Load your data")
    print(f"with open('my_rudradb_data.json', 'r') as f:")
    print(f"    data = json.load(f)")
    print(f"")
    print(f"# Create full RudraDB instance")
    print(f"db_full = rudradb.RudraDB()  # Now with 100,000+ vector capacity!")
    print(f"")
    print(f"# Import your learning data")
    print(f"db_full.import_data(data)")
    print(f"")
    print(f"print('🎉 Migration complete!')")
    print(f"print(f'Ready for production with {{db_full.vector_count()}} vectors')")
    
    # Step 4: Scale up
    print(f"\n# Step 4: Scale up (same API, more capacity)")
    print(f"# Your existing code works unchanged!")
    print(f"# Just add more data:")
    print(f"")
    print(f"# Add thousands more vectors")
    print(f"for i in range(1000):")
    print(f"    embedding = model.encode([f'Document {{i}}'])[0].astype(np.float32)")
    print(f"    db_full.add_vector(f'prod_doc_{{i}}', embedding)")
    print(f"")
    print(f"# Build complex relationship networks")
    print(f"# No more capacity limits!")
    
    print(f"\n✅ Benefits After Upgrade:")
    benefits = [
        "100,000+ vectors (1000x more than Opin)",
        "250,000+ relationships (500x more than Opin)", 
        "Same API - no code changes needed",
        "Production-ready performance",
        "Advanced features and optimizations",
        "Enterprise support options"
    ]
    
    for benefit in benefits:
        print(f"   • {benefit}")
    
    print(f"\n🔄 Zero Code Changes Needed:")
    print(f"   Your RudraDB-Opin code will work exactly the same")
    print(f"   Just with much higher capacity limits!")

def create_migration_helper():
    """Create helper functions for smooth migration"""
    
    class MigrationHelper:
        def __init__(self, opin_db):
            self.opin_db = opin_db
        
        def export_with_metadata(self, filename="rudradb_migration.json"):
            """Export with additional migration metadata"""
            # Get the standard export
            data = self.opin_db.export_data()
            
            # Add migration metadata
            migration_info = {
                "migrated_from": "rudradb-opin",
                "opin_version": rudradb.__version__,
                "migration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "original_stats": self.opin_db.get_statistics(),
                "learning_summary": self._create_learning_summary()
            }
            
            data["migration_info"] = migration_info
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Migration data exported to {filename}")
            print(f"📊 Exported {data['migration_info']['original_stats']['vector_count']} vectors")
            print(f"📊 Exported {data['migration_info']['original_stats']['relationship_count']} relationships")
            
            return filename
        
        def _create_learning_summary(self):
            """Create a summary of what was learned with Opin"""
            stats = self.opin_db.get_statistics()
            
            # Analyze relationship types used
            relationship_types = set()
            relationship_strengths = []
            
            for vec_id in self.opin_db.list_vectors():
                relationships = self.opin_db.get_relationships(vec_id)
                for rel in relationships:
                    relationship_types.add(rel["relationship_type"])
                    relationship_strengths.append(rel["strength"])
            
            return {
                "vectors_explored": stats["vector_count"],
                "relationships_built": stats["relationship_count"],
                "relationship_types_used": list(relationship_types),
                "avg_relationship_strength": np.mean(relationship_strengths) if relationship_strengths else 0,
                "capacity_utilization": {
                    "vector_percentage": stats["capacity_usage"]["vector_usage_percent"],
                    "relationship_percentage": stats["capacity_usage"]["relationship_usage_percent"]
                },
                "learning_completeness": len(relationship_types) / 5 * 100  # % of relationship types explored
            }
        
        def validate_for_migration(self):
            """Validate that the data is ready for migration"""
            stats = self.opin_db.get_statistics()
            
            validation_results = {
                "ready_for_migration": True,
                "warnings": [],
                "recommendations": []
            }
            
            # Check if significant learning has occurred
            if stats["vector_count"] < 10:
                validation_results["warnings"].append("Less than 10 vectors - consider exploring more")
            
            if stats["relationship_count"] < 5:
                validation_results["warnings"].append("Less than 5 relationships - try building connections")
            
            # Check relationship diversity
            relationship_types = set()
            for vec_id in self.opin_db.list_vectors():
                relationships = self.opin_db.get_relationships(vec_id)
                for rel in relationships:
                    relationship_types.add(rel["relationship_type"])
            
            if len(relationship_types) < 3:
                validation_results["recommendations"].append("Try exploring more relationship types")
            
            # Overall assessment
            if stats["capacity_usage"]["vector_usage_percent"] > 80 or \
               stats["capacity_usage"]["relationship_usage_percent"] > 80:
                validation_results["recommendations"].append("Excellent usage - ready for production scale!")
            
            return validation_results
    
    return MigrationHelper

# Usage
migration_guide()

# Create migration helper
if 'db' in locals():  # If you have an active RudraDB-Opin instance
    helper_class = create_migration_helper()
    migration_helper = helper_class(db)
    
    # Validate readiness
    validation = migration_helper.validate_for_migration()
    print(f"\n🔍 Migration Readiness:")
    print(f"   Ready: {validation['ready_for_migration']}")
    if validation['warnings']:
        for warning in validation['warnings']:
            print(f"   ⚠️ {warning}")
    if validation['recommendations']:
        for rec in validation['recommendations']:
            print(f"   💡 {rec}")
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Installation Problems
```python
# Problem: Import error after installation
try:
    import rudradb
    print("✅ RudraDB-Opin imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Solutions:")
    print("   1. Check installation: pip list | grep rudradb")
    print("   2. Reinstall: pip uninstall rudradb-opin && pip install rudradb-opin")
    print("   3. Check Python version: python --version (need 3.8+)")
    print("   4. Try in new environment: python -m venv test_env")
```

#### Issue 2: Dimension Mismatch
```python
def debug_dimension_issues():
    """Debug common dimension-related problems"""
    
    print("🔍 Debugging Dimension Issues")
    print("="*40)
    
    db = rudradb.RudraDB()  # Auto-detect dimension
    
    # Test 1: Check if dimension is set
    print(f"1️⃣ Current dimension: {db.dimension()}")
    
    if db.dimension() is None:
        print("   No dimension set yet - will auto-detect from first vector")
    
    # Test 2: Add first vector
    try:
        first_embedding = np.random.rand(384).astype(np.float32)
        db.add_vector("first", first_embedding)
        print(f"✅ First vector added, dimension auto-detected: {db.dimension()}")
    except Exception as e:
        print(f"❌ Error adding first vector: {e}")
        return
    
    # Test 3: Try different dimensions
    test_dimensions = [256, 384, 512, 768, 1536]
    
    for dim in test_dimensions:
        try:
            test_embedding = np.random.rand(dim).astype(np.float32)
            db.add_vector(f"test_{dim}", test_embedding)
            print(f"❌ Unexpected: Added {dim}-dim vector to {db.dimension()}-dim database")
        except Exception as e:
            if "dimension" in str(e).lower():
                print(f"✅ Correctly rejected {dim}-dim vector (expected {db.dimension()})")
            else:
                print(f"❌ Unexpected error for {dim}-dim: {e}")

# Common dimension solutions
def dimension_best_practices():
    """Best practices for handling dimensions"""
    
    print("💡 Dimension Best Practices:")
    print("="*35)
    
    practices = [
        ("Use auto-detection", "db = rudradb.RudraDB()  # Detects from first vector"),
        ("Consistent embeddings", "Always use same model for all vectors"),
        ("Check before adding", "Verify embedding.shape[0] matches db.dimension()"),
        ("Handle mixed sources", "Normalize all embeddings to same dimension"),
        ("Debug with shape", "Print embedding.shape before adding")
    ]
    
    for practice, example in practices:
        print(f"\n✅ {practice}:")
        print(f"   {example}")

debug_dimension_issues()
dimension_best_practices()
```

#### Issue 3: Capacity Limit Confusion
```python
def debug_capacity_limits():
    """Debug capacity-related confusion"""
    
    print("🔍 Understanding RudraDB-Opin Limits")
    print("="*45)
    
    # Clear explanation of limits
    print(f"📊 RudraDB-Opin Specifications:")
    print(f"   Vector Limit: {rudradb.MAX_VECTORS} (perfect for tutorials)")
    print(f"   Relationship Limit: {rudradb.MAX_RELATIONSHIPS} (rich modeling)")
    print(f"   Max Hops: {rudradb.MAX_HOPS} (multi-hop discovery)")
    
    # Test capacity behavior
    db = rudradb.RudraDB()
    
    print(f"\n🧪 Testing Capacity Behavior:")
    
    # Add vectors up to limit
    print(f"   Adding vectors...")
    added_vectors = 0
    for i in range(rudradb.MAX_VECTORS + 2):  # Try to exceed limit
        try:
            embedding = np.random.rand(384).astype(np.float32)
            db.add_vector(f"test_{i}", embedding)
            added_vectors += 1
        except RuntimeError as e:
            if "RudraDB-Opin Vector Limit Reached" in str(e):
                print(f"   ✅ Vector limit correctly enforced at {added_vectors}")
                print(f"   📝 Error message shows upgrade path")
                break
            else:
                print(f"   ❌ Unexpected error: {e}")
                break
    
    # Test relationship limits
    print(f"   Adding relationships...")
    added_relationships = 0
    
    # First add some vectors for relationships
    for i in range(10):
        embedding = np.random.rand(384).astype(np.float32) 
        try:
            db.add_vector(f"rel_test_{i}", embedding)
        except RuntimeError:
            pass  # May hit vector limit
    
    # Try to add many relationships
    for i in range(rudradb.MAX_RELATIONSHIPS + 10):
        try:
            source = i % 10
            target = (i + 1) % 10
            if source != target:
                db.add_relationship(f"rel_test_{source}", f"rel_test_{target}", "semantic", 0.8)
                added_relationships += 1
        except RuntimeError as e:
            if "RudraDB-Opin Relationship Limit Reached" in str(e):
                print(f"   ✅ Relationship limit correctly enforced at {added_relationships}")
                break
            else:
                # Might be vector not found if we hit vector limit first
                continue
    
    print(f"\n💡 Key Points:")
    print(f"   • Limits are by design - perfect for learning")
    print(f"   • Error messages guide you to upgrade path")
    print(f"   • All features available, just capacity limited")
    print(f"   • Upgrade gives you 1000x more capacity")

debug_capacity_limits()
```

#### Issue 4: Relationship Building Problems
```python
def debug_relationship_issues():
    """Debug common relationship problems"""
    
    print("🔍 Debugging Relationship Issues")
    print("="*40)
    
    db = rudradb.RudraDB()
    
    # Add test vectors
    test_vectors = {
        "vec1": np.random.rand(384).astype(np.float32),
        "vec2": np.random.rand(384).astype(np.float32),
        "vec3": np.random.rand(384).astype(np.float32)
    }
    
    for vec_id, embedding in test_vectors.items():
        db.add_vector(vec_id, embedding)
    
    print(f"✅ Added {len(test_vectors)} test vectors")
    
    # Test 1: Valid relationship
    try:
        db.add_relationship("vec1", "vec2", "semantic", 0.8)
        print("✅ Valid relationship added successfully")
    except Exception as e:
        print(f"❌ Unexpected error with valid relationship: {e}")
    
    # Test 2: Invalid relationship type
    try:
        db.add_relationship("vec1", "vec3", "invalid_type", 0.8)
        print("❌ Should have rejected invalid relationship type")
    except Exception as e:
        print(f"✅ Correctly rejected invalid relationship type: {e}")
    
    # Test 3: Self-referencing relationship
    try:
        db.add_relationship("vec1", "vec1", "semantic", 0.8)
        print("❌ Should have rejected self-referencing relationship")
    except Exception as e:
        print(f"✅ Correctly rejected self-referencing relationship: {e}")
    
    # Test 4: Non-existent vectors
    try:
        db.add_relationship("vec1", "nonexistent", "semantic", 0.8)
        print("❌ Should have rejected relationship with non-existent vector")
    except Exception as e:
        print(f"✅ Correctly rejected relationship with non-existent vector: {e}")
    
    # Test 5: Invalid strength values
    invalid_strengths = [-0.5, 1.5, float('nan'), float('inf')]
    
    for strength in invalid_strengths:
        try:
            db.add_relationship("vec2", "vec3", "semantic", strength)
            print(f"❌ Should have rejected invalid strength: {strength}")
        except Exception as e:
            print(f"✅ Correctly rejected invalid strength {strength}: {type(e).__name__}")
    
    print(f"\n📋 Relationship Best Practices:")
    practices = [
        "Use valid relationship types: semantic, hierarchical, temporal, causal, associative",
        "Strength must be between 0.0 and 1.0", 
        "Both source and target vectors must exist",
        "Avoid self-referencing relationships",
        "Choose appropriate relationship types for your connections"
    ]
    
    for practice in practices:
        print(f"   • {practice}")

debug_relationship_issues()
```

#### Issue 5: Search Problems
```python
def debug_search_issues():
    """Debug search-related problems"""
    
    print("🔍 Debugging Search Issues")
    print("="*35)
    
    db = rudradb.RudraDB()
    
    # Set up test data
    documents = [
        ("doc1", "Machine learning fundamentals"),
        ("doc2", "Deep learning with neural networks"),
        ("doc3", "Python programming basics"),
        ("doc4", "Data science introduction"),
        ("doc5", "Artificial intelligence concepts")
    ]
    
    # Use a simple embedding model for testing
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Add documents
        for doc_id, text in documents:
            embedding = model.encode([text])[0].astype(np.float32)
            db.add_vector(doc_id, embedding, {"text": text})
        
        # Add relationships
        db.add_relationship("doc1", "doc2", "hierarchical", 0.8)
        db.add_relationship("doc2", "doc5", "semantic", 0.7)
        db.add_relationship("doc3", "doc4", "associative", 0.6)
        
        print(f"✅ Test data: {db.vector_count()} vectors, {db.relationship_count()} relationships")
        
    except ImportError:
        print("⚠️ sentence-transformers not available, using random embeddings")
        
        for doc_id, text in documents:
            embedding = np.random.rand(384).astype(np.float32)
            db.add_vector(doc_id, embedding, {"text": text})
        
        db.add_relationship("doc1", "doc2", "hierarchical", 0.8)
        db.add_relationship("doc2", "doc5", "semantic", 0.7)
    
    # Test different search scenarios
    query_embedding = np.random.rand(db.dimension()).astype(np.float32)
    
    # Test 1: Basic search
    try:
        results = db.search(query_embedding)
        print(f"✅ Basic search: {len(results)} results")
    except Exception as e:
        print(f"❌ Basic search failed: {e}")
    
    # Test 2: Search with parameters
    try:
        params = rudradb.SearchParams(
            top_k=3,
            include_relationships=True,
            max_hops=2
        )
        results = db.search(query_embedding, params)
        print(f"✅ Parameterized search: {len(results)} results")
        
        # Analyze results
        direct_results = [r for r in results if r.hop_count == 0]
        relationship_results = [r for r in results if r.hop_count > 0]
        
        print(f"   Direct results: {len(direct_results)}")
        print(f"   Relationship results: {len(relationship_results)}")
        
    except Exception as e:
        print(f"❌ Parameterized search failed: {e}")
    
    # Test 3: Search with invalid parameters
    try:
        invalid_params = rudradb.SearchParams(
            top_k=0,  # Invalid
            include_relationships=True
        )
        results = db.search(query_embedding, invalid_params)
        print(f"❌ Should have rejected invalid parameters")
    except Exception as e:
        print(f"✅ Correctly rejected invalid parameters: {type(e).__name__}")
    
    # Test 4: Empty database search
    empty_db = rudradb.RudraDB()
    try:
        results = empty_db.search(query_embedding)
        print(f"✅ Empty database search: {len(results)} results (expected 0)")
    except Exception as e:
        print(f"❌ Empty database search failed: {e}")
    
    print(f"\n💡 Search Troubleshooting Tips:")
    tips = [
        "Check that vectors exist before searching",
        "Verify query embedding has correct dimension",
        "Use reasonable top_k values (1-50 typical)",
        "Set similarity_threshold between 0.0-1.0",
        "Relationship_weight between 0.0-1.0",
        "Max_hops should be 1-2 for Opin"
    ]
    
    for tip in tips:
        print(f"   • {tip}")

debug_search_issues()
```

### Performance Issues

```python
def debug_performance_issues():
    """Debug and optimize performance problems"""
    
    print("⚡ Performance Troubleshooting")
    print("="*35)
    
    db = rudradb.RudraDB()
    
    # Test vector addition performance
    print("🔍 Testing vector addition performance...")
    
    start_time = time.time()
    test_vectors = 50
    
    for i in range(test_vectors):
        embedding = np.random.rand(384).astype(np.float32)
        db.add_vector(f"perf_test_{i}", embedding)
    
    add_time = time.time() - start_time
    print(f"   Added {test_vectors} vectors in {add_time:.3f}s ({test_vectors/add_time:.0f}/sec)")
    
    if add_time > 5:  # If slower than expected
        print("   ⚠️ Vector addition seems slow")
        print("   💡 Tips:")
        print("     • Use np.float32 for embeddings")
        print("     • Keep metadata small") 
        print("     • Consider batch operations")
    
    # Test search performance
    print("\n🔍 Testing search performance...")
    
    query = np.random.rand(384).astype(np.float32)
    search_iterations = 20
    
    start_time = time.time()
    for _ in range(search_iterations):
        results = db.search(query)
    
    search_time = time.time() - start_time
    print(f"   {search_iterations} searches in {search_time:.3f}s ({search_iterations/search_time:.0f}/sec)")
    
    if search_time > 2:
        print("   ⚠️ Search seems slow")
        print("   💡 Tips:")
        print("     • Reduce top_k if not needed")
        print("     • Use similarity_threshold to filter")
        print("     • Limit max_hops for relationship search")
    
    # Memory usage estimation
    stats = db.get_statistics()
    estimated_memory = (
        stats['vector_count'] * stats['dimension'] * 4 +  # vectors (4 bytes per float32)
        stats['relationship_count'] * 100  # relationships (estimated)
    ) / (1024 * 1024)  # Convert to MB
    
    print(f"\n💾 Estimated memory usage: {estimated_memory:.2f} MB")
    
    if estimated_memory > 100:
        print("   ⚠️ High memory usage")
        print("   💡 Tips:")
        print("     • Consider lower dimensional embeddings")
        print("     • Remove unused vectors/relationships")
        print("     • Monitor capacity usage")
    
    print(f"\n✅ Performance is optimized for Opin limits")
    print(f"   RudraDB-Opin handles 100 vectors efficiently")
    print(f"   Full RudraDB optimized for 100K+ vectors")

debug_performance_issues()
```

### Getting Help

```python
def get_help_resources():
    """Show available help resources"""
    
    print("🆘 RudraDB-Opin Help Resources")
    print("="*40)
    
    resources = [
        ("📚 Documentation", "https://docs.rudradb.com/opin"),
        ("🐛 Bug Reports", "https://github.com/rudradb/rudradb-opin/issues"),
        ("💬 Community", "https://discord.gg/rudradb"),
        ("📧 Support", "support@rudradb.com"),
        ("🚀 Upgrade Info", rudradb.UPGRADE_URL),
        ("📖 Tutorials", "https://rudradb.com/tutorials"),
        ("🎥 Videos", "https://youtube.com/@rudradb"),
        ("📝 Examples", "https://github.com/rudradb/examples")
    ]
    
    for resource, link in resources:
        print(f"   {resource}: {link}")
    
    print(f"\n🔍 Quick Diagnostics:")
    print(f"   Version: {rudradb.__version__}")
    print(f"   Edition: {rudradb.EDITION}")
    print(f"   Python: {sys.version}")
    print(f"   NumPy: {np.__version__}")
    
    # System info
    import platform
    print(f"   Platform: {platform.system()} {platform.release()}")
    
    print(f"\n💡 Before Getting Help:")
    checklist = [
        "Check this troubleshooting guide",
        "Try the examples in this documentation",
        "Test with minimal code example",
        "Check version compatibility",
        "Search existing issues on GitHub"
    ]
    
    for item in checklist:
        print(f"   ☐ {item}")

get_help_resources()
```

---

## 🎉 Conclusion

**Congratulations!** You now have a comprehensive understanding of RudraDB-Opin, the world's first free relationship-aware vector database. 

### 🎯 What You've Learned

- **Installation and Setup**: Get RudraDB-Opin running in minutes
- **Core Concepts**: Understand relationship-aware vector search
- **Complete API**: Master all vector and relationship operations  
- **Relationship Types**: Use all 5 types for rich modeling
- **Search Patterns**: From basic similarity to complex multi-hop discovery
- **ML Integration**: Work with OpenAI, HuggingFace, Sentence Transformers
- **Best Practices**: Build production-quality applications
- **Performance**: Optimize for the 100-vector, 500-relationship sweet spot
- **Capacity Management**: Handle limits gracefully with upgrade guidance
- **Troubleshooting**: Solve common issues quickly

### 🚀 Next Steps

1. **Start Building**: Use the tutorial examples to build your first relationship-aware application
2. **Explore Use Cases**: Try knowledge bases, RAG systems, recommendation engines
3. **Master Relationships**: Experiment with all 5 relationship types
4. **Hit the Limits**: Explore the full capacity to understand when to upgrade
5. **Share & Learn**: Join the community, contribute examples, help others

### 💝 RudraDB-Opin Value

RudraDB-Opin gives you **100% of the features** of relationship-aware vector search with **perfect learning capacity**:

- **No feature restrictions** - All algorithms included
- **Production code quality** - Same codebase as full RudraDB
- **Perfect tutorial size** - 100 vectors, 500 relationships
- **Zero friction** - `pip install rudradb-opin` and start immediately
- **Clear upgrade path** - Same API, just higher capacity

### 🌟 Ready for More?

When you've mastered relationship-aware search with RudraDB-Opin:

```bash
# Seamless upgrade to production scale
pip uninstall rudradb-opin
pip install rudradb

# Same code, 1000x more capacity!
```

**🧬 Welcome to the future of vector databases - where relationships matter!**

---

*RudraDB-Opin: Perfect for learning, hackathons, POCs, ready for production? go for upgrade. Experience relationship-aware vector search today!*