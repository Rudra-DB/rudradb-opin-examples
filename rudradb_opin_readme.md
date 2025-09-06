# 🧬 RudraDB-Opin - Free Relationship-Aware Vector Database

<div align="center">

![RudraDB-Opin Logo](https://img.shields.io/badge/RudraDB-Opin-blue?style=for-the-badge&logo=database&logoColor=white)
[![PyPI version](https://img.shields.io/pypi/v/rudradb-opin.svg?style=for-the-badge)](https://pypi.org/project/rudradb-opin/)
[![Python versions](https://img.shields.io/pypi/pyversions/rudradb-opin.svg?style=for-the-badge)](https://pypi.org/project/rudradb-opin/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)](LICENSE)

**🌟 The World's First Free Relationship-Aware Vector Database**  
*Perfect for Learning, Tutorials, and AI Development*

</div>

---

## 🎯 Why RudraDB-Opin?

Traditional vector databases only find similar vectors. **RudraDB-Opin** understands **relationships between your data**, enabling AI applications that discover connections, not just similarity.

### 🆓 **100% Free Version**
- **100 vectors** - Perfect tutorial and learning size
- **500 relationships** - Rich relationship modeling capability  
- **Complete feature set** - All relationship types and algorithms
- **No usage tracking** - Complete privacy and freedom
- **Production-quality code** - Same codebase as enterprise RudraDB

### 🧠 **Relationship-Aware Intelligence**
```python
# Traditional vector search: "Find similar documents"
results = db.search(query_embedding)  # Only similarity

# RudraDB-Opin: "Find related documents through connections"  
results = db.search(query_embedding, include_relationships=True)
# Discovers: Similar + Hierarchically related + Causally connected + More!
```

### 🚀 **Ready for Production?**
When you outgrow the 100-vector limit, upgrade seamlessly:
```bash
pip uninstall rudradb-opin
pip install rudradb  # Get 100,000+ vectors, same API!
```

---

## 📦 Installation

### Quick Install
```bash
pip install rudradb-opin
```

### From Source
```bash
git clone https://github.com/rudradb/rudradb-opin
cd rudradb-opin
pip install -r requirements.txt
maturin develop --release
```

### Verify Installation
```python
import rudradb
print(f"🎉 RudraDB-Opin {rudradb.__version__} ready!")
print(f"📊 Free tier: {rudradb.MAX_VECTORS} vectors, {rudradb.MAX_RELATIONSHIPS} relationships")
```

---

## 🚀 Quick Start

### 30-Second Demo
```python
import rudradb
import numpy as np

# Create database (auto-detects dimensions!)
db = rudradb.RudraDB()

# Add some AI/ML documents
papers = [
    ("gpt_paper", "Attention Is All You Need - Transformer architecture"),
    ("bert_paper", "BERT: Bidirectional Encoder Representations"),  
    ("resnet_paper", "Deep Residual Learning for Image Recognition"),
    ("gan_paper", "Generative Adversarial Networks")
]

embeddings = {}
for doc_id, content in papers:
    # Use your favorite embedding model (OpenAI, HuggingFace, etc.)
    embedding = np.random.rand(384).astype(np.float32)  # Replace with real embeddings
    embeddings[doc_id] = embedding
    db.add_vector(doc_id, embedding, {
        "title": content,
        "domain": "AI/ML",
        "type": "research_paper"
    })

# Add relationships that traditional vector DBs can't understand
db.add_relationship("bert_paper", "gpt_paper", "temporal", 0.9)        # BERT came before GPT
db.add_relationship("gpt_paper", "bert_paper", "hierarchical", 0.8)    # GPT builds on BERT ideas
db.add_relationship("resnet_paper", "gan_paper", "causal", 0.7)        # ResNet enables better GANs

# Search with relationship awareness
query = embeddings["bert_paper"]  # Find papers related to BERT
results = db.search(query, rudradb.SearchParams(
    top_k=5,
    include_relationships=True,  # 🔥 This is the magic!
    max_hops=2                   # Find indirectly connected papers
))

print("🔍 Traditional similarity search would only find content-similar papers.")
print("🧠 RudraDB-Opin discovers papers through relationships:")
for result in results:
    vector = db.get_vector(result.vector_id)
    connection = "Direct" if result.hop_count == 0 else f"{result.hop_count}-hop connection"
    print(f"   📄 {vector['metadata']['title'][:50]}... ({connection})")
```

---

## 🌟 Core Features

### 🔗 **Five Relationship Types**
```python
# 1. Semantic - Content similarity and meaning
db.add_relationship("python_tutorial", "programming_guide", "semantic", 0.9)

# 2. Hierarchical - Parent-child, category structures  
db.add_relationship("machine_learning", "supervised_learning", "hierarchical", 0.95)

# 3. Temporal - Time-based, sequential relationships
db.add_relationship("data_collection", "model_training", "temporal", 0.8)

# 4. Causal - Cause-effect, problem-solution
db.add_relationship("overfitting_problem", "regularization_solution", "causal", 0.9)

# 5. Associative - General associations, recommendations
db.add_relationship("tensorflow", "pytorch", "associative", 0.7)
```

### 🔄 **Multi-Hop Discovery**
```python
# Find documents through relationship chains
connected = db.get_connected_vectors("starting_document", max_hops=2)

for vector_data, hop_count in connected:
    print(f"📄 {vector_data['metadata']['title']} - {hop_count} hops away")
    
# Example discovery chain:
# Document A → (semantic) → Document B → (causal) → Document C
# "Find solutions to problems mentioned in documents similar to A"
```

### ⚡ **Advanced Search Parameters**
```python
params = rudradb.SearchParams(
    top_k=20,                           # Number of results
    include_relationships=True,         # Enable relationship search
    max_hops=2,                        # Maximum relationship traversal depth
    similarity_threshold=0.1,          # Minimum similarity score
    relationship_weight=0.3,           # How much relationships influence results
    relationship_types=["semantic", "causal"]  # Filter specific relationship types
)

results = db.search(query_embedding, params)
```

---

## 🎓 Perfect Learning Examples

### 📚 **Tutorial: Building a Learning RAG System**
```python
import rudradb
import numpy as np
from sentence_transformers import SentenceTransformer

# Perfect for learning RAG concepts!
class LearningRAG:
    def __init__(self):
        self.db = rudradb.RudraDB()  # Auto-detects embedding dimensions
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
    
    def add_learning_content(self, content_items):
        """Add educational content with relationships"""
        for item in content_items:
            embedding = self.encoder.encode(item['text'])
            self.db.add_vector(
                item['id'], 
                embedding.astype(np.float32),
                {
                    "text": item['text'],
                    "topic": item.get('topic', 'general'),
                    "difficulty": item.get('difficulty', 'beginner'),
                    "type": item.get('type', 'concept')
                }
            )
    
    def add_learning_relationships(self):
        """Create educational relationships"""
        # Prerequisites (hierarchical)
        self.db.add_relationship("basics", "intermediate", "hierarchical", 0.9)
        
        # Learning sequence (temporal) 
        self.db.add_relationship("theory", "practice", "temporal", 0.8)
        
        # Problem-solution (causal)
        self.db.add_relationship("problem", "solution", "causal", 0.95)
        
        # Related concepts (semantic)
        self.db.add_relationship("concept_a", "concept_b", "semantic", 0.7)
    
    def intelligent_search(self, query, learning_style="comprehensive"):
        """Search with educational intelligence"""
        query_embedding = self.encoder.encode(query).astype(np.float32)
        
        if learning_style == "comprehensive":
            # Include prerequisites and related concepts
            params = rudradb.SearchParams(
                top_k=10,
                include_relationships=True,
                max_hops=2,
                relationship_types=["hierarchical", "semantic"]
            )
        elif learning_style == "sequential":
            # Focus on learning sequence
            params = rudradb.SearchParams(
                top_k=8,
                include_relationships=True,
                max_hops=1,
                relationship_types=["temporal", "hierarchical"]
            )
        
        return self.db.search(query_embedding, params)

# Example usage
rag = LearningRAG()

# Add AI/ML learning content (perfect 100-vector size!)
learning_content = [
    {"id": "ml_basics", "text": "Machine learning fundamentals and core concepts", 
     "topic": "ML", "difficulty": "beginner"},
    {"id": "supervised_learning", "text": "Supervised learning algorithms and applications",
     "topic": "ML", "difficulty": "intermediate"},
    {"id": "neural_networks", "text": "Neural networks and deep learning principles",
     "topic": "ML", "difficulty": "advanced"},
    # ... up to 100 learning items
]

rag.add_learning_content(learning_content)
rag.add_learning_relationships()

# Intelligent learning search
results = rag.intelligent_search("neural networks", learning_style="comprehensive")
```

### 🤖 **OpenAI Integration**
```python
import openai
import rudradb
import numpy as np

class OpenAI_RudraDB_Tutorial:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.db = rudradb.RudraDB()  # Auto-detects OpenAI's 1536 dimensions
    
    def add_document_with_openai(self, doc_id, text, metadata=None):
        """Add document using OpenAI embeddings"""
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
        
        self.db.add_vector(doc_id, embedding, {
            "text": text,
            **(metadata or {})
        })
        
        return embedding
    
    def create_knowledge_base(self):
        """Create a sample knowledge base with relationships"""
        documents = [
            ("ai_intro", "Artificial Intelligence is transforming every industry"),
            ("ml_basics", "Machine Learning is a subset of AI focusing on learning from data"),
            ("dl_overview", "Deep Learning uses neural networks with multiple layers"),
            ("nlp_intro", "Natural Language Processing helps computers understand human language"),
            ("cv_basics", "Computer Vision enables machines to interpret visual information")
        ]
        
        # Add documents
        for doc_id, text in documents:
            self.add_document_with_openai(doc_id, text, {"domain": "AI/ML"})
        
        # Add educational relationships
        self.db.add_relationship("ai_intro", "ml_basics", "hierarchical", 0.9)     # AI contains ML
        self.db.add_relationship("ml_basics", "dl_overview", "hierarchical", 0.85)  # ML contains DL
        self.db.add_relationship("dl_overview", "nlp_intro", "semantic", 0.8)      # Related fields
        self.db.add_relationship("dl_overview", "cv_basics", "semantic", 0.8)      # Related fields
        
        print(f"✅ Created knowledge base with {self.db.vector_count()} documents")
        print(f"🔗 Added {self.db.relationship_count()} relationships")
    
    def intelligent_qa(self, question):
        """Answer questions using relationship-aware search"""
        # Get question embedding
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=question
        )
        query_embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
        
        # Search with relationships
        results = self.db.search(query_embedding, rudradb.SearchParams(
            top_k=5,
            include_relationships=True,
            max_hops=2
        ))
        
        # Build context from results
        context = []
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            context.append(vector['metadata']['text'])
        
        # Generate answer using GPT with relationship-aware context
        gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer questions using the provided context from a relationship-aware knowledge base."},
                {"role": "user", "content": f"Context: {' '.join(context)}\n\nQuestion: {question}"}
            ]
        )
        
        return gpt_response.choices[0].message.content

# Usage example
tutorial = OpenAI_RudraDB_Tutorial("your-api-key")
tutorial.create_knowledge_base()
answer = tutorial.intelligent_qa("How does deep learning relate to AI?")
print(f"🤖 Answer: {answer}")
```

### 🤗 **HuggingFace Integration**
```python
from transformers import AutoTokenizer, AutoModel
import torch
import rudradb
import numpy as np

class HuggingFace_RudraDB_Tutorial:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.db = rudradb.RudraDB()
    
    def encode_text(self, text):
        """Create embeddings using HuggingFace model"""
        inputs = self.tokenizer(text, return_tensors="pt", 
                               padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
        
        return embedding.astype(np.float32)
    
    def create_ml_course_database(self):
        """Create a mini ML course with relationships"""
        lessons = [
            ("lesson_1", "Introduction to Machine Learning fundamentals"),
            ("lesson_2", "Supervised learning algorithms and techniques"),
            ("lesson_3", "Unsupervised learning and clustering methods"),
            ("lesson_4", "Neural networks and backpropagation"),
            ("lesson_5", "Deep learning architectures and applications"),
            ("exercise_1", "Practice: Linear regression implementation"),
            ("exercise_2", "Practice: Classification with decision trees"),
            ("project_1", "Final project: Build a recommendation system")
        ]
        
        # Add all lessons
        for lesson_id, content in lessons:
            embedding = self.encode_text(content)
            lesson_type = "theory" if lesson_id.startswith("lesson") else \
                         "practice" if lesson_id.startswith("exercise") else "project"
                         
            self.db.add_vector(lesson_id, embedding, {
                "content": content,
                "type": lesson_type,
                "difficulty": self._get_difficulty(lesson_id)
            })
        
        # Add learning relationships
        # Sequential learning path (temporal)
        lesson_sequence = ["lesson_1", "lesson_2", "lesson_3", "lesson_4", "lesson_5"]
        for i in range(len(lesson_sequence) - 1):
            self.db.add_relationship(
                lesson_sequence[i], lesson_sequence[i+1], 
                "temporal", 0.9
            )
        
        # Theory to practice connections (causal)
        theory_practice = [
            ("lesson_2", "exercise_1"),  # Supervised learning → Linear regression
            ("lesson_2", "exercise_2"),  # Supervised learning → Classification
            ("lesson_5", "project_1")    # Deep learning → Recommendation system
        ]
        for theory, practice in theory_practice:
            self.db.add_relationship(theory, practice, "causal", 0.85)
        
        print(f"📚 Created ML course with {self.db.vector_count()} items")
    
    def _get_difficulty(self, lesson_id):
        if "1" in lesson_id or "2" in lesson_id:
            return "beginner"
        elif "3" in lesson_id or "4" in lesson_id:
            return "intermediate"
        else:
            return "advanced"
    
    def find_learning_path(self, topic):
        """Find optimal learning path for a topic"""
        query_embedding = self.encode_text(topic)
        
        # Search for relevant content with learning relationships
        results = self.db.search(query_embedding, rudradb.SearchParams(
            top_k=8,
            include_relationships=True,
            max_hops=2,
            relationship_types=["temporal", "causal"]  # Focus on learning sequence
        ))
        
        learning_path = []
        for result in results:
            vector = self.db.get_vector(result.vector_id)
            learning_path.append({
                "content": vector['metadata']['content'],
                "type": vector['metadata']['type'],
                "relevance_score": result.combined_score,
                "connection": "direct" if result.hop_count == 0 else f"{result.hop_count}-hop"
            })
        
        return learning_path

# Example usage
hf_tutorial = HuggingFace_RudraDB_Tutorial()
hf_tutorial.create_ml_course_database()

# Find learning path for neural networks
path = hf_tutorial.find_learning_path("neural networks and deep learning")
print("🎓 Recommended Learning Path:")
for step in path:
    print(f"   {step['type'].upper()}: {step['content']} (relevance: {step['relevance_score']:.2f})")
```

---

## 📖 Complete API Reference

### 🏗️ **Core Classes**

#### `RudraDB`
```python
# Database creation
db = rudradb.RudraDB()                    # Auto-dimension detection
db = rudradb.RudraDB(dimension=384)       # Fixed dimension (optional)

# Properties
db.dimension()           # Current embedding dimension
db.vector_count()        # Number of vectors (max 100 in Opin)
db.relationship_count()  # Number of relationships (max 500 in Opin)
db.is_empty()           # Check if database is empty
```

#### `SearchParams`
```python
params = rudradb.SearchParams(
    top_k=10,                          # Number of results to return
    include_relationships=True,         # Enable relationship-aware search
    max_hops=2,                        # Maximum relationship traversal hops
    similarity_threshold=0.0,          # Minimum similarity score (0.0-1.0)
    relationship_weight=0.3,           # Relationship influence (0.0-1.0)
    relationship_types=["semantic"]    # Filter specific relationship types
)
```

### 📊 **Vector Operations**

#### Add Vectors
```python
# Basic vector addition
db.add_vector(vector_id, embedding_array, metadata_dict)

# Example
embedding = np.random.rand(384).astype(np.float32)
db.add_vector("doc_1", embedding, {
    "title": "Introduction to AI",
    "author": "Dr. Smith",
    "tags": ["ai", "introduction"],
    "timestamp": "2024-01-01"
})

# Simplified addition (convenience method)
db.add_vector_simple("doc_2", embedding.tolist())
```

#### Retrieve Vectors
```python
# Get vector with metadata
vector_data = db.get_vector("doc_1")
print(vector_data['embedding'])   # numpy array
print(vector_data['metadata'])    # dictionary

# Check existence
exists = db.vector_exists("doc_1")  # Returns boolean

# List all vector IDs
all_ids = db.list_vectors()  # Returns list of strings
```

#### Update and Remove
```python
# Update metadata only
db.update_vector_metadata("doc_1", {"updated": True, "version": 2})

# Remove vector (also removes its relationships)
db.remove_vector("doc_1")
```

### 🔗 **Relationship Operations**

#### Add Relationships
```python
# Full relationship with metadata
db.add_relationship(
    source_id="parent_doc",
    target_id="child_doc", 
    relationship_type="hierarchical",
    strength=0.9,
    metadata={"created_by": "system", "confidence": 0.95}
)

# Simple relationship
db.add_relationship("doc_a", "doc_b", "semantic", 0.8)
```

#### Query Relationships
```python
# Get all relationships for a vector
relationships = db.get_relationships("doc_1")

# Filter by relationship type  
semantic_rels = db.get_relationships("doc_1", rel_type="semantic")

# Check if relationship exists
exists = db.has_relationship("doc_a", "doc_b")

# Get connected vectors with traversal
connected = db.get_connected_vectors("starting_doc", max_hops=2)
for vector_data, hop_count in connected:
    print(f"Connected: {vector_data['id']} ({hop_count} hops)")
```

#### Remove Relationships
```python
# Remove specific relationship
db.remove_relationship("doc_a", "doc_b")

# Remove all relationships for a vector
db.remove_all_relationships("doc_1")
```

### 🔍 **Search Operations**

#### Basic Search
```python
# Simple similarity search
results = db.search(query_embedding)

# With custom parameters
results = db.search(query_embedding, rudradb.SearchParams(
    top_k=20,
    similarity_threshold=0.5
))
```

#### Relationship-Aware Search
```python
# Enable relationship discovery
params = rudradb.SearchParams(
    top_k=15,
    include_relationships=True,    # 🔥 Enable relationship search
    max_hops=2,                   # Traverse up to 2 relationship hops
    relationship_weight=0.3       # Relationship influence on scoring
)

results = db.search(query_embedding, params)

# Process results
for result in results:
    print(f"Vector: {result.vector_id}")
    print(f"Similarity: {result.similarity_score:.3f}")
    print(f"Combined Score: {result.combined_score:.3f}")
    print(f"Hops: {result.hop_count}")
    print(f"Relationship Path: {result.relationship_path}")
    
    # Get full vector data
    vector = db.get_vector(result.vector_id)
    print(f"Content: {vector['metadata']}")
```

### 📊 **Database Management**

#### Statistics and Monitoring
```python
# Get comprehensive statistics
stats = db.get_statistics()

print(f"Vectors: {stats['vector_count']}/{stats['max_vectors']}")
print(f"Relationships: {stats['relationship_count']}/{stats['max_relationships']}")
print(f"Dimension: {stats['dimension']}")
print(f"Memory usage: {stats['memory_usage_mb']:.1f} MB")

# Capacity usage (Opin-specific)
capacity = stats['capacity_usage']
print(f"Vector usage: {capacity['vector_usage_percent']:.1f}%")
print(f"Remaining vectors: {capacity['vector_capacity_remaining']}")
print(f"Relationship usage: {capacity['relationship_usage_percent']:.1f}%")
```

#### Data Export/Import
```python
# Export database to JSON
export_data = db.export_data()
with open("my_database.json", "w") as f:
    json.dump(export_data, f)

# Import from JSON
with open("my_database.json", "r") as f:
    import_data = json.load(f)
db.import_data(import_data)
```

#### Database Maintenance
```python
# Clear all data
db.clear()

# Optimize database (rebuild indices)
db.optimize()

# Check database integrity
integrity_ok = db.verify_integrity()
print(f"Database integrity: {'✅ OK' if integrity_ok else '❌ Issues found'}")
```

### 🎯 **Constants and Auto-Features (Opin-Specific)**
```python
# Opin limits and info
print(f"Max vectors: {rudradb.MAX_VECTORS}")              # 100
print(f"Max relationships: {rudradb.MAX_RELATIONSHIPS}")  # 500  
print(f"Max hops: {rudradb.MAX_HOPS}")                   # 2
print(f"Edition: {rudradb.EDITION}")                     # "opin-free"
print(f"Is free version: {rudradb.IS_FREE_VERSION}")     # True
print(f"Upgrade info: {rudradb.UPGRADE_MESSAGE}")        # Upgrade guidance

# 🤖 Auto-feature flags
print(f"Auto-dimension detection: {rudradb.AUTO_DIMENSION_DETECTION}")  # True
print(f"Auto-relationships: {rudradb.AUTO_RELATIONSHIPS}")              # True
print(f"Auto-optimization: {rudradb.AUTO_OPTIMIZATION}")                # True

# 🎯 Check auto-detection status
print(f"Current dimension: {db.dimension()}")            # None (until first vector)
print(f"Is dimension locked: {db.is_dimension_locked()}") # False (until first vector)
```

---

## 🏗️ Building from Source

### Prerequisites
- **Python 3.8+**
- **Rust 1.70+**  
- **maturin** (`pip install maturin`)

### Build Process
```bash
# Clone repository
git clone https://github.com/rudradb/rudradb-opin
cd rudradb-opin

# Windows - Automated build
setup_and_test_opin.bat

# Linux/macOS - Manual build
cd python
maturin build --release --out ../target/wheels

# Install the wheel
pip install ../target/wheels/rudradb_opin-*.whl --force-reinstall

# Test installation  
python test_opin_limits.py
```

### Development Setup
```bash
# Development install (editable)
cd python
maturin develop

# Run tests
python -m pytest tests/

# Build documentation
cd ../docs
make html
```

---

## 🧪 Testing & Validation

### Automated Tests
```bash
# Test capacity limits
python test_opin_limits.py

# Run comprehensive demo
python demo_opin.py

# Unit tests
cd rudradb
cargo test

# Python tests
cd python
python -m pytest
```

### Manual Validation
```python
import rudradb
import numpy as np

# Test 1: Auto-dimension detection
db = rudradb.RudraDB()
assert db.dimension() is None

embedding = np.random.rand(256).astype(np.float32)
db.add_vector("test", embedding)
assert db.dimension() == 256

# Test 2: Capacity limits
try:
    for i in range(101):  # Try to exceed 100 vectors
        db.add_vector(f"vec_{i}", np.random.rand(256).astype(np.float32))
    assert False, "Should have hit vector limit"
except Exception as e:
    assert "RudraDB-Opin Vector Limit Reached" in str(e)
    assert "upgrade" in str(e).lower()

# Test 3: All relationship types work
db2 = rudradb.RudraDB()
for i in range(5):
    db2.add_vector(f"v{i}", np.random.rand(256).astype(np.float32))

relationship_types = ["semantic", "hierarchical", "temporal", "causal", "associative"]
for i, rel_type in enumerate(relationship_types):
    db2.add_relationship("v0", f"v{i+1}", rel_type, 0.8)

assert db2.relationship_count() == 5
print("✅ All tests passed!")
```

---

## 🎓 Learning Resources

### 📚 **Tutorials & Guides**
1. **[5-Minute Quickstart](docs/quickstart.md)** - Get running immediately
2. **[Relationship Types Guide](docs/relationships.md)** - Understand all 5 relationship types  
3. **[Multi-Hop Discovery](docs/multi-hop.md)** - Master relationship traversal
4. **[ML Framework Integration](docs/integrations.md)** - OpenAI, HuggingFace, Sentence Transformers
5. **[Production Upgrade Guide](docs/upgrade.md)** - Scaling to full RudraDB

### 🎯 **Use Case Examples**
- **[Educational RAG System](examples/educational_rag.py)** - Learning-focused retrieval
- **[Research Paper Discovery](examples/paper_discovery.py)** - Academic research relationships
- **[E-commerce Recommendations](examples/ecommerce_rec.py)** - Product relationship modeling  
- **[Content Management](examples/content_cms.py)** - Document hierarchy and connections
- **[Multi-language Search](examples/multilang_search.py)** - Cross-language relationships

### 🏆 **Best Practices**
```python
# 1. Design relationships thoughtfully
# Don't just connect everything - model real semantic relationships
db.add_relationship("concept", "example", "hierarchical", 0.9)      # ✅ Clear parent-child
db.add_relationship("problem", "solution", "causal", 0.85)          # ✅ Clear cause-effect
db.add_relationship("random_doc", "other_doc", "associative", 0.1)  # ❌ Weak, noisy connection

# 2. Use appropriate relationship strengths
# High strength (0.8-1.0): Strong, definitive connections
# Medium strength (0.5-0.8): Moderate, useful connections  
# Low strength (0.1-0.5): Weak connections, use sparingly

# 3. Monitor your capacity usage
stats = db.get_statistics()
if stats['capacity_usage']['vector_usage_percent'] > 80:
    print("⚠️  Approaching vector limit - consider upgrade or data curation")

# 4. Use multi-hop search strategically  
# max_hops=1: Direct relationships only (faster)
# max_hops=2: Good balance of discovery vs performance
# max_hops=3+: Only for full RudraDB, can be expensive
```

---

## 🚀 Production Upgrade Path

### When to Upgrade
- ✅ **Need more than 100 vectors**
- ✅ **Need more than 500 relationships**  
- ✅ **Ready for production deployment**
- ✅ **Need enterprise features and support**

### How to Upgrade
```bash
# Step 1: Export your Opin data
python -c "
import rudradb, json
db = rudradb.RudraDB()
# ... load your database ...
data = db.export_data()
with open('my_data.json', 'w') as f:
    json.dump(data, f)
"

# Step 2: Upgrade package
pip uninstall rudradb-opin
pip install rudradb

# Step 3: Import your data  
python -c "
import rudradb, json
db = rudradb.RudraDB()
with open('my_data.json', 'r') as f:
    data = json.load(f)
db.import_data(data)
print(f'✅ Upgraded! Now have {db.vector_count()} vectors')
"
```

### What You Get
| Feature | RudraDB-Opin (Free) | RudraDB (Full) |
|---------|-------------------|-----------------|
| **Vectors** | 100 | 100,000+ |  
| **Relationships** | 500 | 250,000+ |
| **Multi-hop traversal** | 2 hops | Unlimited |
| **Relationship types** | All 5 types | All 5 types + custom |
| **Performance** | Great | Enterprise-optimized |
| **Support** | Community | Enterprise support |
| **Advanced features** | Core features | Analytics, monitoring, custom integrations |
| **Commercial use** | Limited | Full commercial license |

### 📞 **Upgrade Support**
- **Email**: upgrade@rudradb.com
- **Website**: [rudradb.com/upgrade](https://rudradb.com/upgrade)  
- **Enterprise**: Custom solutions available

---

## 🌟 Why Choose RudraDB-Opin?

### 🆚 **vs Traditional Vector Databases**

| Capability | Traditional VectorDBs | RudraDB-Opin |
|------------|----------------------|--------------|
| **Similarity Search** | ✅ Yes | ✅ Yes |
| **Relationship Modeling** | ❌ No | ✅ 5 types |
| **Multi-hop Discovery** | ❌ No | ✅ 2 hops |
| **Learning-focused** | ❌ Complex enterprise tools | ✅ Perfect for education |
| **Free tier** | ❌ Limited trials | ✅ 100% free forever |
| **Production upgrade** | ❌ Vendor lock-in | ✅ Seamless upgrade path |

### 🏆 **Unique Advantages**
1. **Educational Excellence** - Perfect 100-vector size for tutorials
2. **Relationship Intelligence** - Discover connections, not just similarity  
3. **Zero Vendor Lock-in** - Open source with clear upgrade path
4. **ML Framework Agnostic** - Works with any embedding model
5. **Production Quality** - Same codebase as enterprise RudraDB

### 🎯 **Perfect For**
- 🎓 **Students & Researchers** - Learn vector databases with relationships
- 👨‍💻 **Developers** - Prototype relationship-aware applications  
- 🏫 **Educators** - Teach advanced vector search concepts
- 📝 **Content Creators** - Build tutorials and demonstrations
- 🚀 **Startups** - Validate ideas before scaling to production

---

## 📄 License

MIT License - Free for learning, tutorials, and non-commercial use.

```
Copyright (c) 2025 RudraDB

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

This is the FREE version (RudraDB-Opin) with the following limits:
- Maximum 100 vectors
- Maximum 500 relationships  
- 2-hop relationship traversal

For production use without limits, upgrade to full RudraDB.
```

---

## 🤝 Contributing

We welcome contributions to make RudraDB-Opin even better for learning!

### 🎯 **Priority Areas**
- **Educational examples** - More tutorial content
- **ML framework integrations** - Additional model support
- **Documentation** - Clearer guides and examples  
- **Performance optimizations** - Faster operations within Opin limits
- **Testing** - More comprehensive test coverage

### 📝 **Contribution Process**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-tutorial`
3. Add your improvement with tests
4. Submit pull request with clear description

### 🧪 **Example Contribution**
```python
def test_new_learning_feature():
    """Test educational-focused feature"""
    db = rudradb.RudraDB()
    
    # Your amazing educational example here
    # Focus on learning value and clear documentation
    
    assert db.vector_count() <= rudradb.MAX_VECTORS
    print("✅ New learning feature works perfectly!")
```

---

## 💬 Community & Support

### 🆓 **Free Community Support**
- **GitHub Issues** - Bug reports and feature requests
- **Discussions** - Community Q&A and examples  
- **Documentation** - Comprehensive guides and tutorials

### 📞 **Enterprise & Upgrade**
- **Email**: upgrade@rudradb.com
- **Website**: [rudradb.com](https://rudradb.com)
- **Enterprise**: Custom solutions and dedicated support

### 🌟 **Stay Connected**
- **Star the repo** - Help others discover relationship-aware search
- **Share tutorials** - Create content using RudraDB-Opin
- **Join discussions** - Help build the community

---

## 🎉 Get Started Today!

### Quick Start Checklist
- [ ] `pip install rudradb-opin`
- [ ] Run first example: `python -c "import rudradb; print('🎉 Ready with Auto-Features!')"`
- [ ] Verify auto-features: `python -c "import rudradb; print(f'🤖 Auto-Dimension Detection: {rudradb.AUTO_DIMENSION_DETECTION}'); print(f'🧠 Auto-Relationship Detection: {rudradb.AUTO_RELATIONSHIP_DETECTION}')"`
- [ ] Try the 30-second auto-enhanced demo above
- [ ] Test auto-dimension detection with your favorite ML model  
- [ ] Build your first auto-relationship-aware application  
- [ ] Experience auto-optimized search in action
- [ ] Share your auto-intelligent experience and learn from others
- [ ] Upgrade when you're ready for advanced auto-features at production scale

### 🚀 **Ready to Build the Future?**

**RudraDB-Opin** isn't just a database - it's your gateway to **relationship-aware AI**. While others are still thinking in terms of similarity, you'll be building applications that understand **connections, context, and meaning**.

**Start your journey into relationship-aware vector search today!**

---

<div align="center">

**🧬 RudraDB-Opin: Where Relationship-Aware AI Begins**

[![Install now](https://img.shields.io/badge/pip%20install-rudradb--opin-blue?style=for-the-badge&logo=python)](https://pypi.org/project/rudradb-opin/)
[![Star on GitHub](https://img.shields.io/github/stars/rudradb/rudradb-opin?style=for-the-badge&logo=github)](https://github.com/Rudra-DB/rudradb-opin-examples.git)
[![Join Community](https://img.shields.io/badge/Join-Community-green?style=for-the-badge&logo=discord)](https://discord.gg/ztzDE2tf)

**Made with ❤️ for developers learning the future of AI**

</div>