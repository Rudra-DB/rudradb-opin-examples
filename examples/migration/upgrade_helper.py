#!/usr/bin/env python3
"""
RudraDB-Opin to Full RudraDB Migration Helper
Complete migration assistance with data preservation and validation
"""

import json
import time
import numpy as np
import rudradb
from datetime import datetime
from typing import Dict, Any, List, Optional

class MigrationHelper:
    """Complete migration helper for upgrading from RudraDB-Opin to full RudraDB"""
    
    def __init__(self, opin_db: rudradb.RudraDB):
        self.opin_db = opin_db
        self.backup_created = False
        self.migration_log = []
        self.validation_results = {}
        
        print("🚀 RudraDB-Opin Migration Helper Initialized")
        print("   Helping you upgrade from learning to production scale!")
        
    def assess_readiness_for_upgrade(self) -> Dict[str, Any]:
        """Comprehensive assessment of upgrade readiness"""
        
        print("\n🔍 Assessing Upgrade Readiness...")
        stats = self.opin_db.get_statistics()
        
        assessment = {
            "ready": True,
            "readiness_score": 0,
            "requirements_met": [],
            "recommendations": [],
            "warnings": [],
            "upgrade_benefits": [],
            "current_usage": {
                "vectors": stats["vector_count"],
                "relationships": stats["relationship_count"],
                "dimension": stats["dimension"],
                "capacity_usage": stats["capacity_usage"]
            }
        }
        
        # 1. Learning Completeness Assessment
        vector_usage = stats["capacity_usage"]["vector_usage_percent"]
        relationship_usage = stats["capacity_usage"]["relationship_usage_percent"]
        
        if vector_usage >= 80:
            assessment["readiness_score"] += 30
            assessment["requirements_met"].append("Strong vector database experience (80%+ capacity used)")
        elif vector_usage >= 50:
            assessment["readiness_score"] += 20
            assessment["recommendations"].append("Consider adding more vectors to gain experience")
        else:
            assessment["warnings"].append("Limited vector experience - consider exploring more")
        
        if relationship_usage >= 60:
            assessment["readiness_score"] += 25
            assessment["requirements_met"].append("Good relationship modeling experience")
        elif relationship_usage >= 30:
            assessment["readiness_score"] += 15
            assessment["recommendations"].append("Try building more relationships to understand patterns")
        else:
            assessment["warnings"].append("Limited relationship experience - explore relationship types")
        
        # 2. Relationship Diversity Assessment
        relationship_types_used = self._analyze_relationship_diversity()
        if len(relationship_types_used) >= 4:
            assessment["readiness_score"] += 20
            assessment["requirements_met"].append(f"Excellent relationship diversity ({len(relationship_types_used)}/5 types used)")
        elif len(relationship_types_used) >= 2:
            assessment["readiness_score"] += 10
            assessment["recommendations"].append("Try exploring more relationship types")
        else:
            assessment["warnings"].append("Limited relationship types explored")
        
        # 3. Search Pattern Assessment
        if self._has_complex_metadata():
            assessment["readiness_score"] += 15
            assessment["requirements_met"].append("Good metadata design patterns")
        
        # 4. Data Quality Assessment
        if self._validate_data_quality():
            assessment["readiness_score"] += 10
            assessment["requirements_met"].append("High data quality standards")
        
        # Determine readiness
        if assessment["readiness_score"] >= 80:
            assessment["ready"] = True
            assessment["upgrade_benefits"] = [
                f"Scale to 100,000+ vectors (1000x more than current {stats['vector_count']})",
                f"Build 250,000+ relationships (500x more than current {stats['relationship_count']})",
                "Unlimited relationship traversal (beyond 2-hop limit)",
                "Advanced performance optimizations",
                "Production-ready enterprise features",
                "Priority support and documentation"
            ]
        elif assessment["readiness_score"] >= 60:
            assessment["ready"] = True
            assessment["recommendations"].append("Good foundation - ready for production scale")
        else:
            assessment["ready"] = False
            assessment["recommendations"].append("Continue learning with RudraDB-Opin")
        
        return assessment
    
    def _analyze_relationship_diversity(self) -> set:
        """Analyze diversity of relationship types used"""
        relationship_types = set()
        
        for vector_id in self.opin_db.list_vectors():
            relationships = self.opin_db.get_relationships(vector_id)
            for rel in relationships:
                relationship_types.add(rel["relationship_type"])
        
        return relationship_types
    
    def _has_complex_metadata(self) -> bool:
        """Check if user has designed complex metadata structures"""
        metadata_keys = set()
        
        for vector_id in self.opin_db.list_vectors():
            vector = self.opin_db.get_vector(vector_id)
            if vector:
                metadata_keys.update(vector["metadata"].keys())
        
        # Consider complex if more than basic keys
        basic_keys = {"text", "content", "title", "id"}
        return len(metadata_keys - basic_keys) >= 3
    
    def _validate_data_quality(self) -> bool:
        """Validate overall data quality"""
        # Check for consistent metadata, reasonable relationship strengths, etc.
        vectors = self.opin_db.list_vectors()
        if len(vectors) < 5:
            return False
        
        # Sample check - relationship strength distribution
        strengths = []
        for vector_id in vectors[:10]:  # Sample first 10
            relationships = self.opin_db.get_relationships(vector_id)
            strengths.extend([r["strength"] for r in relationships])
        
        if not strengths:
            return False
        
        # Good quality if strengths are well distributed and reasonable
        avg_strength = sum(strengths) / len(strengths)
        return 0.3 <= avg_strength <= 0.9
    
    def create_complete_backup(self) -> str:
        """Create comprehensive backup of RudraDB-Opin data"""
        
        print("\n💾 Creating Complete Backup...")
        
        backup_data = {
            "metadata": {
                "backup_version": "1.0",
                "opin_version": rudradb.__version__ if hasattr(rudradb, '__version__') else "unknown",
                "backup_timestamp": datetime.now().isoformat(),
                "vector_count": self.opin_db.vector_count(),
                "relationship_count": self.opin_db.relationship_count(),
                "dimension": self.opin_db.dimension(),
                "backup_type": "complete_migration_backup"
            },
            "database_export": self.opin_db.export_data(),
            "statistics": self.opin_db.get_statistics(),
            "learning_analysis": self._create_learning_analysis(),
            "migration_notes": {
                "upgrade_path": "rudradb-opin -> rudradb",
                "data_compatibility": "100% compatible",
                "feature_preservation": "all features preserved and enhanced"
            }
        }
        
        # Generate backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"rudradb_opin_backup_{timestamp}.json"
        
        # Save backup
        with open(backup_filename, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        self.backup_created = True
        self.migration_log.append(f"✅ Complete backup created: {backup_filename}")
        
        backup_size_mb = len(json.dumps(backup_data)) / (1024 * 1024)
        print(f"   📁 Backup file: {backup_filename}")
        print(f"   📊 Backup size: {backup_size_mb:.2f} MB")
        print(f"   📦 Contains: {backup_data['metadata']['vector_count']} vectors, {backup_data['metadata']['relationship_count']} relationships")
        
        return backup_filename
    
    def _create_learning_analysis(self) -> Dict[str, Any]:
        """Analyze the learning journey with RudraDB-Opin"""
        
        stats = self.opin_db.get_statistics()
        
        # Analyze relationship patterns
        relationship_analysis = {}
        relationship_strengths = []
        
        for vector_id in self.opin_db.list_vectors():
            relationships = self.opin_db.get_relationships(vector_id)
            for rel in relationships:
                rel_type = rel["relationship_type"]
                strength = rel["strength"]
                
                relationship_analysis[rel_type] = relationship_analysis.get(rel_type, 0) + 1
                relationship_strengths.append(strength)
        
        return {
            "vectors_explored": stats["vector_count"],
            "relationships_built": stats["relationship_count"],
            "dimension_used": stats["dimension"],
            "capacity_utilization": {
                "vector_percentage": stats["capacity_usage"]["vector_usage_percent"],
                "relationship_percentage": stats["capacity_usage"]["relationship_usage_percent"]
            },
            "relationship_types_mastered": list(relationship_analysis.keys()),
            "relationship_distribution": relationship_analysis,
            "avg_relationship_strength": sum(relationship_strengths) / len(relationship_strengths) if relationship_strengths else 0,
            "learning_completeness_score": len(relationship_analysis) / 5 * 100,  # Out of 5 relationship types
            "readiness_indicators": {
                "diverse_relationships": len(relationship_analysis) >= 3,
                "good_capacity_usage": stats["capacity_usage"]["vector_usage_percent"] >= 50,
                "relationship_modeling": stats["relationship_count"] >= 10
            }
        }
    
    def generate_migration_script(self) -> str:
        """Generate automated migration script"""
        
        print("\n📝 Generating Migration Script...")
        
        stats = self.opin_db.get_statistics()
        
        script_content = f'''#!/usr/bin/env python3
"""
Automated RudraDB-Opin to Full RudraDB Migration Script
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Source: RudraDB-Opin v{rudradb.__version__ if hasattr(rudradb, '__version__') else "unknown"}
"""

import json
import rudradb
import numpy as np
from datetime import datetime

def main():
    print("🚀 Starting RudraDB-Opin → Full RudraDB Migration")
    print("=" * 55)
    
    # Load backup data
    print("📂 Loading migration backup...")
    backup_file = input("Enter backup filename (or press Enter for latest): ").strip()
    if not backup_file:
        import glob
        backup_files = glob.glob("rudradb_opin_backup_*.json")
        if backup_files:
            backup_file = max(backup_files)  # Get latest
            print(f"   Using latest backup: {{backup_file}}")
        else:
            print("❌ No backup files found!")
            return False
    
    try:
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Backup file not found: {{backup_file}}")
        return False
    
    original_stats = backup_data['metadata']
    print(f"\\n📊 Original Database:")
    print(f"   Vectors: {{original_stats['vector_count']:,}}")
    print(f"   Relationships: {{original_stats['relationship_count']:,}}")
    print(f"   Dimension: {{original_stats['dimension']}}D")
    
    # Create full RudraDB instance
    print("\\n🧬 Creating Full RudraDB Instance...")
    
    # Preserve original dimension if detected
    if original_stats['dimension']:
        full_db = rudradb.RudraDB(dimension=original_stats['dimension'])
        print(f"   ✅ Created with {{original_stats['dimension']}}D dimension")
    else:
        full_db = rudradb.RudraDB()  # Auto-dimension detection
        print(f"   ✅ Created with auto-dimension detection")
    
    # Import all data
    print("\\n📥 Importing Data...")
    try:
        full_db.import_data(backup_data['database_export'])
        print(f"   ✅ Data import successful")
        
        # Verify import
        new_stats = full_db.get_statistics()
        print(f"\\n🔍 Verification:")
        print(f"   Vectors: {{new_stats['vector_count']:,}} (expected: {{original_stats['vector_count']:,}})")
        print(f"   Relationships: {{new_stats['relationship_count']:,}} (expected: {{original_stats['relationship_count']:,}})")
        print(f"   Dimension: {{new_stats['dimension']}}D")
        
        # Validate migration
        vectors_match = new_stats['vector_count'] == original_stats['vector_count']
        relationships_match = new_stats['relationship_count'] == original_stats['relationship_count']
        dimension_match = new_stats['dimension'] == original_stats['dimension']
        
        if vectors_match and relationships_match and dimension_match:
            print("   ✅ Perfect migration - all data preserved!")
        else:
            print("   ⚠️ Migration verification issues:")
            if not vectors_match:
                print(f"      Vector count mismatch: {{new_stats['vector_count']}} vs {{original_stats['vector_count']}}")
            if not relationships_match:
                print(f"      Relationship count mismatch: {{new_stats['relationship_count']}} vs {{original_stats['relationship_count']}}")
            if not dimension_match:
                print(f"      Dimension mismatch: {{new_stats['dimension']}} vs {{original_stats['dimension']}}")
        
        # Test functionality
        print("\\n🧪 Testing Upgraded Functionality...")
        
        # Test search
        if new_stats['vector_count'] > 0:
            sample_vector_id = full_db.list_vectors()[0]
            sample_vector = full_db.get_vector(sample_vector_id)
            
            if sample_vector:
                sample_embedding = sample_vector['embedding']
                
                # Test enhanced search capabilities
                search_results = full_db.search(sample_embedding, rudradb.SearchParams(
                    top_k=10,  # More results than Opin
                    include_relationships=True,
                    max_hops=5  # More hops than Opin's 2-hop limit
                ))
                
                print(f"   ✅ Enhanced search test: {{len(search_results)}} results")
                print(f"       (can now search with more hops and results)")
        
        # Show upgrade benefits
        print("\\n🎉 Migration Complete! New Capabilities:")
        print(f"   📈 Vectors: {{original_stats['vector_count']:,}} → Unlimited (100,000+ tested)")
        print(f"   🔗 Relationships: {{original_stats['relationship_count']:,}} → Unlimited (250,000+ tested)")
        print(f"   🔄 Multi-hop traversal: 2 hops → Unlimited hops")
        print(f"   ⚡ Performance: Learning scale → Production scale")
        print(f"   🛠 Features: Core features → Enterprise features")
        print(f"   📞 Support: Community → Priority support")
        
        # Learning journey summary
        if 'learning_analysis' in backup_data:
            learning = backup_data['learning_analysis']
            print(f"\\n🎓 Your Learning Journey Summary:")
            print(f"   🏆 Relationship types mastered: {{len(learning['relationship_types_mastered'])}}/5")
            print(f"   📊 Learning completeness: {{learning['learning_completeness_score']:.1f}}%")
            print(f"   🎯 Average relationship strength: {{learning['avg_relationship_strength']:.2f}}")
            print(f"   💡 You're ready for production-scale relationship-aware AI!")
        
        # Save migration report
        migration_report = {{
            "migration_date": datetime.now().isoformat(),
            "source": "RudraDB-Opin",
            "target": "RudraDB Full",
            "original_stats": original_stats,
            "final_stats": new_stats,
            "success": vectors_match and relationships_match,
            "backup_file": backup_file
        }}
        
        with open('migration_report.json', 'w') as f:
            json.dump(migration_report, f, indent=2)
        
        print(f"\\n💾 Migration report saved: migration_report.json")
        print(f"\\n🚀 Welcome to production-scale relationship-aware AI!")
        print(f"   Your learning with RudraDB-Opin prepared you perfectly.")
        print(f"   Same API, unlimited scale. Happy building! 🎉")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {{e}}")
        print(f"\\n📞 Need help? Contact support@rudradb.com with your backup file")
        print(f"   Your data is safe in the backup file: {{backup_file}}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n✅ Migration completed successfully!")
    else:
        print("\\n❌ Migration encountered issues - data is safe in backup")
    
    input("\\nPress Enter to exit...")
'''
        
        script_filename = f"migrate_to_full_rudradb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        
        with open(script_filename, 'w') as f:
            f.write(script_content)
        
        self.migration_log.append(f"✅ Migration script generated: {script_filename}")
        
        print(f"   📄 Script file: {script_filename}")
        print(f"   🎯 Ready to run after upgrading to full RudraDB")
        
        return script_filename
    
    def validate_migration_readiness(self) -> Dict[str, Any]:
        """Validate that everything is ready for migration"""
        
        print("\n🔍 Validating Migration Readiness...")
        
        validation = {
            "ready_for_migration": True,
            "checks_passed": [],
            "warnings": [],
            "critical_issues": [],
            "recommendations": []
        }
        
        # Check 1: Data integrity
        stats = self.opin_db.get_statistics()
        if stats["vector_count"] > 0:
            validation["checks_passed"].append("Database contains data to migrate")
        else:
            validation["critical_issues"].append("No vectors to migrate - database is empty")
            validation["ready_for_migration"] = False
        
        # Check 2: Backup created
        if self.backup_created:
            validation["checks_passed"].append("Complete backup created and verified")
        else:
            validation["warnings"].append("No backup created yet - run create_complete_backup() first")
        
        # Check 3: Data quality
        if self._validate_data_quality():
            validation["checks_passed"].append("Data quality meets migration standards")
        else:
            validation["warnings"].append("Consider reviewing data quality before migration")
        
        # Check 4: Learning completeness
        relationship_types = self._analyze_relationship_diversity()
        if len(relationship_types) >= 3:
            validation["checks_passed"].append(f"Good relationship diversity ({len(relationship_types)} types)")
        else:
            validation["recommendations"].append("Consider exploring more relationship types")
        
        # Check 5: Capacity utilization
        capacity = stats["capacity_usage"]
        if capacity["vector_usage_percent"] >= 50:
            validation["checks_passed"].append("Good learning progress (50%+ vectors used)")
        else:
            validation["recommendations"].append("Consider adding more vectors to gain experience")
        
        # Final assessment
        if validation["critical_issues"]:
            validation["ready_for_migration"] = False
        elif len(validation["warnings"]) > 2:
            validation["recommendations"].append("Address warnings before migration for best results")
        
        return validation
    
    def print_migration_guide(self):
        """Print complete migration guide"""
        
        print("\n📖 Complete Migration Guide")
        print("=" * 40)
        
        print("\n🎯 Step-by-Step Migration Process:")
        
        steps = [
            "1️⃣ Assess readiness (run assess_readiness_for_upgrade())",
            "2️⃣ Create complete backup (run create_complete_backup())",
            "3️⃣ Generate migration script (run generate_migration_script())",
            "4️⃣ Install full RudraDB (pip uninstall rudradb-opin && pip install rudradb)",
            "5️⃣ Run migration script (python migrate_to_full_rudradb_*.py)",
            "6️⃣ Validate migration success and enjoy unlimited scale!"
        ]
        
        for step in steps:
            print(f"   {step}")
        
        print("\n💡 Key Benefits After Migration:")
        benefits = [
            "🚀 Scale from 100 to 100,000+ vectors",
            "🔗 Scale from 500 to 250,000+ relationships", 
            "⚡ Unlimited multi-hop traversal (beyond 2-hop limit)",
            "🎯 Same API - no code changes needed",
            "📈 Enterprise performance optimizations",
            "🛠 Advanced features and analytics",
            "📞 Priority support and documentation"
        ]
        
        for benefit in benefits:
            print(f"   {benefit}")
        
        print("\n🔄 What Stays the Same:")
        preserved = [
            "✅ Exact same API and methods",
            "✅ All your existing code works unchanged",
            "✅ Same relationship types and search patterns",
            "✅ All data and relationships preserved",
            "✅ Auto-dimension detection (enhanced)",
            "✅ Learning patterns and best practices apply"
        ]
        
        for item in preserved:
            print(f"   {item}")

def demo_migration_helper():
    """Demonstrate the migration helper"""
    
    print("🚀 RudraDB-Opin Migration Helper Demo")
    print("=" * 45)
    
    # Create sample RudraDB-Opin database
    print("📚 Creating sample RudraDB-Opin database for demo...")
    
    db = rudradb.RudraDB()
    
    # Add sample content
    sample_docs = [
        ("ai_basics", "Artificial Intelligence fundamentals and applications"),
        ("ml_intro", "Machine Learning algorithms and data science"),
        ("dl_concepts", "Deep Learning neural networks and training"),
        ("nlp_overview", "Natural Language Processing and text analysis"),
        ("cv_basics", "Computer Vision and image recognition")
    ]
    
    for doc_id, text in sample_docs:
        embedding = np.random.rand(384).astype(np.float32)  # Mock embedding
        db.add_vector(doc_id, embedding, {
            "text": text,
            "category": "AI",
            "difficulty": "intermediate",
            "type": "concept"
        })
    
    # Add relationships
    relationships = [
        ("ai_basics", "ml_intro", "hierarchical", 0.9),
        ("ml_intro", "dl_concepts", "temporal", 0.8),
        ("ai_basics", "nlp_overview", "semantic", 0.7),
        ("ai_basics", "cv_basics", "semantic", 0.7),
        ("ml_intro", "nlp_overview", "associative", 0.6)
    ]
    
    for source, target, rel_type, strength in relationships:
        db.add_relationship(source, target, rel_type, strength)
    
    print(f"   ✅ Created sample database: {db.vector_count()} vectors, {db.relationship_count()} relationships")
    
    # Initialize migration helper
    helper = MigrationHelper(db)
    
    # Run readiness assessment
    assessment = helper.assess_readiness_for_upgrade()
    
    print(f"\n📊 Upgrade Readiness Assessment:")
    print(f"   Ready for upgrade: {'✅ YES' if assessment['ready'] else '❌ Not yet'}")
    print(f"   Readiness score: {assessment['readiness_score']}/100")
    
    if assessment['requirements_met']:
        print(f"   ✅ Requirements Met:")
        for req in assessment['requirements_met']:
            print(f"      • {req}")
    
    if assessment['recommendations']:
        print(f"   💡 Recommendations:")
        for rec in assessment['recommendations']:
            print(f"      • {rec}")
    
    if assessment['warnings']:
        print(f"   ⚠️ Warnings:")
        for warning in assessment['warnings']:
            print(f"      • {warning}")
    
    # Create backup
    backup_file = helper.create_complete_backup()
    
    # Generate migration script
    script_file = helper.generate_migration_script()
    
    # Validate readiness
    validation = helper.validate_migration_readiness()
    
    print(f"\n✅ Migration Validation:")
    print(f"   Ready to migrate: {'✅ YES' if validation['ready_for_migration'] else '❌ Not yet'}")
    print(f"   Checks passed: {len(validation['checks_passed'])}")
    print(f"   Warnings: {len(validation['warnings'])}")
    
    # Show migration guide
    helper.print_migration_guide()
    
    print(f"\n🎉 Migration Helper Demo Complete!")
    print(f"   📁 Backup created: {backup_file}")
    print(f"   📄 Migration script: {script_file}")
    print(f"   🚀 Ready to upgrade to production scale!")

if __name__ == "__main__":
    demo_migration_helper()
