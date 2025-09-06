#!/usr/bin/env python3
"""
RudraDB-Opin to RudraDB Upgrade Assistant
=========================================

Complete migration guide and tools for upgrading from RudraDB-Opin to full RudraDB.
Preserves all your learning data and relationships with seamless scaling.
"""

import rudradb
import json
import numpy as np
import time
from typing import Dict, Any, List
from datetime import datetime
import os

class RudraDB_Upgrade_Assistant:
    """Complete upgrade workflow with data preservation"""
    
    def __init__(self, opin_db):
        self.opin_db = opin_db
        self.backup_created = False
        self.migration_log = []
        
        print("🚀 RudraDB-Opin → RudraDB Upgrade Assistant")
        print("=" * 50)
        
    def should_upgrade_assessment(self) -> Dict[str, Any]:
        """Assess if it's time to upgrade from Opin to full RudraDB"""
        
        stats = self.opin_db.get_statistics()
        usage = stats['capacity_usage']
        
        print("🔍 Upgrade Assessment")
        print("=" * 30)
        
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
        for vec_id in self.opin_db.list_vectors():
            relationships = self.opin_db.get_relationships(vec_id)
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
        recommendation = ""
        should_upgrade = False
        
        if len(upgrade_indicators) >= 3:
            recommendation = "🚀 RECOMMENDATION: Time to upgrade!"
            should_upgrade = True
            print(f"\n{recommendation}")
            print(f"   You've mastered relationship-aware vector search with RudraDB-Opin")
            print(f"   Ready for production scale!")
            
        elif len(upgrade_indicators) >= 1:
            recommendation = "💡 RECOMMENDATION: Consider upgrading soon"
            print(f"\n{recommendation}")
            print(f"   You're making great progress with relationship-aware search")
            print(f"   When ready for more scale, upgrade seamlessly!")
            
        else:
            recommendation = "📖 RECOMMENDATION: Keep learning!"
            print(f"\n{recommendation}")
            print(f"   RudraDB-Opin is perfect for your current learning phase")
        
        return {
            "should_upgrade": should_upgrade,
            "indicators": upgrade_indicators,
            "recommendation": recommendation,
            "stats": {
                "vectors": vector_count,
                "relationships": relationship_count,
                "avg_relationships": avg_relationships_per_vector,
                "types_used": len(relationship_types_used)
            }
        }
    
    def create_upgrade_backup(self) -> Dict[str, Any]:
        """Create comprehensive backup before upgrade"""
        
        print("\n📦 Creating Comprehensive Upgrade Backup")
        print("=" * 45)
        
        backup_data = {
            "metadata": {
                "opin_version": rudradb.__version__,
                "export_timestamp": datetime.now().isoformat(),
                "vector_count": self.opin_db.vector_count(),
                "relationship_count": self.opin_db.relationship_count(),
                "dimension": self.opin_db.dimension()
            },
            "database": self.opin_db.export_data(),
            "learning_summary": self._create_learning_summary()
        }
        
        filename = f"rudradb_opin_upgrade_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        self.backup_created = True
        self.migration_log.append(f"✅ Complete backup created: {filename}")
        
        print(f"✅ Backup created: {filename}")
        print(f"   📊 Vectors: {backup_data['metadata']['vector_count']}")
        print(f"   🔗 Relationships: {backup_data['metadata']['relationship_count']}")
        print(f"   🎯 Dimension: {backup_data['metadata']['dimension']}")
        print(f"   📚 Learning summary included")
        
        return {"filename": filename, "data": backup_data}
    
    def _create_learning_summary(self) -> Dict[str, Any]:
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
            "learning_completeness": len(relationship_types) / 5 * 100,  # % of relationship types explored
            "mastery_indicators": {
                "vector_operations": "mastered" if stats["vector_count"] > 20 else "learning",
                "relationship_modeling": "mastered" if stats["relationship_count"] > 50 else "learning",
                "search_patterns": "mastered" if len(relationship_types) > 2 else "learning",
                "capacity_management": "experienced" if stats["capacity_usage"]["vector_usage_percent"] > 50 else "beginner"
            }
        }
    
    def validate_for_migration(self) -> Dict[str, Any]:
        """Validate that the data is ready for migration"""
        
        print("\n🔍 Migration Readiness Validation")
        print("=" * 35)
        
        stats = self.opin_db.get_statistics()
        
        validation_results = {
            "ready_for_migration": True,
            "warnings": [],
            "recommendations": [],
            "checks_passed": 0,
            "total_checks": 5
        }
        
        # Check 1: Minimum data threshold
        if stats["vector_count"] >= 5:
            validation_results["checks_passed"] += 1
            print("✅ Check 1: Sufficient vectors for migration")
        else:
            validation_results["warnings"].append("Less than 5 vectors - consider adding more content")
            print("⚠️ Check 1: Low vector count")
        
        # Check 2: Relationship existence
        if stats["relationship_count"] >= 1:
            validation_results["checks_passed"] += 1
            print("✅ Check 2: Relationships exist")
        else:
            validation_results["warnings"].append("No relationships - consider building connections")
            print("⚠️ Check 2: No relationships found")
        
        # Check 3: Dimension consistency
        if self.opin_db.dimension() and self.opin_db.dimension() > 0:
            validation_results["checks_passed"] += 1
            print(f"✅ Check 3: Consistent dimension ({self.opin_db.dimension()}D)")
        else:
            validation_results["warnings"].append("No dimension detected - add vectors first")
            print("⚠️ Check 3: No dimension set")
        
        # Check 4: Data integrity
        try:
            export_test = self.opin_db.export_data()
            if export_test and 'vectors' in export_test:
                validation_results["checks_passed"] += 1
                print("✅ Check 4: Data export successful")
            else:
                validation_results["warnings"].append("Data export issues detected")
                print("⚠️ Check 4: Export problems")
        except Exception as e:
            validation_results["warnings"].append(f"Export error: {str(e)}")
            print(f"⚠️ Check 4: Export failed - {e}")
        
        # Check 5: Learning progress
        relationship_types = set()
        for vec_id in self.opin_db.list_vectors():
            relationships = self.opin_db.get_relationships(vec_id)
            for rel in relationships:
                relationship_types.add(rel["relationship_type"])
        
        if len(relationship_types) >= 2:
            validation_results["checks_passed"] += 1
            print(f"✅ Check 5: Multiple relationship types explored ({len(relationship_types)}/5)")
        else:
            validation_results["recommendations"].append("Try exploring more relationship types")
            print(f"💡 Check 5: Limited relationship diversity ({len(relationship_types)}/5)")
        
        # Overall assessment
        validation_results["ready_for_migration"] = validation_results["checks_passed"] >= 3
        
        print(f"\n📊 Validation Results:")
        print(f"   Checks passed: {validation_results['checks_passed']}/{validation_results['total_checks']}")
        print(f"   Ready for migration: {'✅ Yes' if validation_results['ready_for_migration'] else '⚠️ Needs attention'}")
        
        if validation_results["warnings"]:
            print(f"   ⚠️ Warnings:")
            for warning in validation_results["warnings"]:
                print(f"      • {warning}")
        
        if validation_results["recommendations"]:
            print(f"   💡 Recommendations:")
            for rec in validation_results["recommendations"]:
                print(f"      • {rec}")
        
        return validation_results
    
    def generate_migration_script(self) -> str:
        """Generate complete migration script"""
        
        print("\n🛠️ Generating Migration Script")
        print("=" * 35)
        
        script_template = '''#!/usr/bin/env python3
"""
Automated RudraDB-Opin to RudraDB Upgrade Script
Generated automatically to preserve all your data and relationships.
"""

import json
import numpy as np
from datetime import datetime

def main():
    print("🚀 Starting RudraDB-Opin → RudraDB Upgrade")
    print("=" * 50)
    
    # Load backup data
    print("📂 Loading backup data...")
    backup_file = input("Enter backup filename (or press Enter for latest): ").strip()
    
    if not backup_file:
        import glob
        backup_files = glob.glob("rudradb_opin_upgrade_backup_*.json")
        if backup_files:
            backup_file = max(backup_files)  # Latest backup
            print(f"   Using latest backup: {backup_file}")
        else:
            print("❌ No backup files found!")
            return False
    
    try:
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Backup file not found: {backup_file}")
        return False
    
    original_stats = backup_data['metadata']
    print(f"   Original database: {original_stats['vector_count']} vectors, {original_stats['relationship_count']} relationships")
    print(f"   Dimension: {original_stats['dimension']}")
    
    # Import full RudraDB (user must install it first)
    print("\\n🧬 Importing full RudraDB...")
    try:
        import rudradb
        print("   ✅ Full RudraDB imported successfully")
    except ImportError:
        print("   ❌ Full RudraDB not installed!")
        print("   Please run: pip uninstall rudradb-opin && pip install rudradb")
        return False
    
    # Create new full RudraDB instance
    print("\\n🚀 Creating full RudraDB instance...")
    
    # Preserve original dimension if detected
    if original_stats['dimension']:
        full_db = rudradb.RudraDB(dimension=original_stats['dimension'])
    else:
        full_db = rudradb.RudraDB()  # Auto-dimension detection
    
    print(f"   ✅ Full RudraDB created (capacity: unlimited)")
    
    # Import all data
    print("\\n📥 Importing data...")
    try:
        full_db.import_data(backup_data['database'])
        print(f"   ✅ Data import successful")
        
        # Verify import
        new_stats = full_db.get_statistics()
        print(f"   📊 Verification: {new_stats['vector_count']} vectors, {new_stats['relationship_count']} relationships")
        
        if new_stats['vector_count'] == original_stats['vector_count']:
            print("   ✅ All vectors successfully migrated")
        else:
            print(f"   ⚠️ Vector count mismatch: {new_stats['vector_count']} vs {original_stats['vector_count']}")
            
        if new_stats['relationship_count'] == original_stats['relationship_count']:
            print("   ✅ All relationships successfully migrated")
        else:
            print(f"   ⚠️ Relationship count mismatch: {new_stats['relationship_count']} vs {original_stats['relationship_count']}")
        
        # Test functionality
        print("\\n🔍 Testing upgraded functionality...")
        
        # Test search
        if new_stats['vector_count'] > 0:
            sample_vector_id = full_db.list_vectors()[0]
            sample_vector = full_db.get_vector(sample_vector_id)
            sample_embedding = sample_vector['embedding']
            
            search_results = full_db.search(sample_embedding, rudradb.SearchParams(
                top_k=5,
                include_relationships=True
            ))
            
            print(f"   ✅ Search test: {len(search_results)} results returned")
        
        # Show upgrade benefits
        print("\\n🎉 Upgrade Complete! New Capabilities:")
        print(f"   📊 Vectors: 100 → Unlimited ({new_stats.get('max_vectors', 'No limit')})")
        print(f"   🔗 Relationships: 500 → Unlimited ({new_stats.get('max_relationships', 'No limit')})")
        print(f"   🎯 Multi-hop traversal: 2 → Unlimited hops")
        print(f"   ✨ All auto-features preserved and enhanced")
        print(f"   🚀 Production-ready with enterprise features")
        
        # Learning summary
        if 'learning_summary' in backup_data:
            learning = backup_data['learning_summary']
            print(f"\\n📚 Your Learning Journey:")
            print(f"   🎓 Vectors explored: {learning['vectors_explored']}")
            print(f"   🧠 Relationships built: {learning['relationships_built']}")
            print(f"   🔄 Relationship types mastered: {len(learning['relationship_types_used'])}/5")
            print(f"   📈 Learning completeness: {learning['learning_completeness']:.0f}%")
        
        print("\\n💾 Upgrade completed successfully!")
        print("Your RudraDB-Opin learning data is now running on full RudraDB with unlimited capacity!")
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        print("   💡 Contact support for assistance")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
'''
        
        script_filename = f"upgrade_to_rudradb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        
        with open(script_filename, 'w') as f:
            f.write(script_template)
        
        self.migration_log.append(f"✅ Migration script generated: {script_filename}")
        
        print(f"✅ Migration script created: {script_filename}")
        print("   🛠️ Automated upgrade process")
        print("   📊 Data validation included")
        print("   🔍 Error handling built-in")
        
        return script_filename
    
    def execute_pre_upgrade_checklist(self) -> bool:
        """Complete pre-upgrade checklist"""
        
        print("\n📋 Pre-Upgrade Checklist")
        print("=" * 30)
        
        # Step 1: Assess readiness
        print("1. 🔍 Assessing upgrade readiness...")
        assessment = self.should_upgrade_assessment()
        
        if not assessment["should_upgrade"] and len(assessment["indicators"]) < 2:
            print("   💡 Consider exploring RudraDB-Opin more before upgrading")
            return False
        
        # Step 2: Create backup
        print("\n2. 📦 Creating backup...")
        if not self.backup_created:
            backup_result = self.create_upgrade_backup()
        else:
            print("   ✅ Backup already created")
        
        # Step 3: Validate readiness
        print("\n3. ✅ Validating migration readiness...")
        validation = self.validate_for_migration()
        
        if not validation["ready_for_migration"]:
            print("   ⚠️ System needs attention before migration")
            return False
        
        # Step 4: Generate migration script
        print("\n4. 🛠️ Generating migration script...")
        script_path = self.generate_migration_script()
        
        # Step 5: Final instructions
        print("\n🎯 Ready to Upgrade!")
        print("Execute these commands in order:")
        print("   1. pip uninstall rudradb-opin")
        print("   2. pip install rudradb") 
        print(f"   3. python {script_path}")
        
        print("\n📚 What You've Accomplished:")
        learning = assessment["stats"]
        print(f"   🎓 Mastered {learning['vectors']} vector concepts")
        print(f"   🧠 Built {learning['relationships']} relationships")
        print(f"   🔄 Explored {learning['types_used']}/5 relationship types")
        print(f"   📈 Ready for production-scale relationship-aware AI!")
        
        return True

def simulate_upgrade_scenario():
    """Simulate a typical upgrade scenario with sample data"""
    
    print("🧪 Simulating Upgrade Scenario")
    print("=" * 35)
    
    # Create sample RudraDB-Opin database
    db = rudradb.RudraDB()
    
    # Add sample learning data
    learning_data = [
        ("ai_basics", [0.8, 0.2, 0.9, 0.1], {"topic": "AI", "level": "beginner"}),
        ("ml_advanced", [0.7, 0.3, 0.8, 0.2], {"topic": "AI", "level": "advanced"}),
        ("python_intro", [0.3, 0.7, 0.4, 0.6], {"topic": "Programming", "level": "beginner"}),
        ("data_science", [0.6, 0.4, 0.7, 0.3], {"topic": "Data", "level": "intermediate"}),
        ("deep_learning", [0.9, 0.1, 0.95, 0.05], {"topic": "AI", "level": "expert"})
    ]
    
    for doc_id, embedding, metadata in learning_data:
        db.add_vector(doc_id, np.array(embedding, dtype=np.float32), metadata)
    
    # Add relationships
    relationships = [
        ("ai_basics", "ml_advanced", "hierarchical", 0.9),
        ("python_intro", "data_science", "temporal", 0.8),
        ("ml_advanced", "deep_learning", "hierarchical", 0.85),
        ("data_science", "ml_advanced", "associative", 0.7)
    ]
    
    for source, target, rel_type, strength in relationships:
        db.add_relationship(source, target, rel_type, strength)
    
    print(f"✅ Created sample database:")
    print(f"   📊 Vectors: {db.vector_count()}")
    print(f"   🔗 Relationships: {db.relationship_count()}")
    
    # Run upgrade assistant
    assistant = RudraDB_Upgrade_Assistant(db)
    
    # Execute checklist
    ready = assistant.execute_pre_upgrade_checklist()
    
    if ready:
        print("\n🎉 Upgrade simulation successful!")
        print("   Real upgrade would preserve all your learning data")
        print("   and provide unlimited capacity for production use!")
    else:
        print("\n💡 Upgrade simulation shows areas for improvement")
        print("   Continue learning with RudraDB-Opin!")
    
    return ready

def main():
    """Main upgrade demonstration"""
    
    # Run simulation
    simulation_success = simulate_upgrade_scenario()
    
    print("\n" + "=" * 60)
    print("📖 How to Use This Upgrade Assistant:")
    print("=" * 60)
    
    print("\n1️⃣ With Your Own RudraDB-Opin Database:")
    print("   from migration.upgrade_assistant import RudraDB_Upgrade_Assistant")
    print("   assistant = RudraDB_Upgrade_Assistant(your_db)")
    print("   assistant.execute_pre_upgrade_checklist()")
    
    print("\n2️⃣ Assessment Only:")
    print("   assessment = assistant.should_upgrade_assessment()")
    print("   print(assessment['recommendation'])")
    
    print("\n3️⃣ Backup Only:")
    print("   backup = assistant.create_upgrade_backup()")
    print("   # Safe to upgrade after backup created")
    
    print("\n🚀 Benefits of Upgrading:")
    print("   • 100,000+ vectors (1000x more than Opin)")
    print("   • 250,000+ relationships (500x more than Opin)")
    print("   • Same API - no code changes needed")
    print("   • Production-ready performance")
    print("   • Advanced features and optimizations")
    
    return simulation_success

if __name__ == "__main__":
    main()
