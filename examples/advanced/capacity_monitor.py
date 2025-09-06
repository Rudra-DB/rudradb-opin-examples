#!/usr/bin/env python3
"""
RudraDB-Opin Advanced Capacity Monitor and Management

This example demonstrates advanced capacity monitoring, predictive analytics,
and intelligent capacity management for RudraDB-Opin, helping users understand
when to upgrade and how to optimize their usage.

Requirements:
    pip install rudradb-opin

Usage:
    python capacity_monitor.py
"""

import rudradb
import numpy as np
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import statistics


class AdvancedCapacityMonitor:
    """Advanced capacity monitoring and management for RudraDB-Opin"""
    
    def __init__(self, db: rudradb.RudraDB):
        self.db = db
        self.usage_history = []
        self.monitoring_start_time = time.time()
        
        print("📊 Advanced Capacity Monitor initialized")
        print(f"   🎯 Monitoring database with {self.db.vector_count()} vectors, {self.db.relationship_count()} relationships")
    
    def get_comprehensive_usage_report(self) -> Dict[str, Any]:
        """Get detailed capacity usage report with analytics"""
        
        stats = self.db.get_statistics()
        usage = stats['capacity_usage']
        
        # Calculate usage trends if we have history
        vector_trend = self._calculate_trend("vectors") if len(self.usage_history) > 1 else 0
        relationship_trend = self._calculate_trend("relationships") if len(self.usage_history) > 1 else 0
        
        # Predict time to capacity
        vector_prediction = self._predict_time_to_capacity("vectors") if vector_trend > 0 else None
        relationship_prediction = self._predict_time_to_capacity("relationships") if relationship_trend > 0 else None
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_duration_hours": (time.time() - self.monitoring_start_time) / 3600,
            
            "current_usage": {
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
                "dimension": stats.get('dimension', 'Not detected'),
                "overall_health": self._get_overall_health(stats)
            },
            
            "usage_trends": {
                "vectors_per_hour": vector_trend,
                "relationships_per_hour": relationship_trend,
                "monitoring_data_points": len(self.usage_history)
            },
            
            "predictions": {
                "vectors_capacity_reached": vector_prediction,
                "relationships_capacity_reached": relationship_prediction
            },
            
            "efficiency_metrics": self._calculate_efficiency_metrics(stats),
            "upgrade_analysis": self._analyze_upgrade_readiness(stats)
        }
        
        # Store this report in history
        self.usage_history.append({
            "timestamp": time.time(),
            "vectors": stats['vector_count'],
            "relationships": stats['relationship_count']
        })
        
        # Keep only recent history (last 100 data points)
        if len(self.usage_history) > 100:
            self.usage_history = self.usage_history[-100:]
        
        return report
    
    def _get_status(self, percentage: float) -> str:
        """Get status based on usage percentage"""
        if percentage >= 95:
            return "CRITICAL"
        elif percentage >= 85:
            return "WARNING"
        elif percentage >= 70:
            return "MODERATE"
        elif percentage >= 50:
            return "ACTIVE"
        else:
            return "HEALTHY"
    
    def _get_overall_health(self, stats: Dict[str, Any]) -> str:
        """Get overall database health assessment"""
        usage = stats['capacity_usage']
        max_usage = max(usage['vector_usage_percent'], usage['relationship_usage_percent'])
        vector_count = stats['vector_count']
        relationship_count = stats['relationship_count']
        
        if max_usage >= 95:
            return "Ready to upgrade - You've mastered RudraDB-Opin!"
        elif max_usage >= 85:
            return "Approaching capacity - Consider upgrade planning"
        elif max_usage >= 70:
            return "Actively learning - Great progress!"
        elif max_usage >= 30:
            return "Building knowledge - Good usage patterns"
        else:
            return "Getting started - Plenty of capacity to explore"
    
    def _calculate_trend(self, resource_type: str) -> float:
        """Calculate usage trend (items per hour)"""
        if len(self.usage_history) < 2:
            return 0
        
        # Get recent data points
        recent_history = self.usage_history[-10:]  # Last 10 data points
        
        # Calculate time differences and usage changes
        time_diffs = []
        usage_diffs = []
        
        for i in range(1, len(recent_history)):
            time_diff = recent_history[i]['timestamp'] - recent_history[i-1]['timestamp']
            usage_diff = recent_history[i][resource_type] - recent_history[i-1][resource_type]
            
            if time_diff > 0:
                time_diffs.append(time_diff)
                usage_diffs.append(usage_diff)
        
        if not time_diffs:
            return 0
        
        # Calculate average rate per hour
        total_time_hours = sum(time_diffs) / 3600
        total_usage_change = sum(usage_diffs)
        
        if total_time_hours > 0:
            return total_usage_change / total_time_hours
        else:
            return 0
    
    def _predict_time_to_capacity(self, resource_type: str) -> Optional[Dict[str, Any]]:
        """Predict when capacity will be reached"""
        trend = self._calculate_trend(resource_type)
        
        if trend <= 0:
            return None
        
        current_stats = self.db.get_statistics()
        
        if resource_type == "vectors":
            current_count = current_stats['vector_count']
            max_count = rudradb.MAX_VECTORS
        else:
            current_count = current_stats['relationship_count']
            max_count = rudradb.MAX_RELATIONSHIPS
        
        remaining = max_count - current_count
        
        if remaining <= 0:
            return {"status": "Already at capacity"}
        
        hours_to_capacity = remaining / trend
        days_to_capacity = hours_to_capacity / 24
        
        prediction_time = datetime.now() + timedelta(hours=hours_to_capacity)
        
        return {
            "hours_remaining": hours_to_capacity,
            "days_remaining": days_to_capacity,
            "predicted_date": prediction_time.isoformat(),
            "confidence": "high" if len(self.usage_history) >= 5 else "low",
            "current_trend": f"{trend:.1f} {resource_type}/hour"
        }
    
    def _calculate_efficiency_metrics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate efficiency and optimization metrics"""
        
        vector_count = stats['vector_count']
        relationship_count = stats['relationship_count']
        dimension = stats.get('dimension', 0)
        
        metrics = {
            "relationships_per_vector": relationship_count / max(vector_count, 1),
            "capacity_utilization_balance": abs(stats['capacity_usage']['vector_usage_percent'] - 
                                               stats['capacity_usage']['relationship_usage_percent']),
            "dimension_efficiency": dimension if dimension else "Unknown",
            "storage_efficiency": "Optimal" if vector_count > 0 else "Underutilized"
        }
        
        # Relationship density analysis
        if relationship_count > 0 and vector_count > 0:
            max_possible_relationships = vector_count * (vector_count - 1) // 2
            relationship_density = relationship_count / max_possible_relationships * 100
            
            metrics["relationship_density_percent"] = relationship_density
            
            if relationship_density < 5:
                metrics["relationship_density_status"] = "Sparse - Consider more connections"
            elif relationship_density < 20:
                metrics["relationship_density_status"] = "Moderate - Good balance"
            else:
                metrics["relationship_density_status"] = "Dense - Rich relationship network"
        
        return metrics
    
    def _analyze_upgrade_readiness(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if user is ready to upgrade to full RudraDB"""
        
        usage = stats['capacity_usage']
        vector_count = stats['vector_count']
        relationship_count = stats['relationship_count']
        
        readiness_score = 0
        factors = []
        
        # Factor 1: Capacity utilization
        max_usage = max(usage['vector_usage_percent'], usage['relationship_usage_percent'])
        if max_usage > 90:
            readiness_score += 40
            factors.append("High capacity utilization (90%+)")
        elif max_usage > 70:
            readiness_score += 20
            factors.append("Significant capacity utilization (70%+)")
        
        # Factor 2: Understanding of relationships
        if relationship_count > 100:
            readiness_score += 30
            factors.append("Substantial relationship modeling experience")
        elif relationship_count > 50:
            readiness_score += 15
            factors.append("Good relationship modeling understanding")
        
        # Factor 3: Database complexity
        if vector_count > 50:
            readiness_score += 20
            factors.append("Significant database size")
        
        # Factor 4: Relationship diversity
        relationship_types_used = set()
        for vec_id in self.db.list_vectors()[:10]:  # Sample check
            relationships = self.db.get_relationships(vec_id)
            for rel in relationships:
                relationship_types_used.add(rel["relationship_type"])
        
        if len(relationship_types_used) >= 4:
            readiness_score += 10
            factors.append("Explored most relationship types")
        elif len(relationship_types_used) >= 2:
            readiness_score += 5
            factors.append("Using multiple relationship types")
        
        # Determine readiness level
        if readiness_score >= 80:
            readiness_level = "READY"
            recommendation = "You've mastered RudraDB-Opin! Time to upgrade for production scale."
        elif readiness_score >= 50:
            readiness_level = "NEARLY_READY"
            recommendation = "You're making great progress. Consider upgrade when you need more capacity."
        elif readiness_score >= 25:
            readiness_level = "LEARNING"
            recommendation = "Keep exploring RudraDB-Opin's features. You're building good understanding."
        else:
            readiness_level = "STARTING"
            recommendation = "Great start! Continue learning with RudraDB-Opin's perfect tutorial capacity."
        
        return {
            "readiness_score": readiness_score,
            "readiness_level": readiness_level,
            "recommendation": recommendation,
            "contributing_factors": factors,
            "relationship_types_explored": list(relationship_types_used),
            "upgrade_benefits": [
                f"1,000x more vectors ({rudradb.MAX_VECTORS} → 100,000+)",
                f"500x more relationships ({rudradb.MAX_RELATIONSHIPS} → 250,000+)",
                "Unlimited relationship hops (2 → ∞)",
                "Advanced performance optimizations",
                "Enterprise support and features"
            ]
        }
    
    def print_detailed_report(self):
        """Print comprehensive capacity report"""
        
        report = self.get_comprehensive_usage_report()
        
        print(f"\n📊 Advanced RudraDB-Opin Capacity Report")
        print("=" * 55)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ Monitoring duration: {report['monitoring_duration_hours']:.1f} hours")
        
        # Current usage
        print(f"\n📈 Current Usage Status:")
        current = report['current_usage']
        
        vec = current['vectors']
        print(f"   📄 Vectors: {vec['used']:,}/{vec['max']:,} ({vec['percentage']:.1f}%)")
        print(f"      Status: {vec['status']} | Remaining: {vec['remaining']:,}")
        self._print_progress_bar(vec['percentage'], "📄")
        
        rel = current['relationships']
        print(f"   🔗 Relationships: {rel['used']:,}/{rel['max']:,} ({rel['percentage']:.1f}%)")
        print(f"      Status: {rel['status']} | Remaining: {rel['remaining']:,}")
        self._print_progress_bar(rel['percentage'], "🔗")
        
        print(f"   🎯 Dimension: {current['dimension']}")
        print(f"   🏥 Overall Health: {current['overall_health']}")
        
        # Usage trends
        if report['monitoring_data_points'] > 1:
            print(f"\n📈 Usage Trends:")
            trends = report['usage_trends']
            print(f"   📄 Vector growth: {trends['vectors_per_hour']:.1f} vectors/hour")
            print(f"   🔗 Relationship growth: {trends['relationships_per_hour']:.1f} relationships/hour")
            print(f"   📊 Data points: {trends['monitoring_data_points']}")
        
        # Predictions
        predictions = report['predictions']
        if predictions['vectors_capacity_reached'] or predictions['relationships_capacity_reached']:
            print(f"\n🔮 Capacity Predictions:")
            
            if predictions['vectors_capacity_reached']:
                pred = predictions['vectors_capacity_reached']
                if isinstance(pred, dict) and 'days_remaining' in pred:
                    print(f"   📄 Vector capacity: {pred['days_remaining']:.1f} days remaining")
                    print(f"      Predicted date: {pred['predicted_date'][:10]}")
                    print(f"      Confidence: {pred['confidence']}")
            
            if predictions['relationships_capacity_reached']:
                pred = predictions['relationships_capacity_reached']
                if isinstance(pred, dict) and 'days_remaining' in pred:
                    print(f"   🔗 Relationship capacity: {pred['days_remaining']:.1f} days remaining")
                    print(f"      Predicted date: {pred['predicted_date'][:10]}")
                    print(f"      Confidence: {pred['confidence']}")
        
        # Efficiency metrics
        print(f"\n⚡ Efficiency Metrics:")
        efficiency = report['efficiency_metrics']
        print(f"   🔗 Relationships per vector: {efficiency['relationships_per_vector']:.1f}")
        print(f"   ⚖️ Capacity balance: {efficiency['capacity_utilization_balance']:.1f}% difference")
        print(f"   💾 Dimension efficiency: {efficiency['dimension_efficiency']}")
        
        if 'relationship_density_percent' in efficiency:
            print(f"   🕸️ Relationship density: {efficiency['relationship_density_percent']:.1f}%")
            print(f"      Status: {efficiency['relationship_density_status']}")
        
        # Upgrade analysis
        print(f"\n🚀 Upgrade Analysis:")
        upgrade = report['upgrade_analysis']
        print(f"   📊 Readiness Score: {upgrade['readiness_score']}/100")
        print(f"   🎯 Readiness Level: {upgrade['readiness_level']}")
        print(f"   💡 Recommendation: {upgrade['recommendation']}")
        
        if upgrade['contributing_factors']:
            print(f"   ✅ Contributing Factors:")
            for factor in upgrade['contributing_factors']:
                print(f"      • {factor}")
        
        if upgrade['relationship_types_explored']:
            print(f"   🧠 Relationship Types Used: {', '.join(upgrade['relationship_types_explored'])}")
        
        if upgrade['readiness_level'] in ['READY', 'NEARLY_READY']:
            print(f"\n🎁 Upgrade Benefits:")
            for benefit in upgrade['upgrade_benefits'][:3]:  # Show top 3
                print(f"      • {benefit}")
            print(f"      • ...and more enterprise features")
    
    def _print_progress_bar(self, percentage: float, icon: str, width: int = 30):
        """Print visual progress bar"""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        print(f"      {icon} [{bar}] {percentage:.1f}%")
    
    def save_monitoring_data(self, filename: str = "capacity_monitoring_data.json"):
        """Save monitoring data to file"""
        
        report = self.get_comprehensive_usage_report()
        report['usage_history'] = self.usage_history
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Monitoring data saved to: {filename}")
    
    def load_monitoring_data(self, filename: str = "capacity_monitoring_data.json"):
        """Load previous monitoring data"""
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if 'usage_history' in data:
                self.usage_history = data['usage_history']
                print(f"📂 Loaded {len(self.usage_history)} historical data points")
            
        except FileNotFoundError:
            print(f"📄 No previous monitoring data found")
        except Exception as e:
            print(f"⚠️ Error loading monitoring data: {e}")


def demo_capacity_monitoring():
    """Demo advanced capacity monitoring"""
    
    print("📊 RudraDB-Opin Advanced Capacity Monitoring Demo")
    print("=" * 55)
    
    # Create database and add some sample data
    db = rudradb.RudraDB()
    
    print("\n🏗️ Creating sample database for monitoring demo...")
    
    # Add sample vectors
    for i in range(25):
        embedding = np.random.rand(384).astype(np.float32)
        metadata = {
            "index": i,
            "category": f"category_{i % 5}",
            "difficulty": ["beginner", "intermediate", "advanced"][i % 3],
            "tags": [f"tag_{j}" for j in range(i % 3 + 1)]
        }
        db.add_vector(f"doc_{i}", embedding, metadata)
    
    # Add sample relationships
    relationships_added = 0
    for i in range(15):
        source = f"doc_{i}"
        target = f"doc_{(i + 1) % 20}"
        rel_type = ["semantic", "hierarchical", "associative"][i % 3]
        strength = 0.5 + np.random.random() * 0.5
        
        try:
            db.add_relationship(source, target, rel_type, strength)
            relationships_added += 1
        except:
            break
    
    print(f"   ✅ Created database: {db.vector_count()} vectors, {db.relationship_count()} relationships")
    
    # Initialize capacity monitor
    monitor = AdvancedCapacityMonitor(db)
    
    # Load any previous monitoring data
    monitor.load_monitoring_data()
    
    # Simulate some usage over time
    print(f"\n⏳ Simulating usage patterns for monitoring...")
    
    for i in range(5):
        # Add a few more vectors
        for j in range(3):
            if db.vector_count() >= rudradb.MAX_VECTORS:
                break
            
            embedding = np.random.rand(384).astype(np.float32)
            db.add_vector(f"sim_{i}_{j}", embedding, {"simulation": True, "batch": i})
        
        # Add some relationships
        vector_ids = db.list_vectors()
        for j in range(2):
            if db.relationship_count() >= rudradb.MAX_RELATIONSHIPS:
                break
            
            try:
                source = np.random.choice(vector_ids)
                target = np.random.choice(vector_ids)
                if source != target:
                    db.add_relationship(source, target, "associative", 0.6)
            except:
                break
        
        # Take a monitoring snapshot
        monitor.get_comprehensive_usage_report()
        time.sleep(0.1)  # Small delay to simulate time passage
    
    # Generate and display comprehensive report
    print(f"\n📋 Generating comprehensive capacity report...")
    monitor.print_detailed_report()
    
    # Save monitoring data
    monitor.save_monitoring_data()
    
    # Show specific recommendations
    print(f"\n💡 Specific Recommendations:")
    
    stats = db.get_statistics()
    usage = stats['capacity_usage']
    
    if usage['vector_usage_percent'] > 80:
        print(f"   📄 Vector Usage High ({usage['vector_usage_percent']:.1f}%):")
        print(f"      • You've learned vector operations well!")
        print(f"      • Consider cleanup or upgrade for more capacity")
    
    if usage['relationship_usage_percent'] > 80:
        print(f"   🔗 Relationship Usage High ({usage['relationship_usage_percent']:.1f}%):")
        print(f"      • Excellent relationship modeling skills!")
        print(f"      • Ready for production-scale relationship networks")
    
    if usage['vector_usage_percent'] > 70 or usage['relationship_usage_percent'] > 70:
        print(f"   🚀 Upgrade Path Available:")
        print(f"      • You've mastered the core concepts of relationship-aware search")
        print(f"      • Full RudraDB offers 1000x more capacity with same API")
        print(f"      • Seamless data migration preserves all your work")
    
    print(f"\n🎉 Advanced capacity monitoring complete!")
    print("    📊 Use this data to optimize your RudraDB-Opin usage")
    print("    🔮 Predictive analytics help plan your upgrade timing")
    print("    💾 Monitoring data saved for trend analysis")


if __name__ == "__main__":
    demo_capacity_monitoring()
