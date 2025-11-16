"""
Technology Knowledge Graph
Builds and maintains relationships between technologies, concepts, and resources
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class TechNodeType(Enum):
    TECHNOLOGY = "technology"
    FRAMEWORK = "framework"
    LANGUAGE = "language"
    TOOL = "tool"
    CONCEPT = "concept"
    PLATFORM = "platform"
    LIBRARY = "library"
    SERVICE = "service"
    STANDARD = "standard"
    METHODOLOGY = "methodology"

class RelationshipType(Enum):
    USES = "uses"
    BUILT_WITH = "built_with"
    COMPATIBLE_WITH = "compatible_with"
    DEPENDS_ON = "depends_on"
    SIMILAR_TO = "similar_to"
    ALTERNATIVE_TO = "alternative_to"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    FOLLOWS = "follows"
    PART_OF = "part_of"

@dataclass
class TechNode:
    id: str
    name: str
    node_type: TechNodeType
    description: str
    domain: str
    popularity_score: float
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class TechRelationship:
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float
    context: str
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class TechKnowledgeGraph:
    nodes: Dict[str, TechNode]
    relationships: List[TechRelationship]
    graph: nx.DiGraph
    domain_clusters: Dict[str, Set[str]]
    popularity_rankings: Dict[str, int]

class TechnologyKnowledgeGraph:
    """Knowledge graph for technology relationships and insights"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.relationships = []
        self.domain_clusters = defaultdict(set)
        self.popularity_rankings = {}
        
        # Initialize with common technology relationships
        self._initialize_base_graph()
    
    def _initialize_base_graph(self):
        """Initialize the knowledge graph with common technology relationships"""
        
        # Programming Languages
        languages = [
            ("python", "Python", TechNodeType.LANGUAGE, "High-level programming language", "general", 0.9),
            ("javascript", "JavaScript", TechNodeType.LANGUAGE, "Web programming language", "web_dev", 0.95),
            ("java", "Java", TechNodeType.LANGUAGE, "Object-oriented programming language", "general", 0.8),
            ("typescript", "TypeScript", TechNodeType.LANGUAGE, "Typed superset of JavaScript", "web_dev", 0.7),
            ("go", "Go", TechNodeType.LANGUAGE, "Systems programming language", "devops", 0.6),
            ("rust", "Rust", TechNodeType.LANGUAGE, "Memory-safe systems language", "systems", 0.5),
            ("csharp", "C#", TechNodeType.LANGUAGE, "Microsoft's object-oriented language", "general", 0.7),
            ("cpp", "C++", TechNodeType.LANGUAGE, "General-purpose programming language", "systems", 0.6),
            ("php", "PHP", TechNodeType.LANGUAGE, "Server-side scripting language", "web_dev", 0.6),
            ("ruby", "Ruby", TechNodeType.LANGUAGE, "Dynamic programming language", "web_dev", 0.4),
            ("swift", "Swift", TechNodeType.LANGUAGE, "Apple's programming language", "mobile_dev", 0.5),
            ("kotlin", "Kotlin", TechNodeType.LANGUAGE, "JVM-based language", "mobile_dev", 0.6),
            ("r", "R", TechNodeType.LANGUAGE, "Statistical computing language", "data_science", 0.5),
            ("scala", "Scala", TechNodeType.LANGUAGE, "Functional programming on JVM", "data_science", 0.4)
        ]
        
        # Web Frameworks
        web_frameworks = [
            ("react", "React", TechNodeType.FRAMEWORK, "JavaScript UI library", "web_dev", 0.9),
            ("vue", "Vue.js", TechNodeType.FRAMEWORK, "Progressive JavaScript framework", "web_dev", 0.7),
            ("angular", "Angular", TechNodeType.FRAMEWORK, "TypeScript web framework", "web_dev", 0.6),
            ("nextjs", "Next.js", TechNodeType.FRAMEWORK, "React framework for production", "web_dev", 0.8),
            ("nuxtjs", "Nuxt.js", TechNodeType.FRAMEWORK, "Vue.js framework", "web_dev", 0.5),
            ("svelte", "Svelte", TechNodeType.FRAMEWORK, "Compile-time optimized framework", "web_dev", 0.4),
            ("express", "Express.js", TechNodeType.FRAMEWORK, "Node.js web framework", "web_dev", 0.8),
            ("django", "Django", TechNodeType.FRAMEWORK, "Python web framework", "web_dev", 0.7),
            ("flask", "Flask", TechNodeType.FRAMEWORK, "Lightweight Python web framework", "web_dev", 0.6),
            ("spring", "Spring", TechNodeType.FRAMEWORK, "Java application framework", "web_dev", 0.7),
            ("laravel", "Laravel", TechNodeType.FRAMEWORK, "PHP web framework", "web_dev", 0.6),
            ("rails", "Ruby on Rails", TechNodeType.FRAMEWORK, "Ruby web framework", "web_dev", 0.5)
        ]
        
        # AI/ML Frameworks
        ai_frameworks = [
            ("tensorflow", "TensorFlow", TechNodeType.FRAMEWORK, "Machine learning platform", "ai_ml", 0.9),
            ("pytorch", "PyTorch", TechNodeType.FRAMEWORK, "Deep learning framework", "ai_ml", 0.9),
            ("keras", "Keras", TechNodeType.FRAMEWORK, "High-level neural networks API", "ai_ml", 0.7),
            ("scikit-learn", "Scikit-learn", TechNodeType.LIBRARY, "Machine learning library", "ai_ml", 0.8),
            ("pandas", "Pandas", TechNodeType.LIBRARY, "Data manipulation library", "data_science", 0.9),
            ("numpy", "NumPy", TechNodeType.LIBRARY, "Numerical computing library", "data_science", 0.9),
            ("matplotlib", "Matplotlib", TechNodeType.LIBRARY, "Plotting library", "data_science", 0.8),
            ("seaborn", "Seaborn", TechNodeType.LIBRARY, "Statistical visualization library", "data_science", 0.6),
            ("opencv", "OpenCV", TechNodeType.LIBRARY, "Computer vision library", "ai_ml", 0.7),
            ("spacy", "spaCy", TechNodeType.LIBRARY, "Natural language processing library", "ai_ml", 0.6),
            ("nltk", "NLTK", TechNodeType.LIBRARY, "Natural language toolkit", "ai_ml", 0.5),
            ("transformers", "Transformers", TechNodeType.LIBRARY, "Hugging Face transformers library", "ai_ml", 0.8)
        ]
        
        # Mobile Frameworks
        mobile_frameworks = [
            ("react-native", "React Native", TechNodeType.FRAMEWORK, "Cross-platform mobile framework", "mobile_dev", 0.8),
            ("flutter", "Flutter", TechNodeType.FRAMEWORK, "Google's UI toolkit", "mobile_dev", 0.7),
            ("xamarin", "Xamarin", TechNodeType.FRAMEWORK, "Microsoft's mobile framework", "mobile_dev", 0.4),
            ("ionic", "Ionic", TechNodeType.FRAMEWORK, "Hybrid mobile framework", "mobile_dev", 0.5),
            ("cordova", "Apache Cordova", TechNodeType.FRAMEWORK, "Mobile app development platform", "mobile_dev", 0.3)
        ]
        
        # DevOps Tools
        devops_tools = [
            ("docker", "Docker", TechNodeType.TOOL, "Containerization platform", "devops", 0.9),
            ("kubernetes", "Kubernetes", TechNodeType.TOOL, "Container orchestration", "devops", 0.8),
            ("jenkins", "Jenkins", TechNodeType.TOOL, "CI/CD automation server", "devops", 0.7),
            ("gitlab", "GitLab", TechNodeType.TOOL, "DevOps platform", "devops", 0.6),
            ("github", "GitHub", TechNodeType.TOOL, "Code hosting platform", "devops", 0.9),
            ("terraform", "Terraform", TechNodeType.TOOL, "Infrastructure as code", "devops", 0.7),
            ("ansible", "Ansible", TechNodeType.TOOL, "Configuration management", "devops", 0.6),
            ("prometheus", "Prometheus", TechNodeType.TOOL, "Monitoring system", "devops", 0.6),
            ("grafana", "Grafana", TechNodeType.TOOL, "Monitoring and visualization", "devops", 0.6)
        ]
        
        # Cloud Platforms
        cloud_platforms = [
            ("aws", "Amazon Web Services", TechNodeType.PLATFORM, "Cloud computing platform", "cloud", 0.9),
            ("azure", "Microsoft Azure", TechNodeType.PLATFORM, "Microsoft's cloud platform", "cloud", 0.8),
            ("gcp", "Google Cloud Platform", TechNodeType.PLATFORM, "Google's cloud platform", "cloud", 0.7),
            ("digitalocean", "DigitalOcean", TechNodeType.PLATFORM, "Cloud infrastructure provider", "cloud", 0.5),
            ("heroku", "Heroku", TechNodeType.PLATFORM, "Platform as a service", "cloud", 0.6)
        ]
        
        # Databases
        databases = [
            ("mysql", "MySQL", TechNodeType.TOOL, "Relational database", "database", 0.8),
            ("postgresql", "PostgreSQL", TechNodeType.TOOL, "Advanced relational database", "database", 0.7),
            ("mongodb", "MongoDB", TechNodeType.TOOL, "NoSQL document database", "database", 0.8),
            ("redis", "Redis", TechNodeType.TOOL, "In-memory data store", "database", 0.7),
            ("elasticsearch", "Elasticsearch", TechNodeType.TOOL, "Search and analytics engine", "database", 0.6),
            ("cassandra", "Apache Cassandra", TechNodeType.TOOL, "Distributed NoSQL database", "database", 0.4),
            ("sqlite", "SQLite", TechNodeType.TOOL, "Embedded SQL database", "database", 0.6)
        ]
        
        # Add all nodes
        all_nodes = languages + web_frameworks + ai_frameworks + mobile_frameworks + devops_tools + cloud_platforms + databases
        
        for node_id, name, node_type, description, domain, popularity in all_nodes:
            self.add_node(
                node_id=node_id,
                name=name,
                node_type=node_type,
                description=description,
                domain=domain,
                popularity_score=popularity
            )
        
        # Add relationships
        self._add_initial_relationships()
    
    def _add_initial_relationships(self):
        """Add initial relationships between technologies"""
        
        # Web framework relationships
        web_relationships = [
            ("react", "javascript", RelationshipType.BUILT_WITH, 0.9),
            ("vue", "javascript", RelationshipType.BUILT_WITH, 0.9),
            ("angular", "typescript", RelationshipType.BUILT_WITH, 0.8),
            ("nextjs", "react", RelationshipType.EXTENDS, 0.9),
            ("nuxtjs", "vue", RelationshipType.EXTENDS, 0.9),
            ("express", "javascript", RelationshipType.BUILT_WITH, 0.9),
            ("django", "python", RelationshipType.BUILT_WITH, 0.9),
            ("flask", "python", RelationshipType.BUILT_WITH, 0.9),
            ("spring", "java", RelationshipType.BUILT_WITH, 0.9),
            ("laravel", "php", RelationshipType.BUILT_WITH, 0.9),
            ("rails", "ruby", RelationshipType.BUILT_WITH, 0.9)
        ]
        
        # AI/ML relationships
        ai_relationships = [
            ("tensorflow", "python", RelationshipType.BUILT_WITH, 0.9),
            ("pytorch", "python", RelationshipType.BUILT_WITH, 0.9),
            ("keras", "tensorflow", RelationshipType.BUILT_WITH, 0.8),
            ("scikit-learn", "python", RelationshipType.BUILT_WITH, 0.9),
            ("pandas", "python", RelationshipType.BUILT_WITH, 0.9),
            ("numpy", "python", RelationshipType.BUILT_WITH, 0.9),
            ("matplotlib", "python", RelationshipType.BUILT_WITH, 0.9),
            ("seaborn", "matplotlib", RelationshipType.BUILT_WITH, 0.8),
            ("opencv", "python", RelationshipType.BUILT_WITH, 0.8),
            ("spacy", "python", RelationshipType.BUILT_WITH, 0.9),
            ("nltk", "python", RelationshipType.BUILT_WITH, 0.9),
            ("transformers", "pytorch", RelationshipType.BUILT_WITH, 0.8)
        ]
        
        # Mobile framework relationships
        mobile_relationships = [
            ("react-native", "javascript", RelationshipType.BUILT_WITH, 0.8),
            ("flutter", "dart", RelationshipType.BUILT_WITH, 0.9),
            ("xamarin", "csharp", RelationshipType.BUILT_WITH, 0.9),
            ("ionic", "javascript", RelationshipType.BUILT_WITH, 0.8),
            ("cordova", "javascript", RelationshipType.BUILT_WITH, 0.8)
        ]
        
        # DevOps relationships
        devops_relationships = [
            ("kubernetes", "docker", RelationshipType.USES, 0.9),
            ("jenkins", "docker", RelationshipType.USES, 0.7),
            ("gitlab", "git", RelationshipType.USES, 0.9),
            ("github", "git", RelationshipType.USES, 0.9),
            ("terraform", "aws", RelationshipType.USES, 0.8),
            ("terraform", "azure", RelationshipType.USES, 0.8),
            ("terraform", "gcp", RelationshipType.USES, 0.8),
            ("ansible", "python", RelationshipType.BUILT_WITH, 0.8)
        ]
        
        # Cloud relationships
        cloud_relationships = [
            ("aws", "docker", RelationshipType.USES, 0.8),
            ("azure", "docker", RelationshipType.USES, 0.8),
            ("gcp", "docker", RelationshipType.USES, 0.8),
            ("heroku", "git", RelationshipType.USES, 0.9)
        ]
        
        # Database relationships
        database_relationships = [
            ("django", "postgresql", RelationshipType.USES, 0.8),
            ("django", "mysql", RelationshipType.USES, 0.7),
            ("rails", "postgresql", RelationshipType.USES, 0.8),
            ("rails", "mysql", RelationshipType.USES, 0.7),
            ("spring", "mysql", RelationshipType.USES, 0.8),
            ("spring", "postgresql", RelationshipType.USES, 0.8),
            ("laravel", "mysql", RelationshipType.USES, 0.8),
            ("laravel", "postgresql", RelationshipType.USES, 0.7)
        ]
        
        # Alternative relationships
        alternative_relationships = [
            ("vue", "react", RelationshipType.ALTERNATIVE_TO, 0.7),
            ("angular", "react", RelationshipType.ALTERNATIVE_TO, 0.7),
            ("flutter", "react-native", RelationshipType.ALTERNATIVE_TO, 0.8),
            ("azure", "aws", RelationshipType.ALTERNATIVE_TO, 0.8),
            ("gcp", "aws", RelationshipType.ALTERNATIVE_TO, 0.7),
            ("mongodb", "mysql", RelationshipType.ALTERNATIVE_TO, 0.6),
            ("postgresql", "mysql", RelationshipType.ALTERNATIVE_TO, 0.7)
        ]
        
        # Add all relationships
        all_relationships = (web_relationships + ai_relationships + mobile_relationships + 
                           devops_relationships + cloud_relationships + database_relationships + 
                           alternative_relationships)
        
        for source, target, rel_type, strength in all_relationships:
            # Only add relationship if both nodes exist
            if source in self.nodes and target in self.nodes:
                self.add_relationship(
                    source_id=source,
                    target_id=target,
                    relationship_type=rel_type,
                    strength=strength,
                    context=f"Initial relationship between {source} and {target}"
                )
    
    def add_node(
        self,
        node_id: str,
        name: str,
        node_type: TechNodeType,
        description: str,
        domain: str,
        popularity_score: float = 0.5,
        metadata: Dict[str, Any] = None
    ) -> TechNode:
        """Add a new technology node to the graph"""
        
        if metadata is None:
            metadata = {}
        
        node = TechNode(
            id=node_id,
            name=name,
            node_type=node_type,
            description=description,
            domain=domain,
            popularity_score=popularity_score,
            metadata=metadata,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **asdict(node))
        self.domain_clusters[domain].add(node_id)
        
        logger.info(f"Added node: {name} ({node_type.value})")
        return node
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        strength: float = 0.5,
        context: str = "",
        metadata: Dict[str, Any] = None
    ) -> TechRelationship:
        """Add a relationship between two technology nodes"""
        
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both source and target nodes must exist")
        
        if metadata is None:
            metadata = {}
        
        relationship = TechRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            strength=strength,
            context=context,
            metadata=metadata,
            created_at=datetime.now()
        )
        
        self.relationships.append(relationship)
        self.graph.add_edge(source_id, target_id, **asdict(relationship))
        
        logger.info(f"Added relationship: {source_id} -> {target_id} ({relationship_type.value})")
        return relationship
    
    def find_related_technologies(
        self,
        node_id: str,
        relationship_types: List[RelationshipType] = None,
        max_depth: int = 2,
        min_strength: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """Find technologies related to a given node"""
        
        if node_id not in self.nodes:
            return []
        
        related = []
        
        if relationship_types is None:
            relationship_types = list(RelationshipType)
        
        # Find direct relationships
        for rel in self.relationships:
            if (rel.source_id == node_id and rel.relationship_type in relationship_types and 
                rel.strength >= min_strength):
                related.append((rel.target_id, rel.relationship_type.value, rel.strength))
            elif (rel.target_id == node_id and rel.relationship_type in relationship_types and 
                  rel.strength >= min_strength):
                related.append((rel.source_id, rel.relationship_type.value, rel.strength))
        
        # Find indirect relationships (up to max_depth)
        if max_depth > 1:
            try:
                # Use NetworkX to find paths
                for target_id in self.nodes:
                    if target_id != node_id:
                        try:
                            path_length = nx.shortest_path_length(self.graph, node_id, target_id)
                            if path_length <= max_depth and path_length > 1:
                                # Calculate path strength
                                path_strength = self._calculate_path_strength(node_id, target_id)
                                if path_strength >= min_strength:
                                    related.append((target_id, f"indirect_{path_length}", path_strength))
                        except nx.NetworkXNoPath:
                            continue
            except Exception as e:
                logger.error(f"Error finding indirect relationships: {e}")
        
        # Sort by strength
        related.sort(key=lambda x: x[2], reverse=True)
        return related
    
    def _calculate_path_strength(self, source_id: str, target_id: str) -> float:
        """Calculate the strength of a path between two nodes"""
        
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            if len(path) < 2:
                return 0.0
            
            strength = 1.0
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                strength *= edge_data.get('strength', 0.5)
            
            return strength
        except nx.NetworkXNoPath:
            return 0.0
    
    def get_technology_ecosystem(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get the complete ecosystem around a technology"""
        
        if node_id not in self.nodes:
            return {}
        
        node = self.nodes[node_id]
        ecosystem = {
            'central_technology': {
                'id': node.id,
                'name': node.name,
                'type': node.node_type.value,
                'domain': node.domain,
                'popularity': node.popularity_score,
                'description': node.description
            },
            'dependencies': [],
            'dependents': [],
            'alternatives': [],
            'compatible_with': [],
            'used_by': [],
            'uses': [],
            'ecosystem_strength': 0.0
        }
        
        # Categorize relationships
        for rel in self.relationships:
            if rel.source_id == node_id:
                if rel.relationship_type == RelationshipType.DEPENDS_ON:
                    ecosystem['dependencies'].append({
                        'id': rel.target_id,
                        'name': self.nodes[rel.target_id].name,
                        'strength': rel.strength
                    })
                elif rel.relationship_type == RelationshipType.USES:
                    ecosystem['uses'].append({
                        'id': rel.target_id,
                        'name': self.nodes[rel.target_id].name,
                        'strength': rel.strength
                    })
                elif rel.relationship_type == RelationshipType.COMPATIBLE_WITH:
                    ecosystem['compatible_with'].append({
                        'id': rel.target_id,
                        'name': self.nodes[rel.target_id].name,
                        'strength': rel.strength
                    })
            
            elif rel.target_id == node_id:
                if rel.relationship_type == RelationshipType.DEPENDS_ON:
                    ecosystem['dependents'].append({
                        'id': rel.source_id,
                        'name': self.nodes[rel.source_id].name,
                        'strength': rel.strength
                    })
                elif rel.relationship_type == RelationshipType.USES:
                    ecosystem['used_by'].append({
                        'id': rel.source_id,
                        'name': self.nodes[rel.source_id].name,
                        'strength': rel.strength
                    })
                elif rel.relationship_type == RelationshipType.ALTERNATIVE_TO:
                    ecosystem['alternatives'].append({
                        'id': rel.source_id,
                        'name': self.nodes[rel.source_id].name,
                        'strength': rel.strength
                    })
        
        # Calculate ecosystem strength
        ecosystem['ecosystem_strength'] = self._calculate_ecosystem_strength(node_id)
        
        return ecosystem
    
    def _calculate_ecosystem_strength(self, node_id: str) -> float:
        """Calculate the strength of a technology's ecosystem"""
        
        if node_id not in self.nodes:
            return 0.0
        
        # Count connections
        in_degree = self.graph.in_degree(node_id)
        out_degree = self.graph.out_degree(node_id)
        
        # Calculate average relationship strength
        total_strength = 0.0
        relationship_count = 0
        
        for rel in self.relationships:
            if rel.source_id == node_id or rel.target_id == node_id:
                total_strength += rel.strength
                relationship_count += 1
        
        avg_strength = total_strength / relationship_count if relationship_count > 0 else 0.0
        
        # Combine metrics
        ecosystem_strength = (in_degree + out_degree) * 0.1 + avg_strength * 0.9
        return min(1.0, ecosystem_strength)
    
    def find_learning_path(
        self,
        start_technology: str,
        target_technology: str,
        max_steps: int = 5
    ) -> List[Dict[str, Any]]:
        """Find a learning path from one technology to another"""
        
        if start_technology not in self.nodes or target_technology not in self.nodes:
            return []
        
        try:
            # Find shortest path
            path = nx.shortest_path(self.graph, start_technology, target_technology)
            
            if len(path) > max_steps:
                return []
            
            learning_path = []
            for i, node_id in enumerate(path):
                node = self.nodes[node_id]
                
                step_info = {
                    'step': i + 1,
                    'technology': {
                        'id': node.id,
                        'name': node.name,
                        'type': node.node_type.value,
                        'domain': node.domain,
                        'description': node.description
                    },
                    'relationship': None,
                    'strength': 0.0
                }
                
                # Add relationship info if not the first step
                if i > 0:
                    prev_node_id = path[i - 1]
                    for rel in self.relationships:
                        if ((rel.source_id == prev_node_id and rel.target_id == node_id) or
                            (rel.source_id == node_id and rel.target_id == prev_node_id)):
                            step_info['relationship'] = rel.relationship_type.value
                            step_info['strength'] = rel.strength
                            break
                
                learning_path.append(step_info)
            
            return learning_path
            
        except nx.NetworkXNoPath:
            return []
    
    def get_domain_technologies(self, domain: str) -> List[Dict[str, Any]]:
        """Get all technologies in a specific domain"""
        
        domain_technologies = []
        
        for node_id in self.domain_clusters.get(domain, []):
            if node_id in self.nodes:
                node = self.nodes[node_id]
                domain_technologies.append({
                    'id': node.id,
                    'name': node.name,
                    'type': node.node_type.value,
                    'description': node.description,
                    'popularity': node.popularity_score,
                    'ecosystem_strength': self._calculate_ecosystem_strength(node_id)
                })
        
        # Sort by popularity and ecosystem strength
        domain_technologies.sort(key=lambda x: (x['popularity'], x['ecosystem_strength']), reverse=True)
        
        return domain_technologies
    
    def find_trending_technologies(self, domain: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Find trending technologies based on ecosystem strength and popularity"""
        
        trending = []
        
        for node_id, node in self.nodes.items():
            if domain and node.domain != domain:
                continue
            
            ecosystem_strength = self._calculate_ecosystem_strength(node_id)
            trend_score = node.popularity_score * 0.6 + ecosystem_strength * 0.4
            
            trending.append({
                'id': node.id,
                'name': node.name,
                'type': node.node_type.value,
                'domain': node.domain,
                'trend_score': trend_score,
                'popularity': node.popularity_score,
                'ecosystem_strength': ecosystem_strength
            })
        
        # Sort by trend score
        trending.sort(key=lambda x: x['trend_score'], reverse=True)
        
        return trending[:limit]
    
    def suggest_technologies(
        self,
        current_technologies: List[str],
        target_domain: str = None,
        max_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """Suggest technologies based on current tech stack"""
        
        suggestions = []
        current_set = set(current_technologies)
        
        # Find technologies related to current stack
        related_tech = set()
        for tech in current_technologies:
            if tech in self.nodes:
                related = self.find_related_technologies(tech, max_depth=2)
                for related_id, _, strength in related:
                    if related_id not in current_set and strength >= 0.5:
                        related_tech.add(related_id)
        
        # Score suggestions
        for tech_id in related_tech:
            if tech_id in self.nodes:
                node = self.nodes[tech_id]
                
                if target_domain and node.domain != target_domain:
                    continue
                
                # Calculate suggestion score
                score = 0.0
                for current_tech in current_technologies:
                    if current_tech in self.nodes:
                        related = self.find_related_technologies(current_tech)
                        for related_id, rel_type, strength in related:
                            if related_id == tech_id:
                                score += strength
                
                suggestions.append({
                    'id': node.id,
                    'name': node.name,
                    'type': node.node_type.value,
                    'domain': node.domain,
                    'description': node.description,
                    'suggestion_score': score,
                    'popularity': node.popularity_score
                })
        
        # Sort by suggestion score
        suggestions.sort(key=lambda x: x['suggestion_score'], reverse=True)
        
        return suggestions[:max_suggestions]
    
    def export_graph(self) -> Dict[str, Any]:
        """Export the knowledge graph as a dictionary"""
        
        return {
            'nodes': {node_id: asdict(node) for node_id, node in self.nodes.items()},
            'relationships': [asdict(rel) for rel in self.relationships],
            'domain_clusters': {domain: list(nodes) for domain, nodes in self.domain_clusters.items()},
            'metadata': {
                'total_nodes': len(self.nodes),
                'total_relationships': len(self.relationships),
                'domains': list(self.domain_clusters.keys()),
                'exported_at': datetime.now().isoformat()
            }
        }
    
    def import_graph(self, graph_data: Dict[str, Any]):
        """Import a knowledge graph from a dictionary"""
        
        # Clear existing data
        self.nodes.clear()
        self.relationships.clear()
        self.domain_clusters.clear()
        self.graph.clear()
        
        # Import nodes
        for node_id, node_data in graph_data.get('nodes', {}).items():
            node = TechNode(**node_data)
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **asdict(node))
            self.domain_clusters[node.domain].add(node_id)
        
        # Import relationships
        for rel_data in graph_data.get('relationships', []):
            rel = TechRelationship(**rel_data)
            self.relationships.append(rel)
            self.graph.add_edge(rel.source_id, rel.target_id, **asdict(rel))
        
        logger.info(f"Imported graph with {len(self.nodes)} nodes and {len(self.relationships)} relationships")
