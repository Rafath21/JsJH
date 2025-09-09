#!/usr/bin/env python3
"""
Resume Parser for Mariah's Resume
Extracts structured sections for LLM tailoring
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ResumeParser:
    """Parser for Mariah's resume content"""
    
    def __init__(self):
        # Mariah's resume content (structured)
        self.resume_data = {
            "summary": """Ships production Computer Vision and LLM/agentic systems; designs reliable training/serving pipelines, evaluation harnesses, and autoscaled inference on Kubernetes/EKS. Partners with product/SRE on SLOs, rollback safety, and observability; improves model quality and iteration speed while balancing cost/latency tradeoffs. Builds FastAPI services, MLflow registries, and request shaping for near real-time latency and high throughput; drives guardrails, drift detection, and incident response quality.""",
            
            "skills": [
                # Languages
                "Python", "SQL", "Java", "Bash",
                # ML/DL
                "PyTorch", "TensorFlow", "Keras", "scikit-learn", "XGBoost", "LightGBM", "Hugging Face",
                # MLOps/Platforms  
                "MLflow", "Airflow", "Kubeflow", "SageMaker", "Vertex AI", "Azure ML", "Weights & Biases",
                # Serving/Systems
                "FastAPI", "Docker", "Kubernetes", "EKS", "KServe", "Seldon", "BentoML", "ONNX", "TensorRT", "gRPC",
                # Data/Compute
                "Pandas", "NumPy", "Spark", "Ray", "Parquet", "Feature Store", "Feast", "Great Expectations",
                # LLM/RAG
                "Transformers", "PEFT", "LoRA", "retrieval pipelines", "vector DBs", "FAISS", "Pinecone", 
                "LangChain", "LlamaIndex", "prompt engineering", "evaluation harnesses",
                # Experimentation & Eval
                "A/B testing", "guardrails", "drift detection", "Evidently",
                # Observability & Ops
                "OpenTelemetry", "Prometheus", "Grafana", "Datadog", "Splunk", "incident response",
                # DevOps/Cloud
                "AWS", "EKS", "S3", "Lambda", "DynamoDB", "GCP", "Azure", "Terraform", "Helm", "Argo CD", 
                "CI/CD", "GitHub Actions", "Jenkins"
            ],
            
            "experience_bullets": [
                # Comfortliv LLC
                "Delivered end-to-end CV classification pipeline (data curation → PyTorch training → FastAPI inference on EKS) with significant accuracy and recall improvement in production",
                "Built LLM/agentic workflows (prompting, tool-calling, RAG) for data extraction, annotation/labeling, and report generation; dramatically reduced manual effort and sped turnaround",
                "Productionized inference with autoscaling and request shaping for near real-time latency and high-throughput workloads",
                "Implemented MLflow experiments and model registry to substantially shorten iteration and deployment time",
                "Authored deployment runbooks/checklists that improve stability and reduce change-related issues",
                
                # Cedar Bluff Technologies
                "Led CV pipeline (annotation → training/fine-tuning → real-time inference); improved internal benchmarks from mAP@50 21.1% / Precision 17.8% / Recall 5.7% → mAP@50 99.2% / Precision 98.4% / Recall 96.7% on internal dataset",
                "Operated EKS-hosted inference with feature-flagged releases and automated rollback; raised reliability and reduced incident risk",
                "Delivered a stakeholder-facing product demo in 10 days, resulting in funding approval and a new client acquisition",
                "Built dashboards and on-call runbooks that accelerated triage and improved overall service stability",
                
                # SIUE Research
                "Created a novel segmentation framework integrating deep learning techniques with correction factors to overcome estimation challenges",
                "Improved yield estimation accuracy, achieving R^2 = 0.96 and reducing mean absolute error (MAE) by 10% compared to previous methods",
                "Built reproducible training/evaluation harness with ablations and cross-validation; improved experimental clarity and decisions",
                "Implemented lightweight APIs exposing datasets, predictions, and metrics to collaborators; significantly reduced time-to-insight",
                
                # Charter Communications
                "Automated 103 BDD cases (Cucumber, Selenium, REST Assured) in CI/CD; significantly reduced regression escapes and enabled rapid releases",
                "Shipped Elasticsearch-based release-health and failure-triage dashboards; materially shortened detection-to-mitigation time",
                "Implemented DynamoDB token-bucket rate limiting to protect downstream services during traffic spikes; improved resiliency",
                "Integrated Okta (OIDC, SCIM, JIT) into test flows; reduced auth-related flakiness",
                
                # ServiceNow
                "Built Java/Spring Boot microservices for Virtual Agent on Kubernetes with Kafka; resiliency patterns improved responsiveness and service reliability",
                "Standardized canary releases and rollback runbooks; deployment time decreased about 30% and release safety improved",
                "Participated in on-call and authored knowledge-base articles; reduced repeat incidents through proactive documentation and fixes"
            ],
            
            "projects": [
                "CV Inference Service (Production): PyTorch → ONNX/TensorRT with FastAPI on EKS; request batching and autoscaling deliver near real-time p95 latency with high throughput; SLO dashboards and rollback playbooks improve operational safety",
                "LLM Agent for Report Automation: Prompt toolkit with tool-calling and retrieval that substantially reduces analyst review time and improves output consistency; guardrails and evaluation harness",
                "Data Prep & Evaluation Suite: Reproducible curation/augmentation pipelines plus ablation harnesses that materially shorten iteration cycles and improve experiment quality",
                "RAG Knowledge Assistant: Retrieval-augmented QA over internal docs with guardrails and eval harness; noticeably improves answer relevance and reduces search time"
            ],
            
            "education": "M.S., Computer Science (CV/ML) — Southern Illinois University Edwardsville — GPA 4.0",
            
            "certifications": [
                "IBM Data Science Professional Certificate",
                "AWS Solutions Architect – Associate (in progress)",
                "AWS Certified Machine Learning Engineer – Associate (in progress)"
            ],
            
            "publications": [
                "MDPI: Journal of Imaging (Plant Phenotyping)",
                "Authorea: GCNet yield estimation",
                "NAPPN People's Choice — Top Fast Forward Speaker"
            ]
        }
        
        # Categorized skills for easier tailoring
        self.skill_categories = {
            "ml_frameworks": ["PyTorch", "TensorFlow", "Keras", "scikit-learn", "XGBoost", "LightGBM", "Hugging Face"],
            "computer_vision": ["computer vision", "CV", "object detection", "segmentation", "YOLO", "OpenCV", "ONNX", "TensorRT"],
            "llm_rag": ["Transformers", "PEFT", "LoRA", "LangChain", "LlamaIndex", "RAG", "vector DBs", "FAISS", "Pinecone"],
            "mlops": ["MLflow", "Airflow", "Kubeflow", "SageMaker", "Vertex AI", "Azure ML", "Weights & Biases"],
            "serving": ["FastAPI", "Docker", "Kubernetes", "EKS", "KServe", "Seldon", "BentoML", "gRPC"],
            "cloud_platforms": ["AWS", "GCP", "Azure", "S3", "Lambda", "DynamoDB"],
            "data_tools": ["Pandas", "NumPy", "Spark", "Ray", "Parquet", "Feature Store", "Feast"],
            "observability": ["OpenTelemetry", "Prometheus", "Grafana", "Datadog", "Splunk"],
            "devops": ["Terraform", "Helm", "Argo CD", "CI/CD", "GitHub Actions", "Jenkins"],
            "languages": ["Python", "SQL", "Java", "Bash"]
        }
    
    def get_resume_sections(self) -> Dict[str, Any]:
        """Get all resume sections"""
        return self.resume_data.copy()
    
    def get_skills_by_category(self, category: str) -> List[str]:
        """Get skills for a specific category"""
        return self.skill_categories.get(category, [])
    
    def get_relevant_experience(self, job_focus: str) -> List[str]:
        """Get experience bullets relevant to job focus"""
        focus_keywords = {
            "cv": ["cv", "computer vision", "detection", "segmentation", "map@50", "precision", "recall", "image", "visual"],
            "llm": ["llm", "rag", "agentic", "prompt", "retrieval", "tool-calling", "language model", "generative"],
            "mlops": ["mlflow", "kubernetes", "eks", "deployment", "pipeline", "inference", "serving", "infrastructure"],
            "research": ["research", "novel", "framework", "accuracy", "r^2", "mae", "evaluation", "experimental", "publication"],
            "general": ["machine learning", "model", "training", "production", "pipeline", "system"]
        }
        
        keywords = focus_keywords.get(job_focus.lower(), focus_keywords["general"])
        
        # Score bullets by relevance
        scored_bullets = []
        for bullet in self.resume_data["experience_bullets"]:
            score = sum(1 for keyword in keywords if keyword.lower() in bullet.lower())
            scored_bullets.append((score, bullet))
        
        # Sort by relevance and return
        scored_bullets.sort(key=lambda x: x[0], reverse=True)
        return [bullet for _, bullet in scored_bullets]
    
    def get_relevant_skills(self, job_focus: str) -> List[str]:
        """Get skills relevant to job focus"""
        focus_categories = {
            "cv": ["ml_frameworks", "computer_vision", "serving", "cloud_platforms"],
            "llm": ["ml_frameworks", "llm_rag", "serving", "cloud_platforms"],
            "mlops": ["mlops", "serving", "cloud_platforms", "devops", "observability"],
            "research": ["ml_frameworks", "data_tools", "languages"],
            "general": ["ml_frameworks", "serving", "cloud_platforms", "languages"]
        }
        
        categories = focus_categories.get(job_focus.lower(), focus_categories["general"])
        
        relevant_skills = []
        for category in categories:
            relevant_skills.extend(self.get_skills_by_category(category))
        
        return relevant_skills[:15]  # Top 15 most relevant
    
    def get_tailored_summary(self, job_focus: str) -> str:
        """Get a tailored summary for specific job focus"""
        focus_summaries = {
            "cv": "Ships production Computer Vision systems with 99.2% mAP@50; designs reliable CV training/serving pipelines, evaluation harnesses, and autoscaled inference on Kubernetes/EKS.",
            "llm": "Ships production LLM/agentic systems; designs reliable training/serving pipelines, RAG workflows, and autoscaled inference on Kubernetes/EKS.",
            "mlops": "Ships production ML systems; designs reliable MLOps pipelines, model serving infrastructure, and autoscaled inference on Kubernetes/EKS.",
            "research": "Research-focused Machine Learning Engineer; creates novel frameworks, conducts rigorous evaluations, and publishes in peer-reviewed journals.",
            "general": self.resume_data["summary"]
        }
        
        return focus_summaries.get(job_focus.lower(), focus_summaries["general"])
    
    def get_relevant_projects(self, job_focus: str) -> List[str]:
        """Get projects relevant to job focus"""
        all_projects = self.resume_data["projects"]
        
        focus_keywords = {
            "cv": ["cv", "computer vision", "inference", "onnx", "tensorrt"],
            "llm": ["llm", "rag", "prompt", "tool-calling", "retrieval"],
            "mlops": ["mlflow", "kubernetes", "pipeline", "serving", "infrastructure"],
            "research": ["evaluation", "experimental", "reproducible", "ablation"],
            "general": ["production", "system", "pipeline"]
        }
        
        keywords = focus_keywords.get(job_focus.lower(), focus_keywords["general"])
        
        # Score projects by relevance
        scored_projects = []
        for project in all_projects:
            score = sum(1 for keyword in keywords if keyword.lower() in project.lower())
            scored_projects.append((score, project))
        
        # Sort by relevance
        scored_projects.sort(key=lambda x: x[0], reverse=True)
        return [project for _, project in scored_projects]
    
    def validate_resume_data(self) -> bool:
        """Validate that resume data is complete"""
        required_fields = ["summary", "skills", "experience_bullets", "projects"]
        
        for field in required_fields:
            if field not in self.resume_data or not self.resume_data[field]:
                logger.error(f"Missing required field: {field}")
                return False
        
        logger.info("Resume data validation passed")
        return True
