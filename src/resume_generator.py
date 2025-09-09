#!/usr/bin/env python3
"""
Resume Generator for Tailored Resumes
Creates tailored resume files with company name prefixes
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import re

from llm_providers import TailoringResponse

logger = logging.getLogger(__name__)

class ResumeGenerator:
    """Generates tailored resume files based on LLM responses"""
    
    def __init__(self, base_resume_path: str = "resume/", output_format: str = "txt"):
        self.base_resume_path = Path(base_resume_path)
        self.output_format = output_format.lower()
        
        # Ensure resume directory exists
        self.base_resume_path.mkdir(parents=True, exist_ok=True)
        
        # Base resume template
        self.base_resume_template = """Syeda Mariah Banu
mariaahbanu@gmail.com | +1 (636) 399-0293 | linkedin.com/in/mariah-banu

MACHINE LEARNING ENGINEER — {focus_areas}

{summary}

CORE SKILLS
{skills_section}

EXPERIENCE
{experience_section}

SELECTED PROJECTS
• CV Inference Service (Production): PyTorch → ONNX/TensorRT with FastAPI on EKS; request batching and autoscaling deliver near real-time p95 latency with high throughput; SLO dashboards and rollback playbooks improve operational safety.

• LLM Agent for Report Automation: Prompt toolkit with tool-calling and retrieval that substantially reduces analyst review time and improves output consistency; guardrails and evaluation harness.

• Data Prep & Evaluation Suite: Reproducible curation/augmentation pipelines plus ablation harnesses that materially shorten iteration cycles and improve experiment quality.

• RAG Knowledge Assistant: Retrieval-augmented QA over internal docs with guardrails and eval harness; noticeably improves answer relevance and reduces search time.

EDUCATION
M.S., Computer Science (CV/ML) — Southern Illinois University Edwardsville — GPA 4.0

CERTIFICATIONS
• IBM Data Science Professional Certificate
• AWS Solutions Architect – Associate (in progress)
• AWS Certified Machine Learning Engineer – Associate (in progress)

AWARDS & PUBLICATIONS
• MDPI: Journal of Imaging (Plant Phenotyping)
• Authorea: GCNet yield estimation
• NAPPN People's Choice — Top Fast Forward Speaker
"""
        
        # Base experience bullets (will be reordered based on tailoring)
        self.base_experience = {
            "comfortliv": [
                "Delivered end-to-end CV classification pipeline (data curation → PyTorch training → FastAPI inference on EKS) with significant accuracy and recall improvement in production.",
                "Built LLM/agentic workflows (prompting, tool-calling, RAG) for data extraction, annotation/labeling, and report generation; dramatically reduced manual effort and sped turnaround.",
                "Productionized inference with autoscaling and request shaping for near real-time latency and high-throughput workloads.",
                "Implemented MLflow experiments and model registry to substantially shorten iteration and deployment time.",
                "Authored deployment runbooks/checklists that improve stability and reduce change-related issues."
            ],
            "cedar_bluff": [
                "Led CV pipeline (annotation → training/fine-tuning → real-time inference); improved internal benchmarks from mAP@50 21.1% / Precision 17.8% / Recall 5.7% → mAP@50 99.2% / Precision 98.4% / Recall 96.7% on internal dataset.",
                "Operated EKS-hosted inference with feature-flagged releases and automated rollback; raised reliability and reduced incident risk.",
                "Delivered a stakeholder-facing product demo in 10 days, resulting in funding approval and a new client acquisition.",
                "Built dashboards and on-call runbooks that accelerated triage and improved overall service stability."
            ],
            "siue": [
                "Created a novel segmentation framework integrating deep learning techniques with correction factors to overcome estimation challenges.",
                "Improved yield estimation accuracy, achieving R^2 = 0.96 and reducing mean absolute error (MAE) by 10% compared to previous methods.",
                "Built reproducible training/evaluation harness with ablations and cross-validation; improved experimental clarity and decisions.",
                "Implemented lightweight APIs exposing datasets, predictions, and metrics to collaborators; significantly reduced time-to-insight."
            ],
            "charter": [
                "Automated 103 BDD cases (Cucumber, Selenium, REST Assured) in CI/CD; significantly reduced regression escapes and enabled rapid releases.",
                "Shipped Elasticsearch-based release-health and failure-triage dashboards; materially shortened detection-to-mitigation time.",
                "Implemented DynamoDB token-bucket rate limiting to protect downstream services during traffic spikes; improved resiliency.",
                "Integrated Okta (OIDC, SCIM, JIT) into test flows; reduced auth-related flakiness."
            ],
            "servicenow": [
                "Built Java/Spring Boot microservices for Virtual Agent on Kubernetes with Kafka; resiliency patterns improved responsiveness and service reliability.",
                "Standardized canary releases and rollback runbooks; deployment time decreased about 30% and release safety improved.",
                "Participated in on-call and authored knowledge-base articles; reduced repeat incidents through proactive documentation and fixes."
            ]
        }
    
    def sanitize_filename(self, company_name: str) -> str:
        """Sanitize company name for use in filename"""
        # Remove special characters and spaces, keep alphanumeric and hyphens
        sanitized = re.sub(r'[^a-zA-Z0-9\-_]', '_', company_name)
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Limit length
        return sanitized[:50]
    
    def generate_focus_areas(self, job_focus: str, tailoring_response: TailoringResponse) -> str:
        """Generate focus areas line based on job focus"""
        focus_map = {
            "cv": "Computer Vision · LLM/Agentic · FastAPI/Kubernetes · MLflow · AWS",
            "llm": "LLM/Agentic · RAG · Computer Vision · FastAPI/Kubernetes · MLflow · AWS", 
            "mlops": "MLOps/Kubernetes · FastAPI · Computer Vision · LLM/Agentic · MLflow · AWS",
            "research": "Research · Computer Vision · LLM/Agentic · FastAPI/Kubernetes · MLflow",
            "general": "Computer Vision · LLM/Agentic · FastAPI/Kubernetes · MLflow · AWS"
        }
        
        return focus_map.get(job_focus, focus_map["general"])
    
    def generate_skills_section(self, tailoring_response: TailoringResponse) -> str:
        """Generate skills section with emphasized skills first"""
        # Get emphasized skills
        emphasized_skills = tailoring_response.skills_reordered[:10]
        
        # Base skills categorized
        base_skills = {
            "Languages": ["Python", "SQL", "(Java)", "Bash"],
            "ML/DL": ["PyTorch", "TensorFlow/Keras", "scikit-learn", "XGBoost", "LightGBM", "Hugging Face"],
            "MLOps/Platforms": ["MLflow", "Airflow", "Kubeflow", "SageMaker", "Vertex AI", "Azure ML", "Weights & Biases"],
            "Serving/Systems": ["FastAPI", "Docker", "Kubernetes/EKS", "KServe/Seldon", "BentoML", "ONNX/TensorRT", "gRPC"],
            "Data/Compute": ["Pandas/NumPy", "Spark/Ray", "Parquet", "Feature Store (Feast)", "Data Quality (Great Expectations)"],
            "LLM/RAG": ["Transformers", "PEFT/LoRA", "retrieval pipelines", "vector DBs (FAISS/Pinecone)", "LangChain/LlamaIndex", "prompt/eval harnesses"],
            "Experimentation & Eval": ["offline/online eval", "A/B testing", "guardrails", "drift detection (Evidently)"],
            "Observability & Ops": ["OpenTelemetry", "Prometheus/Grafana", "Datadog", "Splunk", "incident response/on-call"],
            "DevOps/Cloud": ["AWS (EKS, S3, Lambda, DynamoDB)", "GCP", "Azure", "Terraform/Helm", "Argo CD", "CI/CD (GitHub Actions/Jenkins)"]
        }
        
        # Build skills section with emphasis
        skills_lines = []
        for category, skills in base_skills.items():
            # Highlight emphasized skills in this category
            highlighted_skills = []
            for skill in skills:
                skill_clean = skill.lower().replace('(', '').replace(')', '')
                if any(emp_skill.lower() in skill_clean for emp_skill in emphasized_skills):
                    highlighted_skills.append(f"**{skill}**")  # Bold emphasis
                else:
                    highlighted_skills.append(skill)
            
            skills_line = f"{category}: {', '.join(highlighted_skills)}"
            skills_lines.append(skills_line)
        
        return '\n'.join(skills_lines)
    
    def generate_experience_section(self, tailoring_response: TailoringResponse) -> str:
        """Generate experience section with prioritized bullets"""
        # Use the prioritized experience bullets from tailoring response
        prioritized_bullets = tailoring_response.experience_bullets
        
        # Company sections with titles
        experience_sections = [
            ("Comfortliv LLC — Machine Learning Engineer • May 2025 – Present", "comfortliv"),
            ("Cedar Bluff Technologies — Computer Vision Lead • Jul 2024 – May 2025", "cedar_bluff"),
            ("Southern Illinois University Edwardsville — Research Assistant (Plant Vision Lab) • Jan 2023 – May 2024", "siue"),
            ("Charter Communications (Consultant via Infosys) — Development & Automation Engineer • Aug 2023 – Dec 2023", "charter"),
            ("ServiceNow — Software Engineer • Jul 2020 – May 2021; Jul 2021 – Jul 2022", "servicenow")
        ]
        
        experience_text = []
        used_bullets = set()
        
        for title, company_key in experience_sections:
            experience_text.append(title)
            
            # Get bullets for this company, prioritizing the tailored ones
            company_bullets = self.base_experience.get(company_key, [])
            bullets_to_use = []
            
            # First, add prioritized bullets that belong to this company
            for bullet in prioritized_bullets:
                if bullet not in used_bullets:
                    # Check if this bullet belongs to this company
                    for company_bullet in company_bullets:
                        if self._bullets_similar(bullet, company_bullet):
                            bullets_to_use.append(bullet)
                            used_bullets.add(bullet)
                            break
            
            # Then add remaining company bullets if we need more
            for bullet in company_bullets:
                if len(bullets_to_use) < 3 and bullet not in used_bullets:  # Max 3 bullets per company
                    bullets_to_use.append(bullet)
                    used_bullets.add(bullet)
            
            # Add bullets with proper formatting
            for bullet in bullets_to_use[:3]:  # Max 3 per company
                experience_text.append(f"• {bullet}")
            
            experience_text.append("")  # Empty line between companies
        
        return '\n'.join(experience_text)
    
    def _bullets_similar(self, bullet1: str, bullet2: str, threshold: float = 0.7) -> bool:
        """Check if two bullets are similar (simple word overlap)"""
        words1 = set(bullet1.lower().split())
        words2 = set(bullet2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2) / len(words1 | words2)
        return overlap >= threshold
    
    def generate_tailored_resume(
        self, 
        company: str, 
        job_title: str,
        job_focus: str,
        tailoring_response: TailoringResponse,
        job_id: str = None
    ) -> str:
        """Generate a complete tailored resume"""
        
        # Generate sections
        focus_areas = self.generate_focus_areas(job_focus, tailoring_response)
        skills_section = self.generate_skills_section(tailoring_response)
        experience_section = self.generate_experience_section(tailoring_response)
        
        # Fill template
        tailored_resume = self.base_resume_template.format(
            focus_areas=focus_areas,
            summary=tailoring_response.summary,
            skills_section=skills_section,
            experience_section=experience_section
        )
        
        # Add tailoring metadata as comment at the end
        metadata = f"""

<!-- TAILORING METADATA
Company: {company}
Job Title: {job_title}
Job Focus: {job_focus}
Job ID: {job_id or 'N/A'}
Confidence: {tailoring_response.confidence:.2f}
Keywords Added: {', '.join(tailoring_response.keywords_added)}
Reasoning: {tailoring_response.reasoning}
-->"""
        
        return tailored_resume + metadata
    
    def save_tailored_resume(
        self,
        company: str,
        job_title: str,
        job_focus: str,
        tailoring_response: TailoringResponse,
        job_id: str = None
    ) -> str:
        """Save tailored resume to file with company prefix"""
        
        # Generate resume content
        resume_content = self.generate_tailored_resume(
            company, job_title, job_focus, tailoring_response, job_id
        )
        
        # Generate filename
        company_clean = self.sanitize_filename(company)
        job_title_clean = self.sanitize_filename(job_title)
        
        if job_id:
            filename = f"{company_clean}_{job_title_clean}_{job_id[:8]}.{self.output_format}"
        else:
            filename = f"{company_clean}_{job_title_clean}.{self.output_format}"
        
        # Full path
        file_path = self.base_resume_path / filename
        
        # Save file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(resume_content)
            
            logger.info(f"Saved tailored resume: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save resume to {file_path}: {str(e)}")
            raise
    
    def get_existing_resumes(self) -> Dict[str, str]:
        """Get list of existing tailored resumes"""
        resumes = {}
        
        if not self.base_resume_path.exists():
            return resumes
        
        for file_path in self.base_resume_path.glob(f"*.{self.output_format}"):
            if file_path.name.startswith('.'):  # Skip hidden files
                continue
            
            # Extract company name from filename
            company_name = file_path.stem.split('_')[0]
            resumes[company_name] = str(file_path)
        
        return resumes
    
    def cleanup_old_resumes(self, keep_latest: int = 10):
        """Clean up old resume files, keeping only the latest N"""
        if not self.base_resume_path.exists():
            return
        
        # Get all resume files sorted by modification time
        resume_files = []
        for file_path in self.base_resume_path.glob(f"*.{self.output_format}"):
            if not file_path.name.startswith('.'):
                resume_files.append((file_path.stat().st_mtime, file_path))
        
        # Sort by modification time (newest first)
        resume_files.sort(reverse=True)
        
        # Delete old files beyond the limit
        for _, file_path in resume_files[keep_latest:]:
            try:
                file_path.unlink()
                logger.info(f"Cleaned up old resume: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {str(e)}")
    
    def validate_resume_content(self, resume_content: str) -> bool:
        """Validate that resume content looks reasonable"""
        required_sections = [
            "Syeda Mariah Banu",
            "MACHINE LEARNING ENGINEER",
            "CORE SKILLS",
            "EXPERIENCE",
            "EDUCATION"
        ]
        
        for section in required_sections:
            if section not in resume_content:
                logger.warning(f"Missing required section: {section}")
                return False
        
        # Check minimum length
        if len(resume_content) < 1000:
            logger.warning("Resume content seems too short")
            return False
        
        return True
