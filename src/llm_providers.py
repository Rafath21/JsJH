#!/usr/bin/env python3
"""
LLM Provider Abstraction Layer for Resume Tailoring
Supports multiple models with consistent interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
import time
import re
from enum import Enum

class JobFocus(Enum):
    COMPUTER_VISION = "cv"
    LLM_RAG = "llm"
    MLOPS = "mlops"
    RESEARCH = "research"
    GENERAL_ML = "general"

@dataclass
class TailoringRequest:
    job_description: str
    resume_sections: Dict[str, Any]
    job_focus: JobFocus
    job_title: str
    company: str

@dataclass 
class TailoringResponse:
    summary: str
    experience_bullets: List[str]
    skills_reordered: List[str]
    keywords_added: List[str]
    confidence: float
    reasoning: str = ""

class LLMProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self._initialized = False
    
    @abstractmethod
    def _initialize(self):
        """Initialize the model (lazy loading)"""
        pass
    
    @abstractmethod
    def _generate_raw(self, prompt: str, **kwargs) -> str:
        """Generate raw text from prompt"""
        pass
    
    def generate_tailoring(self, request: TailoringRequest) -> TailoringResponse:
        """Main interface method - generates tailored resume content"""
        if not self._initialized:
            self._initialize()
            
        prompt = self._build_tailoring_prompt(request)
        
        try:
            raw_response = self._generate_raw(
                prompt, 
                max_tokens=1024,
                temperature=0.3,
                **self.kwargs
            )
            
            return self._parse_tailoring_response(raw_response, request)
            
        except Exception as e:
            # Fallback response
            return TailoringResponse(
                summary=self._fallback_summary(request),
                experience_bullets=self._fallback_bullets(request),
                skills_reordered=self._fallback_skills(request),
                keywords_added=[],
                confidence=0.1,
                reasoning=f"Error occurred: {str(e)}"
            )
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass
    
    def _build_tailoring_prompt(self, request: TailoringRequest) -> str:
        """Build the tailoring prompt"""
        
        job_keywords = self._extract_job_keywords(request.job_description)
        focus_guidance = self._get_focus_guidance(request.job_focus)
        
        prompt = f"""You are an expert resume tailoring specialist. Your task is to tailor Mariah's resume for a specific job.

JOB DETAILS:
Title: {request.job_title}
Company: {request.company}
Focus Area: {request.job_focus.value}

JOB DESCRIPTION (first 2000 chars):
{request.job_description[:2000]}

CURRENT RESUME SUMMARY:
{request.resume_sections.get('summary', '')}

CURRENT KEY SKILLS:
{', '.join(request.resume_sections.get('skills', [])[:20])}

FOCUS GUIDANCE:
{focus_guidance}

IMPORTANT JOB KEYWORDS IDENTIFIED:
{', '.join(job_keywords[:10])}

Generate tailoring instructions as valid JSON:
{{
    "summary": "Rewritten 2-3 line summary emphasizing {request.job_focus.value} experience and relevant keywords",
    "skills_to_emphasize": ["skill1", "skill2", "skill3", "skill4", "skill5"],
    "keywords_to_include": ["keyword1", "keyword2", "keyword3"],
    "confidence": 0.85,
    "reasoning": "Brief explanation of tailoring decisions"
}}

RULES:
- Keep all content truthful - only emphasize existing experience
- Include 3-5 keywords from job description naturally
- Maintain professional tone
- Focus on quantified achievements when possible
- Prioritize {request.job_focus.value} experience"""

        return prompt
    
    def _extract_job_keywords(self, job_description: str) -> List[str]:
        """Extract key technical terms from job description"""
        # Common ML/AI keywords to look for
        tech_keywords = [
            'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'xgboost',
            'computer vision', 'cv', 'object detection', 'segmentation', 'yolo',
            'llm', 'large language model', 'rag', 'retrieval', 'transformers',
            'mlops', 'mlflow', 'kubeflow', 'airflow', 'kubernetes', 'docker',
            'fastapi', 'rest api', 'microservices', 'aws', 'gcp', 'azure',
            'python', 'sql', 'spark', 'pandas', 'numpy', 'jupyter',
            'machine learning', 'deep learning', 'neural network', 'ai',
            'data science', 'analytics', 'statistics', 'modeling'
        ]
        
        job_lower = job_description.lower()
        found_keywords = []
        
        for keyword in tech_keywords:
            if keyword in job_lower:
                found_keywords.append(keyword)
        
        # Also extract company-specific terms
        company_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', job_description)
        found_keywords.extend(company_terms[:5])
        
        return list(set(found_keywords))[:15]
    
    def _get_focus_guidance(self, job_focus: JobFocus) -> str:
        """Get tailoring guidance based on job focus"""
        guidance = {
            JobFocus.COMPUTER_VISION: "Emphasize CV pipeline work, object detection, segmentation, real-time inference, and model accuracy metrics",
            JobFocus.LLM_RAG: "Highlight LLM/agentic workflows, RAG systems, prompt engineering, and generative AI experience",
            JobFocus.MLOPS: "Focus on MLflow, Kubernetes deployment, CI/CD, model serving, and production infrastructure",
            JobFocus.RESEARCH: "Emphasize research publications, novel methods, experimental design, and academic achievements",
            JobFocus.GENERAL_ML: "Balance all ML experience - modeling, deployment, and business impact"
        }
        return guidance.get(job_focus, guidance[JobFocus.GENERAL_ML])
    
    def _parse_tailoring_response(self, raw_response: str, request: TailoringRequest) -> TailoringResponse:
        """Parse the LLM response into structured format"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            return TailoringResponse(
                summary=response_data.get('summary', self._fallback_summary(request)),
                experience_bullets=self._select_relevant_bullets(request),
                skills_reordered=response_data.get('skills_to_emphasize', [])[:10],
                keywords_added=response_data.get('keywords_to_include', [])[:5],
                confidence=float(response_data.get('confidence', 0.5)),
                reasoning=response_data.get('reasoning', 'Standard tailoring applied')
            )
            
        except Exception as e:
            return TailoringResponse(
                summary=self._fallback_summary(request),
                experience_bullets=self._select_relevant_bullets(request),
                skills_reordered=request.resume_sections.get('skills', [])[:10],
                keywords_added=[],
                confidence=0.3,
                reasoning=f"Parse error: {str(e)}"
            )
    
    def _fallback_summary(self, request: TailoringRequest) -> str:
        """Generate fallback summary if LLM fails"""
        focus_map = {
            JobFocus.COMPUTER_VISION: "Computer Vision",
            JobFocus.LLM_RAG: "LLM/Agentic",
            JobFocus.MLOPS: "MLOps/Infrastructure",
            JobFocus.RESEARCH: "Research",
            JobFocus.GENERAL_ML: "Machine Learning"
        }
        
        focus_area = focus_map.get(request.job_focus, "Machine Learning")
        
        return f"Machine Learning Engineer specializing in {focus_area} systems; designs reliable training/serving pipelines, evaluation harnesses, and autoscaled inference on Kubernetes/EKS."
    
    def _fallback_bullets(self, request: TailoringRequest) -> List[str]:
        """Generate fallback experience bullets"""
        return request.resume_sections.get('experience_bullets', [])[:8]
    
    def _fallback_skills(self, request: TailoringRequest) -> List[str]:
        """Generate fallback skills list"""
        return request.resume_sections.get('skills', [])[:10]
    
    def _select_relevant_bullets(self, request: TailoringRequest) -> List[str]:
        """Select most relevant experience bullets based on job focus"""
        all_bullets = request.resume_sections.get('experience_bullets', [])
        
        # Keywords to prioritize based on job focus
        priority_keywords = {
            JobFocus.COMPUTER_VISION: ['cv', 'computer vision', 'detection', 'segmentation', 'map@50', 'precision', 'recall'],
            JobFocus.LLM_RAG: ['llm', 'rag', 'agentic', 'prompt', 'retrieval', 'tool-calling'],
            JobFocus.MLOPS: ['mlflow', 'kubernetes', 'eks', 'fastapi', 'deployment', 'pipeline', 'inference'],
            JobFocus.RESEARCH: ['research', 'novel', 'framework', 'accuracy', 'r^2', 'mae', 'evaluation'],
            JobFocus.GENERAL_ML: ['machine learning', 'model', 'training', 'production', 'pipeline']
        }
        
        keywords = priority_keywords.get(request.job_focus, [])
        
        # Score bullets by keyword relevance
        scored_bullets = []
        for bullet in all_bullets:
            score = sum(1 for keyword in keywords if keyword.lower() in bullet.lower())
            scored_bullets.append((score, bullet))
        
        # Sort by score and return top bullets
        scored_bullets.sort(key=lambda x: x[0], reverse=True)
        return [bullet for _, bullet in scored_bullets[:8]]

# Job classification function
def classify_job_type(job_description: str, job_title: str) -> JobFocus:
    """Classify job focus based on description and title"""
    desc_lower = (job_description + " " + job_title).lower()
    
    # Check for specific focus areas
    if any(term in desc_lower for term in ['computer vision', 'cv', 'object detection', 'segmentation', 'image']):
        return JobFocus.COMPUTER_VISION
    
    if any(term in desc_lower for term in ['llm', 'large language model', 'rag', 'generative ai', 'chatbot', 'conversational']):
        return JobFocus.LLM_RAG
    
    if any(term in desc_lower for term in ['mlops', 'ml ops', 'deployment', 'infrastructure', 'kubernetes', 'devops']):
        return JobFocus.MLOPS
    
    if any(term in desc_lower for term in ['research', 'scientist', 'phd', 'publication', 'paper']):
        return JobFocus.RESEARCH
    
    return JobFocus.GENERAL_ML
