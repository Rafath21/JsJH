#!/usr/bin/env python3
"""
Model Comparison Framework for Resume Tailoring
Tests multiple LLM models and evaluates their performance
"""

import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import statistics
from pathlib import Path

from llm_providers import LLMProvider, TailoringRequest, TailoringResponse, JobFocus, classify_job_type
from transformers_provider import ModelFactory
from resume_parser import ResumeParser
from resume_generator import ResumeGenerator

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    model_name: str
    job_id: str
    job_title: str
    company: str
    job_focus: str
    response_time: float
    output_quality: Dict[str, float]
    success: bool
    error_msg: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None

@dataclass
class ComparisonResults:
    model_stats: Dict[str, Dict[str, Any]]
    winner: str
    recommendation: str
    detailed_results: List[ModelPerformance]
    test_summary: Dict[str, Any]

class ModelComparator:
    """Compares multiple LLM models on resume tailoring tasks"""
    
    def __init__(self, models_to_test: Optional[List[str]] = None, generate_resumes: bool = True):
        self.models_to_test = models_to_test or ["phi35", "llama32", "qwen25"]
        self.models: Dict[str, LLMProvider] = {}
        self.results: List[ModelPerformance] = []
        self.resume_parser = ResumeParser()
        self.resume_generator = ResumeGenerator() if generate_resumes else None
        self.generated_resumes: List[Dict[str, Any]] = []
        
    def initialize_models(self):
        """Initialize all models for testing"""
        logger.info("Initializing models for comparison...")
        
        for model_key in self.models_to_test:
            try:
                logger.info(f"Initializing {model_key}...")
                model = ModelFactory.create_model(model_key)
                
                # Test if model is available
                if model.is_available():
                    self.models[model_key] = model
                    logger.info(f"✅ {model_key} initialized successfully")
                else:
                    logger.warning(f"❌ {model_key} not available")
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize {model_key}: {str(e)}")
        
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def run_comparison(self, jobs: List[Dict[str, Any]], limit: int = 5) -> ComparisonResults:
        """Run comparison on first N jobs"""
        if not self.models:
            self.initialize_models()
        
        if not self.models:
            raise RuntimeError("No models available for comparison")
        
        test_jobs = jobs[:limit]
        logger.info(f"🧪 Starting comparison on {len(test_jobs)} jobs with {len(self.models)} models")
        
        # Test each job with each model
        for i, job in enumerate(test_jobs, 1):
            logger.info(f"\n📋 Testing job {i}/{len(test_jobs)}: {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}")
            
            for model_name, model in self.models.items():
                logger.info(f"  🤖 Testing {model_name}...")
                
                try:
                    performance = self._test_single_model(model, model_name, job)
                    self.results.append(performance)
                    
                    # Log quick results
                    if performance.success:
                        quality = performance.output_quality.get('overall', 0)
                        logger.info(f"    ✅ Success - Quality: {quality:.2f}, Time: {performance.response_time:.1f}s")
                    else:
                        logger.warning(f"    ❌ Failed - {performance.error_msg}")
                        
                except Exception as e:
                    logger.error(f"    💥 Exception testing {model_name}: {str(e)}")
                    # Add failed result
                    self.results.append(ModelPerformance(
                        model_name=model_name,
                        job_id=job.get('job_id', 'unknown'),
                        job_title=job.get('title', 'Unknown'),
                        company=job.get('company', 'Unknown'),
                        job_focus='unknown',
                        response_time=0.0,
                        output_quality={},
                        success=False,
                        error_msg=str(e)
                    ))
        
        # Analyze results
        return self._analyze_results()
    
    def _test_single_model(self, model: LLMProvider, model_name: str, job: Dict[str, Any]) -> ModelPerformance:
        """Test a single model on a single job"""
        start_time = time.time()
        
        try:
            # Prepare request
            job_focus = classify_job_type(
                job.get('description', ''),
                job.get('title', '')
            )
            
            request = TailoringRequest(
                job_description=job.get('description', ''),
                resume_sections=self.resume_parser.get_resume_sections(),
                job_focus=job_focus,
                job_title=job.get('title', ''),
                company=job.get('company', '')
            )
            
            # Generate tailoring
            response = model.generate_tailoring(request)
            response_time = time.time() - start_time
            
            # Evaluate quality
            quality_scores = self._evaluate_response(response, job, request)
            
            # Generate tailored resume if enabled
            resume_path = None
            if self.resume_generator:
                try:
                    resume_path = self.resume_generator.save_tailored_resume(
                        company=job.get('company', 'Unknown'),
                        job_title=job.get('title', 'Unknown'),
                        job_focus=job_focus.value,
                        tailoring_response=response,
                        job_id=job.get('job_id')
                    )
                    
                    # Track generated resume
                    self.generated_resumes.append({
                        'model': model_name,
                        'company': job.get('company', 'Unknown'),
                        'job_title': job.get('title', 'Unknown'),
                        'job_id': job.get('job_id'),
                        'resume_path': resume_path,
                        'quality_score': quality_scores.get('overall', 0)
                    })
                    
                    logger.info(f"Generated resume: {resume_path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate resume for {model_name}: {str(e)}")
                    resume_path = None
            
            return ModelPerformance(
                model_name=model_name,
                job_id=job.get('job_id', 'unknown'),
                job_title=job.get('title', ''),
                company=job.get('company', ''),
                job_focus=job_focus.value,
                response_time=response_time,
                output_quality=quality_scores,
                success=True,
                response_data={
                    'summary': response.summary,
                    'skills_count': len(response.skills_reordered),
                    'keywords_count': len(response.keywords_added),
                    'confidence': response.confidence,
                    'reasoning': response.reasoning,
                    'resume_path': resume_path
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            
            return ModelPerformance(
                model_name=model_name,
                job_id=job.get('job_id', 'unknown'),
                job_title=job.get('title', ''),
                company=job.get('company', ''),
                job_focus='unknown',
                response_time=response_time,
                output_quality={},
                success=False,
                error_msg=str(e)
            )
    
    def _evaluate_response(self, response: TailoringResponse, job: Dict[str, Any], request: TailoringRequest) -> Dict[str, float]:
        """Evaluate the quality of a tailoring response"""
        
        job_desc = job.get('description', '').lower()
        job_title = job.get('title', '').lower()
        summary = response.summary.lower()
        
        # 1. Keyword Coverage Score
        important_keywords = self._extract_evaluation_keywords(job_desc, job_title)
        summary_words = set(summary.split())
        matched_keywords = [kw for kw in important_keywords if kw in summary]
        keyword_score = len(matched_keywords) / max(len(important_keywords), 1)
        
        # 2. Length Appropriateness Score
        summary_length = len(response.summary.split())
        if 40 <= summary_length <= 120:
            length_score = 1.0
        elif 30 <= summary_length < 40 or 120 < summary_length <= 150:
            length_score = 0.8
        elif 20 <= summary_length < 30 or 150 < summary_length <= 200:
            length_score = 0.6
        else:
            length_score = 0.3
        
        # 3. Technical Coherence Score
        required_terms = ['machine learning', 'ml', 'engineer', 'systems', 'production']
        coherence_score = sum(1 for term in required_terms if term in summary) / len(required_terms)
        
        # 4. Job Focus Relevance Score
        focus_keywords = {
            'cv': ['computer vision', 'cv', 'detection', 'segmentation', 'image', 'visual'],
            'llm': ['llm', 'language model', 'rag', 'generative', 'agentic', 'prompt'],
            'mlops': ['mlops', 'kubernetes', 'deployment', 'infrastructure', 'serving', 'pipeline'],
            'research': ['research', 'novel', 'framework', 'accuracy', 'evaluation', 'experimental'],
            'general': ['model', 'training', 'data', 'algorithm', 'performance']
        }
        
        job_focus = request.job_focus.value
        relevant_keywords = focus_keywords.get(job_focus, focus_keywords['general'])
        focus_score = sum(1 for kw in relevant_keywords if kw in summary) / len(relevant_keywords)
        
        # 5. Skills Quality Score
        skills_provided = len(response.skills_reordered)
        skills_score = min(skills_provided / 5.0, 1.0)  # Target 5+ skills
        
        # 6. Keywords Added Score
        keywords_added = len(response.keywords_added)
        keywords_score = min(keywords_added / 3.0, 1.0)  # Target 3+ keywords
        
        # 7. Confidence Score (from model)
        confidence_score = response.confidence
        
        # Weighted overall score
        overall_score = (
            keyword_score * 0.25 +      # 25% - keyword coverage
            length_score * 0.15 +       # 15% - appropriate length  
            coherence_score * 0.20 +    # 20% - technical coherence
            focus_score * 0.20 +        # 20% - job focus relevance
            skills_score * 0.10 +       # 10% - skills provided
            keywords_score * 0.05 +     # 5% - keywords added
            confidence_score * 0.05     # 5% - model confidence
        )
        
        return {
            'keyword_coverage': round(keyword_score, 3),
            'length_score': round(length_score, 3),
            'coherence': round(coherence_score, 3),
            'focus_relevance': round(focus_score, 3),
            'skills_quality': round(skills_score, 3),
            'keywords_added': round(keywords_score, 3),
            'model_confidence': round(confidence_score, 3),
            'overall': round(overall_score, 3)
        }
    
    def _extract_evaluation_keywords(self, job_desc: str, job_title: str) -> List[str]:
        """Extract key terms for evaluation"""
        text = f"{job_desc} {job_title}".lower()
        
        # High-value keywords to look for
        important_keywords = [
            'pytorch', 'tensorflow', 'keras', 'scikit-learn',
            'computer vision', 'cv', 'llm', 'rag', 'mlops',
            'kubernetes', 'docker', 'fastapi', 'aws', 'gcp',
            'machine learning', 'deep learning', 'ai',
            'python', 'sql', 'spark', 'pandas'
        ]
        
        found = []
        for keyword in important_keywords:
            if keyword in text:
                found.append(keyword)
        
        return found[:10]  # Top 10 most important
    
    def _analyze_results(self) -> ComparisonResults:
        """Analyze all test results and determine winner"""
        if not self.results:
            raise ValueError("No results to analyze")
        
        # Group results by model
        model_results = {}
        for result in self.results:
            model_name = result.model_name
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)
        
        # Calculate statistics for each model
        model_stats = {}
        for model_name, results in model_results.items():
            successful_results = [r for r in results if r.success]
            
            if successful_results:
                # Quality metrics
                overall_scores = [r.output_quality.get('overall', 0) for r in successful_results]
                response_times = [r.response_time for r in successful_results]
                confidence_scores = [r.response_data.get('confidence', 0) for r in successful_results if r.response_data]
                
                model_stats[model_name] = {
                    'success_rate': len(successful_results) / len(results),
                    'total_tests': len(results),
                    'successful_tests': len(successful_results),
                    'avg_response_time': statistics.mean(response_times) if response_times else 0,
                    'median_response_time': statistics.median(response_times) if response_times else 0,
                    'avg_quality': statistics.mean(overall_scores) if overall_scores else 0,
                    'median_quality': statistics.median(overall_scores) if overall_scores else 0,
                    'quality_std': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                    'avg_confidence': statistics.mean(confidence_scores) if confidence_scores else 0,
                    'quality_breakdown': self._calculate_quality_breakdown(successful_results)
                }
            else:
                model_stats[model_name] = {
                    'success_rate': 0,
                    'total_tests': len(results),
                    'successful_tests': 0,
                    'avg_response_time': 0,
                    'median_response_time': 0,
                    'avg_quality': 0,
                    'median_quality': 0,
                    'quality_std': 0,
                    'avg_confidence': 0,
                    'quality_breakdown': {}
                }
        
        # Determine winner based on composite score
        winner = self._determine_winner(model_stats)
        
        # Generate recommendation
        if winner and winner in model_stats:
            winner_stats = model_stats[winner]
            recommendation = (
                f"{winner} performs best with {winner_stats['avg_quality']:.3f} average quality, "
                f"{winner_stats['success_rate']:.1%} success rate, and "
                f"{winner_stats['avg_response_time']:.1f}s average response time"
            )
        else:
            recommendation = "No clear winner - all models had issues"
        
        # Test summary
        test_summary = {
            'total_tests': len(self.results),
            'successful_tests': sum(1 for r in self.results if r.success),
            'models_tested': list(model_stats.keys()),
            'jobs_tested': len(set(r.job_id for r in self.results)),
            'avg_response_time_all': statistics.mean([r.response_time for r in self.results if r.success]),
            'best_overall_quality': max([r.output_quality.get('overall', 0) for r in self.results if r.success], default=0)
        }
        
        return ComparisonResults(
            model_stats=model_stats,
            winner=winner,
            recommendation=recommendation,
            detailed_results=self.results,
            test_summary=test_summary
        )
    
    def _calculate_quality_breakdown(self, results: List[ModelPerformance]) -> Dict[str, float]:
        """Calculate average scores for each quality metric"""
        if not results:
            return {}
        
        metrics = ['keyword_coverage', 'length_score', 'coherence', 'focus_relevance', 
                  'skills_quality', 'keywords_added', 'model_confidence']
        
        breakdown = {}
        for metric in metrics:
            scores = [r.output_quality.get(metric, 0) for r in results]
            breakdown[metric] = statistics.mean(scores) if scores else 0
        
        return breakdown
    
    def _determine_winner(self, model_stats: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Determine the winning model based on composite score"""
        if not model_stats:
            return None
        
        # Calculate composite score for each model
        model_scores = {}
        for model_name, stats in model_stats.items():
            # Composite score: quality (50%) + success_rate (30%) + speed bonus (20%)
            quality_score = stats['avg_quality']
            success_score = stats['success_rate']
            
            # Speed bonus: faster is better, but diminishing returns
            avg_time = stats['avg_response_time']
            speed_bonus = max(0, 1.0 - (avg_time / 30.0))  # 30s baseline
            
            composite_score = (
                quality_score * 0.5 + 
                success_score * 0.3 + 
                speed_bonus * 0.2
            )
            
            model_scores[model_name] = composite_score
        
        # Return model with highest composite score
        if model_scores:
            return max(model_scores.keys(), key=lambda k: model_scores[k])
        
        return None
    
    def save_results(self, output_path: str):
        """Save comparison results to JSON file"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        results = self._analyze_results()
        
        # Convert to JSON-serializable format
        output_data = {
            'model_stats': results.model_stats,
            'winner': results.winner,
            'recommendation': results.recommendation,
            'test_summary': results.test_summary,
            'detailed_results': [asdict(result) for result in results.detailed_results],
            'generated_resumes': self.generated_resumes
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved comparison results to {output_path}")
    
    def print_resume_generation_summary(self):
        """Print summary of generated resumes"""
        if not self.generated_resumes:
            print("    No resumes generated during comparison")
            return
        
        print(f"\n📄 Resume Generation Summary:")
        print(f"    Total resumes generated: {len(self.generated_resumes)}")
        
        # Group by model
        by_model = {}
        for resume in self.generated_resumes:
            model = resume['model']
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(resume)
        
        for model, resumes in by_model.items():
            avg_quality = sum(r['quality_score'] for r in resumes) / len(resumes)
            print(f"    {model}: {len(resumes)} resumes, avg quality: {avg_quality:.3f}")
            
            # Show best resume for this model
            best_resume = max(resumes, key=lambda x: x['quality_score'])
            print(f"      Best: {best_resume['company']} - {best_resume['job_title']} (quality: {best_resume['quality_score']:.3f})")
            print(f"      Path: {best_resume['resume_path']}")
        
        # Show overall best resume
        best_overall = max(self.generated_resumes, key=lambda x: x['quality_score'])
        print(f"\n    🏆 Best Resume Overall:")
        print(f"      Model: {best_overall['model']}")
        print(f"      Company: {best_overall['company']}")
        print(f"      Job: {best_overall['job_title']}")
        print(f"      Quality: {best_overall['quality_score']:.3f}")
        print(f"      Path: {best_overall['resume_path']}")
    
    def cleanup(self):
        """Clean up all models"""
        for model in self.models.values():
            if hasattr(model, 'cleanup'):
                model.cleanup()
        
        self.models.clear()
        logger.info("Cleaned up all models")
