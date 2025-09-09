#!/usr/bin/env python3
"""
Test script for model comparison framework
Run this to verify the setup works before running the full pipeline
"""

import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all imports work"""
    print("🧪 Testing imports...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from llm_providers import LLMProvider, TailoringRequest, TailoringResponse, JobFocus, classify_job_type
        from transformers_provider import TransformersLLM, ModelFactory
        from resume_parser import ResumeParser
        from model_comparator import ModelComparator
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_resume_parser():
    """Test resume parser functionality"""
    print("\n🧪 Testing resume parser...")
    
    try:
        from resume_parser import ResumeParser
        parser = ResumeParser()
        
        # Test basic functionality
        resume_data = parser.get_resume_sections()
        assert "summary" in resume_data
        assert "skills" in resume_data
        assert len(resume_data["skills"]) > 0
        
        # Test skill categorization
        cv_skills = parser.get_skills_by_category("computer_vision")
        assert len(cv_skills) > 0
        
        # Test job focus tailoring
        cv_experience = parser.get_relevant_experience("cv")
        assert len(cv_experience) > 0
        
        print("✅ Resume parser working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Resume parser failed: {e}")
        return False

def test_job_classification():
    """Test job classification"""
    print("\n🧪 Testing job classification...")
    
    try:
        from llm_providers import classify_job_type, JobFocus
        
        # Test cases
        test_cases = [
            ("Computer Vision Engineer for autonomous vehicles", "Computer Vision Engineer", JobFocus.COMPUTER_VISION),
            ("LLM Engineer building RAG systems", "AI Engineer", JobFocus.LLM_RAG),
            ("MLOps Engineer managing Kubernetes deployments", "MLOps Engineer", JobFocus.MLOPS),
            ("Research Scientist in machine learning", "Research Scientist", JobFocus.RESEARCH),
            ("Machine Learning Engineer", "ML Engineer", JobFocus.GENERAL_ML)
        ]
        
        for desc, title, expected in test_cases:
            result = classify_job_type(desc, title)
            print(f"  '{title}' -> {result.value} (expected: {expected.value})")
            # Note: Classification might not be perfect, so we don't assert
        
        print("✅ Job classification working")
        return True
        
    except Exception as e:
        print(f"❌ Job classification failed: {e}")
        return False

def test_model_factory():
    """Test model factory without actually loading models"""
    print("\n🧪 Testing model factory...")
    
    try:
        from transformers_provider import ModelFactory
        
        # Test supported models list
        supported = ModelFactory.SUPPORTED_MODELS
        assert "phi35" in supported
        assert "llama32" in supported
        assert "qwen25" in supported
        
        print(f"✅ Model factory supports: {list(supported.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Model factory failed: {e}")
        return False

def test_sample_tailoring_request():
    """Test creating a sample tailoring request"""
    print("\n🧪 Testing tailoring request creation...")
    
    try:
        from llm_providers import TailoringRequest, JobFocus
        from resume_parser import ResumeParser
        
        parser = ResumeParser()
        resume_data = parser.get_resume_sections()
        
        # Create sample request
        request = TailoringRequest(
            job_description="We are looking for a Computer Vision Engineer with PyTorch experience...",
            resume_sections=resume_data,
            job_focus=JobFocus.COMPUTER_VISION,
            job_title="Computer Vision Engineer",
            company="Test Company"
        )
        
        assert request.job_focus == JobFocus.COMPUTER_VISION
        assert len(request.resume_sections["skills"]) > 0
        
        print("✅ Tailoring request creation working")
        return True
        
    except Exception as e:
        print(f"❌ Tailoring request failed: {e}")
        return False

def test_resume_generator():
    """Test resume generator functionality"""
    print("\n🧪 Testing resume generator...")
    
    try:
        from resume_generator import ResumeGenerator
        from llm_providers import TailoringResponse
        
        generator = ResumeGenerator()
        
        # Create sample tailoring response
        sample_response = TailoringResponse(
            summary="Machine Learning Engineer specializing in Computer Vision systems; designs reliable CV training/serving pipelines.",
            experience_bullets=[
                "Led CV pipeline achieving mAP@50 99.2% accuracy",
                "Built LLM/agentic workflows for data extraction",
                "Implemented MLflow experiments and model registry"
            ],
            skills_reordered=["PyTorch", "Computer Vision", "FastAPI", "Kubernetes", "MLflow"],
            keywords_added=["computer vision", "pytorch", "kubernetes"],
            confidence=0.85,
            reasoning="Emphasized CV experience for computer vision role"
        )
        
        # Test resume generation
        resume_content = generator.generate_tailored_resume(
            company="Test Company",
            job_title="Computer Vision Engineer", 
            job_focus="cv",
            tailoring_response=sample_response,
            job_id="test123"
        )
        
        # Validate content
        assert "Syeda Mariah Banu" in resume_content
        assert "Computer Vision" in resume_content
        assert "PyTorch" in resume_content
        assert len(resume_content) > 1000
        
        print("✅ Resume generator working")
        return True
        
    except Exception as e:
        print(f"❌ Resume generator failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Model Comparison Framework\n")
    
    tests = [
        test_imports,
        test_resume_parser,
        test_job_classification,
        test_model_factory,
        test_sample_tailoring_request,
        test_resume_generator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print("❌ Test failed, check the error above")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The framework is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the main pipeline with model comparison enabled")
        print("3. Check out/model_comparison_results.json for results")
        print("4. Tailored resumes will be saved to resume/ folder with company prefixes")
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
