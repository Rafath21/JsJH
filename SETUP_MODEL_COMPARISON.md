# Model Comparison Framework Setup Guide

## 🎯 Overview

This framework compares **Phi-3.5-mini**, **Llama 3.2 3B**, and **Qwen 2.5 7B** models for resume tailoring on the first 5 jobs to determine which performs best.

**✨ NEW: Automatic Resume Generation** - Each model will generate tailored resumes saved to `resume/` folder with company name prefixes like `OpenAI_Computer_Vision_Engineer_abc123.txt`.

## 📋 Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended for 7B model)
- GPU optional but recommended

## 🚀 Installation Steps

### 1. Install Dependencies

```bash
# Install PyTorch (CPU version - faster setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or install GPU version if you have CUDA
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python test_model_comparison.py
```

You should see:
```
🎉 All tests passed! The framework is ready to use.
```

### 3. Configure Model Comparison

Edit `config/mariah.yaml`:

```yaml
model_comparison:
  enabled: true                    # Enable comparison
  models: ["phi35", "llama32", "qwen25"]  # Models to test
  job_limit: 5                     # Test on first 5 jobs
  auto_select_winner: false        # Don't auto-update config
```

## 🏃‍♂️ Running the Comparison

### Option 1: Full Pipeline with Comparison
```bash
python src/main.py --config config/mariah.yaml --out out/jobs_with_comparison.xlsx
```

### Option 2: Test Comparison Only
```bash
# Set MAX_JOBS_LIMIT to 5 in src/main.py first
python src/main.py --config config/mariah.yaml --out out/test_comparison.xlsx
```

## 📊 Results

After running, check:
- `out/model_comparison_results.json` - Detailed comparison results
- `resume/` folder - Generated tailored resumes with company prefixes
- Console output for winner announcement and resume generation summary

### Sample Console Output:
```
🏆 Model Comparison Results:
    Winner: phi35
    phi35 performs best with 0.847 average quality, 100.0% success rate, and 12.3s average response time
    Detailed results saved to: out/model_comparison_results.json

📄 Resume Generation Summary:
    Total resumes generated: 15
    phi35: 5 resumes, avg quality: 0.847
      Best: OpenAI - Computer Vision Engineer (quality: 0.923)
      Path: resume/OpenAI_Computer_Vision_Engineer_abc123.txt
    
    🏆 Best Resume Overall:
      Model: phi35
      Company: OpenAI
      Job: Computer Vision Engineer
      Quality: 0.923
      Path: resume/OpenAI_Computer_Vision_Engineer_abc123.txt
```

### Sample Resume Files:
```
resume/
├── OpenAI_Computer_Vision_Engineer_abc123.txt
├── Databricks_ML_Engineer_def456.txt
├── Stripe_AI_Engineer_ghi789.txt
└── ...
```

## 🔧 Troubleshooting

### Memory Issues
- Use CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Reduce models tested: `models: ["phi35"]` (test just one)
- Close other applications

### Model Download Failures
- Models auto-download on first use (~1-7GB each)
- Ensure stable internet connection
- Check HuggingFace Hub status

### Import Errors
- Run `python test_model_comparison.py` to diagnose
- Ensure all dependencies installed
- Check Python version compatibility

## ⚡ Performance Tips

1. **Start with CPU-only** for initial testing
2. **Test one model first**: Set `models: ["phi35"]`
3. **Use GPU if available**: Install CUDA PyTorch version
4. **Monitor memory**: Use `htop` or Activity Monitor

## 🎯 Expected Timeline

- **Setup**: 10-15 minutes
- **First run**: 20-30 minutes (includes model downloads)
- **Subsequent runs**: 5-10 minutes

## 📈 Next Steps After Comparison

1. **Review results** in `out/model_comparison_results.json`
2. **Update config** with winning model:
   ```yaml
   llm:
     selected_model: "phi35"  # Use winner
   ```
3. **Disable comparison** for future runs:
   ```yaml
   model_comparison:
     enabled: false
   ```
4. **Proceed to Milestone B Phase 2**: Resume tailoring implementation

## 🛠️ Development Notes

### File Structure
```
src/
├── llm_providers.py          # Abstract LLM interface
├── transformers_provider.py  # HuggingFace implementation
├── model_comparator.py       # Comparison framework
├── resume_parser.py          # Mariah's resume data
└── main.py                   # Updated main pipeline
```

### Key Classes
- `LLMProvider`: Abstract base for all models
- `TransformersLLM`: HuggingFace implementation
- `ModelComparator`: Runs comparison and evaluation
- `ResumeParser`: Mariah's structured resume data

### Evaluation Metrics
- **Keyword Coverage**: Job-relevant terms in output
- **Length Score**: Appropriate summary length
- **Technical Coherence**: ML terminology usage
- **Focus Relevance**: Matches job type (CV/LLM/MLOps)
- **Overall Quality**: Weighted composite score

## 🔮 Future Enhancements

1. **Add more models**: Mistral, CodeLlama, etc.
2. **Fine-tuning**: Train on job description → resume pairs
3. **A/B testing**: Real-world response rate comparison
4. **Cost optimization**: Model size vs. quality tradeoffs
