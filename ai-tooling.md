# AI Tooling Documentation

## AI Tools Used

### 1. GitHub Copilot

**What worked well:**
- Generated boilerplate code for Flask routes
- Suggested error handling patterns

**What didn't work well:**
- Suggested deprecated LangChain imports
- Outdated model names for Groq

### 2. ChatGPT (GPT-4)

**What worked well:**
- Step-by-step debugging of complex errors
- Explaining version conflicts
- Generating documentation

**What didn't work well:**
- Hallucinated that langchain-groq was included in base packages

## Key Problems Solved

### Problem 1: Sentence-Transformers Import Error
**Solution:** Downgraded to huggingface-hub==0.19.4 and sentence-transformers==2.2.2

### Problem 2: Groq Model Decommissioned
**Solution:** Updated model from mixtral-8x7b to llama-3.3-70b-versatile

### Problem 3: LangChain Version Conflicts
**Solution:** Pinned compatible versions in requirements.txt

### Problem 4: ModuleNotFoundError with 'app' Imports
**Solution:** Run with python -m app.main from parent directory

## Lessons Learned

1. Always pin dependency versions
2. Test AI suggestions before integrating
3. Verify API model names with provider documentation
4. Use official documentation as source of truth

## Disclosure

AI tools were used as coding assistants. All final code was reviewed and tested by the developer.