# Design and Evaluation Document

## Part i: Design and Architecture Decisions

### Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| LLM Provider | Groq (Llama 3.3 70B) | Free tier, fast inference |
| Embeddings | Sentence Transformers | Local, no API cost |
| Vector Database | ChromaDB | Embedded, easy setup |
| Framework | LangChain 0.1.0 | Standard RAG abstractions |
| Backend | Flask | Lightweight |

### Key Architecture Decisions

**Decision 1: Topic Guard Before LLM Call**
- Filters off-topic questions BEFORE calling LLM
- Saves API costs and reduces latency

**Decision 2: Modular Pipeline Design**
- Separate modules for loading, storage, and orchestration
- Enables independent testing and debugging

**Decision 3: Local Embeddings Over Cloud APIs**
- Zero cost after initial 90MB download
- Privacy-preserving (no data leaves machine)

**Decision 4: Source Citation in Responses**
- Retrieved documents include filename metadata
- Builds user trust and enables verification

## Part ii: Evaluation Approach and Results

### Test Questions (7 total)

| # | Question | Expected Answer | Source |
|---|----------|----------------|--------|
| 1 | How many PTO days do new employees get? | 15 days | pto_policy.md |
| 2 | Minimum internet speed for remote work? | 25 Mbps | remote_work_policy.md |
| 3 | How often must I change my password? | 90 days | security_policy.md |
| 4 | What is company 401k match? | 4% | benefits_policy.md |
| 5 | How many weeks of parental leave? | 16 weeks | leave_policy.md |
| 6 | Maximum meal reimbursement? | $75 per person | expense_policy.md |
| 7 | What is weather like today? | Polite refusal | Topic guard |

### Results Summary

| Metric | Result |
|--------|--------|
| Accuracy on policy questions | 6/6 (100%) |
| Off-topic detection | 1/1 (100%) |
| Average response time | 1-4 seconds |
| Source citation accuracy | 100% |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Documents indexed | 9 |
| Chunks created | 85 |
| Average retrieval time | ~0.5 sec |
| Average LLM generation | ~1-3 sec |
| End-to-end latency | ~1.5-3.5 sec |

### Evaluation Conclusion

**Overall Assessment: PASS — Ready for submission**