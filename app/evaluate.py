import time
import json
from typing import List, Dict, Any
from app.rag_pipeline import RAGPipeline

EVALUATION_QUESTIONS = [
    {
        "question": "How many PTO days do new employees get per year?",
        "gold_answer": "15 days PTO per year",
        "topic": "PTO"
    },
    {
        "question": "What is the maximum number of PTO days I can carry over to the next year?",
        "gold_answer": "5 unused PTO days",
        "topic": "PTO"
    },
    {
        "question": "How many days of sick leave am I entitled to per year?",
        "gold_answer": "10 sick days per year",
        "topic": "PTO"
    },
    {
        "question": "Can I work remotely full-time?",
        "gold_answer": "Up to 3 days per week remotely, minimum 2 days in-office",
        "topic": "Remote Work"
    },
    {
        "question": "What are the internet speed requirements for remote work?",
        "gold_answer": "Minimum 25 Mbps",
        "topic": "Remote Work"
    },
    {
        "question": "How often must I change my password according to security policy?",
        "gold_answer": "Every 90 days",
        "topic": "Security"
    },
    {
        "question": "Is multi-factor authentication required?",
        "gold_answer": "Yes, required for all employee accounts",
        "topic": "Security"
    },
    {
        "question": "What is the maximum meal reimbursement for client dinners?",
        "gold_answer": "Up to $75 per person",
        "topic": "Expense"
    },
    {
        "question": "How long do I have to submit expense reports?",
        "gold_answer": "Within 30 days of expense",
        "topic": "Expense"
    },
    {
        "question": "What is the maximum hotel reimbursement per night domestically?",
        "gold_answer": "$200/night domestic",
        "topic": "Expense"
    },
    {
        "question": "How much is the company 401k match?",
        "gold_answer": "4% company match",
        "topic": "Benefits"
    },
    {
        "question": "What percentage of health insurance premiums does the company pay?",
        "gold_answer": "80% of premiums",
        "topic": "Benefits"
    },
    {
        "question": "How much is the annual wellness program benefit?",
        "gold_answer": "$500 annual",
        "topic": "Benefits"
    },
    {
        "question": "What is the tuition reimbursement limit per year?",
        "gold_answer": "$5000/year",
        "topic": "Benefits"
    },
    {
        "question": "How many weeks of parental leave for primary caregivers?",
        "gold_answer": "16 weeks paid leave",
        "topic": "Leave"
    },
    {
        "question": "What is the FMLA leave entitlement?",
        "gold_answer": "12 weeks unpaid leave per year",
        "topic": "Leave"
    },
    {
        "question": "How many days of bereavement leave for immediate family?",
        "gold_answer": "5 days paid",
        "topic": "Leave"
    },
    {
        "question": "What is the performance review cycle?",
        "gold_answer": "Annual in January, mid-year check-in in July",
        "topic": "Performance"
    },
    {
        "question": "When do merit increases take effect?",
        "gold_answer": "April 1 each year",
        "topic": "Performance"
    },
    {
        "question": "What is the rating scale for performance reviews?",
        "gold_answer": "1-5 scale (1=Unsatisfactory to 5=Exceptional)",
        "topic": "Performance"
    }
]


def evaluate_groundedness(answer: str, context: List[str]) -> float:
    """Evaluate if the answer is grounded in the retrieved context."""
    answer_lower = answer.lower()
    context_combined = " ".join(context).lower()
    
    key_claims = []
    if "15" in answer and "day" in answer:
        key_claims.append("15 days")
    if "5" in answer and "carry" in answer:
        key_claims.append("5")
    if "10" in answer and "sick" in answer:
        key_claims.append("10 sick")
    if "80%" in answer or "80 percent" in answer:
        key_claims.append("80")
    if "4%" in answer or "4 percent" in answer:
        key_claims.append("4")
    
    if not key_claims:
        return 0.5
    
    grounded_claims = sum(1 for claim in key_claims if claim in context_combined)
    return grounded_claims / len(key_claims)


def evaluate_citation_accuracy(answer: str, citations: List[Dict]) -> float:
    """Evaluate if citations correctly point to supporting documents."""
    if not citations:
        return 0.0
    
    answer_lower = answer.lower()
    
    relevant_sources = 0
    for citation in citations:
        source = citation.get('source', '').lower()
        content = citation.get('content', '').lower()
        
        if any(keyword in content for keyword in ['pto', 'vacation', 'day', 'policy']):
            if any(keyword in answer_lower for keyword in ['pto', 'vacation', 'day', 'time off']):
                relevant_sources += 1
        elif any(keyword in content for keyword in ['remote', 'work', 'home']):
            if any(keyword in answer_lower for keyword in ['remote', 'work from home', 'hybrid']):
                relevant_sources += 1
        elif any(keyword in content for keyword in ['security', 'password', 'mfa']):
            if any(keyword in answer_lower for keyword in ['security', 'password', 'mfa']):
                relevant_sources += 1
        elif any(keyword in content for keyword in ['expense', 'reimburs', 'travel']):
            if any(keyword in answer_lower for keyword in ['expense', 'reimburs', 'travel']):
                relevant_sources += 1
        elif any(keyword in content for keyword in ['benefit', 'insurance', '401k']):
            if any(keyword in answer_lower for keyword in ['benefit', 'insurance', '401k']):
                relevant_sources += 1
    
    return min(relevant_sources / max(len(citations), 1), 1.0)


def run_evaluation(rag_pipeline: RAGPipeline, questions: List[Dict] = None) -> Dict[str, Any]:
    """Run evaluation on the RAG pipeline."""
    if questions is None:
        questions = EVALUATION_QUESTIONS
    
    results = []
    latencies = []
    
    for item in questions:
        start_time = time.time()
        result = rag_pipeline.answer(item['question'])
        latency = time.time() - start_time
        
        groundedness = evaluate_groundedness(
            result.get('answer', ''),
            [c.get('content', '') for c in result.get('citations', [])]
        )
        
        citation_accuracy = evaluate_citation_accuracy(
            result.get('answer', ''),
            result.get('citations', [])
        )
        
        results.append({
            'question': item['question'],
            'gold_answer': item.get('gold_answer', ''),
            'answer': result.get('answer', ''),
            'groundedness': groundedness,
            'citation_accuracy': citation_accuracy,
            'latency': latency,
            'success': result.get('success', False)
        })
        
        latencies.append(latency)
    
    latencies_sorted = sorted(latencies)
    p50_idx = int(len(latencies_sorted) * 0.5)
    p95_idx = int(len(latencies_sorted) * 0.95)
    
    avg_groundedness = sum(r['groundedness'] for r in results) / len(results)
    avg_citation_accuracy = sum(r['citation_accuracy'] for r in results) / len(results)
    avg_latency = sum(latencies) / len(latencies)
    
    return {
        'total_questions': len(results),
        'groundedness': {
            'average': avg_groundedness,
            'percentage': round(avg_groundedness * 100, 1)
        },
        'citation_accuracy': {
            'average': avg_citation_accuracy,
            'percentage': round(avg_citation_accuracy * 100, 1)
        },
        'latency': {
            'average': round(avg_latency, 2),
            'p50': round(latencies_sorted[p50_idx], 2) if latencies_sorted else 0,
            'p95': round(latencies_sorted[p95_idx], 2) if latencies_sorted else 0
        },
        'individual_results': results
    }


def save_evaluation_results(results: Dict[str, Any], output_file: str = "evaluation_results.json"):
    """Save evaluation results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {output_file}")
