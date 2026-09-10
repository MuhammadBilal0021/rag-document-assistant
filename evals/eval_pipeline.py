import argparse
import json
import os
import sys
from metrics import LLMJudge

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_evaluation(mock: bool, no_judge: bool):
    print("Starting RAG Evaluation Harness...")
    print(f"Mode: {'Mock' if mock else 'Live'} Generation")
    print(f"Judge: {'Disabled' if no_judge else 'Enabled'}\n")

    test_cases_path = os.path.join(os.path.dirname(__file__), 'test_cases.json')
    baseline_path = os.path.join(os.path.dirname(__file__), 'baseline.json')

    cases = load_json(test_cases_path)
    baseline = load_json(baseline_path)

    judge = LLMJudge(use_mock=mock, enable_judge=not no_judge)
    
    print(f"Loaded {len(cases)} test cases across {len(baseline) - 1} categories.\n")

    results = {}
    
    for case in cases:
        cat = case['category']
        if cat not in results:
            results[cat] = {'relevance': [], 'faithfulness': [], 'precision': []}
            
        # Simulate generation/retrieval
        if mock:
            context = "\n".join(case['mock_context'])
            answer = case['mock_generation']
        else:
            # Here it would call app.rag.pipeline.RAGPipeline
            print("Error: Live generation requires API keys and Vector DB. Run with --mock.")
            sys.exit(1)

        # Evaluate
        rel = judge.evaluate_relevance(case['question'], answer)
        fai = judge.evaluate_faithfulness(context, answer)
        pre = judge.evaluate_retrieval_precision(case['question'], context)

        if rel != "n/a": results[cat]['relevance'].append(rel)
        if fai != "n/a": results[cat]['faithfulness'].append(fai)
        if pre != "n/a": results[cat]['precision'].append(pre)

    print("=== Evaluation Results ===")
    regression_found = False

    for cat, metrics in results.items():
        print(f"\n[{cat}]")
        for metric, values in metrics.items():
            if not values or values[0] == "n/a":
                avg = "n/a"
                print(f"  {metric}: n/a")
            else:
                avg = sum(values) / len(values)
                print(f"  {metric}: {avg:.2f}")
                
                # Check baseline
                base_val = baseline.get(cat, {}).get(metric)
                if base_val != "n/a" and base_val is not None:
                    if avg < base_val:
                        print(f"    -> REGRESSION: {avg:.2f} < baseline {base_val}")
                        regression_found = True

    print("\n==========================")
    if regression_found:
        print("FAIL: Baseline regression detected.")
        sys.exit(1)
    else:
        print("PASS: No regressions detected.")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Evaluation Harness")
    parser.add_argument("--mock", action="store_true", help="Use mock generation and retrieval")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judge (returns n/a)")
    
    args = parser.parse_args()
    run_evaluation(args.mock, args.no_judge)
