from typing import List, Dict, Union
from datasets import load_dataset
from rouge_score import rouge_scorer
import numpy as np
import argparse
import json
import re
import random

class RandomSentenceSelectionSummary:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def select_random_sentences(self, sentences: List[str], compression_ratio: float) -> List[int]:
        """Randomly select sentences based on compression ratio."""
        num_sentences = len(sentences)
        num_to_select = max(1, int(num_sentences * compression_ratio))
        return sorted(random.sample(range(num_sentences), num_to_select))

    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores"""
        scores = self.scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def calculate_qa_accuracy(self, summary: str, qa_pairs: List[Dict]) -> float:
        """Calculate QA exact match accuracy using the summary"""
        correct = 0
        for qa in qa_pairs:
            question = qa['question']
            true_answer = qa['answer']
            if true_answer.lower() in summary.lower():
                correct += 1
        return correct / len(qa_pairs) if qa_pairs else 0

    def process_sample(self, sample: Dict, compression_ratio: float = 0.16) -> Dict:
        """Process a single MeetingBank sample using random selection."""
        transcript = sample["prompt"]
        reference_summary = sample["gpt4_summary"]
        qa_pairs = sample["QA_pairs"]
        
        # Split transcript into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', transcript) if s.strip()]
        
        # Randomly select sentences
        selected_indices = self.select_random_sentences(sentences, compression_ratio)
        summary_sentences = [sentences[idx] for idx in selected_indices]
        summary = " ".join(summary_sentences)
        
        # Print selected sentences
        print("\nRandomly Selected Sentences:")
        for idx in selected_indices:
            print(f"Sentence {idx}: {sentences[idx]}")
        
        # Calculate metrics
        rouge_scores = self.calculate_rouge(reference_summary, summary)
        qa_accuracy = self.calculate_qa_accuracy(summary, qa_pairs)
        
        return {
            "original_length": len(sentences),
            "summary_length": len(summary_sentences),
            "compression_ratio": len(summary_sentences) / len(sentences),
            "rouge_scores": rouge_scores,
            "qa_accuracy": qa_accuracy,
            "summary": summary,
            "reference_summary": reference_summary
        }

    def evaluate(self, num_samples: int = 10, compression_ratio: float = 0.50, start_from_end: bool = False) -> List[Dict]:
        """Evaluate on MeetingBank-QA-Summary dataset using random selection."""
        meeting_bank_qa = load_dataset("microsoft/MeetingBank-QA-Summary", split="test")
        
        results = []
        if start_from_end:
            start_idx = max(0, len(meeting_bank_qa) - num_samples)
        else:
            start_idx = 0
        
        for i in range(start_idx, min(start_idx + num_samples, len(meeting_bank_qa))):
            print(f"\nProcessing sample {i + 1}/{len(meeting_bank_qa)}")
            sample = meeting_bank_qa[i]
            result = self.process_sample(sample, compression_ratio)
            if result:
                results.append(result)
                
                print(f"Original: {result['original_length']} sentences")
                print(f"Summary: {result['summary_length']} sentences")
                print(f"ROUGE-1: {result['rouge_scores']['rouge1']:.3f}")
                print(f"ROUGE-2: {result['rouge_scores']['rouge2']:.3f}")
                print(f"ROUGE-L: {result['rouge_scores']['rougeL']:.3f}")
                print(f"QA Accuracy: {result['qa_accuracy']:.3f}")
        
        if results:
            avg_rouge1 = np.mean([r['rouge_scores']['rouge1'] for r in results])
            avg_rouge2 = np.mean([r['rouge_scores']['rouge2'] for r in results])
            avg_rougeL = np.mean([r['rouge_scores']['rougeL'] for r in results])
            avg_qa_acc = np.mean([r['qa_accuracy'] for r in results])
            
            print("\nAggregate Results:")
            print(f"Average ROUGE-1: {avg_rouge1:.3f}")
            print(f"Average ROUGE-2: {avg_rouge2:.3f}")
            print(f"Average ROUGE-L: {avg_rougeL:.3f}")
            print(f"Average QA Accuracy: {avg_qa_acc:.3f}")
            
            with open("random_selection_evaluation.json", "w") as f:
                json.dump(results, f, indent=2)
            print("Results saved to random_selection_evaluation.json")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Run random sentence selection evaluation')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to evaluate')
    parser.add_argument('--compression_ratio', type=float, default=0.31, help='Compression ratio for summarization')
    parser.add_argument('--output_file', type=str, default='random_selection_evaluation.json', help='Output file for results')
    parser.add_argument('--start_from_end', action='store_true', help='Start evaluation from the end of the dataset')
    args = parser.parse_args()
    
    print(f"Starting random selection evaluation with {args.num_samples} samples and compression ratio {args.compression_ratio}")
    summarizer = RandomSentenceSelectionSummary()
    results = summarizer.evaluate(num_samples=args.num_samples, compression_ratio=args.compression_ratio, start_from_end=args.start_from_end)
    
    if results:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 