from typing import List, Dict, Union
from datasets import load_dataset
from rouge_score import rouge_scorer
import numpy as np
from BaseSummarizer import BaseSummarizer
import os
import sys
import argparse
import json
import re
import time

class SummarizerEvaluation(BaseSummarizer):
    def __init__(self, url: str, api_key: str):
        super().__init__(url, api_key)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def score_sentences(self, sentences: List[str], max_tokens: int = None) -> Dict[int, int]:
        """Ranks sentences for MeetingBank evaluation."""
        all_ranks = {}
        
        # Get chunks that fit within token limits
        chunks = self.chunk_sentences(sentences, max_tokens)
        
        for chunk_idx, (chunk_sentences, (start_idx, end_idx)) in enumerate(chunks):
            print(f"\nProcessing chunk {chunk_idx + 1}/{len(chunks)} (sentences {start_idx}-{end_idx})")
            
            prompt = self._get_base_prompt(chunk_sentences, chunk_idx + 1, (start_idx, end_idx))
            
            max_retries = 3
            for attempt in range(max_retries):
                print(f"\nAttempt {attempt + 1}/{max_retries}")
                response = self.call_gpt(prompt, timeout=240)
                
                if response.startswith("error:"):
                    print(f"API error: {response}")
                    if attempt == max_retries - 1:
                        return {}
                    time.sleep(10)
                    continue
                    
                parsed = self.clean_and_parse_json(response, chunk_sentences)
                if not parsed:
                    if attempt == max_retries - 1:
                        print("Failed to parse JSON after all retries")
                        return {}
                    time.sleep(5)
                    continue
                    
                batch_ranks = parsed["sentence_ranks"]
                if not batch_ranks:
                    if attempt == max_retries - 1:
                        print("No valid ranks in response")
                        return {}
                    time.sleep(5)
                    continue
                    
                # Print GPT's ranks for this chunk
                print("\nGPT Ranks for this chunk:")
                print("ID | Rank | Sentence")
                print("-" * 80)
                
                # Sort ranks by index, handling both string and integer indices
                def get_index(idx):
                    if isinstance(idx, str) and idx.startswith("ID: "):
                        return int(idx[4:])
                    return int(idx)
                
                for idx_str, rank in sorted(batch_ranks.items(), key=lambda x: get_index(x[0])):
                    local_idx = get_index(idx_str)
                    if local_idx < len(chunk_sentences):
                        print(f"{local_idx:2d} | {rank:4d} | {chunk_sentences[local_idx][:50]}...")
                
                # Process valid ranks
                for idx_str, rank in batch_ranks.items():
                    try:
                        local_idx = get_index(idx_str)
                        if local_idx < 0 or local_idx >= len(chunk_sentences):
                            print(f"Invalid local index {local_idx} in chunk")
                            continue
                            
                        global_idx = start_idx + local_idx
                        all_ranks[global_idx] = rank
                    except ValueError:
                        print(f"Invalid index format: {idx_str}")
                        continue
                
                # If we got here, we have valid ranks
                break
                
            time.sleep(2)  # Small delay between chunks
        
        return all_ranks

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
        """Process a single MeetingBank sample and evaluate results."""
        transcript = sample["prompt"]
        reference_summary = sample["gpt4_summary"]
        qa_pairs = sample["QA_pairs"]
        
        # Split transcript into sentences 
        sentences = self.clean_and_split_sentences(transcript)
        
        # Score sentences
        scores = self.score_sentences(sentences)
        if not scores:
            return None
        
        # Calculate max duration based on compression ratio
        max_duration = len(sentences) * compression_ratio
        
        # Generate summary
        selected_indices = self.optimize_summary(
            [{"sentence": s, "duration": 1} for s in sentences], 
            scores, 
            max_duration
        )
        summary_sentences = [sentences[idx] for idx in selected_indices]
        summary = " ".join(summary_sentences)
        
        # Print compressed sentences
        print("\nCompressed Sentences:")
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
        """Evaluate on MeetingBank-QA-Summary dataset."""
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
            
            with open("meeting_bank_evaluation.json", "w") as f:
                json.dump(results, f, indent=2)
            print("Results saved to meeting_bank_evaluation.json")
        
        return results 

def main():
    parser = argparse.ArgumentParser(description='Run MeetingBank evaluation')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to evaluate')
    parser.add_argument('--compression_ratio', type=float, default=0.31, help='Compression ratio for summarization')
    parser.add_argument('--output_file', type=str, default='meeting_bank_evaluation.json', help='Output file for results')
    parser.add_argument('--start_from_end', action='store_true', help='Start evaluation from the end of the dataset')
    args = parser.parse_args()

    # Get API credentials from environment variables
    url = os.getenv("URL_ENDPOINT")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not url or not api_key:
        print("Error: Missing API credentials")
        print("Please set URL_ENDPOINT and OPENAI_API_KEY environment variables")
        sys.exit(1)
    
    print(f"Starting evaluation with {args.num_samples} samples and compression ratio {args.compression_ratio}")
    summarizer = SummarizerEvaluation(url, api_key)
    results = summarizer.evaluate(num_samples=args.num_samples, compression_ratio=args.compression_ratio, start_from_end=args.start_from_end)
    
    if results:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 