from typing import List, Dict, Union
import json
import re
from BaseSummarizer import BaseSummarizer
import os
import sys
import argparse
import time

class TranscriptSummarizer(BaseSummarizer):
    def __init__(self, url: str, api_key: str):
        super().__init__(url, api_key)

    def score_sentences(self, sentences: List[Dict], max_tokens: int = None) -> Dict[int, int]:
        """Ranks sentences for video summarization."""
        all_ranks = {}
        
        # Extract just the sentence text for chunking
        sentence_texts = [s['sentence'] for s in sentences]
        
        # Get chunks that fit within token limits
        chunks = self.chunk_sentences(sentence_texts, max_tokens)
        
        for chunk_idx, (chunk_sentences, (start_idx, end_idx)) in enumerate(chunks):
            print(f"\nProcessing chunk {chunk_idx + 1}/{len(chunks)} (sentences {start_idx}-{end_idx})")
            
            prompt = self._get_base_prompt(chunk_sentences, chunk_idx + 1, (start_idx, end_idx))
            
            max_retries = 3  
            chunk_success = False
            
            for attempt in range(max_retries):
                print(f"\nAttempt {attempt + 1}/{max_retries}")
                response = self.call_gpt(prompt, timeout=240)
                
                if response.startswith("error:"):
                    print(f"API error: {response}")
                    if attempt == max_retries - 1:
                        print("Max retries reached for API errors, moving to next chunk")
                        break
                    time.sleep(5) 
                    continue
                    
                parsed = self.clean_and_parse_json(response, chunk_sentences)
                if not parsed:
                    if attempt == max_retries - 1:
                        print("Failed to parse JSON after all retries, moving to next chunk")
                        break
                    time.sleep(5) 
                    continue
                    
                batch_ranks = parsed["sentence_ranks"]
                if not batch_ranks:
                    if attempt == max_retries - 1:
                        print("No valid ranks in response, moving to next chunk")
                        break
                    time.sleep(10)
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
                
                # Sort by rank (ascending) instead of index
                for idx_str, rank in sorted(batch_ranks.items(), key=lambda x: x[1]):
                    local_idx = get_index(idx_str)
                    if local_idx < len(chunk_sentences):
                        print(f"{local_idx:2d} | {rank:4d} | {chunk_sentences[local_idx][:50]}...")
                
                # Process valid ranks
                chunk_ranks = {}
                for idx_str, rank in batch_ranks.items():
                    try:
                        local_idx = get_index(idx_str)
                        if local_idx < 0 or local_idx >= len(chunk_sentences):
                            print(f"Invalid local index {local_idx} in chunk")
                            continue
                            
                        global_idx = start_idx + local_idx
                        chunk_ranks[global_idx] = rank
                    except ValueError:
                        print(f"Invalid index format: {idx_str}")
                        continue
                
                # Validate we got ranks for all sentences in this chunk
                if len(chunk_ranks) == len(chunk_sentences):
                    all_ranks.update(chunk_ranks)
                    chunk_success = True
                    break
                else:
                    print(f"Warning: Only got {len(chunk_ranks)} ranks out of {len(chunk_sentences)} sentences")
                    if attempt == max_retries - 1:
                        print("Max retries reached, using partial ranks for this chunk")
                        all_ranks.update(chunk_ranks)  # Use partial results
                        break
                    time.sleep(10)
                    continue
                
            if not chunk_success:
                print(f"Warning: Failed to process chunk {chunk_idx + 1}, continuing with next chunk")
            
            time.sleep(5)  # Small delay between chunks
        
        # Validate we have ranks for all sentences
        if len(all_ranks) < len(sentences):
            print(f"Warning: Only got ranks for {len(all_ranks)} out of {len(sentences)} sentences")
            # Continue with partial results instead of failing completely
        
        return all_ranks

    def summarize(self, sentences: List[Dict], max_duration: float) -> Dict:
        """Generate a summary from video transcript sentences."""
        print(f"Starting summarization of {len(sentences)} sentences...")
        
        # Score sentences
        ranks = self.score_sentences(sentences)
        if not ranks:
            return None
        
        # Generate summary
        selected_indices = self.optimize_summary(sentences, ranks, max_duration)
        summary_sentences = [sentences[idx] for idx in selected_indices]
        
        # Calculate total duration
        total_duration = sum(s['duration'] for s in summary_sentences)
        
        return {
            "sentences": summary_sentences,
            "total_duration": total_duration,
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(summary_sentences),
            "compression_ratio": len(summary_sentences) / len(sentences)
        }

    def save_summary_to_file(self, summary: Dict, output_file: str):
        """Save the summary to a file."""
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {output_file}")

def load_transcript(file_path: str) -> List[Dict]:
    """Load transcript from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to expected format if needed
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'sentences' in data:
            return data['sentences']
        else:
            print("Error: Invalid transcript format")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading transcript: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Summarize a transcript')
    parser.add_argument('--input', type=str, required=True, help='Input transcript JSON file')
    parser.add_argument('--max_duration', type=float, required=True, help='Maximum duration for summary in seconds')
    parser.add_argument('--output', type=str, default='summary.json', help='Output file for summary')
    args = parser.parse_args()

    # Get API credentials from environment variables
    url = os.getenv("URL_ENDPOINT")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not url or not api_key:
        print("Error: Missing API credentials")
        print("Please set URL_ENDPOINT and OPENAI_API_KEY environment variables")
        sys.exit(1)
    
    # Load transcript
    print(f"Loading transcript from {args.input}")
    sentences = load_transcript(args.input)
    
    # Create summarizer and generate summary
    print(f"Generating summary with max duration of {args.max_duration} seconds")
    summarizer = TranscriptSummarizer(url, api_key)
    summary = summarizer.summarize(sentences, args.max_duration)
    
    if summary:
        summarizer.save_summary_to_file(summary, args.output)
    else:
        print("Error: Failed to generate summary")
        sys.exit(1)

if __name__ == "__main__":
    main() 