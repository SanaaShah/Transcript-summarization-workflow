import requests
import json
import re
from typing import List, Dict, Union, Optional, Tuple
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np

class BaseSummarizer:
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self.encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 Turbo's encoding
        self.MAX_TOKENS = 120000  # Conservative limit for GPT-4 Turbo (128K - buffer for response)
        self.OVERLAP_TOKENS = 2000  # Target overlap size in tokens (about 1-2 paragraphs)
        self.MAX_SENTENCES_PER_CHUNK = 200  # Maximum sentences per chunk (reduced from 250)
        # Initialize Sentence-BERT for semantic similarity
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.SIMILARITY_THRESHOLD = 0.85  # Adjust this threshold as needed

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string."""
        return len(self.encoder.encode(text))

    def find_context_boundary(self, sentences: List[str], target_tokens: int) -> int:
        """
        Find a natural boundary point for context overlap.
        Looks for sentence boundaries that are close to the target token count.
        """
        current_tokens = 0
        best_boundary = 0
        min_diff = float('inf')

        for i, sentence in enumerate(sentences):
            current_tokens += self.estimate_tokens(sentence)
            diff = abs(current_tokens - target_tokens)
            
            # Prefer boundaries at the end of sentences
            if diff < min_diff:
                min_diff = diff
                best_boundary = i + 1  # Include this sentence in the overlap
            
            # If we've gone significantly past the target, stop
            if current_tokens > target_tokens * 1.2:
                break

        return best_boundary

    def clean_and_parse_json(self, response: str, sentences: List[str] = None) -> Optional[Dict]:
        """Cleans and parses a JSON response from GPT with better error handling."""
        try:
            # Clean response
            response = re.sub(r'```json|```', '', response).strip()
            
            # Handle truncated JSON
            if response.endswith('...'):
                last_brace = response.rfind('}')
                if last_brace != -1:
                    response = response[:last_brace + 1]
            
            # Clean JSON
            response = response.replace("'", '"')
            response = re.sub(r',\s*}', '}', response)
            response = re.sub(r',\s*]', ']', response)
            
            # Parse and validate
            parsed = json.loads(response)
            if isinstance(parsed, dict) and "sentence_ranks" in parsed:
                ranks = parsed["sentence_ranks"]
                if isinstance(ranks, dict):
                    valid_ranks = {}
                    used_ranks = set()
                    missing_indices = set(range(len(sentences))) if sentences else set()
                    
                    # First pass: collect valid ranks and track used/missing indices
                    for idx_str, rank in ranks.items():
                        try:
                            # Convert index to integer, handling both string and integer indices
                            if isinstance(idx_str, str) and idx_str.startswith("ID: "):
                                local_idx = int(idx_str[4:])
                            else:
                                local_idx = int(idx_str)
                                
                            # Validate index is within bounds
                            if sentences is not None:
                                if local_idx < 0 or local_idx >= len(sentences):
                                    print(f"Warning: Index {local_idx} out of bounds for chunk of size {len(sentences)}")
                                    continue
                                    
                            # Validate rank is positive
                            rank_int = int(rank)
                            if rank_int <= 0:
                                print(f"Warning: Invalid rank {rank_int} for index {local_idx}")
                                continue
                                
                            # Track used indices
                            if local_idx in missing_indices:
                                missing_indices.remove(local_idx)
                                
                            # Handle duplicate ranks by incrementing
                            while rank_int in used_ranks:
                                rank_int += 1
                                
                            used_ranks.add(rank_int)
                            valid_ranks[local_idx] = rank_int
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Error processing index {idx_str}: {str(e)}")
                            continue
                    
                    # Second pass: assign ranks to missing indices
                    if missing_indices:
                        print(f"Assigning ranks to {len(missing_indices)} missing indices")
                        max_rank = max(used_ranks) if used_ranks else 0
                        for idx in missing_indices:
                            max_rank += 1
                            valid_ranks[idx] = max_rank
                    
                    # Final validation
                    if len(valid_ranks) != len(sentences):
                        print(f"Warning: Still missing ranks. Expected {len(sentences)}, got {len(valid_ranks)}")
                        return None
                        
                    # Normalize ranks to be sequential
                    sorted_ranks = sorted(valid_ranks.items(), key=lambda x: x[1])
                    for i, (idx, _) in enumerate(sorted_ranks, 1):
                        valid_ranks[idx] = i
                    
                    parsed["sentence_ranks"] = valid_ranks
                    return parsed
        except Exception as e:
            print(f"Error parsing JSON: {str(e)}")
            return None

    def calculate_optimal_chunk_size(self, total_sentences: int, compression_ratio: float = None) -> int:
        """
        Calculate optimal chunk size based on compression ratio or max duration.
        """
        if compression_ratio is not None:
            # For evaluation: use compression ratio to determine chunk size
            target_sentences = max(10, int(total_sentences * compression_ratio * 2))
            return min(target_sentences, self.MAX_SENTENCES_PER_CHUNK)
        else:
            # For summarization: use fixed maximum
            return self.MAX_SENTENCES_PER_CHUNK

    def chunk_sentences(self, sentences: List[str], max_tokens: int = None, compression_ratio: float = None) -> List[Tuple[List[str], Tuple[int, int]]]:
        """
        Chunk sentences into groups that fit within token limits, with context-aware overlap.
        """
        if max_tokens is None:
            max_tokens = self.MAX_TOKENS

        # Calculate optimal chunk size
        chunk_size = self.calculate_optimal_chunk_size(len(sentences), compression_ratio)
        
        chunks = []
        current_chunk = []
        current_start_idx = 0
        current_tokens = 0

        # Estimate tokens for the base prompt template
        base_prompt = self._get_base_prompt([], 1, (0, 0))
        base_tokens = self.estimate_tokens(base_prompt)

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.estimate_tokens(sentence)
            
            # Check if adding this sentence would exceed limits
            if (current_tokens + sentence_tokens + base_tokens > max_tokens or 
                len(current_chunk) >= chunk_size):
                
                if current_chunk:
                    # Find overlap boundary
                    overlap_size = self.find_context_boundary(
                        current_chunk[-10:],
                        self.OVERLAP_TOKENS
                    )
                    
                    # Add current chunk
                    chunks.append((current_chunk, (current_start_idx, i)))
                    
                    # Prepare next chunk with overlap
                    if overlap_size > 0:
                        current_chunk = current_chunk[-overlap_size:]
                        current_start_idx = i - overlap_size
                        current_tokens = sum(self.estimate_tokens(s) for s in current_chunk)
                    else:
                        current_chunk = []
                        current_start_idx = i
                        current_tokens = 0
                    continue
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            i += 1

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append((current_chunk, (current_start_idx, len(sentences))))

        return chunks

    def _get_base_prompt(self, sentences: List[str], batch_num: int, batch_range: tuple) -> str:
        """Generate a prompt for ranking sentences by their contribution to a coherent narrative summary.
        
        Args:
            sentences: List of transcript sentences to rank
            batch_num: Current batch number (for multi-batch processing)
            batch_range: Tuple of (start_idx, end_idx) for current batch
            
        Returns:
            Structured prompt focused on narrative flow
        """
        
        # Context for narrative continuity across batches
        if batch_num == 1:
            additional_context = ""
        else:
            additional_context = """
Narrative Continuity Guidance:
- Identify sentences that bridge previous and current batches
- Prioritize transitional phrases that maintain flow (e.g., "As we discussed earlier...")
- Note recurring narrative threads that gain momentum""".format(
            batch_num=batch_num,
            start=batch_range[0],
            end=batch_range[1]
        )

        sentences_list = [f"ID: {idx} | Sentence: \"{sentence}\"" 
                         for idx, sentence in enumerate(sentences)]
        
        return f"""You are a narrative flow expert. Rank these {len(sentences)} sentences by their contribution to a coherent, story-like summary.

### Input Format:
`ID: X | Sentence: "<text>"` (X = 0-indexed unique ID)

### Narrative Priority Criteria:
1. **Core Narrative**: Sentences advancing the main story/argument (key events, turning points)
2. **Connective Tissue**: Transitions, context-setting, and logical bridges between ideas  
3. **Impactful Details**: Vivid examples or data that reinforce the narrative  
4. **Filler**: Repetitive, off-topic, or socially-driven content (greetings, small talk)

### Ranking Rules:
1. Assign UNIQUE ranks 1-{len(sentences)} (1 = most narratively important)
2. Preserve cause-effect relationships where possible
3. Prioritize sentences that would appear in a "highlight reel" of the transcript

### Output Format (JSON):
{{
  "sentence_ranks": {{
    "ID: 0": <int>,  // Rank for sentence 0
    ...
    "ID: {len(sentences)-1}": <int>
  }},
  "narrative_arc": {{
    "opening": ["ID: X", ...],  // 1-2 sentences setting the stage
    "climax": ["ID: Y", ...],  // 1-3 peak momentum sentences
    "resolution": ["ID: Z", ...]  // 1-2 concluding sentences
  }}
}}

{additional_context.strip()}

### Sentences to Rank:
{chr(10).join(sentences_list)}"""

    def call_gpt(self, prompt: str, timeout: int = 180) -> str:
        """Calls the GPT API with the given prompt."""
        print("Debug - callGPT URL:", self.url)

        payload = {
            "messages": [
                {"role": "system", "content": "You are a professional extractive transcript summarization assistant that responds with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000,
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }

        try:
            r = requests.post(
                self.url,
                headers={
                    "api-key": self.api_key,
                    "Content-Type": "application/json"
                },
                stream=False,
                json=payload,
                timeout=timeout
            )
            r.raise_for_status()
            response_json = r.json()

            if not response_json:
                return "error: empty response"
            if 'choices' not in response_json:
                return f"error: missing choices field - {response_json.get('error', {}).get('message', 'unknown error')}"
            if not response_json['choices']:
                return "error: empty choices array"

            return response_json['choices'][0]['message']['content']

        except requests.exceptions.RequestException as e:
            return f"error: request failed - {str(e)}"
        except Exception as e:
            return f"error: unexpected - {str(e)}"

    def score_sentences(self, sentences: List[Dict], batch_size: int = 390, overlap: int = 20) -> Dict[int, float]:
        """Base method for scoring sentences. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement score_sentences")

    def calculate_semantic_similarity(self, sentence1: str, sentence2: str) -> float:
        """Calculate semantic similarity between two sentences using Sentence-BERT."""
        embeddings = self.model.encode([sentence1, sentence2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return similarity

    def remove_similar_sentences(self, sentences: List[Dict], selected_indices: List[int]) -> List[int]:
        """Remove semantically similar sentences from the selection."""
        if not selected_indices:
            return selected_indices

        # Sort indices to process in chronological order
        selected_indices.sort()
        filtered_indices = [selected_indices[0]]  # Keep the first sentence

        for i in range(1, len(selected_indices)):
            current_idx = selected_indices[i]
            current_sentence = sentences[current_idx]['sentence']
            
            # Check similarity with previous sentences
            is_similar = False
            for prev_idx in filtered_indices[-3:]:  # Compare with last 3 sentences
                prev_sentence = sentences[prev_idx]['sentence']
                similarity = self.calculate_semantic_similarity(current_sentence, prev_sentence)
                if similarity > self.SIMILARITY_THRESHOLD:
                    is_similar = True
                    print(f"Removing similar sentence: {current_sentence[:50]}... (similarity: {similarity:.2f})")
                    break

            if not is_similar:
                filtered_indices.append(current_idx)

        return filtered_indices

    def optimize_summary(self, sentences: List[Dict], ranks: Dict[int, int], max_duration: float) -> List[int]:
        """Selects sentences based on their ranks while staying under duration."""
        sentence_data = []
        
        for idx, sentence in enumerate(sentences):
            rank = ranks.get(idx, len(sentences))
            duration = sentence.get('duration', 1)
            if duration <= 0:
                continue
            
            sentence_data.append((idx, rank, duration))
        
        # Sort by rank (ascending - lower rank = more important)
        sentence_data.sort(key=lambda x: x[1])
        
        selected_indices = []
        total_duration = 0
        
        # First pass: select sentences based on rank
        for idx, rank, duration in sentence_data:
            if total_duration + duration <= max_duration:
                selected_indices.append(idx)
                total_duration += duration

        # Second pass: remove similar sentences
        selected_indices = self.remove_similar_sentences(sentences, selected_indices)
        
        # Final pass: ensure we're under duration limit
        final_indices = []
        total_duration = 0
        for idx in selected_indices:
            duration = sentences[idx].get('duration', 1)
            if total_duration + duration <= max_duration:
                final_indices.append(idx)
                total_duration += duration

        return final_indices

    def clean_and_split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # First, normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        return sentences 