# Transcript Summarization Workflow

This project provides tools for summarizing video transcripts using different approaches, including GPT-based summarization and random sentence selection for comparison.

## Features

- **Base Summarizer**: Core functionality for processing and chunking transcripts
- **Transcript Summarizer**: GPT-based extractive summarization
- **Random Summarizer**: Random sentence selection for baseline comparison
- **Summarizer Evaluation**: Tools for evaluating summarization quality
- **MeetingBank Dataset Integration**: Support for the MeetingBank-QA-Summary dataset

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Transcript-Summarization-Workflow.git
cd Transcript-Summarization-Workflow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export URL_ENDPOINT="your_api_endpoint"
export OPENAI_API_KEY="your_api_key"
```

## Usage

### 1. GPT-based Summarization

```bash
python3 TranscriptSummarizer.py --input your_file.json --max_duration 300
```

This will:
- Process the transcript in chunks
- Use GPT to rank sentences by importance
- Generate a summary within the specified duration limit
- Maintain context between chunks

### 2. Random Sentence Selection (Baseline)

```bash
python3 RandomSentenceSelectionSummary.py --input your_file.json --compression_ratio 0.31
```

This provides a baseline by:
- Randomly selecting sentences
- Maintaining the same compression ratio as GPT summaries
- Useful for comparing against GPT-based summarization

### 3. Evaluation

```bash
python3 SummarizerEvaluation.py --num_samples 10 --compression_ratio 0.31
```

This will:
- Evaluate GPT based summarization on MeetingBankQA dataset
- Calculate ROUGE scores and QA accuracy
- Generate comparison metrics
- Save results to `results.json`

## Output Format

Results are saved in JSON format with:
- Original transcript length
- Summary length
- Compression ratio
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- QA accuracy
- Generated summary
- Reference summary

## Requirements

- OpenAI API access
- Sentence Transformers
- ROUGE score calculation
- [MeetingBankQA dataset](https://huggingface.co/datasets/meetingbank/meetingbank-qa-summary)




