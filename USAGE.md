# LIVE Benchmark Usage Guide

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Copy configuration template
cp config_example.py config.py

# Edit config.py with your API credentials
```

### 2. Download MS-COCO Images
```bash
# Create data directory
mkdir -p data/coco_val2014

# Download MS-COCO val2014 images (approximately 41K images)
# From official website or use script
```

### 3. Load Dataset
```bash
# Load LIVE benchmark dataset from Hugging Face
python load_data.py

# Or save to local JSON
python load_data.py > LIVE_dataset.json
```

### 4. Run Evaluation
```bash
# Basic evaluation
python evaluation.py \
    --input LIVE_dataset.json \
    --output results.json \
    --api-key YOUR_API_KEY

# With specific model
python evaluation.py \
    --input LIVE_dataset.json \
    --output results_gpt4o.json \
    --api-key YOUR_API_KEY \
    --model gpt-4o \
    --base-url https://your-api.com/v1

# Custom settings
python evaluation.py \
    --input LIVE_dataset.json \
    --output results.json \
    --workers 30 \
    --batch-size 1000 \
    --checkpoint-interval 100
```

### 5. Run Example Evaluation
```bash
# Quick test with sample data
python example_evaluation.py \
    --api-key YOUR_API_KEY \
    --model gpt-4o

# Analyze existing results
python example_evaluation.py \
    --analyze results.json
```

## Advanced Usage

### Resume Interrupted Evaluation
```bash
python evaluation.py \
    --input LIVE_dataset.json \
    --output results.json \
    --api-key YOUR_API_KEY \
    --resume
```

### Filter Dataset for Specific Tasks
```python
# In Python
from load_data import LIVEDataLoader

loader = LIVEDataLoader()
dataset = loader.load_dataset()

# Get sentiment evaluation samples
sentiment_samples = loader.get_samples_by_question_class('Sentiment')

# Get UIC (Uniform Image Context) samples
uic_samples = loader.get_samples_by_type('UIC')

# Save filtered data
loader.save_to_json("sentiment_samples.json")
```

### Custom API Integration
Edit `evaluation.py` or create a new script based on the evaluation module:

```python
from evaluation import query_vlm, create_vlm_request

# Your custom implementation
class CustomEvaluator:
    def evaluate_model(self, question, images):
        # Your model-specific logic here
        pass
```

## Configuration Options

### API Settings (config.py)
- `API_KEY`: Your VLM service API key
- `BASE_URL`: API endpoint URL
- `MODEL_NAME`: Specific model to use
- `MAX_RETRIES`: Number of retry attempts (default: 3)
- `TIMEOUT`: Request timeout in seconds (default: 60)

### Processing Parameters
- `--workers`: Concurrent request threads (default: 20)
- `--batch-size`: Entries per batch (default: 5000)
- `--checkpoint-interval`: Save frequency (default: 50)
- `--batch-delay`: Delay between batches (default: 1s)

### Evaluation Settings
- `--temperature`: Model response randomness (default: 0 for deterministic)
- `--max-tokens`: Maximum response length (default: 4096)
- `--save-stats`: Generate detailed statistics file

## Common Workflows

### 1. Full Benchmark Evaluation
```bash
# Download dataset from Hugging Face
python -c "from load_data import LIVEDataLoader; l=LIVEDataLoader(); l.save_to_json('LIVE_full.json')"

# Run complete evaluation (this will take several hours)
python evaluation.py \
    --input LIVE_full.json \
    --output results_full.json \
    --api-key YOUR_API_KEY \
    --workers 50 \
    --batch-size 5000
```

### 2. Task-Specific Evaluation
```bash
# Extract only color-related questions
python -c "
from load_data import LIVEDataLoader
l = LIVEDataLoader()
l.load_dataset()
color_samples = l.get_samples_by_question_class('Color')
import json
with open('color_samples.json','w') as f:
    json.dump(color_samples, f, indent=2)
"

# Evaluate only color questions
python evaluation.py \
    --input color_samples.json \
    --output color_results.json \
    --api-key YOUR_API_KEY
```

### 3. Multi-Scenario Comparison
```bash
# Extract UIC samples
python -c "
from load_data import LIVEDataLoader
l = LIVEDataLoader()
l.load_dataset()
uic = l.get_samples_by_type('UIC')
with open('uic_samples.json','w') as f:
    json.dump(uic, f, indent=2)
"

# Run UIC evaluation
python evaluation.py \
    --input uic_samples.json \
    --output uic_results.json \
    --api-key YOUR_API_KEY
```

## Results Format

### Main Results File (JSON)
Each entry contains:
```json
{
  "task": "attributes",
  "type": "UIC",
  "qtype": "2",
  "image_id": ["COCO_val2014_xxxx.jpg", "COCO_val2014_yyyy.jpg"],
  "yes_question": "Is the sky blue in image 1?",
  "no_question": "Is the sky red in image 1?",
  "ritem": "sky is blue",
  "hitem": "sky is red",
  "yes_question_class": "Color",
  "no_question_class": "Color",
  "yes_answer": "yes",
  "no_answer": "no",
  "yes_response": "Yes, the sky appears blue in image 1...",
  "no_response": "No, the sky is not red in image 1...",
  "processed_at": "2024-03-25T10:30:00",
  "model_name": "gpt-4o"
}
```

### Statistics File (`_stats.json`)
```json
{
  "total_entries": 32544,
  "completed_entries": 32500,
  "completion_rate": 0.9986,
  "yes_question_accuracy": 0.85,
  "no_question_accuracy": 0.80,
  "overall_accuracy": 0.68,
  "hallucination_rate": 0.32,
  "timestamp": "2024-03-25T12:00:00"
}
```

## Performance Tips

1. **Parallel Processing**: Increase `--workers` for faster evaluation
2. **Batch Size**: Adjust based on API rate limits
3. **Checkpoints**: Use `--resume` to continue interrupted evaluations
4. **Error Handling**: Monitor logs for failed requests and retry if needed
5. **Resource Usage**: Monitor memory usage with large image batches

## Troubleshooting

### Common Issues

1. **Image Not Found**
   - Ensure MS-COCO images are in the correct directory
   - Check file paths in dataset

2. **API Rate Limits**
   - Reduce `--workers`
   - Increase `--batch-delay`
   - Use different API endpoints

3. **Memory Issues**
   - Reduce `--batch-size`
   - Process in smaller chunks
   - Use image resizing if needed

4. **Authentication Errors**
   - Verify API key is correct
   - Check base URL format
   - Ensure API permissions

### Getting Help
- Check existing issues on GitHub
- Read the full README.md
- Examine example scripts in the repository