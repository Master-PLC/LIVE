"""
LIVE Benchmark Evaluation Script
A comprehensive evaluation tool for multi-image visual hallucination testing in Vision Language Models
"""

import json
from openai import OpenAI
from typing import List, Dict, Set, Optional, Tuple
import os
from tqdm import tqdm
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import httpx
from datetime import datetime

# Try to import local config
CONFIG_FILE = None
try:
    from config import *
    CONFIG_FILE = True
except ImportError:
    CONFIG_FILE = False
    print("No config.py found, using default values. Please copy config_example.py to config.py and configure your API settings.")

# Default values if config not found
if not CONFIG_FILE:
    API_KEY = "your-api-key"
    BASE_URL = "http://localhost:8000/v1"
    MODEL_NAME = ""
    TIMEOUT = 60.0
    CONNECT_TIMEOUT = 10.0
    MAX_RETRIES = 3
    INITIAL_DELAY = 2
    TEMPERATURE = 0
    MAX_TOKENS = 4096
    DEFAULT_NUM_WORKERS = 20
    DEFAULT_BATCH_SIZE = 5000
    DEFAULT_CHECKPOINT_INTERVAL = 50
    DEFAULT_BATCH_DELAY = 1
    DEFAULT_IMAGE_DIR = "/ossfs/workspace/LIVE_benchmark/data/COCO/val2014"
    DEFAULT_INSTRUCTION = "Answer with 'Yes' or 'No' first, and then provide your reasoning."
    SAVE_FULL_RESPONSES = True

def load_json(file_path: str) -> List[Dict]:
    """Load JSON file safely"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return []

def save_json(data: List[Dict], file_path: str):
    """Save JSON file safely"""
    try:
        abs_path = os.path.abspath(file_path)
        dir_path = os.path.dirname(abs_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(abs_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")

def load_checkpoint(output_file: str) -> Dict[int, Dict]:
    """Load existing results as checkpoint"""
    if os.path.exists(output_file):
        try:
            existing_data = load_json(output_file)
            print(f"Found existing results with {len(existing_data)} entries")
            # Convert to dictionary format with index as key
            return {i: entry for i, entry in enumerate(existing_data)}
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return {}
    return {}

def is_entry_complete(entry: Dict) -> bool:
    """Check if entry has been completely processed"""
    return ('yes_answer' in entry and 'no_answer' in entry and
            entry['yes_answer'] not in ['error', None] and
            entry['no_answer'] not in ['error', None])

def get_local_image_path(original_path: str, local_base_dir: str) -> str:
    """Get local image path from original path"""
    file_name = os.path.basename(original_path)
    return os.path.join(local_base_dir, file_name)

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encode image to base64 string"""
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def create_vlm_request(question: str, base64_images: List[str],
                      instruction: str = DEFAULT_INSTRUCTION) -> List[Dict]:
    """Create VLM request content with images and question"""
    content = []

    # Add images
    for base64_image in base64_images:
        content.append({
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        })

    # Add question with instruction
    content.append({
        'type': 'text',
        'text': f'''{question}\n\n{instruction}'''
    })

    return content

def query_vlm(client: OpenAI, model_name: str, question: str,
              base64_images: List[str], max_retries: int = MAX_RETRIES,
              initial_delay: int = INITIAL_DELAY) -> Tuple[str, str]:
    """Query VLM with retry mechanism"""

    for attempt in range(max_retries):
        try:
            content = create_vlm_request(question, base64_images)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': content
                }],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )

            answer = response.choices[0].message.content.strip().lower()
            answer_prefix = answer[:20]  # Check first 20 characters

            # Extract yes/no from response
            if 'yes' in answer_prefix:
                simple_answer = 'yes'
            elif 'no' in answer_prefix:
                simple_answer = 'no'
            else:
                # Model didn't follow instruction, return full response
                simple_answer = answer

            return simple_answer, answer if SAVE_FULL_RESPONSES else simple_answer

        except Exception as e:
            print(f"\n[Warning] VLM query error (Attempt {attempt + 1}/{max_retries}): {e}")

            if attempt == max_retries - 1:
                print(f"Failed to query VLM after {max_retries} attempts")
                return "error", "error"

            # Exponential backoff
            delay = initial_delay * (2 ** attempt)
            print(f"Waiting {delay}s before retry...")
            time.sleep(delay)

    return "error", "error"

def process_data_entry(client: OpenAI, model_name: str, entry: Dict,
                      index: int, base64_images: List[str]) -> Tuple[int, Dict]:
    """Process single data entry with both questions"""

    if not base64_images:
        print(f"Warning: No images found for entry {index}")
        entry.update({
            'yes_answer': 'error',
            'no_answer': 'error',
            'error': 'Missing images'
        })
        return (index, entry)

    # Get both questions
    yes_question = entry.get('yes_question', '').strip()
    no_question = entry.get('no_question', '').strip()

    if not yes_question or not no_question:
        print(f"Warning: Missing questions in entry {index}")
        entry.update({
            'yes_answer': 'error',
            'no_answer': 'error',
            'error': 'Missing questions'
        })
        return (index, entry)

    # Process yes question
    yes_answer, yes_full = query_vlm(client, model_name, yes_question, base64_images)
    entry['yes_answer'] = yes_answer

    # Process no question
    no_answer, no_full = query_vlm(client, model_name, no_question, base64_images)
    entry['no_answer'] = no_answer

    # Save full responses if configured
    if SAVE_FULL_RESPONSES:
        entry['yes_response'] = yes_full
        entry['no_response'] = no_full

    # Add processing metadata
    entry['processed_at'] = datetime.now().isoformat()
    entry['model_name'] = model_name

    return (index, entry)

def preprocess_and_cache_images(data: List[Dict], local_base_dir: str) -> Dict[str, str]:
    """Pre-encode all unique images for faster evaluation"""

    print("Discovering unique images...")
    unique_paths = set()
    for entry in data:
        for img_path in entry.get('image_id', []):
            local_path = get_local_image_path(img_path, local_base_dir)
            unique_paths.add(local_path)

    print(f"Found {len(unique_paths)} unique images. Starting encoding...")

    cache = {}
    missing_files = 0

    for local_path in tqdm(unique_paths, desc="Encoding images"):
        if not os.path.exists(local_path):
            print(f"Warning: Image not found: {local_path}")
            missing_files += 1
            continue

        encoded = encode_image_to_base64(local_path)
        if encoded:
            cache[local_path] = encoded
        else:
            missing_files += 1

    print(f"Successfully encoded {len(cache)} images. {missing_files} files missing or failed.")
    return cache

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze evaluation results and compute statistics"""

    total = len(results)
    completed = sum(1 for r in results if is_entry_complete(r))

    stats = {
        'total_entries': total,
        'completed_entries': completed,
        'completion_rate': completed / total if total > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }

    # Task type breakdown
    yes_correct = 0
    no_correct = 0
    both_correct = 0

    for entry in results:
        if not is_entry_complete(entry):
            continue

        # Check if answers match expected (yes_question should be 'yes', no_question should be 'no')
        yes_correct += (entry['yes_answer'] == 'yes')
        no_correct += (entry['no_answer'] == 'no')
        both_correct += (entry['yes_answer'] == 'yes' and entry['no_answer'] == 'no')

    if completed > 0:
        stats.update({
            'yes_question_accuracy': yes_correct / completed,
            'no_question_accuracy': no_correct / completed,
            'overall_accuracy': both_correct / completed,
            'hallucination_rate': 1 - (both_correct / completed)
        })

    return stats

def main():
    """Main evaluation function"""

    parser = argparse.ArgumentParser(
        description='LIVE Benchmark - Multi-Image VLM Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluation.py --input questions.json --output results.json --api-key your-key

  # Resume evaluation from checkpoint
  python evaluation.py --input questions.json --output results.json --resume

  # Custom batch size and workers
  python evaluation.py --input questions.json --output results.json --workers 30 --batch-size 1000
        """
    )

    # Required arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input JSON file with questions')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to output JSON file for results')

    # API configuration
    parser.add_argument('--api-key', type=str, default=API_KEY,
                        help='API key for VLM service')
    parser.add_argument('--base-url', type=str, default=BASE_URL,
                        help='Base URL for VLM API')
    parser.add_argument('--model', type=str, default=MODEL_NAME,
                        help='Specific model name to use')

    # Paths
    parser.add_argument('--image-dir', type=str, default=DEFAULT_IMAGE_DIR,
                        help='Directory containing MS-COCO images')

    # Processing settings
    parser.add_argument('--workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help='Number of concurrent workers (default: 20)')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Process entries in batches (default: 5000)')
    parser.add_argument('--checkpoint-interval', type=int, default=DEFAULT_CHECKPOINT_INTERVAL,
                        help='Save checkpoint every N entries (default: 50)')
    parser.add_argument('--batch-delay', type=int, default=DEFAULT_BATCH_DELAY,
                        help='Delay between batches in seconds (default: 1)')

    # Other options
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file')
    parser.add_argument('--instruction', type=str, default=DEFAULT_INSTRUCTION,
                        help='Instruction appended to questions')
    parser.add_argument('--save-stats', action='store_true',
                        help='Save statistics summary')
    parser.add_argument('--stats-file', type=str, default="",
                        help='Path for statistics file (default: output_stats.json)')

    args = parser.parse_args()

    # Validate arguments
    if not args.api_key or args.api_key == "your-api-key":
        print("Error: Please provide a valid API key with --api-key or set it in config.py")
        return

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return

    # Create output directory
    out_abs = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Set stats file path
    if args.save_stats and not args.stats_file:
        base_name = os.path.splitext(args.output)[0]
        args.stats_file = f"{base_name}_stats.json"

    print("="*60)
    print("LIVE Benchmark Evaluation")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print(f"Resume: {args.resume}")
    print("="*60)

    # Load data
    print(f"Loading data from {args.input}...")
    data = load_json(args.input)
    print(f"Loaded {len(data)} entries")

    if not data:
        print("No data to process")
        return

    # Pre-encode images
    print("Pre-encoding images...")
    image_cache = preprocess_and_cache_images(data, args.image_dir)

    if not image_cache:
        print("No images could be loaded. Check --image-dir path")
        return

    # Load checkpoint
    results = {}
    if args.resume:
        results = load_checkpoint(args.output)
        completed = sum(1 for entry in results.values() if is_entry_complete(entry))
        print(f"Resumed from checkpoint: {completed}/{len(data)} entries completed")

    # Prepare tasks
    tasks_to_process = []
    skipped = 0

    print("Preparing tasks...")
    for i, entry in enumerate(data):
        # Skip if already processed
        if i in results and is_entry_complete(results[i]):
            continue

        # Prepare base64 images
        base64_images = []
        image_paths = entry.get('image_id', [])
        has_missing = False

        for img_path in image_paths:
            local_path = get_local_image_path(img_path, args.image_dir)
            if local_path in image_cache and image_cache[local_path]:
                base64_images.append(image_cache[local_path])
            else:
                has_missing = True
                break

        if not has_missing and base64_images:
            tasks_to_process.append((i, entry, base64_images))
        else:
            skipped += 1

    print(f"Prepared {len(tasks_to_process)} tasks, {skipped} skipped (missing images)")

    if not tasks_to_process:
        print("No tasks to process")
        return

    # Create batches
    def create_batches(tasks, batch_size):
        for i in range(0, len(tasks), batch_size):
            yield tasks[i:i+batch_size]

    batches = list(create_batches(tasks_to_process, args.batch_size))
    total_batches = len(batches)

    print(f"Created {total_batches} batches")

    # Process batches
    lock = threading.Lock()
    processed_count = 0

    for batch_num, current_batch in enumerate(batches):
        print(f"\nProcessing batch {batch_num + 1}/{total_batches} ({len(current_batch)} tasks)")

        # Create new client for this batch
        client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=httpx.Timeout(TIMEOUT, connect=CONNECT_TIMEOUT)
        )

        # Detect model name if not specified
        if not args.model:
            try:
                model_name = client.models.list().data[0].id
                print(f"Using model: {model_name}")
            except Exception as e:
                print(f"Error detecting model: {e}")
                client.close()
                continue
        else:
            model_name = args.model
            if batch_num == 0:
                print(f"Using model: {model_name}")

        # Process batch
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_data_entry, client, model_name,
                              entry, idx, base64_images)
                for idx, entry, base64_images in current_batch
            ]

            # Process results with progress bar
            batch_results = []
            for future in tqdm(as_completed(futures), total=len(futures),
                             desc=f"Batch {batch_num + 1}"):
                try:
                    idx, processed_entry = future.result()

                    with lock:
                        results[idx] = processed_entry
                        processed_count += 1

                        # Save checkpoint
                        if processed_count % args.checkpoint_interval == 0:
                            sorted_results = [results[i] for i in sorted(results.keys())]
                            save_json(sorted_results, args.output)

                except Exception as e:
                    print(f"\nError processing task: {e}")

        # Close client
        client.close()

        # Save batch results
        print(f"Batch {batch_num + 1} complete. Saving checkpoint...")
        sorted_results = [results[i] for i in sorted(results.keys()) if i in results]

        # Merge with original data for final output
        final_results = []
        for i, original_entry in enumerate(data):
            if i in results:
                final_results.append(results[i])
            else:
                final_results.append(original_entry)

        save_json(final_results, args.output)

        # Delay between batches (except last)
        if batch_num < total_batches - 1:
            print(f"Waiting {args.batch_delay}s before next batch...")
            time.sleep(args.batch_delay)

    # Final processing
    print("\nProcessing complete!")

    # Create final results
    final_results = []
    for i, original_entry in enumerate(data):
        if i in results:
            final_results.append(results[i])
        else:
            final_results.append(original_entry)

    save_json(final_results, args.output)

    # Analyze results
    completed = sum(1 for e in final_results if is_entry_complete(e))
    print(f"\nEvaluation summary:")
    print(f"Total entries: {len(final_results)}")
    print(f"Completed: {completed} ({100*completed/len(final_results):.1f}%)")

    if args.save_stats and completed > 0:
        stats = analyze_results(final_results)
        save_json(stats, args.stats_file)
        print(f"Statistics saved to {args.stats_file}")

        print(f"\nDetailed statistics:")
        print(f"Yes question accuracy: {100*stats.get('yes_question_accuracy', 0):.2f}%")
        print(f"No question accuracy: {100*stats.get('no_question_accuracy', 0):.2f}%")
        print(f"Overall accuracy: {100*stats.get('overall_accuracy', 0):.2f}%")
        print(f"Hallucination rate: {100*stats.get('hallucination_rate', 0):.2f}%")

if __name__ == "__main__":
    main()