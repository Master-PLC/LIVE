#!/usr/bin/env python3
"""
Example script for running LIVE benchmark evaluation
Demonstrates how to use the evaluation module with different configurations
"""

import os
import sys
import json
import argparse
from datetime import datetime
import signal

def create_sample_data():
    """Create a small sample dataset for testing"""
    sample_data = [
        {
            "task": "attributes",
            "type": "UIC",
            "qtype": "2",
            "image_id": [
                "COCO_val2014_000000001234.jpg",
                "COCO_val2014_000000004567.jpg"
            ],
            "yes_question": "Is the sky blue in image 1?",
            "no_question": "Is the sky red in image 1?",
            "ritem": "sky is blue",
            "hitem": "sky is red",
            "yes_question_class": "Color",
            "no_question_class": "Color"
        },
        {
            "task": "objects",
            "type": "DIC",
            "qtype": "3",
            "image_id": [
                "COCO_val2014_000000007890.jpg",
                "COCO_val2014_000000001111.jpg",
                "COCO_val2014_000000002222.jpg"
            ],
            "yes_question": "Is there a dog in any of image 1, image 2 and image 3?",
            "no_question": "Is there a cat in any of image 1, image 2 and image 3?",
            "ritem": "dog is present",
            "hitem": "cat is present",
            "yes_question_class": "Object",
            "no_question_class": "Object"
        },
        {
            "task": "sentiment",
            "type": "UIC",
            "qtype": "1",
            "image_id": ["COCO_val2014_000000003333.jpg"],
            "yes_question": "Is the person smiling in image 1?",
            "no_question": "Is the person frowning in image 1?",
            "ritem": "person is smiling",
            "hitem": "person is frowning",
            "yes_question_class": "Sentiment",
            "no_question_class": "Sentiment"
        }
    ]
    return sample_data

def run_evaluation(input_file, output_file, api_key, base_url=None, model=None,
                  image_dir=None, workers=5, batch_size=100, save_stats=True):
    """Run evaluation using the evaluation.py script"""

    cmd = [
        sys.executable, "evaluation.py",
        "--input", input_file,
        "--output", output_file,
        "--api-key", api_key,
        "--workers", str(workers),
        "--batch-size", str(batch_size)
    ]

    if base_url:
        cmd.extend(["--base-url", base_url])

    if model:
        cmd.extend(["--model", model])

    if image_dir:
        cmd.extend(["--image-dir", image_dir])

    if save_stats:
        cmd.append("--save-stats")

    print(f"Running command: {' '.join(cmd)}")

    # Execute the command
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running evaluation: {result.stderr}")
        return False

    print(result.stdout)
    return True

def analyze_results(results_file, stats_file=None):
    """Analyze and display results"""

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        total = len(results)
        completed = sum(1 for r in results if 'yes_answer' in r and r['yes_answer'] != 'error')

        print("\n" + "="*50)
        print("EVALUATION RESULTS SUMMARY")
        print("="*50)
        print(f"Total samples: {total}")
        print(f"Completed samples: {completed}")
        print(f"Success rate: {100*completed/total:.1f}%")

        if completed > 0:
            # Check accuracy (yes questions should be 'yes', no questions should be 'no')
            yes_correct = sum(1 for r in results if r.get('yes_answer') == 'yes')
            no_correct = sum(1 for r in results if r.get('no_answer') == 'no')
            both_correct = sum(1 for r in results if r.get('yes_answer') == 'yes' and r.get('no_answer') == 'no')

            print(f"\nAccuracy Analysis:")
            print(f"  Yes questions correct: {yes_correct}/{completed} ({100*yes_correct/completed:.1f}%)")
            print(f"  No questions correct: {no_correct}/{completed} ({100*no_correct/completed:.1f}%)")
            print(f"  Both questions correct: {both_correct}/{completed} ({100*both_correct/completed:.1f}%)")
            print(f"  Hallucination rate: {100*(1-both_correct/completed):.1f}%")

            # Per-task analysis
            tasks = defaultdict(lambda: {'total': 0, 'both_correct': 0})
            types = defaultdict(lambda: {'total': 0, 'both_correct': 0})
            qtypes = defaultdict(lambda: {'total': 0, 'both_correct': 0})

            for r in results:
                if 'yes_answer' in r and r['yes_answer'] != 'error':
                    task = r.get('task', 'unknown')
                    task_type = r.get('type', 'unknown')
                    qtype = r.get('qtype', 'unknown')

                    tasks[task]['total'] += 1
                    types[task_type]['total'] += 1
                    qtypes[qtype]['total'] += 1

                    if r.get('yes_answer') == 'yes' and r.get('no_answer') == 'no':
                        tasks[task]['both_correct'] += 1
                        types[task_type]['both_correct'] += 1
                        qtypes[qtype]['both_correct'] += 1

            print(f"\nPer-Task Results:")
            for task, stats in sorted(tasks.items()):
                accuracy = 100 * stats['both_correct'] / stats['total']
                print(f"  {task}: {stats['both_correct']}/{stats['total']} ({accuracy:.1f}%)")

            print(f"\nPer-Type Results:")
            for task_type, stats in sorted(types.items()):
                accuracy = 100 * stats['both_correct'] / stats['total']
                print(f"  {task_type}: {stats['both_correct']}/{stats['total']} ({accuracy:.1f}%)")

            print(f"\nPer-Qtype Results:")
            for qtype, stats in sorted(qtypes.items()):
                accuracy = 100 * stats['both_correct'] / stats['total']
                print(f"  {qtype} images: {stats['both_correct']}/{stats['total']} ({accuracy:.1f}%)")

        # Load stats file if provided
        if stats_file and os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            print(f"\nDetailed statistics saved to: {stats_file}")

    except Exception as e:
        print(f"Error analyzing results: {e}")
        return False

    return True

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\nReceived interrupt signal. Exiting gracefully...")
    sys.exit(0)

def main():
    """Main example function"""

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="Example script for LIVE benchmark evaluation",
        epilog="""
Examples:
  # Run with sample data
  python example_evaluation.py --api-key YOUR_API_KEY

  # Run with custom model
  python example_evaluation.py --api-key YOUR_API_KEY --model gpt-4o --base-url YOUR_URL

  # Analyze existing results
  python example_evaluation.py --analyze results.json
        """
    )

    parser.add_argument("--api-key", type=str, help="VLM API key")
    parser.add_argument("--base-url", type=str, default=None,
                        help="VLM API base URL (default: None)")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model name (e.g., gpt-4o, claude-3-opus)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory with COCO images")
    parser.add_argument("--input", type=str, default="sample_data.json",
                        help="Input JSON file (will create sample if not exists)")
    parser.add_argument("--output", type=str, default="sample_results.json",
                        help="Output JSON file")
    parser.add_argument("--create-sample", action="store_true",
                        help="Create sample data and exit")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Analyze existing results file and exit")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of concurrent workers")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for processing")
    parser.add_argument("--no-stats", action="store_true",
                        help="Don't save statistics file")

    args = parser.parse_args()

    # Handle analyze mode
    if args.analyze:
        return analyze_results(args.analyze)

    # Create sample data if requested or if input doesn't exist
    if args.create_sample or not os.path.exists(args.input):
        print("Creating sample data...")
        sample_data = create_sample_data()
        with open(args.input, 'w') as f:
            json.dump(sample_data, f, indent=2)
        print(f"Sample data created: {args.input}")

        if args.create_sample:
            return True

    # Check API key
    if not args.api_key:
        print("Error: Please provide --api-key for the VLM service")
        print("Get a free API key from: https://platform.openai.com or similar service")
        return False

    print("="*60)
    print("LIVE Benchmark - Example Evaluation")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model or 'Auto-detected'}")
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print("="*60)

    # Run evaluation
    success = run_evaluation(
        input_file=args.input,
        output_file=args.output,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        image_dir=args.image_dir,
        workers=args.workers,
        batch_size=args.batch_size,
        save_stats=not args.no_stats
    )

    if success:
        # Analyze results
        stats_file = args.output.replace('.json', '_stats.json')
        analyze_results(args.output, stats_file)

        print(f"\nEvaluation complete! Results saved to:")
        print(f"  - {args.output}")
        if not args.no_stats:
            print(f"  - {stats_file}")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)