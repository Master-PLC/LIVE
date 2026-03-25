"""
Data loading utilities for LIVE benchmark
Provides convenient functions to load and work with the LIVE dataset from Hugging Face
"""

from datasets import load_dataset, Dataset
import json
import os
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import pandas as pd

class LIVEDataLoader:
    """LIVE Benchmark Dataset Loader"""

    def __init__(self, dataset_name: str = "Tong613/LIVE-multi-image-bench", data_dir: str = "main"):
        """
        Initialize the dataset loader

        Args:
            dataset_name: Hugging Face dataset name
            data_dir: Subdirectory within the dataset
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.dataset = None
        self.all_data = None

    def load_dataset(self, split: str = "train") -> Optional[Dataset]:
        """
        Load the dataset from Hugging Face

        Args:
            split: Dataset split to load

        Returns:
            Dataset object or None if loading failed
        """
        try:
            print(f"Loading dataset {self.dataset_name}...")
            self.dataset = load_dataset(self.dataset_name, data_dir=self.data_dir, split=split)
            print(f"Successfully loaded {len(self.dataset)} samples")
            return self.dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def to_list(self) -> List[Dict]:
        """Convert dataset to list of dictionaries"""
        if self.dataset is None:
            self.load_dataset()
        if self.dataset is not None:
            self.all_data = [sample for sample in self.dataset]
            return self.all_data
        return []

    def save_to_json(self, output_path: str) -> bool:
        """Save dataset to JSON file"""
        if self.all_data is None:
            self.to_list()

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.all_data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(self.all_data)} samples to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving to JSON: {e}")
            return False

    def get_samples_by_type(self, type_name: str) -> List[Dict]:
        """Get samples by scenario type (UIC or DIC)"""
        if self.all_data is None:
            self.to_list()
        return [sample for sample in self.all_data if sample['type'] == type_name]

    def get_samples_by_task(self, task_name: str) -> List[Dict]:
        """Get samples by task (objects, relations, attributes, etc.)"""
        if self.all_data is None:
            self.to_list()
        return [sample for sample in self.all_data if sample['task'] == task_name]

    def get_samples_by_qtype(self, qtype: str) -> List[Dict]:
        """Get samples by question granularity (1, 2, 3, or 4)"""
        if self.all_data is None:
            self.to_list()
        return [sample for sample in self.all_data if sample['qtype'] == qtype]

    def get_samples_by_question_class(self, class_name: str) -> List[Dict]:
        """Get samples by question class (Color, Sentiment, etc.)"""
        if self.all_data is None:
            self.to_list()
        return [sample for sample in self.all_data
                if sample.get('yes_question_class') == class_name]

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if self.all_data is None:
            self.to_list()

        stats = {
            'total_samples': len(self.all_data),
            'by_type': defaultdict(int),
            'by_task': defaultdict(int),
            'by_qtype': defaultdict(int),
            'by_class': defaultdict(int),
            'unique_images': set()
        }

        for sample in self.all_data:
            stats['by_type'][sample['type']] += 1
            stats['by_task'][sample['task']] += 1
            stats['by_qtype'][sample['qtype']] += 1

            # Count question classes
            if 'yes_question_class' in sample:
                stats['by_class'][sample['yes_question_class']] += 1

            # Track unique images
            for img_id in sample.get('image_id', []):
                stats['unique_images'].add(img_id)

        stats['num_unique_images'] = len(stats['unique_images'])
        return dict(stats)

    def print_statistics(self):
        """Print dataset statistics"""
        stats = self.get_statistics()

        print("\n=== LIVE Dataset Statistics ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Unique images: {stats['num_unique_images']}")

        print("\nBy Scenario Type:")
        for type_name, count in stats['by_type'].items():
            print(f"  {type_name}: {count}")

        print("\nBy Question Granularity:")
        for qtype, count in sorted(stats['by_qtype'].items()):
            print(f"  {qtype} images: {count}")

        print("\nBy Task:")
        for task, count in stats['by_task'].items():
            print(f"  {task}: {count}")

        print("\nBy Question Class:")
        for class_name, count in sorted(stats['by_class'].items()):
            print(f"  {class_name}: {count}")

    def quick_sample(self, n: int = 5) -> List[Dict]:
        """Get a quick sample of n entries"""
        if self.all_data is None:
            self.to_list()
        return self.all_data[:n] if n <= len(self.all_data) else self.all_data

    def validate_dataset(self) -> Dict:
        """Validate dataset format and content"""
        if self.all_data is None:
            self.to_list()

        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        required_fields = ['task', 'type', 'qtype', 'image_id',
                          'yes_question', 'no_question', 'ritem', 'hitem',
                          'yes_question_class', 'no_question_class']

        for i, sample in enumerate(self.all_data):
            # Check required fields
            for field in required_fields:
                if field not in sample:
                    validation['errors'].append(f"Sample {i}: Missing field '{field}'")
                    validation['valid'] = False

            # Check field types
            if 'image_id' in sample and not isinstance(sample['image_id'], list):
                validation['errors'].append(f"Sample {i}: 'image_id' should be a list")
                validation['valid'] = False

            if 'qtype' in sample and sample['qtype'] not in ['1', '2', '3', '4']:
                validation['warnings'].append(f"Sample {i}: 'qtype'={sample['qtype']} not in [1,2,3,4]")

            if 'type' in sample and sample['type'] not in ['UIC', 'DIC']:
                validation['warnings'].append(f"Sample {i}: 'type'={sample['type']} not in [UIC,DIC]")

        return validation

def load_live_dataset(output_file: Optional[str] = None,
                     sample_tasks: Optional[List[str]] = None,
                     sample_types: Optional[List[str]] = None,
                     max_samples: Optional[int] = None) -> Tuple[Dataset, List[Dict]]:
    """
    Convenience function to load LIVE dataset with filtering options

    Args:
        output_file: Save dataset to this file if provided
        sample_tasks: Filter by specific tasks
        sample_types: Filter by scenario types (UIC/DIC)
        max_samples: Limit number of samples

    Returns:
        Tuple of (Dataset, List[Dict])
    """
    loader = LIVEDataLoader()
    dataset = loader.load_dataset()

    if dataset is None:
        return None, []

    data = loader.to_list()

    # Apply filters
    if sample_tasks:
        data = [d for d in data if d['task'] in sample_tasks]

    if sample_types:
        data = [d for d in data if d['type'] in sample_types]

    if max_samples:
        data = data[:max_samples]

    # Save if requested
    if output_file:
        loader.save_to_json(output_file)

    return loader.dataset, data

def main():
    """Example usage and data loading demo"""

    print("Loading LIVE benchmark dataset...")
    loader = LIVEDataLoader()

    # Load dataset
    loader.load_dataset()

    # Print statistics
    loader.print_statistics()

    # Validate dataset
    print("\n=== Dataset Validation ===")
    validation = loader.validate_dataset()
    print(f"Valid: {validation['valid']}")
    if validation['errors']:
        print(f"Errors: {len(validation['errors'])}")
        for error in validation['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
    if validation['warnings']:
        print(f"Warnings: {len(validation['warnings'])}")
        for warning in validation['warnings'][:5]:  # Show first 5 warnings
            print(f"  - {warning}")

    # Get sample entries
    print("\n=== Sample Entries ===")
    samples = loader.quick_sample(3)
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"  Task: {sample['task']}")
        print(f"  Type: {sample['type']}")
        print(f"  QType: {sample['qtype']}")
        print(f"  Yes Question: {sample['yes_question']}")
        print(f"  No Question: {sample['no_question']}")
        print(f"  Images: {len(sample['image_id'])} images")

    # Save to JSON
    print("\nSaving to JSON...")
    success = loader.save_to_json("LIVE_dataset.json")
    if success:
        print("Dataset saved successfully!")

    # Filter examples
    print("\n=== Filtered Examples ===")
    uic_samples = loader.get_samples_by_type('UIC')
    print(f"UIC samples: {len(uic_samples)}")

    dic_samples = loader.get_samples_by_type('DIC')
    print(f"DIC samples: {len(dic_samples)}")

    sentiment_samples = loader.get_samples_by_question_class('Sentiment')
    print(f"Sentiment samples: {len(sentiment_samples)}")

    color_samples = loader.get_samples_by_question_class('Color')
    print(f"Color samples: {len(color_samples)}")

if __name__ == "__main__":
    main()