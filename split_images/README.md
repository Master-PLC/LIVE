# Image Clustering and Sampling Module

## Overview

This module implements advanced image clustering and sampling algorithms used to construct the multi-image scenarios in the LIVE benchmark. It ensures balanced representation of both UIC (Uniform Image Contexts) and DIC (Diverse Image Contexts) scenarios.

## Pipeline Steps

### Step 1: K-Means Clustering (`step1_k_experiments_results.py`)
Determines optimal number of clusters for MS-COCO images:
- Extracts CLIP features from all validation images
- Tests multiple K values (100, 500, 1000, 2000)
- Analyzes cluster quality using silhouette scores
- Selects K=1000 as optimal based on experiments

**Key Features:**
- Efficient feature extraction using batch processing
- Multiple K value testing with parallel execution
- Quality metrics analysis and visualization

### Step 2: Image Clustering (`step2_split_images.py`)
Clusters images and samples diverse combinations:
- Performs K-means clustering with K=1000
- Creates cluster-to-images mapping
- Generates structured image combinations (AAAA, ABCD patterns)
- Ensures balanced distribution across clusters

**Sampling Patterns:**
- **AAAA**: All 4 images from same cluster (UIC scenario)
- **ABCD**: Each image from different cluster (DIC scenario)
- **AABC**: Mixed patterns for intermediate scenarios

### Step 3: Visualization (`step3_visual_images_and_add_caption.ipynb`)
Jupyter notebook for:
- Visualizing clustered images
- Checking category distribution
- Adding enrichment metadata
- Quality validation of sampled combinations

## Usage

### Run Complete Pipeline
```bash
# Step 1: Find optimal K
python step1_k_experiments_results.py

# Step 2: Cluster and sample
python step2_split_images.py

# Step 3: Visualize (in Jupyter)
jupyter notebook step3_visual_images_and_add_caption.ipynb
```

### Parameters
```python
# Key parameters in step2_split_images.py
CLIP_MODEL = 'ViT-B/32'
NUM_CLUSTERS = 1000
SAMPLING_RATIO = 0.1  # Sample 10% of possible combinations
RANDOM_SEED = 42

# UIC/DIC balance
UIC_PERCENTAGE = 0.5  # 50% UIC scenarios
DIC_MIN_CLUSTER_DIFF = 200  # Minimum cluster distance for DIC
```

## Output Files

### cluster_inverted_index_k1000.json
Maps clusters to list of images:
```json
{
  "0": ["COCO_val2014_000000000001.jpg", "COCO_val2014_000000000002.jpg", ...],
  "1": ["COCO_val2014_000000000003.jpg", "COCO_val2014_000000000004.jpg", ...],
  ...
}
```

### sampled_all_tuples_k1000.json
Sampled image combinations with metadata:
```json
[
  {
    "combination": ["COCO_val2014_xxxx.jpg", ...],
    "clusters": [123, 456, 123, 123],
    "type": "UIC",  // or "DIC"
    "pattern": "AABC"
  },
  ...
]
```

### k_experiments_results/
Experimental results determining optimal K:
- `k_experiment_results.json`: Raw metrics
- `k_analysis_plot.pdf/png`: Visualization

## Technical Details

### CLIP Feature Extraction
- Uses ViT-B/32 model by default
- Features normalized to unit vectors
- Batch size optimized for GPU memory

### K-Means Implementation
- Mini-batch K-means for scalability
- Multiple random initializations
- Convergence monitoring

### Sampling Strategy
- Stratified sampling across clusters
- Ensures geometric diversity
- Balanced UIC/DIC scenario generation

## Performance
- Processes MS-COCO val2014 (~41K images) in <2 hours
- K=1000 clustering: ~15 minutes
- Feature extraction: ~30 minutes
- Sampling: ~5 minutes
- Memory usage: ~8GB for full processing

## Dependencies
- torch & torchvision
- clip (OpenCLIP)
- scikit-learn
- numpy
- pandas
- tqdm
- pillow