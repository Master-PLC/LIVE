# LIVE: An LLM-assisted Multi-Image Visual Hallucination Evaluation Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

**LIVE (Likelihood-based Image Verification and Evaluation)** is a comprehensive benchmark designed to evaluate multi-image visual hallucinations in Large Vision-Language Models (LVLMs). Unlike traditional single-image benchmarks, LIVE systematically addresses the complexities of multi-image understanding with two distinct evaluation scenarios:

- **UIC (Uniform Image Contexts)**: Evaluates content confusion in similar images
- **DIC (Diverse Image Contexts)**: Evaluates context interference across different images

> **📢 Important Notice for Reviewers:** > Detailed dataset statistics, K-means clustering analysis, full LLM prompting templates, and extended experimental results can be found in our **[Supplementary Material (PDF)](./LIVE_Supplementary_Material.pdf)**. 
*(Note: adjust the path `./LIVE_Supplementary_Material.pdf` if you place it in a different folder like `./docs/`)*

## 🌟 Key Features

- **Multi-Image Scenarios**: 488 scenarios (242 UIC + 246 DIC) with 32K+ yes/no questions
- **Multi-Granularity Assessment**: Tests different numbers of target images (1-4 images)
- **Comprehensive Coverage**: 6 visual recognition tasks (Object, Material, Color, Sentiment, Action, Position)
- **Large-Scale Dataset**: Built on MS-COCO validation set with diverse everyday images
- **Benchmark Ready**: Direct integration with Hugging Face dataset hub

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/LIVE_benchmark.git
cd LIVE_benchmark

# Create conda environment
conda create -n live_benchmark python=3.10
conda activate live_benchmark

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Download

```python
from datasets import load_dataset

# Load the LIVE dataset from Hugging Face
dataset = load_dataset("Tong613/LIVE-multi-image-bench", data_dir="main")

# Access dataset splits
train_data = dataset['train']
print(f"Total samples: {len(train_data)}")
```

### 3. Evaluation Pipeline

#### Prepare Your Images
Download MS-COCO validation images and place them in `data/coco_val2014/`

#### Configure Model
Edit `evaluation.py` to set your VLM API credentials:

```python
# Set your API configuration
API_KEY = "your-api-key"
BASE_URL = "https://your-vlm-api.com/v1"
MODEL_NAME = "your-vision-language-model"
```

#### Run Evaluation

```bash
# Evaluate on subset of questions
python evaluation.py \
    --data_path path/to/questions.json \
    --image_dir data/coco_val2014/ \
    --output results.json \
    --num_samples 100

# Full evaluation with multiple workers
python evaluation.py \
    --data_path path/to/questions.json \
    --image_dir data/coco_val2014/ \
    --output results.json \
    --num_workers 4
```

## 📊 Dataset Structure

### Data Format
Each sample in the dataset contains:

```json
{
  "task": "attributes",
  "type": "UIC",  // or "DIC"
  "qtype": "4",  // Number of target images (1-4)
  "image_id": [
    "COCO_val2014_000000239985.jpg",
    "COCO_val2014_000000376628.jpg",
    "COCO_val2014_000000369763.jpg",
    "COCO_val2014_000000176793.jpg"
  ],
  "yes_question": "Is the lady smiling in image 4?",
  "no_question": "Is the lady frowning in image 4?",
  "ritem": "lady is smiling",
  "hitem": "lady is frowning",
  "yes_question_class": "Sentiment",
  "no_question_class": "Sentiment"
}
```

### Evaluation Metrics
- **Hallucination Rate**: Percentage of counterfactual questions answered incorrectly
- **Granularity Analysis**: Performance breakdown by number of images (1-4)
- **Task-Specific Metrics**: Performance per visual recognition category

## 🛠️ Repository Structure

```
LIVE_benchmark/
├── evaluation.py              # Main evaluation script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── parser_oar/               # Scene graph parser for RELATION task
│   ├── coco_parser_oar.json  # Parser configuration
│   ├── demo.py               # Parser demonstration
│   └── parser_coco_oar.py    # COCO-specific parser
└── split_images/             # Image clustering and sampling
    ├── step1_k_experiments_results.py     # K-means clustering experiments
    ├── step2_split_images.py              # Image clustering and sampling
    ├── step3_visual_images_and_add_caption.ipynb  # Visualization notebook
    ├── k_experiments_results/             # K-means experiment results
    │   ├── k_analysis_plot.pdf
    │   ├── k_analysis_plot.png
    │   └── k_experiment_results.json
    └── split_images/                      # Sampled image tuples
        ├── cluster_inverted_index_k1000.json
        └── sampled_all_tuples_k1000.json
```

## 📚 Module Descriptions

### parser_oar/ - Scene Graph Parser
- **Purpose**: Extracts factual relationships from image captions
- **Main Components**: `parser_coco_oar.py`, `demo.py`, `coco_parser_oar.json`
- **Function**: Processes COCO captions to generate scene graphs for VLOG benchmark
- **Usage**: See [parser_oar/README.md](parser_oar/README.md)

### split_images/ - Image Clustering Pipeline
- **Purpose**: Intelligent image clustering and sampling for balanced scenarios
- **Main Steps**:
  - K-means clustering with CLIP features (K=1000 selected)
  - Diverse sampling strategies (AAAA, ABCD patterns)
  - UIC/DIC scenario balancing
- **Usage**: See [split_images/README.md](split_images/README.md)

## 🔍 Usage Examples

### Loading Different Tasks
```python
# Load specific tasks
dataset = load_dataset("Tong613/LIVE-multi-image-bench", data_dir="main")

# Filter by task type
sentiment_samples = [d for d in dataset['train'] if d['yes_question_class'] == 'Sentiment']
uic_samples = [d for d in dataset['train'] if d['type'] == 'UIC']
```

### Custom Evaluation
```python
from evaluation import evaluate_model

# Evaluate on specific subset
results = evaluate_model(
    model_api=your_vlm_api,
    questions=questions_subset,
    image_dir="data/coco_val2014/",
    batch_size=32
)
```

## 📈 Results Format

Evaluation results are saved in JSON format:

```json
[
  {
    "question_id": 0,
    "image_ids": ["COCO_val2014_000000239985.jpg", ...],
    "question": "Is the lady smiling in image 4?",
    "expected_answer": "yes",
    "model_answer": "yes",
    "correct": true,
    "response_time": 1.23
  },
  ...
]
```

## 🤝 Contributing

We welcome contributions to improve the benchmark! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MS-COCO dataset for providing the base images
- Hugging Face for dataset hosting
- The research community for their valuable feedback
