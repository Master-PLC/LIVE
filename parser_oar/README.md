# Scene Graph Parser for LIVE Benchmark

## Overview

This module contains scene graph parsing tools specifically designed for the LIVE benchmark, enabling extraction of factual relationships from image captions and visual content.

## Components

### parser_coco_oar.py
Main parser implementation that:
- Processes COCO dataset captions
- Extracts objects, attributes, and relationships
- Generates scene graphs for visual understanding tasks
- Supports multi-image scenarios in LIVE benchmark

### demo.py
Demonstration script showing:
- Parser usage examples
- Scene graph visualization
- Integration with LIVE dataset format

### coco_parser_oar.json
Configuration file defining:
- Parser parameters and settings
- Object type mappings
- Relationship extraction rules
- COCO-specific customizations

## Usage

### Basic Usage
```python
from parser_coco_oar import COCOParser

# Initialize parser
parser = COCOParser(config_path='coco_parser_oar.json')

# Parse image captions
captions = [...]  # List of captions
scene_graphs = parser.parse_captions(captions)
```

### Advanced Features
- Multi-image graph merging
- Object intersection detection
- Attribute conflict resolution
- Relationship validation

## Integration with LIVE

This parser is specifically designed to support the RELATION task in LIVE benchmark:
- Extracts factual relationships from MS-COCO captions
- Generates hallucinative (counterfactual) relationships
- Validates relationship ground truth
- Supports both UIC and DIC scenarios

## Dependencies
- NLTK
- spaCy
- NetworkX (for graph operations)
- numpy
- json

## Example Output
```json
{
  "objects": ["person", "bicycle", "tree"],
  "attributes": {
    "person": ["standing", "wearing_hat"],
    "bicycle": ["red", "parked"]
  },
  "relationships": [
    {"subject": "person", "predicate": "near", "object": "bicycle"},
    {"subject": "bicycle", "predicate": "under", "object": "tree"}
  ]
}
```