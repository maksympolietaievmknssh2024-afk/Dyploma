# Dataset Information

For training our image generation model with contextual understanding of objects, we recommend the following datasets:

1. **LAION-5B** - A large-scale dataset with 5 billion CLIP-filtered image-text pairs
   - Website: https://laion.ai/blog/laion-5b/
   - Contains diverse object representations with descriptive captions
   - Good for understanding contextual differences (e.g., "flying saucer" as UFO vs tableware)

2. **MS-COCO** - Common Objects in Context
   - Website: https://cocodataset.org/
   - Contains 330K images with object segmentation and 5 captions per image
   - Good for object recognition and contextual understanding

3. **Conceptual Captions** - A dataset of ~3.3M images annotated with captions
   - Paper: https://aclanthology.org/P18-1238/
   - Focuses on natural language descriptions of images

## Dataset Preparation

1. Download the chosen dataset(s) from their respective websites
2. Place the raw data in the `raw/` subdirectory
3. Run the preprocessing script: `python ../utils/preprocess_dataset.py --dataset [dataset_name]`
4. The processed data will be stored in the `processed/` subdirectory

## Dataset Structure

After preprocessing, the dataset should have the following structure:

```
data/
├── raw/                  # Raw dataset files
├── processed/            # Processed dataset files
│   ├── images/           # Processed images
│   ├── captions.json     # Text captions for images
│   └── metadata.json     # Additional metadata
└── README.md             # This file
```

## Custom Dataset

If you want to use your own dataset, ensure it follows this format:

1. Images should be in a common format (JPEG, PNG)
2. Captions should be in a JSON file with the following structure:

```json
{
  "image_id": {
    "captions": ["caption1", "caption2", ...],
    "metadata": { ... }
  },
  ...
}
```

Then run the preprocessing script as described above.