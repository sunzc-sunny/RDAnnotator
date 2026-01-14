# RDAnnotator

A toolset for annotating and verifying drone images using GPT-4-o.

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env_example .env
   ```
4. Edit `.env` with your actual paths and API keys

## Configuration

### Environment Variables

- `VISDRONE_DATA_ROOT`: Root directory for VisDrone dataset
- `OUTPUT_ROOT`: Directory for output files (annotations, captions, etc.)
- `PROMPT_ROOT`: Directory containing prompt templates
- `LOG_DIR`: Directory for log files
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_API_URL`: OpenAI API endpoint (default: https://api.openai.com/v1/chat/completions)

### Directory Structure

```
.
├── data/                  # VisDrone dataset (symlink to VISDRONE_DATA_ROOT)
│   ├── images/             # Training images
│   ├── annotations/        # Annotation files
│   └── ...
├── output/                # Generated outputs (symlink to OUTPUT_ROOT)
│   ├── captions/          # Image captions
│   ├── color_annotations/  # Color annotations
│   └── ...
├── prompts/               # Prompt templates (symlink to PROMPT_ROOT)
│   ├── annotation_example_color_v3/
│   ├── check_color_example/
│   └── ...
└── logs/                  # Log files
```

## Usage

1. Run the RDAnnotator:
   ```python
   python get_annotation/main.py
   ```
2. Or run individual tools:
   ```python
   python get_annotation/color_tools/color_annotation_v3.py
   python get_annotation/color_tools/check_color.py
   python get_annotation/color_tools/check_annotation_chatgpt.py
   python get_annotation/color_tools/regenerate_annotation_color.py
   ```
### Strong Recommendation: run the batch requests of individual tools to reduce the cost of OpenAI API. (about 1/2 of the cost)
   ```python
   python get_annotation/color_tools/batch_color_annotation_pipeline_text.py

   # After the batch requests completed, run the following to get response from OpenAI.
   python get_annotation/color_tools/get_batch_color_annotation.py
   ```

## License

This project is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
