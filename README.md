# HFT News Prompt Hub

A multi-provider LLM prompt management system for generating fictional financial news articles with sentiment scoring. Supports Anthropic Claude, OpenAI GPT, and Google Gemini models.

## Features

- **Multi-Provider LLM Support**: Switch between Claude, GPT, and Gemini models
- **Financial News Generation**: Create realistic fictional news articles with sentiment scoring
- **Flexible Command Interface**: Simple shell scripts with sensible defaults
- **Environment-Based Configuration**: Secure API key management
- **Extensible Architecture**: Easy to add new prompts and providers

## Quick Start

### 1. Installation

```bash
git clone https://github.com/your-username/HFT-news-prompt-hub.git
cd HFT-news-prompt-hub
pip install -r requirements.txt
```

### 2. Environment Setup

Set your API keys:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
```

### 3. Generate News

```bash
# Default: NVDA stock, sentiment 4, Claude model
./scripts/generate_stock_news.sh

# Custom ticker and sentiment
./scripts/generate_stock_news.sh AAPL 7

# Specify different model
./scripts/generate_stock_news.sh AAPL 7 gpt-4
```

## Usage

### Command Line Interface

```bash
python3 PromptManager.py [command] [options]

Commands:
  list                    List available prompts
  run                     Execute a prompt
  create                  Create new prompt template
  init                    Initialize project structure

Options:
  --prompt, -p           Prompt name to execute
  --model, -m            LLM model to use
  --vars, -v             Variables in key=value format
  --output, -o           Output file path
  --config, -c           Config file path
```

### Supported Models

**Anthropic (Default):**
- `claude-3-5-sonnet-20241022` (default)
- `claude-3-5-sonnet`
- `claude-3-sonnet`
- `claude-3-haiku`

**OpenAI:**
- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

**Google:**
- `gemini-pro`
- `gemini-1.5-pro`

### Examples

```bash
# List available prompts
python3 PromptManager.py list

# Generate news with specific parameters
python3 PromptManager.py run \
  --prompt newsGenerator \
  --model claude-3-5-sonnet-20241022 \
  --vars ticker=NVDA sentiment_score=4 \
  --output output/nvda_news.json

# Use different model
python3 PromptManager.py run \
  --prompt newsGenerator \
  --model gpt-4 \
  --vars ticker=AAPL sentiment_score=7
```

## Sentiment Score Guide

- **-10**: Catastrophic developments (bankruptcy, massive fraud)
- **-5**: Significant negative news (major losses, failed products)
- **0**: Neutral/mixed developments (routine updates)
- **+5**: Strong positive news (successful launches, expansion)
- **+10**: Groundbreaking developments (revolutionary breakthroughs)

## Project Structure

```
HFT-news-prompt-hub/
├── PromptManager.py          # Main CLI application
├── README.md                 # This file
├── LICENSE                   # License information
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore patterns
├── prompts/                 # Prompt definitions
│   └── newsGenerator.json   # News generation prompt
├── templates/               # Template files
│   ├── newsGenerator.txt    # Detailed news template
│   └── sample.txt          # Sample template
├── scripts/                 # Shell scripts
│   └── generate_stock_news.sh # News generation script
├── output/                  # Generated outputs
├── wrappers/               # Custom prompt wrappers
├── tests/                  # Test files
└── docs/                   # Documentation
```

## Configuration

Create a `config.yaml` file or let the system generate defaults:

```yaml
prompts_dir: prompts
wrappers_dir: wrappers
output_dir: output
default_model: claude-3-5-sonnet-20241022
api_keys:
  openai: set-OPENAI_API_KEY-env-var
  anthropic: set-ANTHROPIC_API_KEY-env-var
  google: set-GOOGLE_API_KEY-env-var
default_params:
  temperature: 0.7
  max_tokens: 4000
```

## Adding New Prompts

1. Create prompt definition in `prompts/`:

```json
{
  "name": "myPrompt",
  "description": "Description of prompt",
  "template": "Your prompt template with {variables}",
  "parameters": {
    "variable": {
      "type": "string",
      "required": true,
      "description": "Variable description"
    }
  }
}
```

2. Optionally create custom wrapper in `wrappers/`:

```python
def execute(prompt_config, global_config, model=None, **kwargs):
    # Custom logic here
    return result
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

Apache 2.0 License - see LICENSE file for details.

## Disclaimer

This tool generates **fictional** financial news for simulation and testing purposes only. Do not use generated content for actual trading decisions or distribute as real financial news.
