#!/usr/bin/env python3
"""
Prompt CLI Application
A command-line tool for managing and executing AI prompts with Python wrappers.
"""

import argparse
import json
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import importlib.util
import requests
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.base_url = 'https://api.anthropic.com/v1/messages'
        
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, model: str = 'claude-3-5-sonnet-20241022', **kwargs) -> str:
        if not self.is_available():
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
        headers = {
            'x-api-key': self.api_key,
            'content-type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        data = {
            'model': model,
            'max_tokens': kwargs.get('max_tokens', 4000),
            'messages': [{'role': 'user', 'content': prompt}]
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()['content'][0]['text']

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = 'https://api.openai.com/v1/chat/completions'
        
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, model: str = 'gpt-4', **kwargs) -> str:
        if not self.is_available():
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': kwargs.get('max_tokens', 4000)
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()['choices'][0]['message']['content']

class GeminiProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, model: str = 'gemini-pro', **kwargs) -> str:
        if not self.is_available():
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}'
        
        data = {
            'contents': [{
                'parts': [{'text': prompt}]
            }]
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        return response.json()['candidates'][0]['content']['parts'][0]['text']

class PromptManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.prompts_dir = Path(self.config.get('prompts_dir', 'prompts'))
        self.wrappers_dir = Path(self.config.get('wrappers_dir', 'wrappers'))
        
        # Initialize LLM providers
        self.providers = {
            'anthropic': AnthropicProvider(),
            'openai': OpenAIProvider(),
            'gemini': GeminiProvider()
        }
        
        # Model to provider mapping
        self.model_mapping = {
            'claude-3-5-sonnet-20241022': 'anthropic',
            'claude-3-5-sonnet': 'anthropic',
            'claude-3-sonnet': 'anthropic',
            'claude-3-haiku': 'anthropic',
            'gpt-4': 'openai',
            'gpt-4-turbo': 'openai',
            'gpt-3.5-turbo': 'openai',
            'gemini-pro': 'gemini',
            'gemini-1.5-pro': 'gemini'
        }
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return self.create_default_config()
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default configuration file."""
        default_config = {
            'prompts_dir': 'prompts',
            'wrappers_dir': 'wrappers',
            'output_dir': 'output',
            'default_model': 'claude-3-5-sonnet-20241022',
            'api_keys': {
                'openai': 'set-OPENAI_API_KEY-env-var',
                'anthropic': 'set-ANTHROPIC_API_KEY-env-var',
                'google': 'set-GOOGLE_API_KEY-env-var'
            },
            'default_params': {
                'temperature': 0.7,
                'max_tokens': 1000
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    def list_prompts(self) -> List[str]:
        """List all available prompts."""
        if not self.prompts_dir.exists():
            return []
        
        prompts = []
        for file in self.prompts_dir.glob('*.json'):
            prompts.append(file.stem)
        return sorted(prompts)
    
    def load_prompt(self, name: str) -> Dict[str, Any]:
        """Load a specific prompt configuration."""
        prompt_file = self.prompts_dir / f"{name}.json"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt '{name}' not found")
        
        with open(prompt_file, 'r') as f:
            return json.load(f)
    
    def load_wrapper(self, name: str):
        """Dynamically load a prompt wrapper module."""
        wrapper_file = self.wrappers_dir / f"{name}.py"
        if not wrapper_file.exists():
            raise FileNotFoundError(f"Wrapper '{name}' not found")
        
        spec = importlib.util.spec_from_file_location(name, wrapper_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def get_provider_for_model(self, model: str) -> LLMProvider:
        """Get the appropriate provider for a model."""
        provider_name = self.model_mapping.get(model)
        if not provider_name:
            # Default to anthropic for unknown models
            provider_name = 'anthropic'
            
        provider = self.providers[provider_name]
        if not provider.is_available():
            # Fallback to any available provider
            for fallback_provider in self.providers.values():
                if fallback_provider.is_available():
                    return fallback_provider
            raise ValueError("No LLM providers available. Please set API keys in environment variables.")
            
        return provider
    
    def execute_prompt(self, prompt_name: str, model: str = None, **kwargs) -> str:
        """Execute a prompt with LLM provider."""
        prompt_config = self.load_prompt(prompt_name)
        
        # Use specified model or default
        if not model:
            model = self.config.get('default_model', 'claude-3-5-sonnet-20241022')
            
        # Try wrapper first
        wrapper_name = prompt_config.get('wrapper', prompt_name)
        try:
            wrapper_module = self.load_wrapper(wrapper_name)
            if hasattr(wrapper_module, 'execute'):
                return wrapper_module.execute(prompt_config, self.config, model=model, **kwargs)
        except FileNotFoundError:
            pass
            
        # Fallback to LLM execution
        return self.llm_execute(prompt_config, model, **kwargs)
    
    def llm_execute(self, prompt_config: Dict[str, Any], model: str, **kwargs) -> str:
        """Execute prompt using LLM provider."""
        template = prompt_config.get('template', '')
        
        # Simple template substitution
        for key, value in kwargs.items():
            template = template.replace(f"{{{key}}}", str(value))
        
        # Get provider and generate response
        provider = self.get_provider_for_model(model)
        return provider.generate(template, model=model, **kwargs)

def main():
    parser = argparse.ArgumentParser(description='Prompt CLI Application')
    parser.add_argument('command', choices=['list', 'run', 'create', 'init'])
    parser.add_argument('--prompt', '-p', help='Prompt name')
    parser.add_argument('--config', '-c', default='config.yaml', help='Config file path')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--vars', '-v', nargs='*', help='Variables in key=value format')
    parser.add_argument('--model', '-m', help='LLM model to use (default: claude-3-5-sonnet-20241022)')
    
    args = parser.parse_args()
    
    pm = PromptManager(args.config)
    
    if args.command == 'init':
        init_project()
    elif args.command == 'list':
        prompts = pm.list_prompts()
        if prompts:
            print("Available prompts:")
            for prompt in prompts:
                print(f"  - {prompt}")
        else:
            print("No prompts found")
    
    elif args.command == 'run':
        if not args.prompt:
            print("Error: --prompt required for run command")
            sys.exit(1)
        
        # Parse variables
        variables = {}
        if args.vars:
            for var in args.vars:
                if '=' in var:
                    key, value = var.split('=', 1)
                    variables[key] = value
        
        try:
            result = pm.execute_prompt(args.prompt, model=args.model, **variables)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(result)
                print(f"Output saved to {args.output}")
            else:
                print(result)
        
        except Exception as e:
            print(f"Error executing prompt: {e}")
            sys.exit(1)
    
    elif args.command == 'create':
        create_prompt_template(args.prompt or 'new_prompt')

def init_project():
    """Initialize a new prompt project structure."""
    dirs = ['prompts', 'wrappers', 'output', 'tests']
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"Created directory: {dir_name}")
    
    # Create example files
    create_example_files()
    print("Project initialized successfully!")

def create_example_files():
    """Create example prompt, wrapper, and config files."""
    
    # Example prompt
    example_prompt = {
        "name": "code_review",
        "description": "Code review prompt",
        "wrapper": "code_review",
        "template": "Please review the following code and provide feedback:\n\n{code}\n\nFocus on: {focus_areas}",
        "parameters": {
            "code": {"type": "string", "required": True},
            "focus_areas": {"type": "string", "default": "bugs, performance, readability"}
        }
    }
    
    with open('prompts/code_review.json', 'w') as f:
        json.dump(example_prompt, f, indent=2)
    
    # Example wrapper
    wrapper_code = '''"""
Code Review Wrapper
Custom logic for code review prompts
"""

def execute(prompt_config, global_config, **kwargs):
    """Execute code review prompt with custom preprocessing."""
    
    template = prompt_config.get('template', '')
    
    # Custom preprocessing
    code = kwargs.get('code', '')
    focus_areas = kwargs.get('focus_areas', 'bugs, performance, readability')
    
    # Add line numbers if not present
    if code and '\\n' in code:
        lines = code.split('\\n')
        numbered_code = '\\n'.join(f"{i+1:3d}: {line}" for i, line in enumerate(lines))
        kwargs['code'] = numbered_code
    
    # Substitute variables
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", str(value))
    
    return result

def validate_inputs(**kwargs):
    """Validate inputs for code review."""
    if not kwargs.get('code'):
        raise ValueError("Code parameter is required")
    return True
'''
    
    with open('wrappers/code_review.py', 'w') as f:
        f.write(wrapper_code)
    
    # README
    readme_content = '''# Prompt CLI Application

A command-line tool for managing and executing AI prompts with Python wrappers.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Initialize project
```bash
python prompt_cli.py init
```

### List available prompts
```bash
python prompt_cli.py list
```

### Run a prompt
```bash
python prompt_cli.py run --prompt code_review --vars code="print('hello')" focus_areas="style, bugs"
```

### Create new prompt template
```bash
python prompt_cli.py create --prompt new_prompt_name
```

## Directory Structure

- `prompts/` - JSON prompt definitions
- `wrappers/` - Python wrapper modules
- `output/` - Generated outputs
- `config.yaml` - Global configuration
'''
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    # Requirements
    requirements = '''pyyaml>=6.0
requests>=2.28.0
click>=8.0.0
jinja2>=3.1.0
'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)

def create_prompt_template(name: str):
    """Create a new prompt template."""
    template = {
        "name": name,
        "description": f"Description for {name} prompt",
        "wrapper": name,
        "template": "Your prompt template here with {variable} placeholders",
        "parameters": {
            "variable": {"type": "string", "required": True, "description": "Description of variable"}
        }
    }
    
    prompt_file = Path('prompts') / f"{name}.json"
    with open(prompt_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    wrapper_template = f'''"""
{name.title()} Wrapper
Custom logic for {name} prompts
"""

def execute(prompt_config, global_config, **kwargs):
    """Execute {name} prompt."""
    
    template = prompt_config.get('template', '')
    
    # Add your custom logic here
    
    # Basic variable substitution
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", str(value))
    
    return result

def validate_inputs(**kwargs):
    """Validate inputs for {name}."""
    # Add validation logic
    return True
'''
    
    wrapper_file = Path('wrappers') / f"{name}.py"
    with open(wrapper_file, 'w') as f:
        f.write(wrapper_template)
    
    print(f"Created prompt template: {prompt_file}")
    print(f"Created wrapper template: {wrapper_file}")

if __name__ == '__main__':
    main()
