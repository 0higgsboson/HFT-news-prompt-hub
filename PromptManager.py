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

class PromptManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.prompts_dir = Path(self.config.get('prompts_dir', 'prompts'))
        self.wrappers_dir = Path(self.config.get('wrappers_dir', 'wrappers'))
        
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
            'default_model': 'gpt-3.5-turbo',
            'api_keys': {
                'openai': 'your-openai-key-here',
                'anthropic': 'your-anthropic-key-here'
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
    
    def execute_prompt(self, prompt_name: str, **kwargs) -> str:
        """Execute a prompt with its wrapper."""
        prompt_config = self.load_prompt(prompt_name)
        wrapper_name = prompt_config.get('wrapper', prompt_name)
        
        try:
            wrapper_module = self.load_wrapper(wrapper_name)
            if hasattr(wrapper_module, 'execute'):
                return wrapper_module.execute(prompt_config, self.config, **kwargs)
            else:
                raise AttributeError(f"Wrapper '{wrapper_name}' missing execute function")
        except FileNotFoundError:
            # Fallback to basic execution
            return self.basic_execute(prompt_config, **kwargs)
    
    def basic_execute(self, prompt_config: Dict[str, Any], **kwargs) -> str:
        """Basic prompt execution without custom wrapper."""
        template = prompt_config.get('template', '')
        
        # Simple template substitution
        for key, value in kwargs.items():
            template = template.replace(f"{{{key}}}", str(value))
        
        return template

def main():
    parser = argparse.ArgumentParser(description='Prompt CLI Application')
    parser.add_argument('command', choices=['list', 'run', 'create', 'init'])
    parser.add_argument('--prompt', '-p', help='Prompt name')
    parser.add_argument('--config', '-c', default='config.yaml', help='Config file path')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--vars', '-v', nargs='*', help='Variables in key=value format')
    
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
            result = pm.execute_prompt(args.prompt, **variables)
            
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
