from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Type, Any
from pydantic import BaseModel, create_model
from pydantic_ai import Agent, RunContext
from pydantic_ai.format_as_xml import format_as_xml
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rprint

console = Console()

@dataclass
class TableGenerationState:
    prompt: str
    model_name: str | None = None
    pydantic_model: Type[BaseModel] | None = None
    table_data: List[Dict[str, Any]] | None = None
    model_messages: list = field(default_factory=list)

# Agent for generating Pydantic model
model_generator = Agent(
    'openai:gpt-4o-mini',
    system_prompt="""You are an expert at generating Pydantic models based on user requirements.
Generate a Pydantic model with appropriate field types and descriptions.
The response must be valid Python code that can be executed to create a Pydantic model.
Include field descriptions using the description parameter.
Use appropriate Python types (str, int, float, datetime, etc).
Do not include any markdown formatting or code blocks - just return the raw Python code.
""",
    result_type=str
)

# Agent for generating table data
data_generator = Agent(
    'openai:gpt-4o-mini',
    system_prompt="""You are an expert at generating realistic tabular data.
Generate data that matches the provided Pydantic model structure.
Return the data as a list of dictionaries that can be parsed into the Pydantic model.
Make sure all values match the required types from the model.
Do not include any markdown formatting or code blocks - just return the raw Python code.
""",
    result_type=str
)

def extract_code(response: str) -> str:
    """Extract Python code from a possibly markdown-formatted response"""
    # If the response contains a code block, extract just the code
    if "```python" in response:
        # Split on ```python and take everything after it
        code = response.split("```python")[1]
        # If there's a closing ```, remove everything after it
        if "```" in code:
            code = code.split("```")[0]
        return code.strip()
    # If it's already just code, return as is
    return response.strip()

def display_model(model: Type[BaseModel]):
    """Display a Pydantic model in a nice format using rich"""
    table = Table(title=f"Model: {model.__name__}")
    table.add_column("Field")
    table.add_column("Type")
    table.add_column("Description")
    
    for field_name, field in model.model_fields.items():
        table.add_row(
            field_name,
            str(field.annotation),
            field.description or ""
        )
    
    console.print(table)

def display_data(model: Type[BaseModel], data: List[Dict[str, Any]]):
    """Display tabular data using rich"""
    table = Table(title=f"Generated {model.__name__} Data")
    
    # Add columns based on model fields
    for field_name in model.model_fields:
        table.add_column(field_name)
    
    # Add rows
    for item in data:
        table.add_row(*[str(item[field]) for field in model.model_fields])
    
    console.print(table)

async def main():
    state = TableGenerationState(
        prompt=Prompt.ask("Enter your requirements for the table data")
    )
    
    while True:
        # Generate Pydantic model
        model_code = await model_generator.run(
            f"Generate a Pydantic model for: {state.prompt}\n"
            "Return only the model code that can be executed.",
            message_history=state.model_messages
        )
        
        # Extract the code from the response and execute it
        code = extract_code(model_code.data)
        namespace = {}
        exec(code, namespace)
        model_name = next(name for name, cls in namespace.items() 
                         if isinstance(cls, type) and issubclass(cls, BaseModel))
        state.pydantic_model = namespace[model_name]
        state.model_name = model_name
        
        # Display the model
        console.print("\nGenerated Model:")
        display_model(state.pydantic_model)
        
        # Ask for feedback
        if Confirm.ask("Is this model structure ok?"):
            break
            
        # If not ok, get feedback and update prompt
        feedback = Prompt.ask("What changes would you like to make?")
        state.prompt += f"\nChanges requested: {feedback}"
        state.model_messages = model_code.new_messages()
    
    # Generate table data
    data_code = await data_generator.run(
        f"""Generate a list of 5 items matching this model:

{format_as_xml(state.pydantic_model.model_json_schema())}

Return only valid Python code that creates a list of dictionaries."""
    )
    
    # Extract the code and execute it to get the table data
    code = extract_code(data_code.data)
    namespace = {}
    exec(code, namespace)
    state.table_data = namespace.get('data', [])
    
    # Validate and display the data
    validated_data = [state.pydantic_model(**item) for item in state.table_data]
    console.print("\nGenerated Data:")
    display_data(state.pydantic_model, state.table_data)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())