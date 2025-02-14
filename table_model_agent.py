from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict, create_model
from pydantic_ai import Agent, RunContext
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table

console = Console()

class ColumnDefinition(BaseModel):
    """Definition of a single column in the table"""
    name: str = Field(..., description="Name of the column")
    type: str = Field(..., description="Python type for this column (str, int, float, bool, etc.)")
    description: Optional[str] = Field(None, description="Description of what this column represents")

class TableDefinition(BaseModel):
    """Definition of a table structure"""
    name: str = Field(..., description="Name of the table")
    description: str = Field(..., description="Description of what this table represents")
    columns: List[ColumnDefinition] = Field(..., description="List of column definitions")

class SubjectList(BaseModel):
    """List of subjects to generate table entries for"""
    subjects: List[str] = Field(..., description="List of specific subjects/categories to generate table entries for")
    context: str = Field(..., description="Explanation of how these subjects are derived from the table structure")

@dataclass
class TableGenerationState:
    prompt: str
    table_definition: Optional[TableDefinition] = None
    dynamic_model: Optional[type[BaseModel]] = None
    table_data: Optional[List[Dict[str, Any]]] = None
    subject_list: Optional[SubjectList] = None
    model_messages: list = field(default_factory=list)

# Agent for generating table structure
model_generator = Agent(
    'openai:gpt-4o-mini',
    system_prompt="""You are an expert at analyzing requirements and generating table structures.
Create a TableDefinition instance with appropriate columns and types based on the user's requirements.
The TableDefinition model has these fields:
- name: str (name of the table)
- description: str (description of what the table represents)
- columns: List[ColumnDefinition] (list of column definitions)

The ColumnDefinition model has these fields:
- name: str (name of the column)
- type: str (Python type as string: 'str', 'int', 'float', 'bool', etc.)
- description: Optional[str] (description of what the column represents)

Example format:
TableDefinition(
    name="MyTable",
    description="Description of the table",
    columns=[
        ColumnDefinition(name="column1", type="str", description="Description of column1"),
        ColumnDefinition(name="column2", type="int", description="Description of column2")
    ]
)

Return ONLY the raw Python code that creates the TableDefinition instance.
DO NOT include any markdown formatting, code blocks, or explanatory text.
""",
    result_type=str
)

# Agent for generating table data
data_generator = Agent(
    'openai:gpt-4o-mini',
    system_prompt="""You are an expert at generating realistic tabular data.
Generate data that matches the provided table structure.
Return a list of dictionaries where each dictionary represents a row of data.
Make sure all values match the required types from the column definitions.
DO NOT include any markdown formatting, code blocks, or explanatory text.
Return ONLY the raw Python code that creates the list of dictionaries.
""",
    result_type=str
)

# Agent for analyzing table structure and generating subject categories
subject_generator = Agent(
    'openai:gpt-4',
    system_prompt="""You are an expert at analyzing data structures and categorization.
Based on the provided table definition and its columns, determine meaningful categories based on how the data can be divided.

You MUST return ONLY raw Python code that creates a SubjectList instance without any extra text, markdown, or code blocks.

Example output format:
SubjectList(
    subjects=["Category 1", "Category 2", "Category 3"],
    context="These categories represent X division of the data"
)

DO NOT include any explanations, comments, or formatting - just the raw Python code for creating the SubjectList instance.
""",
    result_type=str
)

def display_table_definition(table_def: TableDefinition):
    """Display a table definition in a nice format using rich"""
    table = Table(title=f"Table: {table_def.name}")
    table.add_column("Column")
    table.add_column("Type")
    table.add_column("Description")
    
    for col in table_def.columns:
        table.add_row(
            col.name,
            col.type,
            col.description or ""
        )
    
    console.print("\nTable Description:", style="bold")
    console.print(table_def.description)
    console.print("\nColumns:", style="bold")
    console.print(table)

def display_table_data(table_def: TableDefinition, data: List[Dict[str, Any]]):
    """Display generated table data using rich"""
    table = Table(title=f"{table_def.name} Data")
    
    # Add columns based on table definition
    for col in table_def.columns:
        table.add_column(col.name)
    
    # Add rows
    for row in data:
        table.add_row(*[str(row[col.name]) for col in table_def.columns])
    
    console.print(table)

def create_dynamic_model(table_def: TableDefinition) -> type[BaseModel]:
    """Create a Pydantic model dynamically based on the table definition using create_model"""
    # Convert string type names to actual types
    type_map = {
        'str': str,
        'int': int,
        'float': float, 
        'bool': bool,
        'list': list,
        'dict': dict
    }
    
    # Build fields dictionary for create_model
    fields = {
        col.name: (
            type_map.get(col.type.lower(), str), 
            Field(..., description=col.description)
        )
        for col in table_def.columns
    }
    
    # Use Pydantic's create_model helper
    return create_model(
        table_def.name,
        __config__=ConfigDict(
            title=table_def.name,
            arbitrary_types_allowed=True
        ),
        **fields
    )

async def main():
    state = TableGenerationState(
        prompt=Prompt.ask("Enter your requirements for the table data")
    )
    
    while True:
        # Generate table definition
        table_code = await model_generator.run(
            f"Generate a TableDefinition for: {state.prompt}\n"
            "Return only the Python code that creates a TableDefinition instance.",
            message_history=state.model_messages
        )
        
        # Execute the code to get the table definition
        namespace = {'TableDefinition': TableDefinition, 'ColumnDefinition': ColumnDefinition}
        exec(f"table_def = {table_code.data}", namespace)
        state.table_definition = namespace['table_def']
        
        # Display the table structure
        display_table_definition(state.table_definition)
        
        if Confirm.ask("\nIs this table structure ok?"):
            break
            
        feedback = Prompt.ask("What changes would you like to make?")
        state.prompt += f"\nChanges requested: {feedback}"
        state.model_messages = table_code.new_messages()
    
    # Create dynamic Pydantic model
    state.dynamic_model = create_dynamic_model(state.table_definition)

    # Generate subject categories based on table structure
    while True:
        try:
            subjects_code = await subject_generator.run(
                f"""Analyze this table structure and determine meaningful categories for dividing the data in categories in multiple iterations. Typically this will be based on the natural taxonomy of the names of the subject matter in the table.
                Consider the columns and their types to identify natural groupings.
                
                Table Name: {state.table_definition.name}
                Description: {state.table_definition.description}
                
                Columns:
                {chr(10).join(f'- {col.name} ({col.type}): {col.description}' for col in state.table_definition.columns)}
                
                Original prompt: {state.prompt}
                
                Return ONLY the raw Python code for a SubjectList instance."""
            )
            
            # Test parse the code before executing
            code = subjects_code.data.strip()
            if not code.startswith('SubjectList(') or not code.endswith(')'):
                raise ValueError("Invalid SubjectList format")
                
            # Execute the subjects code
            namespace = {'SubjectList': SubjectList}
            exec(f"subject_list = {code}", namespace)
            state.subject_list = namespace['subject_list']
            
            # Validate we got at least one subject
            if not state.subject_list.subjects:
                raise ValueError("No subjects generated")
                
            break
        except Exception as e:
            console.print(f"[red]Error generating categories: {e}[/red]")
            if not Confirm.ask("Retry category generation?"):
                return

    # Display generated categories
    console.print("\nGenerated Categories:")
    for subject in state.subject_list.subjects:
        console.print(f"- {subject}")
    console.print(f"\nCategorization Context: {state.subject_list.context}")

    if not Confirm.ask("\nAre these categories ok?"):
        return
    
    # Generate table data for each category
    all_data = []
    for subject in state.subject_list.subjects:
        data_code = await data_generator.run(
            f"""Generate entries for the category: {subject}
            
            Context: {state.subject_list.context}
            
            Table Name: {state.table_definition.name}
            Description: {state.table_definition.description}
            
            Current Category: {subject}
            Generate entries that specifically fit into this category.
            
            Required Columns:
            {chr(10).join(f'- {col.name} ({col.type}): {col.description}' for col in state.table_definition.columns)}
            
            Rules:
            1. Generate entries that clearly belong to the category "{subject}"
            2. Ensure all entries are consistent with the category classification
            3. Return only Python code for a list of dictionaries
            4. Each dictionary should contain all required columns
            5. Do not include any explanatory text or code blocks
            """
        )
        
        # Execute the data code to get the table data
        namespace = {}
        exec(f"data = {data_code.data}", namespace)
        subject_data = namespace['data']
        all_data.extend(subject_data)
    
    # Validate and display all data
    state.table_data = all_data
    validated_data = [state.dynamic_model(**row) for row in state.table_data]
    console.print(f"\nGenerated Data ({len(validated_data)} entries):")
    display_table_data(state.table_definition, state.table_data)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())