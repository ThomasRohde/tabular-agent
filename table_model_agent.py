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
Based on the provided table definition, its columns, and user-provided initial categories,
expand and refine the categorization to create a complete list of relevant categories.

You MUST return ONLY raw Python code that creates a SubjectList instance without any extra text, markdown, or code blocks.

Example output format:
SubjectList(
    subjects=["Category 1", "Category 2", "Category 3"],
    context="These categories represent X division of the data, expanded from user suggestions"
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

    # Get initial categories from user
    while True:
        console.print("\nPlease provide some initial categories for the data (comma-separated):")
        initial_categories = [cat.strip() for cat in Prompt.ask("Categories").split(",") if cat.strip()]
        
        if initial_categories:
            break
        console.print("[red]Please enter at least one category[/red]")

    # Generate expanded subject categories based on table structure and user input
    while True:
        try:
            # Generate expanded categories
            prompt = (
                f"Table: {state.table_definition.name}\n"
                f"Description: {state.table_definition.description}\n"
                f"Initial categories: {', '.join(initial_categories)}\n\n"
                "Generate a SubjectList instance that expands these categories while maintaining "
                "relevance to the table structure. Include the initial categories plus related ones."
            )
            
            subject_result = await subject_generator.run(prompt)
            
            # Clean and validate the generated code
            generated_code = subject_result.data.strip()
            if not generated_code.startswith('SubjectList('):
                raise ValueError("Invalid generated code format")
            
            # Create a safe namespace and execute
            namespace = {'SubjectList': SubjectList}
            try:
                exec(f"result = {generated_code}", namespace)
                generated_list = namespace['result']
                
                # Ensure it's a valid SubjectList instance
                if not isinstance(generated_list, SubjectList):
                    raise ValueError("Generated result is not a SubjectList")
                    
                # Merge with initial categories and remove duplicates
                all_subjects = list(dict.fromkeys(
                    initial_categories + generated_list.subjects
                ))
                
                # Create final subject list
                state.subject_list = SubjectList(
                    subjects=all_subjects,
                    context=generated_list.context
                )
                break
                
            except Exception as e:
                raise ValueError(f"Failed to process generated categories: {str(e)}")
                
        except Exception as e:
            console.print(f"[red]Error in category generation: {str(e)}[/red]")
            if not Confirm.ask("Would you like to retry?"):
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
    total_entries = 0
    validation_errors = []

    with console.status(f"[bold green]Generating data for {len(state.subject_list.subjects)} categories...") as status:
        for i, subject in enumerate(state.subject_list.subjects, 1):
            status.update(f"[bold green]Processing category {i}/{len(state.subject_list.subjects)}: {subject}")
            
            data_code = await data_generator.run(
                f"""Generate entries for the category: {subject}
                Table: {state.table_definition.name}
                Description: {state.table_definition.description}
                
                Rules:
                1. Values MUST match these EXACT types:
                {chr(10).join(f'   - {col.name}: {col.type} ({col.description})' for col in state.table_definition.columns)}
                2. For 'int' type, use whole numbers only (e.g. 1, 2, 3)
                3. For 'float' type, decimals are allowed
                4. For 'bool' type, use True/False only
                5. For 'str' type, use text strings
                
                Return ONLY a Python list of dictionaries with these exact column names and types.
                """
            )
            
            try:
                # Execute the data code
                namespace = {}
                exec(f"data = {data_code.data}", namespace)
                subject_data = namespace['data']
                
                # Validate and convert each row
                valid_rows = []
                for row_idx, row in enumerate(subject_data):
                    try:
                        converted_row = {}
                        for col in state.table_definition.columns:
                            value = row.get(col.name)
                            if value is None:
                                raise ValueError(f"Missing required column: {col.name}")
                            
                            # Strict type conversion
                            if col.type.lower() == 'int':
                                if isinstance(value, float):
                                    if value.is_integer():
                                        value = int(value)
                                    else:
                                        raise ValueError(f"Column {col.name} requires an integer, got {value}")
                                elif isinstance(value, str):
                                    value = int(value)
                            elif col.type.lower() == 'float':
                                value = float(value)
                            elif col.type.lower() == 'bool':
                                if isinstance(value, str):
                                    value = value.lower() in ('true', '1', 'yes', 'y')
                                else:
                                    value = bool(value)
                            elif col.type.lower() == 'str':
                                value = str(value)
                                
                            converted_row[col.name] = value
                            
                        # Validate with the model
                        validated_row = state.dynamic_model(**converted_row)
                        valid_rows.append(validated_row.model_dump())
                        
                    except Exception as e:
                        validation_errors.append(f"Category '{subject}' Row {row_idx + 1}: {str(e)}")
                        continue
                
                all_data.extend(valid_rows)
                total_entries += len(valid_rows)
                console.print(f"✓ Added {len(valid_rows)} valid entries for {subject}")
                
            except Exception as e:
                console.print(f"[red]Error processing category {subject}: {str(e)}[/red]")
                continue
            
            # Show progress
            status.update(f"[bold green]Progress: {total_entries} total valid entries ({i}/{len(state.subject_list.subjects)} categories)")
    
    # Show validation summary
    if validation_errors:
        console.print("\n[yellow]Validation Summary:[/yellow]")
        for error in validation_errors[:5]:
            console.print(f"- {error}")
        if len(validation_errors) > 5:
            console.print(f"...and {len(validation_errors) - 5} more errors")
    
    if not all_data:
        console.print("[red]No valid data was generated. Exiting...[/red]")
        return
        
    state.table_data = all_data
    console.print(f"\n[green]Successfully generated {len(all_data)} valid entries across {len(state.subject_list.subjects)} categories[/green]")
    console.print(f"\nGenerated Data Sample (showing up to 10 entries):")
    display_table_data(state.table_definition, all_data[:10])
    
    if len(all_data) > 10:
        console.print(f"\n[dim]...and {len(all_data) - 10} more entries[/dim]")

    console.print("\nValidating generated data...")
    validated_data = []
    validation_errors = []
    
    for i, row in enumerate(state.table_data):
        try:
            validated_row = state.dynamic_model(**row)
            validated_data.append(validated_row)
        except Exception as e:
            validation_errors.append(f"Row {i + 1}: {str(e)}")
    
    if validation_errors:
        console.print("[red]Some entries failed validation:[/red]")
        for error in validation_errors[:5]:  # Show first 5 errors
            console.print(f"- {error}")
        if len(validation_errors) > 5:
            console.print(f"...and {len(validation_errors) - 5} more errors")
        if not Confirm.ask("Continue with valid entries only?"):
            return
    
    state.table_data = [row.model_dump() for row in validated_data]
    console.print(f"\n[green]Successfully generated {len(validated_data)} valid entries[/green]")
    console.print(f"\nGenerated Data ({len(validated_data)} entries):")
    display_table_data(state.table_definition, state.table_data)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())