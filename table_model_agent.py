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
    description: str = Field(..., description="Description of what the table represents")
    columns: List[ColumnDefinition] = Field(..., description="List of column definitions")

class SubjectList(BaseModel):
    """List of subjects to generate table entries for"""
    subjects: List[str] = Field(..., description="List of specific subjects/categories")
    context: str = Field(..., description="Explanation of how these subjects are derived")

@dataclass
class TableGenerationState:
    prompt: str
    table_definition: Optional[TableDefinition] = None
    dynamic_model: Optional[type[BaseModel]] = None
    table_data: Optional[List[Dict[str, Any]]] = None
    subject_list: Optional[SubjectList] = None
    model_messages: list = field(default_factory=list)

def display_table_definition(table_def: TableDefinition):
    """Display the table structure using rich."""
    table = Table(title=f"Table: {table_def.name}")
    table.add_column("Column")
    table.add_column("Type")
    table.add_column("Description")
    for col in table_def.columns:
        table.add_row(col.name, col.type, col.description or "")
    console.print("\nTable Description:", style="bold")
    console.print(table_def.description)
    console.print("\nColumns:", style="bold")
    console.print(table)

def display_table_data(table_def: TableDefinition, data: List[Dict[str, Any]]):
    """Display generated table data using rich."""
    table = Table(title=f"{table_def.name} Data")
    for col in table_def.columns:
        table.add_column(col.name)
    for row in data:
        table.add_row(*[str(row[col.name]) for col in table_def.columns])
    console.print(table)

def create_dynamic_model(table_def: TableDefinition) -> type[BaseModel]:
    """Create a Pydantic model dynamically based on the table definition."""
    type_map = {
        'str': str,
        'int': int,
        'float': float, 
        'bool': bool,
        'list': list,
        'dict': dict
    }
    fields = {
        col.name: (
            type_map.get(col.type.lower(), str),
            Field(..., description=col.description)
        )
        for col in table_def.columns
    }
    return create_model(
        table_def.name,
        __config__=ConfigDict(
            title=table_def.name,
            arbitrary_types_allowed=True
        ),
        **fields
    )

# Agent for generating the table structure
model_generator = Agent(
    'openai:gpt-4o-mini',
    system_prompt="""You are an expert at analyzing requirements and generating table structures.
Create a TableDefinition instance with appropriate columns and types based on the user's requirements.
The TableDefinition model has these fields:
- name: str
- description: str
- columns: List[ColumnDefinition]

The ColumnDefinition model has:
- name: str
- type: str (e.g. 'str', 'int', 'float', 'bool')
- description: Optional[str]

Example:
TableDefinition(
    name="MyTable",
    description="Description of the table",
    columns=[
        ColumnDefinition(name="column1", type="str", description="Description of column1"),
        ColumnDefinition(name="column2", type="int", description="Description of column2")
    ]
)
Return ONLY the Python object for the TableDefinition instance.""",
    result_type=TableDefinition
)

# Agent for generating subject categories
subject_generator = Agent(
    'openai:gpt-4',
    system_prompt="""You are an expert at data categorization.
Based on the provided table definition and initial categories, generate a complete list of relevant categories.
Return ONLY raw Python code that creates a SubjectList instance.
Example:
SubjectList(
    subjects=["Category 1", "Category 2", "Category 3"],
    context="Expanded from user suggestions"
)""",
    result_type=SubjectList
)

async def main():
    state = TableGenerationState(
        prompt=Prompt.ask("Enter your requirements for the table data")
    )
    
    # Loop until the user confirms the table structure
    while True:
        console.print("[blue]model_generator prompt:[/blue]", state.prompt)
        response = await model_generator.run(
            f"Generate a TableDefinition for: {state.prompt}",
            message_history=state.model_messages
        )
        state.table_definition = response.data
        display_table_definition(state.table_definition)
        if Confirm.ask("\nIs this table structure ok?"):
            break
        feedback = Prompt.ask("What changes would you like to make?")
        state.prompt += f"\nChanges requested: {feedback}"
        state.model_messages = response.new_messages()
    
    # Build the dynamic model using Pydantic's create_model
    state.dynamic_model = create_dynamic_model(state.table_definition)
    
    # Get initial categories from the user
    while True:
        console.print("\nPlease provide some initial categories for the data (comma-separated):")
        initial_categories = [cat.strip() for cat in Prompt.ask("Categories").split(",") if cat.strip()]
        if initial_categories:
            break
        console.print("[red]Please enter at least one category[/red]")
    
    # Generate expanded subject categories
    while True:
        try:
            prompt = (
                f"We want to create a dataset with the following description: {state.table_definition.description}\n"
                f"Divide the dataset into categories similar to: {', '.join(initial_categories)}\n\n"
                "Please generate a list of relevant categories."
            )
            console.print("[blue]subject_generator prompt:[/blue]", prompt)
            subject_result = await subject_generator.run(prompt)
            generated_list = subject_result.data
            if not isinstance(generated_list, SubjectList):
                raise ValueError("Generated result is not a SubjectList")
            all_subjects = list(dict.fromkeys(initial_categories + generated_list.subjects))
            state.subject_list = SubjectList(subjects=all_subjects, context=generated_list.context)
            break
        except Exception as e:
            console.print(f"[red]Error in category generation: {str(e)}[/red]")
            if not Confirm.ask("Would you like to retry?"):
                return

    console.print("\nGenerated Categories:")
    for subject in state.subject_list.subjects:
        console.print(f"- {subject}")
    console.print(f"\nCategorization Context: {state.subject_list.context}")
    if not Confirm.ask("\nAre these categories ok?"):
        return

    # Create a new data generator agent that returns a list of dynamic_model instances.
    from typing import List as TList
    dynamic_data_generator = Agent(
        'openai:gpt-4o-mini',
        system_prompt=f"""You are an expert at generating realistic tabular data.
Generate a JSON array of objects that match the table structure.
Each object must have exactly the following keys with values of the correct type:
{chr(10).join(f' - {col.name}: {col.type}' for col in state.table_definition.columns)}
Return ONLY the JSON data.""",
        result_type=TList[state.dynamic_model]
    )

    all_data = []
    total_entries = 0
    validation_errors = []

    with console.status(f"[bold green]Generating data for {len(state.subject_list.subjects)} categories...") as status:
        for i, subject in enumerate(state.subject_list.subjects, 1):
            status.update(f"[bold green]Processing category {i}/{len(state.subject_list.subjects)}: {subject}")
            data_prompt = f"""Generate entries for the category: {subject}
Table: {state.table_definition.name}
Description: {state.table_definition.description}

Rules:
1. Values MUST match these EXACT types:
{chr(10).join(f'   - {col.name}: {col.type} ({col.description})' for col in state.table_definition.columns)}
2. For 'int' type, use whole numbers only.
3. For 'float' type, decimals are allowed.
4. For 'bool' type, use True/False only.
5. For 'str' type, use text strings.

Return ONLY a JSON array of objects with these keys and types.
"""
            console.print("[blue]data_generator prompt for subject:[/blue]", subject, "\n", data_prompt)
            data_response = await dynamic_data_generator.run(data_prompt)
            try:
                # data_response.data is already a list of dynamic_model instances (validated by Pydantic)
                valid_rows = [row.model_dump() for row in data_response.data]
                all_data.extend(valid_rows)
                total_entries += len(valid_rows)
                console.print(f"✓ Added {len(valid_rows)} valid entries for {subject}")
            except Exception as e:
                console.print(f"[red]Error processing category {subject}: {str(e)}[/red]")
                continue
            status.update(f"[bold green]Progress: {total_entries} total valid entries ({i}/{len(state.subject_list.subjects)} categories)")
    
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
    console.print("\nGenerated Data Sample (showing up to 10 entries):")
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
        for error in validation_errors[:5]:
            console.print(f"- {error}")
        if len(validation_errors) > 5:
            console.print(f"...and {len(validation_errors) - 5} more errors")
        if not Confirm.ask("Continue with valid entries only?"):
            return
    state.table_data = [row.model_dump() for row in validated_data]
    console.print(f"\n[green]Successfully generated {len(validated_data)} valid entries[/green]")
    console.print("\nGenerated Data:")
    display_table_data(state.table_definition, state.table_data)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
