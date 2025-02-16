# Tabular Agent

An AI-powered tool for generating table structures and sample data using GPT models.

## Features

- Interactive table structure generation
- Automatic data type inference
- Dynamic Pydantic model creation
- Sample data generation matching the table structure
- Rich console output with formatted tables

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python tabular_agent.py
```

The tool will:
1. Ask for your table requirements in plain English
2. Generate a table structure with appropriate columns and data types
3. Show you the proposed structure and ask for feedback
4. Generate sample data matching the structure
5. Display the results in a nicely formatted table

Example prompt:
```
> Enter your requirements for the table data
Create a table for tracking employee performance reviews
```

## Architecture

Tabular Agent utilizes a modular design with several agents that generate, validate, and present table data dynamically.

### Agent Flow

```mermaid
flowchart TB
    A((__start__)) --> B["Prompt user to enter table requirements"]
    B --> C[model_generator Agent]
    C -- can call --> T((search_data Tool))
    C --> D{User confirms table structure?}
    D -- "No" --> B
    D -- "Yes" --> E["Add 'subject' column & create dynamic model"]
    E --> F["Prompt user for initial categories"]
    F --> G[subject_generator Agent]
    G -- can call --> T
    G --> H{User confirms categories?}
    H -- "No" --> G
    H -- "Yes" --> I["Create dynamic_data_generator Agent"]
    I -- can call --> T
    I --> J["Generate data for each subject"]
    J --> K["Validate generated data"]
    K --> L{Any validation errors?}
    L -- "No" --> M{Save data to Excel?}
    L -- "Yes" --> N{Continue with valid entries only?}
    N -- "No" --> O((__end__))
    N -- "Yes" --> M
    M -- "No" --> O
    M -- "Yes" --> P["Write data to Excel file"]
    P --> O((__end__))
```

## License

MIT

## Contributing

Contributions are welcome! Please submit issues and pull requests through GitHub.
