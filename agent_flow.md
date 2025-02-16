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
