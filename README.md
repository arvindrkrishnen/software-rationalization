# software-rationalization
Transformer code which would help architects to find and group similar software together

Problem Statement: 
In large enterprise, there is always need for accelerating rationalization of technology hardware and software. Often the grouping becomes very difficult.

Solution: The code here should help group similar software together. Based on the grouping, one can start looking at identifying the rationalization opportunities.

This code performs the following steps:

Loads the dataset into a DataFrame.
Preprocesses the data by filling missing descriptions with empty strings.
Uses Hugging Face's DistilBERT tokenizer and model for embedding software descriptions.
Performs KMeans clustering to create groups of software based on common capabilities.
Assigns software names to the appropriate groups based on clustering results.
Creates a DataFrame with Software Name, Group, Match Score, and Description.
Exports the software groups DataFrame to an Excel file named software_groups.xlsx

Hope this helps!
