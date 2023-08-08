# You can run this in Colab. 

# Arvind Radhakrishnen - @arvindrkrishnen

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from gensim import models, corpora



# Load the dataset into a DataFrame. A sample dataset is available in this repository

dataset = pd.read_excel('/content/Tech to Domain Mapping.xlsx')


# Preprocess the data
data = dataset['Description'].fillna('')  # Replace missing descriptions with empty strings

# Use Hugging Face's DistilBERT tokenizer and model for embedding
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Tokenize and encode the software descriptions
encoded_data = tokenizer(data.tolist(), truncation=True, padding=True, return_tensors="pt")
with torch.no_grad():
    embeddings = model(**encoded_data).last_hidden_state.mean(dim=1).numpy()

# Determine the number of clusters (you can adjust this based on your data)
num_clusters = 5

# Perform KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Create a DataFrame for software groups
software_groups = pd.DataFrame(columns=['Software Name', 'Group', 'Match Score', 'Description'])

# Add Software Name to the appropriate groups based on clustering
for idx, (software_name, description) in enumerate(zip(dataset['Software Name'], data)):
    software_group = cluster_labels[idx]
    match_score = 1.0  # You can customize this based on your requirements
    software_groups = software_groups.append({'Software Name': software_name, 'Group': software_group, 'Match Score': match_score, 'Description': description}, ignore_index=True)


merged_df = pd.merge(software_groups, dataset, on='Software Name', how='inner')

merged_df.drop(columns=['Description_y'], inplace=True)

# Export the software groups DataFrame to an Excel file
merged_df.to_excel('software_groups.xlsx', index=False)

print(merged_df)
