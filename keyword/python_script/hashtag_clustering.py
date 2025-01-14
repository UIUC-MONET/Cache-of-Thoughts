import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os 
import sys
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import numpy as np 
from tqdm import tqdm 
from bidict import bidict

import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class TextEmbedder:
    def __init__(self, model_name, task_name_to_instruct=None, max_length=32768, use_instruction=True, requires_tokenizer=False, **kwargs):
        self.valid_params = {key: value for key, value in kwargs.items() if value is not None}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) if requires_tokenizer else None
        self.max_length = max_length
        self.task_name_to_instruct = task_name_to_instruct if task_name_to_instruct else {}
        self.use_instruction = use_instruction
        self.requires_tokenizer = requires_tokenizer

    def _get_instruction(self, task):
        if self.use_instruction and task in self.task_name_to_instruct:
            return "Instruct: " + self.task_name_to_instruct[task] + "\nQuery: "
        return ""

    def encode(self, texts, task="default", batch_size=1):
        prefix = self._get_instruction(task)
        texts = [prefix + text for text in texts]  # Add instructions if applicable

        if self.requires_tokenizer and self.tokenizer:  # For models like gte-base-en-v1.5
            # Tokenize input texts
            batch_dict = self.tokenizer(texts, max_length=self.max_length, **self.valid_params)
            batch_dict = {key: value.to(self.device) for key, value in batch_dict.items()}  # Move tensors to the correct device

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = outputs.last_hidden_state[:, 0]  # Get the embeddings (usually the first token)

            # Normalize the embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
        else:  # For models that don’t require tokenizers explicitly
            if batch_size == 1:
                embeddings = self.model.encode(texts, instruction=prefix, max_length=self.max_length)
            else:
                # Mini-batch
                embeddings = self.model._do_encode(
                    texts, batch_size=batch_size, instruction=prefix, max_length=self.max_length, num_workers=4, return_numpy=True
                )
            return F.normalize(torch.tensor(embeddings), p=2, dim=1)

    def similarity_score(self, query_embeddings, passage_embeddings):
        return (query_embeddings @ passage_embeddings.T) * 100
    

class TextDataset(Dataset):
    def __init__(self, texts, text_prefix=""):
        self.texts = texts
        self.text_prefix = text_prefix

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.text_prefix + self.texts[idx],
        }
    

# begin

# # generate keyword embeddings
# llama_keyword_dir = '/scratch/bczf/zoezheng126/uouo/github-repo/UOUO/keyword/mmmu/new_prompt/test_keyword'

# txt_files = [f for f in os.listdir(llama_keyword_dir) if f.startswith("llama_keywords_") and f.endswith(".txt")]

# txt_files_sorted = sorted(txt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

# all_keywords_sorted = []

# for filename in txt_files_sorted:
#     with open(os.path.join(llama_keyword_dir, filename), 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             kws = line.strip().split(', ')
#             kws = kws[:5] if len(kws) > 5 else kws
#             all_keywords_sorted.extend(kws)

# keyword_dataset = TextDataset(all_keywords_sorted)

# # DataLoader for batching
# batch_size = 64
# keyword_dataloader = DataLoader(keyword_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# # Text Embedder Model
# task_name_to_instruct = {
#     "example": "Given a question, retrieve passages that answer the question",
# }
# # embedder = TextEmbedderWithLoader('nvidia/NV-Embed-v2', task_name_to_instruct)
model_name = 'gte-base-en-v1.5'
# embedder = TextEmbedder(model_name='Alibaba-NLP/gte-base-en-v1.5', max_length=8192, use_instruction=False, requires_tokenizer=True, padding=True, truncation=True, return_tensors='pt')

# all_query_embeddings = []
# all_passage_embeddings = []
# os.environ['TOKENIZERS_PARALLELISM']='True'
# output_path = '/scratch/bczf/zoezheng126/uouo/github-repo/UOUO/keyword/ShareGPT4V/embeddings'
output_path = '/scratch/bczf/zoezheng126/uouo/github-repo/UOUO/keyword/mmmu/new_prompt/embeddings'
# index_dict = dict()
# output_embeddings = []
# output_text = []
# i = 0
# for batch in tqdm(keyword_dataloader):
#     text = batch['text']
#     query_embeddings = embedder.encode(text) #task="example" for nvidia/NV-Embed-v2\
#     query_embeddings_list = query_embeddings.to('cpu').tolist()
#     output_embeddings.extend(query_embeddings_list)
#     output_text.extend(text)
#     for j in range(len(text)):
#         index_dict[text[j]] = i + j
#     i += len(text) + 1

# output_all = [index_dict, output_embeddings, output_text]
    
query_embeddings_file = f'{output_path}/single_keyword_embeddings_dict_{model_name}.pkl'
# # save embeddings
# with open(query_embeddings_file, 'wb') as f:
#     pickle.dump(output_all, f)



# load text embeddings
with open(query_embeddings_file, 'rb') as file:
    data = pickle.load(file)

index_dict, embeddings, texts = data

# filter embeddings - Extract the unique texts and embeddings
unique_texts_embeddings = dict()
for text, embedding in zip(texts, embeddings):
    if text not in unique_texts_embeddings:
        unique_texts_embeddings[text] = embedding

unique_texts = list(unique_texts_embeddings.keys())
unique_embeddings = list(unique_texts_embeddings.values())

texts = unique_texts 
embeddings = unique_embeddings

# grid-search for optimal clustering
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# wcss = []
# k = 0
# for k in range(150, 500, 50):  # Start from 50, go up to 300, in steps of 50
#     kmeans = KMeans(n_clusters=k, random_state=0)
#     kmeans.fit(embeddings)
#     wcss.append(kmeans.inertia_)
#     k += 1

# # Plot the Elbow Method graph
# plt.plot(range(150, 500, 50), wcss)
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.title('Elbow Method to Determine Optimal Number of Clusters')
# plt.show()

from sklearn.cluster import KMeans
num_clusters = 500  # Specify the number of clusters you want
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(embeddings)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Group texts based on their cluster labels
clusters = {i: [] for i in range(num_clusters)}
# for text, label in zip(texts, labels):
#     clusters[label].append(text)

clusters = {}
for i in range(num_clusters):
    clusters[i] = {
        "centroid": centroids[i].tolist(), 
        "texts": [],
        "embeddings": []
    }

for text, embedding, label in zip(texts, embeddings, labels):
    clusters[label]["texts"].append(text)
    clusters[label]["embeddings"].append(embeddings)
    print(len(clusters[label]['texts']))
    
print('begin saving files')
print(clusters[0]['texts'][0], clusters[0]['embeddings'][0], len(clusters[0]['texts']))

with open(os.path.join(output_path,f"clusters_{0}_{model_name}.pkl"), "wb")as f:
    pickle.dump(clusters[0], f)
# output_path = '/scratch/bczf/zoezheng126/uouo/github-repo/UOUO/keyword/mmmu/new_prompt/clusters'
# for key, value in clusters.items():
#     with open(os.path.join(output_path,f"clusters_{key}_{model_name}.json"), "w") as f:
#         json.dump(clusters, f, indent=4)


# Print the texts in each cluster
# for cluster_id, cluster_texts in clusters.items():
#     print(f"Cluster {cluster_id}:")
#     for text in cluster_texts:
#         print(f"  {text}")



print("Clusters saved to clusters.json")







