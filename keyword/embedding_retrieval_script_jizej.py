# %%
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os 
import sys
from torch.utils.data import Dataset, DataLoader
import json
import pickle

# %%
data_path = '/home/jizej/Workspaces/UOUO/keyword/ShareGPT4V'
if data_path not in sys.path:
    sys.path.append(data_path)

os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + f":{data_path}"
os.environ['HF_HOME'] = '/home/jizej/.cache/huggingface'
print(os.getenv('HF_HOME'))

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
# # Each query needs to be accompanied by an corresponding instruction describing the task.
# task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}

# query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
# queries = [
#     'are judo throws allowed in wrestling?', 
#     'how to become a radiology technician in michigan?'
#     ]

# # No instruction needed for retrieval passages
# passage_prefix = ""
# passages = [
#     "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
#     "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan."
# ]

# # load model with tokenizer use fp16
# model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True, torch_dtype=torch.float16)
# model = model.to(device)

# # get the embeddings
# max_length = 32768
# query_embeddings = model.encode(queries, instruction=query_prefix, max_length=max_length)
# passage_embeddings = model.encode(passages, instruction=passage_prefix, max_length=max_length)

# # normalize embeddings
# query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
# passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

# # get the embeddings with DataLoader (spliting the datasets into multiple mini-batches)
# # batch_size=2
# # query_embeddings = model._do_encode(queries, batch_size=batch_size, instruction=query_prefix, max_length=max_length, num_workers=32, return_numpy=True)
# # passage_embeddings = model._do_encode(passages, batch_size=batch_size, instruction=passage_prefix, max_length=max_length, num_workers=32, return_numpy=True)

# scores = (query_embeddings @ passage_embeddings.T) * 100
# print(scores.tolist())
# # [[87.42693328857422, 0.46283677220344543], [0.965264618396759, 86.03721618652344]]


# %%
# need compile session

# %%
class TextEmbedder:
    def __init__(self, model_name, task_name_to_instruct=None, max_length=32768, use_instruction=True):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.task_name_to_instruct = task_name_to_instruct if task_name_to_instruct else {}
        self.use_instruction = use_instruction

    def _get_instruction(self, task):
        if self.use_instruction and task in self.task_name_to_instruct:
            return "Instruct: " + self.task_name_to_instruct[task] + "\nQuery: "
        return ""

    def encode(self, texts, task="default", batch_size=1, num_workers=4):
        prefix = self._get_instruction(task)
        if batch_size == 1:
            embeddings = self.model.encode(texts, instruction=prefix, max_length=self.max_length)
        else:
            # Mini-batch
            embeddings = self.model._do_encode(
                texts, batch_size=batch_size, instruction=prefix, max_length=self.max_length, num_workers=num_workers, return_numpy=True
            )
        return F.normalize(torch.tensor(embeddings), p=2, dim=1)

    def similarity_score(self, query_embeddings, passage_embeddings):
        return (query_embeddings @ passage_embeddings.T) * 100

# %%
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

class TextEmbedderWithLoader:
    def __init__(self, model_name, task_name_to_instruct=None, max_length=32768):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(device)
        self.max_length = max_length
        self.task_name_to_instruct = task_name_to_instruct if task_name_to_instruct else {}

    def _get_instruction(self, task):
        return "Instruct: " + self.task_name_to_instruct.get(task, "") + "\nQuery: "

    def encode_batch(self, batch, task="default"):
        query_embeddings = self.model.encode(batch["text"], instruction=self._get_instruction(task), max_length=self.max_length)
        return F.normalize(torch.tensor(query_embeddings), p=2, dim=1)

    def similarity_score(self, query_embeddings, passage_embeddings):
        return (query_embeddings @ passage_embeddings.T) * 100

# %%
# main

# %%
# [example] Initialize Dataset and DataLoader
# queries = [
#     'are judo throws allowed in wrestling?', 
#     'how to become a radiology technician in michigan?'
# ]

# passages = [
#     "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
#     "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan."
# ]

# %%
llama_keyword_dir = '/home/jizej/Workspaces/UOUO/keyword/llama_sharegpt4v_keywords'

txt_files = [f for f in os.listdir(llama_keyword_dir) if f.startswith("llama_keywords_") and f.endswith(".txt")]

txt_files_sorted = sorted(txt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

all_keywords_sorted = []

for filename in txt_files_sorted:
    with open(os.path.join(llama_keyword_dir, filename), 'r') as file:
        lines = file.readlines()
        for line in lines:
            all_keywords_sorted.append(line.strip().split(', '))

cut_keywords_list = [kws[:5] if len(kws) > 5 else kws for kws in all_keywords_sorted] # only use first five keywords 
cut_keywords_str_list = [','.join(kws) for kws in cut_keywords_list]

# %%
# load json and flatten it into a 1d list
hashtag_json = '/home/jizej/Workspaces/UOUO/keyword/hashtag_1745.json'
with open(hashtag_json, 'r') as file:
    json_data = file.read()
hashtags = json.loads(json_data)

# def extract_all_hashtags(nested_json):
#     flat_list = []

#     def flatten(current):
#         if isinstance(current, dict):
#             for key, value in current.items():
#                 # Append the key (Level 1 or Level 2 hashtags)
#                 flat_list.append(key)
#                 flatten(value)
#         elif isinstance(current, list):
#             for item in current:
#                 # Append Level 3 hashtags
#                 flat_list.append(item)
#                 flatten(item)
    
#     flatten(nested_json)
#     return flat_list

def extract_all_hashtags(nested_json):
    flat_list = []
    keys_list = []

    def flatten(current, key_path=""):
        if isinstance(current, dict):
            for key, value in current.items():
                flatten(value, key_path + key + ".")
        elif isinstance(current, list):
            for index, item in enumerate(current):
                flatten(item, key_path + str(index) + ".")
        else:
            flat_list.append(current)
            keys_list.append(key_path[:-1])  # Remove trailing dot

    flatten(nested_json)
    return flat_list, keys_list


flat_hashtags_list, key_hashtags_list = extract_all_hashtags(hashtags)
print("Flat list of all hashtags:", flat_hashtags_list)




# %%
keyword_dataset = TextDataset(cut_keywords_str_list)
hashtag_dataset = TextDataset(flat_hashtags_list)

# DataLoader for batching
batch_size = 32
keyword_dataloader = DataLoader(keyword_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
hashtag_dataloader = DataLoader(hashtag_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Text Embedder Model
task_name_to_instruct = {
    "example": "Given a question, retrieve passages that answer the question",
}
embedder = TextEmbedderWithLoader('nvidia/NV-Embed-v2', task_name_to_instruct)

# %%
all_query_embeddings = []
all_passage_embeddings = []

for batch in keyword_dataloader:
    query_embeddings = embedder.encode_batch(batch, task="example")
    all_query_embeddings.append(query_embeddings)

query_embeddings_file = 'keyword_embeddings.pkl'
# save embeddings
with open(query_embeddings_file, 'wb') as f:
    pickle.dump(all_query_embeddings, f)

for batch in hashtag_dataloader:
    passage_embeddings = embedder.encode_batch(batch, task="example")
    all_passage_embeddings.append(passage_embeddings)

passage_embeddings_file = 'hashtag_embeddings.pkl'
with open(passage_embeddings_file, 'wb') as f:
    pickle.dump(all_passage_embeddings, f)

# # Stack embeddings from all batches
# all_query_embeddings = torch.cat(all_query_embeddings, dim=0)
# all_passage_embeddings = torch.cat(all_passage_embeddings, dim=0)

# full_similarity_matrix = embedder.similarity_score(all_query_embeddings, all_passage_embeddings)
# print(full_similarity_matrix.tolist())

# %% [markdown]
# # BELOW DO NOT RUN

# %%
# # method to map json to 1d and map it back to json
# def json_to_1d_list_and_keypaths(nested_json):
#     flat_list = []
#     keys_list = []

#     def flatten(current, key_path=""):
#         if isinstance(current, dict):
#             for key, value in current.items():
#                 flatten(value, key_path + key + ".")
#         elif isinstance(current, list):
#             for index, item in enumerate(current):
#                 flatten(item, key_path + str(index) + ".")
#         else:
#             flat_list.append(current)
#             keys_list.append(key_path[:-1])  # Remove trailing dot

#     flatten(nested_json)
#     return flat_list, keys_list


# def list_to_json(flat_list, keys_list):
#     reconstructed_json = {}

#     def insert_value(nested_dict, keys, value):
#         for i, key in enumerate(keys[:-1]):
#             if key.isdigit():  # If the key is a digit, treat it as a list index
#                 key = int(key)
#                 if not isinstance(nested_dict, list):
#                     nested_dict = []  # Make sure it's a list
#                 # Extend the list if necessary
#                 while len(nested_dict) <= key:
#                     nested_dict.append({} if not keys[i + 1].isdigit() else [])
#             if isinstance(nested_dict, list):
#                 if len(nested_dict) <= key:
#                     nested_dict.extend([None] * (key + 1 - len(nested_dict)))
#                 if nested_dict[key] is None:
#                     nested_dict[key] = {} if not keys[i + 1].isdigit() else []
#                 nested_dict = nested_dict[key]
#             else:
#                 if key not in nested_dict:
#                     nested_dict[key] = {} if not keys[i + 1].isdigit() else []
#                 nested_dict = nested_dict[key]

#         if keys[-1].isdigit():  # Handle the last key
#             key = int(keys[-1])
#             if len(nested_dict) <= key:
#                 nested_dict.extend([None] * (key + 1 - len(nested_dict)))
#             nested_dict[key] = value
#         else:
#             nested_dict[keys[-1]] = value

#     for value, key_path in zip(flat_list, keys_list):
#         keys = key_path.split(".")
#         insert_value(reconstructed_json, keys, value)

#     return reconstructed_json


# # # Example usage:

# # nested_json = hashtags

# # # Convert to 1D list without key paths
# # flat_list, keys_list = json_to_1d_list_and_keypaths(nested_json)
# # print("Flat list (without key paths):", flat_list)
# # print("Keys list (to map back):", keys_list)

# # flat_list = ['1' for _ in range(len(flat_list))]
# # # Now you can use the flat_list (without key paths) and keys_list to map it back to the original structure
# # reconstructed_json = list_to_json(flat_list, keys_list)
# # print("Reconstructed JSON:", reconstructed_json)


# %%
# import torch
# import numpy as np

# def calculate_similarity(embedding1, embedding2):
#     """Calculate the cosine similarity between two embeddings."""
#     return (embedding1 @ embedding2.T).item()

# def find_top_k_similar(embedding, candidate_embeddings, candidate_labels, k):
#     """Find top-k most similar embeddings to a given embedding."""
#     similarities = [(label, calculate_similarity(embedding, candidate_embedding))
#                     for label, candidate_embedding in zip(candidate_labels, candidate_embeddings)]
    
#     # Sort by similarity and get the top k
#     similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
#     return similarities[:k]

# def filter_hashtags_by_level(level, flattened_hashtags, embeddings, current_level_index):
#     """Filter hashtags and embeddings by a certain level index."""
#     level_hashtags = [tag for tag in flattened_hashtags if len(tag.split('.')) == level]
#     level_embeddings = [embeddings[idx] for idx, tag in enumerate(flattened_hashtags) if len(tag.split('.')) == level]
#     return level_hashtags, level_embeddings

# def select_important_hashtags_per_level(query_embedding, flattened_hashtags, embeddings):
#     # Step 1: Calculate similarity with Level 1 hashtags
#     level1_hashtags, level1_embeddings = filter_hashtags_by_level(1, flattened_hashtags, embeddings, 1)
#     top_level1 = find_top_k_similar(query_embedding, level1_embeddings, level1_hashtags, 2)

#     # Store selected Level 1 hashtags and their indices
#     selected_level1 = [hashtag for hashtag, _ in top_level1]

#     # Step 2: For each selected Level 1, calculate similarity with Level 2 hashtags
#     selected_level2_hashtags = []
#     selected_level2_embeddings = []
#     for level1_tag in selected_level1:
#         # Get corresponding Level 2 hashtags under each Level 1
#         level2_hashtags = [tag for tag in flattened_hashtags if tag.startswith(level1_tag) and len(tag.split('.')) == 2]
#         level2_embeddings = [embeddings[idx] for idx, tag in enumerate(flattened_hashtags) if tag in level2_hashtags]

#         # Select the top 4 Level 2 hashtags
#         top_level2 = find_top_k_similar(query_embedding, level2_embeddings, level2_hashtags, 4)
#         selected_level2_hashtags.extend([hashtag for hashtag, _ in top_level2])
#         selected_level2_embeddings.extend([embedding for _, embedding in top_level2])

#     # Step 3: For each selected Level 2, calculate similarity with Level 3 hashtags
#     selected_level3_hashtags = []
#     for level2_tag in selected_level2_hashtags:
#         # Get corresponding Level 3 hashtags under each Level 2
#         level3_hashtags = [tag for tag in flattened_hashtags if tag.startswith(level2_tag) and len(tag.split('.')) == 3]
#         level3_embeddings = [embeddings[idx] for idx, tag in enumerate(flattened_hashtags) if tag in level3_hashtags]

#         # Select the top 8 Level 3 hashtags
#         top_level3 = find_top_k_similar(query_embedding, level3_embeddings, level3_hashtags, 8)
#         selected_level3_hashtags.extend([hashtag for hashtag, _ in top_level3])

#     return selected_level1, selected_level2_hashtags, selected_level3_hashtags

# # Example usage:
# flattened_hashtags = [
#     "Level1_Hashtag1", "Level1_Hashtag2", 
#     "Level1_Hashtag1.Level2_Hashtag1", "Level1_Hashtag1.Level2_Hashtag2",
#     "Level1_Hashtag2.Level2_Hashtag3", "Level1_Hashtag2.Level2_Hashtag4",
#     "Level1_Hashtag1.Level2_Hashtag1.Level3_Hashtag1", "Level1_Hashtag1.Level2_Hashtag1.Level3_Hashtag2",
#     "Level1_Hashtag1.Level2_Hashtag2.Level3_Hashtag3", "Level1_Hashtag1.Level2_Hashtag2.Level3_Hashtag4",
#     "Level1_Hashtag2.Level2_Hashtag3.Level3_Hashtag5", "Level1_Hashtag2.Level2_Hashtag3.Level3_Hashtag6",
#     "Level1_Hashtag2.Level2_Hashtag4.Level3_Hashtag7", "Level1_Hashtag2.Level2_Hashtag4.Level3_Hashtag8"
# ]

# # Example embeddings corresponding to the flattened hashtags
# embeddings = [torch.randn(4096).to('cuda') for _ in range(len(flattened_hashtags))]  # Random embeddings for demonstration
# embeddings = torch.stack([torch.randn(4096) for _ in range(len(flattened_hashtags))]).to('cuda')

# for i in range(50000):
#     # Simulated query embedding (for a given keyword)
#     query_embedding = torch.randn(4096).to('cuda')

#     # Find top hashtags for the query
#     selected_level1, selected_level2, selected_level3 = select_important_hashtags_per_level(query_embedding, flattened_hashtags, embeddings)

#     # print("Selected Level 1 hashtags:", selected_level1)
#     # print("Selected Level 2 hashtags:", selected_level2)
#     # print("Selected Level 3 hashtags:", selected_level3)


# %%
# matrix too big out of time

# import torch
# import numpy as np

# def calculate_similarity_matrix(embeddings1, embeddings2):
#     """Calculate the similarity matrix between two sets of embeddings."""
#     return torch.matmul(embeddings1, embeddings2.T)

# def find_top_k_similarities(similarity_matrix, candidate_labels, k):
#     """Find top-k most similar labels for each keyword based on the similarity matrix."""
#     top_k_similarities = []
#     for row in similarity_matrix:
#         # Get the minimum of k or the number of available candidates
#         k = min(k, len(candidate_labels))
#         # Get the top k indices and their labels
#         top_k_indices = row.topk(k=k, largest=True).indices
#         top_k_labels = [(candidate_labels[i], row[i].item()) for i in top_k_indices]
#         top_k_similarities.append(top_k_labels)
#     return top_k_similarities

# def filter_by_level(flattened_hashtags, level):
#     """Filter hashtags by their level (1, 2, or 3)."""
#     return [tag for tag in flattened_hashtags if len(tag.split('.')) == level]

# def select_important_hashtags_per_level(query_embeddings, flattened_hashtags, hashtag_embeddings):
#     # Step 1: Compute the full similarity matrix between all queries and all hashtags
#     similarity_matrix = calculate_similarity_matrix(query_embeddings, hashtag_embeddings)

#     # Step 2: Filter and select hashtags for each level

#     # Filter Level 1 hashtags
#     level1_hashtags = filter_by_level(flattened_hashtags, 1)
#     level1_indices = [i for i, tag in enumerate(flattened_hashtags) if tag in level1_hashtags]
#     level1_similarity_matrix = similarity_matrix[:, level1_indices]

#     # Select the top 2 Level 1 hashtags for each query
#     top_level1 = find_top_k_similarities(level1_similarity_matrix, level1_hashtags, 2)

#     selected_level2_hashtags = []
#     selected_level3_hashtags = []

#     # Step 3: For each query, get the top 2 Level 1 hashtags and continue filtering
#     for i, (query_embedding, top_level1_tags) in enumerate(zip(query_embeddings, top_level1)):
#         current_level2_hashtags = []
#         current_level3_hashtags = []

#         for level1_tag, _ in top_level1_tags:
#             # Filter Level 2 hashtags under the selected Level 1 tag
#             level2_hashtags = [tag for tag in flattened_hashtags if tag.startswith(level1_tag) and len(tag.split('.')) == 2]
#             level2_indices = [idx for idx, tag in enumerate(flattened_hashtags) if tag in level2_hashtags]
#             level2_similarity_matrix = similarity_matrix[i, level2_indices]

#             # Select the top 4 Level 2 hashtags
#             top_level2 = find_top_k_similarities(level2_similarity_matrix.unsqueeze(0), level2_hashtags, 4)[0]
#             current_level2_hashtags.extend([tag for tag, _ in top_level2])

#             # Filter Level 3 hashtags under the selected Level 2 tags
#             for level2_tag, _ in top_level2:
#                 level3_hashtags = [tag for tag in flattened_hashtags if tag.startswith(level2_tag) and len(tag.split('.')) == 3]
#                 level3_indices = [idx for idx, tag in enumerate(flattened_hashtags) if tag in level3_hashtags]
#                 level3_similarity_matrix = similarity_matrix[i, level3_indices]

#                 # Select the top 8 Level 3 hashtags
#                 top_level3 = find_top_k_similarities(level3_similarity_matrix.unsqueeze(0), level3_hashtags, 8)[0]
#                 current_level3_hashtags.extend([tag for tag, _ in top_level3])

#         selected_level2_hashtags.append(current_level2_hashtags)
#         selected_level3_hashtags.append(current_level3_hashtags)

#     return top_level1, selected_level2_hashtags, selected_level3_hashtags

# # Example usage:
# flattened_hashtags = [
#     "Level1_Hashtag1", "Level1_Hashtag2", 
#     "Level1_Hashtag1.Level2_Hashtag1", "Level1_Hashtag1.Level2_Hashtag2",
#     "Level1_Hashtag2.Level2_Hashtag3", "Level1_Hashtag2.Level2_Hashtag4",
#     "Level1_Hashtag1.Level2_Hashtag1.Level3_Hashtag1", "Level1_Hashtag1.Level2_Hashtag1.Level3_Hashtag2",
#     "Level1_Hashtag1.Level2_Hashtag2.Level3_Hashtag3", "Level1_Hashtag1.Level2_Hashtag2.Level3_Hashtag4",
#     "Level1_Hashtag2.Level2_Hashtag3.Level3_Hashtag5", "Level1_Hashtag2.Level2_Hashtag3.Level3_Hashtag6",
#     "Level1_Hashtag2.Level2_Hashtag4.Level3_Hashtag7", "Level1_Hashtag2.Level2_Hashtag4.Level3_Hashtag8"
# ]

# # Example embeddings corresponding to the flattened hashtags
# hashtag_embeddings = torch.randn(len(flattened_hashtags), 4096).to('cuda')  # Random embeddings for demonstration, moved to CUDA
# query_embeddings = torch.randn(50000, 4096).to('cuda')

# # Find top hashtags for the query
# top_level1, selected_level2, selected_level3 = select_important_hashtags_per_level(query_embeddings, flattened_hashtags, hashtag_embeddings)

# print("Selected Level 1 hashtags:", top_level1)
# print("Selected Level 2 hashtags:", selected_level2)
# print("Selected Level 3 hashtags:", selected_level3)


# %%



