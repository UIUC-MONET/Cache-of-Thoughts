import random
import math
import time
from typing import Any, Dict, List, Set
from dataclasses import dataclass

import hnswlib
import numpy as np
import heapq as heap

# hierachical
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


@dataclass
class DataRecord:
    keywords: Set[Any]
    image_clip: np.array
    vector_data: np.array
    text_data: Any
    image_data: Any
    text_clip: np.array


def proximitiy(veca, vecb):
    return math.exp((-(abs(np.linalg.norm(veca - vecb)) ** 2)) / 100)


class HNSWLFU:
    def __init__(self, dim, num_elements, protect_duration = 10, ef_construction = 40, M = 32):
        self.dim = dim
        self.num_elements = num_elements
        self.hnsw = hnswlib.Index(space = 'l2', dim = dim)
        self.hnsw.init_index(max_elements = num_elements, ef_construction = ef_construction, M = M)
        self.hnsw.set_ef(ef_construction)

        self.frequencies: Dict[int, DataRecord] = {}
        self.data: Dict[int, DataRecord] = {}
        self.inserted_time: Dict[int, int] = {}
        self.history: List[int] = []

        self.cur_timestamp = 0
        self.protect_duration = protect_duration

    def insert_data(self, data: DataRecord):
        self.cur_timestamp += 1
        if len(self.data) < self.num_elements:
            new_id = len(self.data)
            self.data[new_id] = data
            self.frequencies[new_id] = 1
            self.inserted_time[new_id] = self.cur_timestamp
            self.hnsw.add_items(data.vector_data, new_id)
        else:
            # We need to evict
            valid_ids_dict = {k: v for k, v in self.frequencies.items()
                              if self.cur_timestamp - self.inserted_time[k] >= self.protect_duration}
            id_to_evict = min(valid_ids_dict, key=self.frequencies.get)
            self.data[id_to_evict] = data
            self.frequencies[id_to_evict] = 1
            self.inserted_time[id_to_evict] = self.cur_timestamp
            self.hnsw.add_items(data.vector_data, id_to_evict)

    def knn_query(self, query_data: DataRecord, k = 1) -> List[Any]:
        # Query data: 1d vector of length dim
        # Allowed_ids: set of IDs of data records passing the filter
        # Returns images corresponding to the top-k search results.
        allowed_ids = {k for k, v in self.data.items() if v.keywords.intersection(query_data.keywords)}
        filter_func = lambda idx: idx in allowed_ids
        labels, _ = self.hnsw.knn_query(query_data.vector_data, k = k, num_threads = 1, filter = filter_func)
        for i in labels.tolist()[0]:
            self.frequencies[i] += 1
        return [self.data[i].image_data for i in labels.tolist()[0]]


class HNSWRandom:
    def __init__(self, dim, num_elements, protect_duration = 10, ef_construction = 40, M = 32):
        self.dim = dim
        self.num_elements = num_elements
        self.hnsw = hnswlib.Index(space = 'l2', dim = dim)
        self.hnsw.init_index(max_elements = num_elements, ef_construction = ef_construction, M = M)
        self.hnsw.set_ef(ef_construction)

        self.data: Dict[int, DataRecord] = {}
        self.inserted_time: Dict[int, int] = {}

        self.cur_timestamp = 0
        self.protect_duration = protect_duration

    def insert_data(self, data: DataRecord):
        self.cur_timestamp += 1
        if len(self.data) < self.num_elements:
            new_id = len(self.data)
            self.data[new_id] = data
            self.inserted_time[new_id] = self.cur_timestamp
            self.hnsw.add_items(data.vector_data, new_id)
        else:
            # We need to evict
            valid_ids = [k for k in self.data.keys()
                         if self.cur_timestamp - self.inserted_time[k] >= self.protect_duration]
            id_to_evict = random.sample(valid_ids, 1)[0]
            self.data[id_to_evict] = data
            self.inserted_time[id_to_evict] = self.cur_timestamp
            self.hnsw.add_items(data.vector_data, id_to_evict)

    def knn_query(self, query_data: DataRecord, k = 1) -> List[Any]:
        # Query data: 1d vector of length dim
        # Allowed_ids: set of IDs of data records passing the filter
        # Returns images corresponding to the top-k search results.
        allowed_ids = {k for k, v in self.data.items() if v.keywords.intersection(query_data.keywords)}
        filter_func = lambda idx: idx in allowed_ids
        labels, _ = self.hnsw.knn_query(query_data.vector_data, k = k, num_threads = 1, filter = filter_func)
        self.history += labels.tolist()[0] 
        return [self.data[i].image_data for i in labels.tolist()[0]]


class HNSW:
    def __init__(self, dim, num_elements, ef_construction = 40, M = 32, evict_method = 'random'):
        # M for speed and approximation trade off M = 16 
        self.dim = dim
        self.num_elements = num_elements
        self.hnsw = hnswlib.Index(space = 'l2', dim = dim)
        self.hnsw.init_index(max_elements = num_elements, ef_construction = ef_construction, M = M)
        self.hnsw.set_ef(ef_construction)
       
        self.data: Dict[int, DataRecord] = {}
        self.history: List[int] = []

        self.loss_scores: Dict[int, float] = {}
        self.loss_scores_initialized = False
        self.evict_method = evict_method    # choose between billy and random


    def choose_id_to_evict(self, data: DataRecord) -> int:
        # Courtesy of Visualization-aware sampling, Yongjoo Park et al., ICDE 2016
        data_list = [data] + list(self.data.values())

        start_time = time.time()

        cond_loss_scores = {}
        tmp_scores = {}
        for k, v in self.data.items():
            if v.keywords.intersection(data.keywords):
                cond_loss_scores[k] = 1 / sum(proximitiy(j.vector_data, v.vector_data) for j in data_list if v.keywords.intersection(j.keywords))
                tmp_scores[k] = cond_loss_scores[k]
            else:
                tmp_scores[k] = self.loss_scores[k]
        tmp_scores[self.num_elements] = 1 / sum(proximitiy(j.vector_data, data.vector_data) for j in data_list if data.keywords.intersection(j.keywords))

        min_index = min(tmp_scores, key=tmp_scores.get)
        end_time = time.time()
        print(f"Eviction time: {end_time - start_time} seconds")
        return min_index


    def insert_data(self, data: DataRecord):
        start_time = time.time()
        if len(self.data) < self.num_elements:
            new_id = len(self.data)
            self.data[new_id] = data
            self.hnsw.add_items(data.vector_data, new_id)
            end_time = time.time()
        else:
            start_time = time.time()
            if self.evict_method == 'billy':
                if not self.loss_scores_initialized:
                    self.loss_scores_initialized = True
                    for k, v in self.data.items():
                        self.loss_scores[k] = 1 / sum(proximitiy(j.vector_data, v.vector_data) for j in self.data.values() if v.keywords.intersection(j.keywords))

                # We need to evict
                id_to_evict = self.choose_id_to_evict(data)
                if id_to_evict != self.num_elements:
                    evicted_keywords = self.data[id_to_evict].keywords
                    # Element to evict is not the incoming element
                    print("evict:", id_to_evict)
                    self.data[id_to_evict] = data
                    self.hnsw.add_items(data.vector_data, id_to_evict)

                    # Update loss scores
                    for k, v in self.data.items():
                        if evicted_keywords.intersection(v.keywords) or self.data[id_to_evict].keywords.intersection(v.keywords):
                            self.loss_scores[k] = 1 / sum(proximitiy(j.vector_data, v.vector_data) for j in self.data.values() if v.keywords.intersection(j.keywords))
            elif self.evict_method == 'random':
                id_to_evict = np.random.randint(0, self.num_elements)
                self.data[id_to_evict] = data
                self.hnsw.add_items(data.vector_data, id_to_evict)
            end_time = time.time()
            print(f"Evict time: {end_time - start_time} seconds")


    def knn_query_filtered_by_keywords(self, query_data: DataRecord, k = 1) -> List[Any]:
        # Query data: see above
        # Allowed_ids: set of IDs of data records passing the filter
        # Returns images corresponding to the top-k search results.

        allowed_ids = {k for k, v in self.data.items() if v.keywords.intersection(query_data.keywords)}
        filter_func = lambda idx: idx in allowed_ids
        if len(allowed_ids) == 0: # incase there is no overlap
            print("No overlap: ", query_data.keywords)
            filter_func = lambda idx: True
        labels, _ = self.hnsw.knn_query(query_data.vector_data, k = k, num_threads = 1, filter = filter_func)
        self.history += labels.tolist()[0] 
        return [(self.data[i].text_data ,self.data[i].image_data) for i in labels.tolist()[0]]
    
    
    def knn_query(self, query_embedding, k = 1) -> List[Any]:
        # Query Embedding: clip embedding of the query
        # Allowed_ids: set of IDs of data records passing the filter
        # Returns images corresponding to the top-k search results.

        labels, _ = self.hnsw.knn_query(query_embedding, k = k, num_threads = 1)
        self.history += labels.tolist()[0] 
        return [(self.data[i].text_data ,self.data[i].image_data) for i in labels.tolist()[0]]

class HNSWHierachical:
    def __init__(self, dim, num_elements, ef_construction=40, M=32, evict_method='random', num_clusters=100, n_components=50, use_pca=True):
        self.dim = dim
        self.num_elements = num_elements
        self.hnsw = hnswlib.Index(space='l2', dim=dim)
        self.hnsw.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
        self.hnsw.set_ef(ef_construction)

        self.data: Dict[int, DataRecord] = {}
        self.loss_scores: Dict[int, float] = {}
        self.loss_scores_initialized = False
        self.evict_method = evict_method

        # PCA and Clustering setup
        self.num_clusters = num_clusters
        self.use_pca = use_pca
        if self.use_pca:
            if n_components is not None:
                self.pca = PCA(n_components=min(dim, n_components))  # Reduce to at most 50 dimensions or dim
                self.n_components = n_components
            else: 
                self.pca = PCA()
                self.n_components = None

        self.kmeans = None
        self.cluster_means = None
        self.cluster_assignments = {}

    def cold_start_clustering(self, initial_data: List[DataRecord]):
        embeddings = np.array([data.text_clip for data in initial_data])
        if self.use_pca:
            if self.n_components is None:
                self.pca.fit(embeddings)
                cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
                self.n_components = np.argmax(cumulative_variance >= 0.95) + 1
                print(f'n_components = {self.n_components} can accounts 95% variation.')
            reduced_embeddings = self.pca.fit_transform(embeddings)
        else:
            reduced_embeddings = embeddings
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.kmeans.fit(reduced_embeddings)

        self.cluster_means = self.kmeans.cluster_centers_
        for idx, data in enumerate(initial_data):
            cluster_id = self.kmeans.labels_[idx]
            self.cluster_assignments[idx] = cluster_id
            data.keywords.add(f"level1_{cluster_id}")
            self.data[idx] = data
            self.hnsw.add_items(data.vector_data, idx)
            self.history: List[int] = []

    def update_cluster_mean(self, cluster_id: int):
        cluster_members = [idx for idx, cid in self.cluster_assignments.items() if cid == cluster_id]
        if not cluster_members:
            self.cluster_means[cluster_id] = np.zeros_like(self.cluster_means[cluster_id])
            return

        embeddings = np.array([self.data[idx].vector_data for idx in cluster_members])
        if self.use_pca:
            reduced_embeddings = self.pca.transform(embeddings)
        else:
            reduced_embeddings = embeddings
        self.cluster_means[cluster_id] = np.mean(reduced_embeddings, axis=0)

    def assign_cluster(self, data: DataRecord):
        if self.use_pca:
            reduced_embedding = self.pca.transform([data.text_clip])[0]
        else:
            reduced_embedding = data.text_clip
        cluster_id = np.argmin(np.linalg.norm(self.cluster_means - reduced_embedding, axis=1))
        return cluster_id
    
    def assign_top_n_clusters(self, data: DataRecord, n: int):
        if self.use_pca:
            reduced_embedding = self.pca.transform(data.text_clip)[0]
        else:
            reduced_embedding = data.text_clip
        distances = np.linalg.norm(self.cluster_means - reduced_embedding, axis=1)
        top_n_clusters = np.argsort(distances)[:n]
        return top_n_clusters

    def insert_data(self, data: DataRecord, top_n=1):
        if len(self.data) < self.num_elements:
            new_id = len(self.data)
            top_clusters = self.assign_top_n_clusters(data, top_n)
            for cluster_id in top_clusters:
                data.keywords.add(f"level1_{cluster_id}")
            self.data[new_id] = data
            self.hnsw.add_items(data.vector_data, new_id)
            self.cluster_assignments[new_id] = top_clusters[0]
            for cluster_id in top_clusters:
                self.update_cluster_mean(cluster_id)
        else:
            if self.evict_method == 'billy':
                id_to_evict = self.choose_id_to_evict(data)
                evicted_data = self.data[id_to_evict]
                evicted_cluster_id = self.cluster_assignments.pop(id_to_evict)
                del self.data[id_to_evict]

                top_clusters = self.assign_top_n_clusters(data, top_n)
                for cluster_id in top_clusters:
                    data.keywords.add(f"level1_{cluster_id}")
                self.data[id_to_evict] = data
                self.hnsw.add_items(data.vector_data, id_to_evict)
                self.cluster_assignments[id_to_evict] = top_clusters[0]

                self.update_cluster_mean(evicted_cluster_id)
                for cluster_id in top_clusters:
                    self.update_cluster_mean(cluster_id)
            elif self.evict_method == 'random':
                id_to_evict = np.random.randint(0, self.num_elements)
                evicted_cluster_id = self.cluster_assignments.pop(id_to_evict)
                del self.data[id_to_evict]

                top_clusters = self.assign_top_n_clusters(data, top_n)
                for cluster_id in top_clusters:
                    data.keywords.add(f"level1_{cluster_id}")
                self.data[id_to_evict] = data
                self.hnsw.add_items(data.vector_data, id_to_evict)
                self.cluster_assignments[id_to_evict] = top_clusters[0]

                self.update_cluster_mean(evicted_cluster_id)
                for cluster_id in top_clusters:
                    self.update_cluster_mean(cluster_id)

    def knn_query_filtered_by_hierarchy(self, query_data: DataRecord, k=1, top_n=1):
        top_clusters = self.assign_top_n_clusters(query_data, top_n)
    
        query_data.keywords.update({f"level1_{cluster_id}" for cluster_id in top_clusters})
        
        allowed_ids = {
            idx
            for idx, cluster_id in self.cluster_assignments.items()
            if cluster_id in top_clusters
        }
        
        filter_func = lambda idx: idx in allowed_ids
        
        try:
            labels, _ = self.hnsw.knn_query(query_data.vector_data, k=k, num_threads=1, filter=filter_func)
            self.history += labels.tolist()[0] 
            results = [(self.data[i].text_data, self.data[i].image_data) for i in labels.tolist()[0]]
        except Exception as e:
            print(f"KNN query failed: {e}")
            results = []
        
        return results

# # Example code
# dim = 512
# #num_elements = 3126 * 16
# temp = len(validation_dataset_single_image)
# hnsw = HNSW(dim, 1000)
# #data = np.float32(np.random.random((num_elements, dim)))
# stuff_list = [DataRecord(set(hashtag_for_filter[i][2]), clip_features[i][0], clip_features[i][1], clip_features[i][0], image_paths[i]) for i in range(temp)]
# #stuff_list = [DataRecord({str(i % 25)}, data[i,:], i) for i in range(num_elements)]

# for i, stuff in tqdm(enumerate(stuff_list), total=1000):
#     hnsw.insert_data(stuff)
#     #print(stuff.keywords)
#     #print(i)
#     #print(image_paths[i])
