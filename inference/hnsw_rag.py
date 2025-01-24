import random
import math
import pickle
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

import hnswlib
import numpy as np
import heapq as heap


@dataclass
class DataRecord:
    keywords: Set[Any]
    image_clip: np.array
    # text_clip: np.array
    vector_data: np.array
    text_data: Any
    image_data: Any


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

        #self.loss_score_heap: List[Tuple[float, int]] = []
        #self.loss_score_dirty: Dict[int, bool] = {}
        self.loss_scores: Dict[int, float] = {}
        self.loss_scores_initialized = False
            # self.data_score_dirty[i] = True
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
            #print(f"Insert time: {end_time - start_time} seconds")
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
        #labels, _ = self.hnsw.knn_query(query_data.vector_data, k = k, num_threads = 1)
        return [(self.data[i].text_data ,self.data[i].image_data) for i in labels.tolist()[0]]
    
    
    def knn_query(self, query_embedding, k = 1) -> List[Any]:
        # Query Embedding: clip embedding of the query
        # Allowed_ids: set of IDs of data records passing the filter
        # Returns images corresponding to the top-k search results.

        labels, _ = self.hnsw.knn_query(query_embedding, k = k, num_threads = 1)
        self.history += labels.tolist()[0] 
        return [(self.data[i].text_data ,self.data[i].image_data) for i in labels.tolist()[0]]



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
