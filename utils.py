import torch
import faiss
import random
import logging
import numpy as np
from enum import Enum
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

import json

class JSONLinesHandler(logging.FileHandler):
    def emit(self, record):
        log_entry = self.format(record)
        with open(self.baseFilename, 'a') as file:
            file.write(f"{log_entry}\n")

def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up jsonlines logger."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w') as file:
        pass  # create the file if it does not exist

    formatter = logging.Formatter('%(message)s')  # Only message gets logged
    handler = JSONLinesHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def parse_pred_text(pred_text: str, label_set: set[str]) -> str:
    """A simple heuristic parsing function for compatibility with the label_set."""
    pred_text = pred_text.strip(" ().:")
    if pred_text[0] in label_set:
        pred_text = pred_text[0]
    return pred_text

def text_in_label_set(text: str, label_set: set[str]) -> bool:
    text = text.lower().strip()
    fuzzy_label_set = {label.lower() for label in label_set}
    return text in fuzzy_label_set

class RetrieveOrder(Enum):
    SIMILAR_AT_TOP = "similar_at_top"  # the most similar retrieved chunk is ordered at the top
    SIMILAR_AT_BOTTOM = "similar_at_bottom"  # reversed
    RANDOM = "random"  # randomly shuffle the retrieved chunks

class RAG:

    def __init__(self, rag_config: dict) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(rag_config["embedding_model"])
        self.embed_model = AutoModel.from_pretrained(rag_config["embedding_model"]).eval()
        
        self.index = None
        self.id2evidence = dict()
        self.embed_dim = len(self.encode_data("Test embedding size"))
        self.insert_acc = 0
        
        self.seed = rag_config["seed"]
        self.top_k = rag_config["top_k"]
        orders = {member.value for member in RetrieveOrder}
        assert rag_config["order"] in orders
        self.retrieve_order = rag_config["order"]
        random.seed(self.seed)
        
        self.create_faiss_index()
        
        # TODO: make a file to save the inserted rows
        '''
        self.rag_filename = rag_config.get("rag_filename", "rag_data.jsonl")
        if Path(self.rag_filename).exists():
            with open(self.rag_filename, 'r', encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line.strip())
                    self.insert(row["key"], row["value"])
        '''

    def create_faiss_index(self):
        # Create a FAISS index
        self.index = faiss.IndexFlatL2(self.embed_dim)

    def encode_data(self, sentence: str) -> np.ndarray:
        # Tokenize the sentence
        encoded_input = self.tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        feature = sentence_embeddings.numpy()[0]
        norm = np.linalg.norm(feature)
        return feature / norm

    def insert(self, key: str, value: str) -> None:
        """Use the key text as the embedding for future retrieval of the value text."""
        embedding = self.encode_data(key).astype('float32')  # Ensure the data type is float32
        self.index.add(np.expand_dims(embedding, axis=0))
        self.id2evidence[str(self.insert_acc)] = value
        
        '''
        with open(self.rag_filename, 'a', encoding='utf-8') as f:
            json.dump({"key": key, "value": value}, f)
            f.write("\n")
        '''
        
        self.insert_acc += 1

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Retrieve top-k text chunks"""
        embedding = self.encode_data(query).astype('float32')  # Ensure the data type is float32
        top_k = min(top_k, self.insert_acc)
        distances, indices = self.index.search(np.expand_dims(embedding, axis=0), top_k)
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        
        results = [{'link': str(idx), '_score': {'faiss': dist}} for dist, idx in zip(distances, indices)]
        # Re-order the sequence based on self.retrieve_order
        if self.retrieve_order == RetrieveOrder.SIMILAR_AT_BOTTOM.value:
            results = list(reversed(results))
        elif self.retrieve_order == RetrieveOrder.RANDOM.value:
            random.shuffle(results)
        
        text_list = [self.id2evidence[result["link"]] for result in results]
        return text_list
    
class AdaptiveRAG:
    def __init__(self, rag_config: dict):
        self.tokenizer = AutoTokenizer.from_pretrained(rag_config["embedding_model"])
        self.embed_model = AutoModel.from_pretrained(rag_config["embedding_model"]).eval()
        self.index = faiss.IndexFlatL2(rag_config.get("embed_dim", 768))
        self.id2evidence = {}
        self.insert_acc = 0
        self.top_k = rag_config["top_k"]
        
        self.default_weight = 1.0
        
        self.retrieve_count = {}
        self.insert_order = {}

    def encode_data(self, text: str) -> np.ndarray:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            embeddings = self.embed_model(**tokens).last_hidden_state[:, 0, :]
        return embeddings.squeeze().numpy()

    def insert(self, key: str, value: str):
        embedding = self.encode_data(key).astype("float32")
        self.index.add(embedding[np.newaxis, :])
        self.id2evidence[str(self.insert_acc)] = value
        self.retrieve_count[str(self.insert_acc)] = 0
        self.insert_order[str(self.insert_acc)] = self.insert_acc
        self.insert_acc += 1

    def retrieve(self, query: str) -> list[tuple[str, float]]:
        embedding = self.encode_data(query).astype("float32")
        distances, indices = self.index.search(embedding[np.newaxis, :], self.top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            evidence = self.id2evidence.get(str(idx), None)
            if evidence:
                results.append((evidence, 1.0 / (dist + 1e-5)))
                self.retrieve_count[str(idx)] = self.retrieve_count.get(str(idx), 0) + 1
        
        return results

    def adjust_weights(self, retrieval_scores):
        max_score = max(retrieval_scores) if retrieval_scores else 1.0
        weights = [score / max_score for score in retrieval_scores]
        
        return weights

    def update_memory(self, top_k: int) -> None:
        sorted_examples = sorted(
            self.id2evidence.keys(),
            key=lambda x: (self.retrieve_count.get(x, 0), self.insert_order.get(x, 0)),
            reverse=True
        )
        keep_indices = set(sorted_examples[:top_k])
        
        self.id2evidence = {k: v for k, v in self.id2evidence.items() if k in keep_indices}
        self.retrieve_count = {k: v for k, v in self.retrieve_count.items() if k in keep_indices}
        self.insert_order = {k: v for k, v in self.insert_order.items() if k in keep_indices}
        
        self.index.reset()
        for idx, evidence in self.id2evidence.items():
            embedding = self.encode_data(evidence).astype("float32")
            self.index.add(embedding[np.newaxis, :])
            
        print('Memory update!')

def extract_json_string(res: str) -> str:
    """Extract the first valid json string from the response string (of LLMs).
    
    Return '' (empty string) if not found. Raise ValueError if an } is found before any {.
    """
    start, end = -1, -1
    cnt = 0  # act as a representation of a stack of '{' '}' pairs
    for i in range(len(res)):
        ch = res[i]
        if ch == '{':
            if cnt == 0:  # the first time '{' is encountered
                start = i
            cnt += 1
        elif ch == '}':
            if cnt <= 0:
                raise ValueError("found } before any { appears")
            cnt -= 1
            if cnt == 0:  # found the end index
                end = i
                break
    return res[start:end+1]

def strip_all_lines(s: str) -> str:
    """Remove all leading and trailing spaces of each line in the string."""
    return '\n'.join([line.strip() for line in s.splitlines()])

if __name__ == "__main__":
# Initialize RAG with a configuration dictionary
    rag_config = {
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "rag_filename": "test_rag_pool",
        "seed": 42,
        "top_k": 5,
        "order": "similar_at_top"  # ["similar_at_top", "similar_at_bottom", "random"]
    }
    rag = RAG(rag_config)

    # Key-value pairs for testing
    key_value_pairs = [
        ("Apple is my favorite fruit", "Oh really?"),
        ("What is your favorite fruit?", "Lettuce, tomato, and spinach."),
        ("What is your favorite vegetable?", "Apple, banana, and watermelon."),
        ("What do you like to read in your free time?", "Sherlock Holmes")
    ]

    # Insert the key-value pairs into the RAG
    for key, value in key_value_pairs:
        rag.insert(key, key + ' ' + value)

    from pprint import pprint

    query = "I like to eat lettuce."
    results = rag.retrieve(query, top_k=rag_config["top_k"])
    pprint(results)

def merge_dicts(dicts: list[dict]) -> dict:
    d = dict()
    for dd in dicts:
        for k, v in dd.items():
            if (k in d) and (d[k] != v):
                print(k, d[k], v)
                raise ValueError("Found duplicated and inconsistent key-value pair.")
            d[k] = v
    return d
