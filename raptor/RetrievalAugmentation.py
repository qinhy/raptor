
import copy
import functools
import logging
import pickle
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import numpy as np
import tiktoken
import umap
from pydantic import BaseModel, Field, model_validator
from scipy import spatial
from sklearn.mixture import GaussianMixture

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def retry_with_exponential_backoff(
    retries: int = 6,
    min_wait: float = 1.0,
    max_wait: float = 20.0,
    multiplier: float = 2.0,
) -> Callable:
    """
    Decorator implementing a simple exponential backoff retry strategy.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wait = max(min_wait, 0.0)
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == retries - 1:
                        raise
                    upper = min(max(wait, min_wait), max_wait)
                    lower = min(min_wait, upper)
                    time.sleep(random.uniform(lower, upper))
                    wait = min(max(upper * multiplier, min_wait), max_wait)

        return wrapper

    return decorator


class Node(BaseModel):
    text: str
    index: int
    children: Set[int] = Field(default_factory=set)
    embeddings: Dict[str, List[float]] # vendor name : vec ==> {"OpenAI":[0.545,..]}

class Tree(BaseModel):
    all_nodes: Dict[int, Node]
    root_nodes: Dict[int, Node]
    leaf_nodes: Dict[int, Node]
    num_layers: int
    layer_to_nodes: Dict[int, List[Node]]

def reverse_mapping(layer_to_nodes: Dict[int, List[Node]]) -> Dict[int, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer


def split_text(
    text: str, tokenizer=tiktoken.get_encoding("cl100k_base"), max_tokens: int=150, overlap: int = 0
):
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.
    """
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)

    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, token_count in zip(sentences, n_tokens):
        if not sentence.strip():
            continue

        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            filtered_sub_sentences = [
                sub.strip() for sub in sub_sentences if sub.strip() != ""
            ]
            sub_token_counts = [
                len(tokenizer.encode(" " + sub_sentence))
                for sub_sentence in filtered_sub_sentences
            ]

            sub_chunk = []
            sub_length = 0

            for sub_sentence, sub_token_count in zip(
                filtered_sub_sentences, sub_token_counts
            ):
                if sub_length + sub_token_count > max_tokens and sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                    if overlap > 0:
                        sub_length = sum(sub_token_counts[-overlap:])
                    else:
                        sub_length = 0

                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count

            if sub_chunk:
                chunks.append(" ".join(sub_chunk))

        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            if overlap > 0 and current_chunk:
                current_length = sum(
                    len(tokenizer.encode(" " + sentence)) for sentence in current_chunk
                )
            else:
                current_length = 0
            current_chunk.append(sentence)
            current_length += token_count

        else:
            current_chunk.append(sentence)
            current_length += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    return [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    indices = sorted(node_dict.keys())
    return [node_dict[index] for index in indices]


def get_embeddings(node_list: List[Node], embedding_model: str) -> List:
    return [node.embeddings[embedding_model] for node in node_list]


def get_children(node_list: List[Node]) -> List[Set[int]]:
    return [node.children for node in node_list]


def get_text(node_list: List[Node]) -> str:
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text


def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    return np.argsort(distances)


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(
        embeddings, min(dim, len(embeddings) - 2)
    )
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters


class BaseEmbeddingModel(BaseModel):
    def create_embedding(self, text):
        raise NotImplementedError

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        raise NotImplementedError

class BaseSummarizationModel(BaseModel):
    def summarize(self, context, max_tokens=150, stop_sequence=None):
        raise NotImplementedError

class OpenAIChat(BaseSummarizationModel):
    system_prompt = "You are a helpful assistant."
    def __init__(self, model: str, client = None):        
        raise NotImplementedError
    
    def answer_question(context, question):
        raise NotImplementedError

class GPT3TurboSummarizationModel(OpenAIChat):
    user_prompt_template = (
        "Write a summary of the following, including as many key details as possible: {context}:"
    )
    def __init__(self, model: str = "gpt-3.5-turbo", client = None):
        super().__init__(model=model, client=client)

class BaseRetriever(BaseModel):
    def retrieve(self, query: str) -> str:
        raise NotImplementedError

class TreeBuilderConfig(BaseModel):
    tokenizer_name: str = "cl100k_base"
    max_tokens: int = 100
    num_layers: int = 5
    threshold: float = 0.5
    top_k: int = 5
    selection_mode: str = "top_k"
    summarization_length: int = 100
    summarization_model: BaseSummarizationModel
    embedding_models: Dict[str, BaseEmbeddingModel]
    cluster_embedding_model: str = "OpenAI"

    def tokenizer(self):
        return tiktoken.get_encoding(self.tokenizer_name)
    def token_len(self, text: str):
        return len(self.tokenizer().encode(text))

class TreeBuilder(BaseModel):
    config: TreeBuilderConfig

    def create_node(
        self, index: int, text: str, children_indices: Optional[Set[int]] = None
    ) -> Tuple[int, Node]:
        if children_indices is None:
            children_indices = set()

        embeddings = {
            model_name: model.create_embedding(text)
            for model_name, model in self.config.embedding_models.items()
        }
        return (index, Node(text, index, children_indices, embeddings))

    def create_embedding(self, text) -> List[float]:
        return self.config.embedding_models[self.config.cluster_embedding_model].create_embedding(
            text
        )

    def summarize(self, context, max_tokens=150) -> str:
        return self.config.summarization_model.summarize(context, max_tokens)

    def get_relevant_nodes(self, current_node:Node, list_nodes:List[Node]) -> List[Node]:
        embeddings = get_embeddings(list_nodes, self.config.cluster_embedding_model)
        distances = distances_from_embeddings(
            current_node.embeddings[self.config.cluster_embedding_model], embeddings
        )
        indices = indices_of_nearest_neighbors_from_distances(distances)

        if self.config.selection_mode == "threshold":
            best_indices = [
                index for index in indices if distances[index] > self.config.threshold
            ]
        elif self.config.selection_mode == "top_k":
            best_indices = indices[: self.config.top_k]
        else:
            best_indices = []

        return [list_nodes[idx] for idx in best_indices]

    def multithreaded_create_leaf_nodes(self, chunks: List[str]) -> Dict[int, Node]:
        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, text): (index, text)
                for index, text in enumerate(chunks)
            }

            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node

        return leaf_nodes

    def build_from_text(self, text: str, use_multithreading: bool = True) -> Tree:
        chunks = split_text(text, self.config.tokenizer(), self.config.max_tokens)

        logging.info("Creating Leaf Nodes")

        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(chunks)
        else:
            leaf_nodes = {}
            for index, chunk in enumerate(chunks):
                __, node = self.create_node(index, chunk)
                leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}
        logging.info(f"Created {len(leaf_nodes)} Leaf Embeddings")

        logging.info("Building All Nodes")
        all_nodes = copy.deepcopy(leaf_nodes)

        root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)

        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.config.num_layers, layer_to_nodes)

        return tree

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        raise NotImplementedError

class ClusterTreeConfig(TreeBuilderConfig):
    reduction_dimension: int = 10
    clustering_algorithm:str = "RAPTOR_Clustering"
    max_length_in_cluster: int = 3500,
    reduction_dimension: int = 10,
    threshold: float = 0.1,
    verbose: bool = False,

    def perform_RAPTOR_clustering(self,
            nodes: List[Node],
            embedding_model_name: str):
        return self.static_perform_RAPTOR_clustering(
            nodes,
            embedding_model_name,
            self.max_length_in_cluster,
            tiktoken.get_encoding(self.tokenizer_name),
            self.reduction_dimension,
            self.threshold,
            self.verbose,
        )

    @staticmethod
    def static_perform_RAPTOR_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
    ) -> List[List[Node]]:
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])

        clusters = perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )

        node_clusters = []

        for label in np.unique(np.concatenate(clusters)):
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]
            cluster_nodes = [nodes[i] for i in indices]

            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )

            if total_length > max_length_in_cluster:
                if verbose:
                    logging.info(
                        f"reclustering cluster with {len(cluster_nodes)} nodes"
                    )
                node_clusters.extend(
                    ClusterTreeConfig.perform_RAPTOR_clustering(
                        cluster_nodes, embedding_model_name, max_length_in_cluster,
                        tokenizer, reduction_dimension, threshold, verbose
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters

class ClusterTreeBuilder(TreeBuilder):
    config:ClusterTreeConfig
    _summary_cache = {}

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)

        def process_cluster(
            cluster:List[Node], new_level_nodes, next_index, summarization_length, lock
        ):
            node_texts = get_text(cluster)

            child_ids = tuple(sorted(n.index for n in cluster))
            if child_ids in self._summary_cache:
                summarized_text = self._summary_cache[child_ids]
            else:
                summarized_text = self.summarize(context=node_texts, max_tokens=summarization_length)
                self._summary_cache[child_ids] = summarized_text

            logging.info(
                f"Node Texts Length: {self.config.token_len(node_texts)}, "
                f"Summarized Text Length: {self.config.token_len(summarized_text)}"
            )

            __, new_parent_node = self.create_node(
                next_index, summarized_text, {node.index for node in cluster}
            )

            with lock:
                new_level_nodes[next_index] = new_parent_node

        for layer in range(self.num_layers):
            new_level_nodes = {}

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.config.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    "Stopping Layer construction: Cannot Create More Layers. "
                    f"Total Layers in tree: {layer}"
                )
                break

            clusters = self.config.perform_RAPTOR_clustering(
                node_list_current_layer,
                self.config.cluster_embedding_model,
            )

            lock = Lock()

            summarization_length = self.config.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)
            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        return current_level_nodes


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 224
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

class TreeRetrieverConfig(BaseModel):
    tokenizer_name: str = "cl100k_base"
    threshold: float = 0.5
    top_k: int = 5
    selection_mode: str = "top_k"
    context_embedding_model: str = "OpenAI"
    embedding_model: BaseEmbeddingModel = Field(default_factory=OpenAIEmbeddingModel)
    num_layers: Optional[int] = None
    start_layer: Optional[int] = None

    def tokenizer(self):
        return tiktoken.get_encoding(self.tokenizer_name)
    def token_len(self, text: str):
        return len(self.tokenizer().encode(text))

class TreeRetriever(BaseModel):
    config:TreeRetrieverConfig
    tree:Tree
    tree_node_index_to_layer:Dict[int,int] = {}

    def model_post_init(self, context):
        tree = self.tree
        config = self.config

        if not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree")

        if not isinstance(config.top_k, int) or config.top_k < 1:
            raise ValueError("top_k must be an integer greater than or equal to 1")
        if not isinstance(config.threshold, (int, float)) or not 0 <= float(
            config.threshold
        ) <= 1:
            raise ValueError("threshold must be a number between 0 and 1")
        if config.selection_mode not in {"top_k", "threshold"}:
            raise ValueError("selection_mode must be either 'top_k' or 'threshold'")
        if not isinstance(config.embedding_model, BaseEmbeddingModel):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        if not isinstance(config.context_embedding_model, str):
            raise ValueError("context_embedding_model must be a string")
        if config.num_layers is not None and (
            not isinstance(config.num_layers, int) or config.num_layers < 0
        ):
            raise ValueError("num_layers must be an integer greater than or equal to 0")
        if config.start_layer is not None and (
            not isinstance(config.start_layer, int) or config.start_layer < 0
        ):
            raise ValueError(
                "start_layer must be an integer greater than or equal to 0"
            )

        if config.num_layers is not None and config.num_layers > tree.num_layers + 1:
            raise ValueError(
                "num_layers in config must be less than or equal to tree.num_layers + 1"
            )

        if config.start_layer is not None and config.start_layer > tree.num_layers:
            raise ValueError(
                "start_layer in config must be less than or equal to tree.num_layers"
            )

        num_layers = (
            config.num_layers if config.num_layers is not None else tree.num_layers + 1
        )
        start_layer = (
            config.start_layer if config.start_layer is not None else tree.num_layers
        )

        if num_layers > start_layer + 1:
            raise ValueError("num_layers must be less than or equal to start_layer + 1")
        
        self.tree_node_index_to_layer = reverse_mapping(self.tree.layer_to_nodes)
        
        return super().model_post_init(context)

    def create_embedding(self, text: str) -> List[float]:
        return self.config.embedding_model.create_embedding(text)

    def retrieve_information_collapse_tree(
        self, query: str, top_k: int, max_tokens: int
    ) -> Tuple[List[Node], str]:
        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = get_node_list(self.tree.all_nodes)

        embeddings = get_embeddings(node_list, self.config.context_embedding_model)

        distances = distances_from_embeddings(query_embedding, embeddings)

        indices = indices_of_nearest_neighbors_from_distances(distances)

        total_tokens = 0
        for idx in indices[:top_k]:
            node = node_list[idx]
            node_tokens = self.config.token_len(node.text)

            if total_tokens + node_tokens > max_tokens:
                break

            selected_nodes.append(node)
            total_tokens += node_tokens

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve_information(
        self, current_nodes: List[Node], query: str, num_layers: int
    ) -> Tuple[List[Node], str]:
        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = current_nodes

        for layer in range(num_layers):
            embeddings = get_embeddings(node_list, self.config.context_embedding_model)

            distances = distances_from_embeddings(query_embedding, embeddings)

            indices = indices_of_nearest_neighbors_from_distances(distances)

            if self.config.selection_mode == "threshold":
                best_indices:List[int] = [
                    index for index in indices if distances[index] > self.config.threshold
                ]
            elif self.config.selection_mode == "top_k":
                best_indices:List[int] = indices[: self.config.top_k].tolist()
            else:
                best_indices:List[int] = []

            nodes_to_add = [node_list[idx] for idx in best_indices]

            selected_nodes.extend(nodes_to_add)

            if layer != num_layers - 1:
                child_nodes = []

                for index in best_indices:
                    child_nodes.extend(node_list[index].children)

                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve(
        self,
        query: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ):
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")

        if not isinstance(collapse_tree, bool):
            raise ValueError("collapse_tree must be a boolean")

        start_layer = self.config.start_layer if start_layer is None else start_layer
        num_layers = self.config.num_layers if num_layers is None else num_layers

        if not isinstance(start_layer, int) or not (
            0 <= start_layer <= self.tree.num_layers
        ):
            raise ValueError(
                "start_layer must be an integer between 0 and tree.num_layers"
            )

        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")

        if num_layers > (start_layer + 1):
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        if collapse_tree:
            logging.info("Using collapsed_tree")
            selected_nodes, context = self.retrieve_information_collapse_tree(
                query, top_k, max_tokens
            )
        else:
            layer_nodes = self.tree.layer_to_nodes[start_layer]
            selected_nodes, context = self.retrieve_information(
                layer_nodes, query, num_layers
            )

        if return_layer_information:
            layer_information = []

            for node in selected_nodes:
                layer_information.append(
                    {
                        "node_index": node.index,
                        "layer_number": self.tree_node_index_to_layer[node.index],
                    }
                )

            return context, layer_information

        return context


# Define a dictionary to map supported tree builders to their respective configs
supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}


class RetrievalAugmentationConfig(BaseModel):    
    tree_builder_config: Optional[TreeBuilderConfig] = None
    tree_retriever_config: Optional[TreeRetrieverConfig] = None
    qa_model: OpenAIChat
    embedding_model: Optional[BaseEmbeddingModel] = None
    summarization_model: Optional[BaseSummarizationModel] = None
    tree_builder_type: str = "cluster"
    tr_tokenizer_name: Optional[str] = None
    tr_threshold: float = 0.5
    tr_top_k: int = 5
    tr_selection_mode: str = "top_k"
    tr_context_embedding_model: str = "OpenAI"
    tr_embedding_model: Optional[BaseEmbeddingModel] = None
    tr_num_layers: Optional[int] = None
    tr_start_layer: Optional[int] = None
    tb_tokenizer_name: Optional[str] = None
    tb_max_tokens: int = 100
    tb_num_layers: int = 5
    tb_threshold: float = 0.5
    tb_top_k: int = 5
    tb_selection_mode: str = "top_k"
    tb_summarization_length: int = 100
    tb_summarization_model: Optional[BaseSummarizationModel] = None
    tb_embedding_models: Optional[Dict[str, BaseEmbeddingModel]] = None
    tb_cluster_embedding_model: Optional[str] = "OpenAI"

class RetrievalAugmentation:
    """
    A Retrieval Augmentation class that combines the TreeBuilder and TreeRetriever classes.
    Enables adding documents to the tree, retrieving information, and answering questions.
    """

    def __init__(self, config=None, tree=None):
        """
        Initializes a RetrievalAugmentation instance with the specified configuration.
        Args:
            config (RetrievalAugmentationConfig): The configuration for the RetrievalAugmentation instance.
            tree: The tree instance or the path to a pickled tree file.
        """
        if config is None:
            config = RetrievalAugmentationConfig()
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError(
                "config must be an instance of RetrievalAugmentationConfig"
            )

        # Check if tree is a string (indicating a path to a pickled tree)
        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("The loaded object is not an instance of Tree")
            except Exception as e:
                raise ValueError(f"Failed to load tree from {tree}: {e}")
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError(
                "tree must be an instance of Tree, a path to a pickled Tree, or None"
            )

        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)

        self.tree_retriever_config = config.tree_retriever_config
        self.qa_model = config.qa_model

        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        logging.info("RetrievalAugmentation initialized with config %s", config)
        
    # --- helper: rebuild a full Tree object given a leaf-node dict ---
    def _build_tree_from_leaf_nodes(self, leaf_nodes: Dict[int, Node],
                                    use_multithreading=False) -> Tree:
        """
        Rebuilds the complete tree (all layers) from a dict of leaf nodes.

        NOTE: This uses the existing TreeBuilder.construct_tree() pipeline so that
        clustering/summarization behavior stays identical to build_from_text().
        """
        # Layer 0 = leaves
        layer_to_nodes = {0: list(leaf_nodes.values())}

        # Start the all_nodes dict with the leaves; construct_tree will append parents
        all_nodes = copy.deepcopy(leaf_nodes)

        # Build upper layers
        root_nodes = self.tree_builder.construct_tree(
            current_level_nodes=leaf_nodes,
            all_tree_nodes=all_nodes,
            layer_to_nodes=layer_to_nodes,
            use_multithreading=use_multithreading,  # keep default behavior consistent; change if you prefer
        )

        # Create a new Tree object
        new_tree = Tree(
            all_nodes=all_nodes,
            root_nodes=root_nodes,
            leaf_nodes=leaf_nodes,
            num_layers=self.tree_builder.config.num_layers,
            layer_to_nodes=layer_to_nodes,
        )
        return new_tree

    def add_to_existing(
        self,
        docs,
        *,
        deduplicate: bool = True,
        use_multithreading: bool = False,
    ) -> None:
        """
        Incrementally add new documents to the existing tree.

        Strategy: Merge old leaf nodes with the new docs' chunks, then rebuild the
        upper layers with the current clustering/summarization pipeline. Existing
        leaf embeddings are reused; new chunk embeddings are computed once.

        Args:
            docs (str | List[str]): Text to add.
            deduplicate (bool): If True, skip new chunks that are identical to
                existing leaf texts (simple exact-string match).
            use_multithreading (bool): Whether to parallelize summarization in
                construct_tree. Defaults to False for reproducibility.
        """
        # If there is no tree yet, just build a new one
        if self.tree is None:
            self.add_documents(docs)
            return

        # Normalize docs input
        if isinstance(docs, list):
            docs = "\n\n".join([d for d in docs if isinstance(d, str) and d.strip()])
        elif not isinstance(docs, str):
            raise ValueError("docs must be a string or a list of strings")

        docs = docs.strip()
        if not docs:
            logging.info("add_to_existing called with empty docs; nothing to add.")
            return

        # Prepare chunking with the same tokenizer/max_tokens as the builder
        tokenizer = self.tree_builder.config.tokenizer()
        max_tokens = self.tree_builder.config.max_tokens
        chunks = split_text(docs, tokenizer=tokenizer, max_tokens=max_tokens)

        if not chunks:
            logging.info("No chunks produced from docs; nothing to add.")
            return

        # Copy existing leaf nodes (reusing their embeddings)
        new_leaf_nodes: Dict[int, Node] = dict(self.tree.leaf_nodes)

        # Optional de-duplication by exact text match against existing leaves
        if deduplicate:
            existing_texts: Set[str] = {
                " ".join(n.text.splitlines()).strip() for n in new_leaf_nodes.values()
            }
            chunks = [c for c in chunks if c.strip() and c.strip() not in existing_texts]
            if not chunks:
                logging.info("All new chunks were duplicates of existing leaves.")
                return

        # Assign unique indices for the new chunks
        start_index = (max(self.tree.all_nodes.keys()) + 1) if self.tree.all_nodes else 0

        # Create nodes (this computes embeddings only for the new chunks)
        for offset, text in enumerate(chunks):
            node_index = start_index + offset
            _, node = self.tree_builder.create_node(index=node_index, text=text)
            new_leaf_nodes[node_index] = node

        # Rebuild the whole tree (parents) from the combined leaves
        # (Reuse the same construct_tree pipeline to keep behavior consistent)
        # Optionally allow multithreading in the construct step
        # (we pass the flag through by temporarily overriding the default parameter)
        tree = self._build_tree_from_leaf_nodes(new_leaf_nodes,use_multithreading)

        # Swap in the new tree and refresher retriever
        self.tree = tree
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)

        logging.info(
            "add_to_existing: added %d new chunks; total leaf nodes: %d; total nodes: %d",
            len(chunks),
            len(self.tree.leaf_nodes),
            len(self.tree.all_nodes),
        )

    def add_documents(self, docs):
        if self.tree is not None:
            logging.warning(
                "Tree already exists; forwarding to add_to_existing(). "
                "If you intended to overwrite, set self.tree=None first."
            )
            self.add_to_existing(docs)
            return

        self.tree = self.tree_builder.build_from_text(text=docs)
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)


    def retrieve(
        self,
        question,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = True,
    ):
        """
        Retrieves information and answers a question using the TreeRetriever instance.

        Args:
            question (str): The question to answer.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            use_all_information (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The context from which the answer can be found.

        Raises:
            ValueError: If the TreeRetriever instance has not been initialized.
        """
        if self.retriever is None:
            raise ValueError(
                "The TreeRetriever instance has not been initialized. Call 'add_documents' first."
            )

        return self.retriever.retrieve(
            question,
            start_layer,
            num_layers,
            top_k,
            max_tokens,
            collapse_tree,
            return_layer_information,
        )

    def answer_question(
        self,
        question,
        top_k: int = 10,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ):
        """
        Retrieves information and answers a question using the TreeRetriever instance.

        Args:
            question (str): The question to answer.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            use_all_information (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The answer to the question.

        Raises:
            ValueError: If the TreeRetriever instance has not been initialized.
        """
        # if return_layer_information:
        context, layer_information = self.retrieve(
            question, start_layer, num_layers, top_k, max_tokens, collapse_tree, True
        )

        answer = self.qa_model.answer_question(context, question)

        if return_layer_information:
            return answer, layer_information

        return answer

    def save(self, path):
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"Tree successfully saved to {path}")
