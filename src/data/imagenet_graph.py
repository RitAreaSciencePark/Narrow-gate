import nltk
custom_nltk_dir = '/orfeo/scratch/dssc/zenocosini/nltk_data'
nltk.data.path.append(custom_nltk_dir)
wn = nltk.corpus.wordnet
# from nltk.corpus import wordnet as wn
import inflect
import datasets
from dataclasses import dataclass
import networkx as nx
from easyroutine.logger import Logger
import os
from src.data.imagenet_classes import IMAGENET2012_CLASSES
import matplotlib.pyplot as plt
from typing import Optional, Union, Literal
import pickle
from collections import defaultdict
import random
# from rich import print



class ImageNetGraph:
    """
    ImageNetGraph class to build a graph of ImageNet classes using WordNet synsets.

    """

    def __init__(
        self,
        root_synset_name: Optional[str] = None,
        root_offset: Optional[str] = None,
        tree: bool = False,
        graph: Optional[nx.DiGraph] = None,
        init_graph: bool = True,
    ):
        self.logger = Logger(
            logname="imagenet_graph",
            log_file_path=f"logs.log",
        )

        self.tree = tree
        self.wordnet = wn
        self.p = inflect.engine()

        self.synsets_to_class_idx = {
            syn: idx for idx, syn in enumerate(IMAGENET2012_CLASSES.keys())
        }
        self.imagenet_synsets = set(self.synsets_to_class_idx.keys())
        if init_graph:
            if root_synset_name:
                self.root_synset = wn.synset(root_synset_name)
            elif root_offset:
                self.root_synset = self._get_synset_from_offset(root_offset)
            else:
                self.root_synset = None

            if graph is None:
                self.graph = self._build_imagenet_wordnet_graph(
                    root_synset=self.root_synset
                )

                if tree:
                    self.graph = self.build_tree_from_graph(
                        self.graph, self.root_synset.name()
                    )  # type: ignore
                    self.logger.info(
                        msg=f"Tree built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges",
                        std_out=True,
                    )
            else:
                self.graph = graph
                self.logger.info(
                    msg=f"Graph loaded with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges",
                    std_out=True,
                )
        self.n_imagenet_class = len(IMAGENET2012_CLASSES.keys())

    def __repr__(self):
        return f"ImageNetGraph(root_synset={self.root_synset.name()})"  # type: ignore

    def __str__(self):
        return f"ImageNetGraph(root_synset={self.root_synset.name()})"  # type: ignore

    def get_all_offsets(self):
        return list(self.graph.nodes)

    def get_all_synsets(self):
        return [self.graph.nodes[offset]["label"] for offset in self.graph.nodes]

    def get_all_definitions(self):
        return [self.get_definition(offset=offset) for offset in self.graph.nodes]

    def get_all_lemma_names(self):
        return [self.graph.nodes[offset]["label"] for offset in self.graph.nodes]

    def children(
        self,
        synset_name: Optional[str] = None,
        offset: Optional[str] = None,
        depth: Union[int, Literal["max"]] = 1,
    ):
        if synset_name:
            synset = self.wordnet.synset(synset_name)
            offset = self._get_offset_from_synset(synset_name)
        elif offset:
            synset = self._get_synset_from_offset(offset)
        else:
            self.logger.error("No synset or offset provided")

        children = set()
        queue = [(synset, 0)]
        while queue:
            node, level = queue.pop(0)
            if level == depth:
                break
            for child in node.hyponyms():  # type: ignore
                children.add(child.name())
                queue.append((child, level + 1))  # type: ignore
        return list(children)

    def parents(
        self,
        synset_name: Optional[str] = None,
        offset: Optional[str] = None,
        depth: Union[int, Literal["max"]] = 1,
    ):
        if synset_name:
            synset = self.wordnet.synset(synset_name)
            offset = self._get_offset_from_synset(synset_name)
        elif offset:
            synset = self._get_synset_from_offset(offset)
        else:
            self.logger.error("No synset or offset provided")

        parents = set()
        queue = [(synset, 0)]
        while queue:
            node, level = queue.pop(0)
            if level == depth:
                break
            for parent in node.hypernyms():  # type: ignore
                parents.add(parent.name())
                queue.append((parent, level + 1))  # type: ignore
        return list(parents)

    def all_nodes(self):
        return self.graph.nodes

    def leaf_nodes(self):
        return [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]

    def root_nodes(self):
        return [node for node in self.graph.nodes if self.graph.in_degree(node) == 0]

    def _get_synset_from_offset(self, offset):
        """
        Given a WordNet offset (e.g. n00001740), return the corresponding synset (e.g. Synset('entity.n.01'))
        """
        try:
            return wn.synset_from_pos_and_offset("n", int(offset[1:]))
        except:
            self.logger.warning(f"Could not find synset for offset {offset}")
            return None

    def get_mapping_dict(self):
        all_info_dict = {}  # offset key ad value is a dict with synset_name, definition and lemma_names
        for offset in self.graph.nodes:
            all_info_dict[offset] = self.get_all(offset=offset)
        return all_info_dict

    def _get_offset_from_synset(self, synset_name: str):
        """
        Given a synset name (e.g. entity.n.01), return the corresponding WordNet offset (e.g. n00001740)
        """
        synset = self.wordnet.synset(synset_name)
        return f"n{synset.offset():08d}"  # type: ignore

    def get_definition(
        self, synset_name: Optional[str] = None, offset: Optional[str] = None
    ):
        """
        Given a synset or offset, return the definition of the synset.
        For example, for the synset 'entity.n.01' the definition is 'that which is perceived or known or inferred to have its own distinct existence (living or nonliving)'
        Args:
            synset_name: str, synset name (e.g. 'entity.n.01')
            offset: str, WordNet offset (e.g. 'n00001740')

        """
        if synset_name:
            return wn.synset(synset_name).definition()  # type: ignore
        elif offset:
            synset = self._get_synset_from_offset(offset).name()  # type: ignore
            return wn.synset(synset).definition()  # type: ignore
        else:
            self.logger.error("No synset or offset provided")

    def get_all(self, synset_name: Optional[str] = None, offset: Optional[str] = None):
        """
        Given a synset or offset, return a dictionary with the offset, synset name, definition and lemma names.

        Args:
            synset_name: str, synset name (e.g. 'entity.n.01')
            offset: str, WordNet offset (e.g. 'n00001740')

        Returns:
            dict: dictionary with the offset, synset name, definition and lemma names.

        """
        if synset_name:
            offset = self._get_offset_from_synset(synset_name)
            synset = self.wordnet.synset(synset_name)
        elif offset:
            synset = self._get_synset_from_offset(offset)
        else:
            self.logger.error("No synset or offset provided")
        return {
            "offset": offset,
            "synset_name": synset.name(),  # type: ignore
            "definition": self.get_definition(offset=offset),
            "lemma_names": synset.lemma_names(),  # type: ignore
        }

    def _build_imagenet_wordnet_graph(self, root_synset=None):
        """
        Given a root synset, build a graph of ImageNet classes using WordNet synsets.

        Args:
            root_synset: str, root synset name (e.g. 'entity.n.01')

        Returns:
            nx.DiGraph: a directed graph of ImageNet classes using WordNet synsets.
        """
        G = nx.DiGraph()
        imagenet_nodes = set()
        non_imagenet_nodes = set()

        if root_synset:
            allowed_synsets = set(self._get_all_descendants(root_synset))
        else:
            allowed_synsets = None

        for offset, _ in IMAGENET2012_CLASSES.items():
            synset = self._get_synset_from_offset(offset)
            if synset is None or (allowed_synsets and synset not in allowed_synsets):
                continue

            G.add_node(offset, label=synset.name())
            imagenet_nodes.add(offset)

            # Add edges to all hypernyms (parents) up to the root
            hypernym_path = synset.hypernym_paths()[0]
            for i in range(len(hypernym_path)):
                child = hypernym_path[i]
                child_offset = f"n{child.offset():08d}"
                if i + 1 < len(hypernym_path):
                    parent = hypernym_path[i + 1]
                    parent_offset = f"n{parent.offset():08d}"
                    if child_offset not in G:
                        G.add_node(child_offset, label=child.name())
                        if child_offset != offset:
                            non_imagenet_nodes.add(child_offset)
                    if parent_offset not in G:
                        G.add_node(parent_offset, label=parent.name())
                        non_imagenet_nodes.add(parent_offset)
                    G.add_edge(child_offset, parent_offset)

        self.logger.info(
            msg=f"ImageNet-WordNet graph built with {len(G.nodes)} nodes and {len(G.edges)} edges",
            std_out=True,
        )
        self.logger.info(msg=f"ImageNet nodes: {len(imagenet_nodes)}", std_out=True)
        self.logger.info(
            msg=f"Non-ImageNet (ancestor) nodes: {len(non_imagenet_nodes)}",
            std_out=True,
        )
        return G

    def _get_all_descendants(self, synset):
        """
        Given a synset, return all its descendants (hyponyms).
        """
        descendants = set()
        for hyponym in synset.hyponyms():
            descendants.add(hyponym)
            descendants.update(self._get_all_descendants(hyponym))
        return descendants

    def get_custom_subgraph(self, root_word):
        """
        Given a root word, build a custom subgraph of ImageNet classes using WordNet synsets.
        """
        root_synset = wn.synset(root_word)
        return self._build_imagenet_wordnet_graph(root_synset=root_synset)

    def build_tree_from_graph(self, graph, root_node):
        """
        Given a graph and a root node, build a tree using a Breadth-First Search (BFS) traversal.
        """
        tree = nx.DiGraph()
        visited = set()
        queue = [self._get_offset_from_synset(root_node)]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                # Assuming each node stores its children in the graph
                for child in graph.successors(node):
                    if child not in visited:
                        tree.add_node(child, label=graph.nodes[child]["label"])
                        tree.add_edge(node, child)
                        queue.append(child)
        return tree

    def visualize_graph(self, graph=None, filename="imagenet_wordnet_graph.png"):
        if graph is None:
            graph = self.graph
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(graph, k=0.5, iterations=50)
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_size=100,
            node_color="lightblue",
            font_size=8,
            font_weight="bold",
            arrows=True,
        )
        labels = nx.get_node_attributes(graph, "label")
        nx.draw_networkx_labels(graph, pos, labels, font_size=6)
        plt.show()
        self.logger.info(msg=f"Graph visualization saved to {filename}", std_out=True)

    def visualize_tree(self, tree, filename="tree_visualization.png"):
        if tree is None or tree.number_of_nodes() == 0:
            self.logger.error("Provided tree is empty or None, cannot visualize.")
            return

        def draw_tree(G, root, pos={}, x=0, y=0, layer=1):
            pos[root] = (x, y)
            children = list(G.successors(root))
            if len(children) != 0:
                dx = 2 ** (-layer)
                nextx = x - sum([dx * (i + 1) for i in range(len(children) - 1)])
                for child in children:
                    nextx += dx
                    pos = draw_tree(G, child, pos, nextx, y - 1, layer + 1)
            return pos

        # Find the root (node with in_degree 0)
        root = [node for node in tree.nodes() if tree.in_degree(node) == 0][0]

        # Create the position dictionary
        pos = draw_tree(tree, root)

        # Create a new figure
        fig, ax = plt.subplots(figsize=(20, 20))

        # Draw nodes
        nx.draw_networkx_nodes(tree, pos, node_size=50, node_color="skyblue", ax=ax)

        # Draw edges
        nx.draw_networkx_edges(tree, pos, ax=ax)

        # Add labels
        labels = nx.get_node_attributes(tree, "label")
        nx.draw_networkx_labels(tree, pos, labels, font_size=8, ax=ax)

        # Invert the y-axis to have the root at the top
        ax.invert_yaxis()

        # Remove axis
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        self.logger.info(msg=f"Tree visualization saved to {filename}", std_out=True)

    def save(self, path: str):
        """
        Save the the grah
        """
        dict = {
            "graph": self.graph,
            "root_synset_name": self.root_synset.name() if self.root_synset else None,
            "tree": self.tree,
        }

        # Save the graph
        with open(path, "wb") as f:
            pickle.dump(dict, f)
        self.logger.info(msg=f"Graph saved to {path}", std_out=True)

    @classmethod
    def load(cls, path: str):
        """
        Load the graph
        """
        with open(path, "rb") as f:
            dict = pickle.load(f)
        return cls(**dict)

    def info(self):
        print(
            f"ImageNetGraph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
        )
        print(f"Root synset: {self.root_synset.name() if self.root_synset else None}")


import random


class SampledImageNetGraph(ImageNetGraph):
    def __init__(self, n_class: int, loaded: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_imagenet_class = n_class
        if loaded is True:
            self.sampled_classes = self._sample_classes()
            self.graph = self._build_sampled_graph()

    def _sample_classes(self):
        """
        Sample n number of classes from IMAGENET2012_CLASSES.
        """
        return random.sample(list(IMAGENET2012_CLASSES.keys()), self.n_imagenet_class)

    def _build_sampled_graph(self):
        """
        Build a graph considering only the sampled leaves.
        """
        G = nx.DiGraph()

        for offset in self.sampled_classes:
            synset = self._get_synset_from_offset(offset)
            if synset is None:
                continue

            current = synset
            while current:
                current_offset = f"n{current.offset():08d}"
                G.add_node(current_offset, label=current.name())

                hypernyms = current.hypernyms()
                if hypernyms:
                    parent = hypernyms[0]
                    parent_offset = f"n{parent.offset():08d}"
                    G.add_node(parent_offset, label=parent.name())
                    G.add_edge(parent_offset, current_offset)
                    current = parent
                else:
                    break

        self.logger.info(
            msg=f"Sampled ImageNet graph built with {len(G.nodes)} nodes and {len(G.edges)} edges",
            std_out=True,
        )
        return G

    def get_sampled_classes(self):
        """
        Return the list of sampled class offsets.
        """
        return self.sampled_classes

    def get_sampled_synsets(self):
        """
        Return the list of sampled synsets.
        """
        return [self._get_synset_from_offset(offset) for offset in self.sampled_classes]

    def get_sampled_lemma_names(self):
        """
        Return the list of lemma names for the sampled classes.
        """
        return [IMAGENET2012_CLASSES[offset] for offset in self.sampled_classes]

    def save(self, path: str):
        """
        Save the the grah
        """
        dict = {
            "graph": self.graph,
            "root_synset_name": self.root_synset.name() if self.root_synset else None,
            "tree": self.tree,
            "n_class": self.n_imagenet_class,
        }

        with open(path, "wb") as f:
            pickle.dump(dict, f)





class BalancedManualSampleImagenet(ImageNetGraph):
    def __init__(self, high_level_classes: dict, high_level_classes_type: Literal["synset_name", "offset"] = "synset_name"):
        super().__init__(init_graph=False)
        if high_level_classes_type == "offset":
            self.high_level_classes = {
                self._get_synset_from_offset(k).name(): v
                for k, v in high_level_classes.items()
            }
        self.high_level_classes = high_level_classes
        self.root_synset = None
        self.high_level_leaves = {}
        self.leaf_image_counts = {}
        self.depth_image_distribution = {}  # Dictionary to hold image distribution by depth
        self.graph = self._build_balanced_manual_graph()

    def _build_balanced_manual_graph(self):
        """
        Constructs a directed graph representing a balanced distribution of images across high-level classes from ImageNet, 
        ensuring the total number of images allocated equals the specified number for each class.
        """
        self.logger.info(
            msg=f"Building balanced manual ImageNet graph with {len(self.high_level_classes)} high-level classes",
            std_out=True,
        )
        G = nx.DiGraph()

        def is_relevant_synset(synset):
            if f"n{synset.offset():08d}" in self.imagenet_synsets:
                return True
            return any(is_relevant_synset(hypo) for hypo in synset.hyponyms())

        def count_relevant_leaves(synset):
            if f"n{synset.offset():08d}" in self.imagenet_synsets and not any(
                f"n{hypo.offset():08d}" in self.imagenet_synsets for hypo in synset.hyponyms()
            ):
                return 1
            return sum(count_relevant_leaves(hypo) for hypo in synset.hyponyms() if is_relevant_synset(hypo))

        for high_level_class, num_images in self.high_level_classes.items():
            high_level_synset = wn.synset(high_level_class)
            high_level_offset = f"n{high_level_synset.offset():08d}"
            G.add_node(high_level_offset, label=high_level_synset.name())

            # Get all relevant nodes in the subgraph
            subgraph_nodes = [
                s for s in high_level_synset.closure(lambda s: s.hyponyms())
                if is_relevant_synset(s)
            ]
            subgraph_nodes.append(high_level_synset)

            # Calculate depth and subtree size for each node
            depths = {}
            subtree_sizes = {}
            for s in subgraph_nodes:
                path = s.hypernym_paths()[0]
                relevant_path = [node for node in path if is_relevant_synset(node)]
                depths[s] = len(relevant_path) - 1
                subtree_sizes[s] = count_relevant_leaves(s)

            max_depth = max(depths.values())

            # Calculate weights based on depth and subtree size
            weights = {
                s: (max_depth - depths[s] + 1) * subtree_sizes[s]
                for s in subgraph_nodes
            }

            # Get ImageNet leaves (nodes in ImageNet with no ImageNet children)
            imagenet_leaves = [
                s for s in subgraph_nodes
                if f"n{s.offset():08d}" in self.imagenet_synsets and subtree_sizes[s] == 1
            ]

            if not imagenet_leaves:
                print(f"Warning: No ImageNet leaves found for {high_level_class}")
                continue

            # Normalize weights for ImageNet leaves
            leaf_weights = {s: weights[s] for s in imagenet_leaves}
            total_leaf_weight = sum(leaf_weights.values())
            normalized_weights = {
                s: w / total_leaf_weight for s, w in leaf_weights.items()
            }

            # Distribute images based on normalized weights
            image_distribution = {}
            total_allocated_images = 0
            for s, weight in normalized_weights.items():
                images = int(num_images * weight)
                image_distribution[s] = images
                total_allocated_images += images

            # Correct any shortfall or surplus in allocated images
            remaining_images = num_images - total_allocated_images
            if remaining_images != 0:
                # Distribute the remaining images proportionally to those with the largest rounding errors
                sorted_leaves = sorted(
                    imagenet_leaves,
                    key=lambda x: normalized_weights[x] - image_distribution[x] / num_images,
                    reverse=True
                )

                for s in sorted_leaves:
                    if remaining_images == 0:
                        break
                    image_distribution[s] += 1
                    remaining_images -= 1

            # Build the graph and assign image counts
            self.high_level_leaves[high_level_class] = []
            for s in subgraph_nodes:
                s_offset = f"n{s.offset():08d}"
                G.add_node(s_offset, label=s.name())

                if s_offset in self.imagenet_synsets:
                    if s in imagenet_leaves:
                        self.leaf_image_counts[s.name()] = image_distribution[s]
                        self.high_level_leaves[high_level_class].append(s.name())

                for hypernym in s.hypernyms():
                    if hypernym in subgraph_nodes:
                        hypernym_offset = f"n{hypernym.offset():08d}"
                        G.add_edge(hypernym_offset, s_offset)

        self.logger.info(
            msg=f"Balanced manual ImageNet graph built with {len(G.nodes)} nodes and {len(G.edges)} edges",
            std_out=True,
        )
        return G

    
    def save(self, path: str):
        """
        Save the the grah
        """
        dict = {
            "graph": self.graph,
            "root_synset_name": self.root_synset.name() if self.root_synset else None,
            "tree": self.tree,
            "high_level_classes": self.high_level_classes,
            "high_level_leaves": self.high_level_leaves,
            "leaf_image_counts": self.leaf_image_counts,
        }

        with open(path, "wb") as f:
            pickle.dump(dict, f)

    @classmethod
    def load(cls, path: str):
        """
        Load the graph and return an initialized BalancedManualSampleImagenet instance.

        Args:
            path (str): The path to the saved graph file.

        Returns:
            BalancedManualSampleImagenet: An initialized instance of the class.
        """
        with open(path, "rb") as f:
            saved_data = pickle.load(f)

        # Create an instance of the class
        instance = cls({}, "synset_name")  # Initialize with empty high_level_classes

        # Populate the instance attributes
        instance.graph = saved_data["graph"]
        instance.root_synset = wn.synset(saved_data["root_synset_name"]) if saved_data["root_synset_name"] else None
        instance.tree = saved_data["tree"]
        instance.high_level_classes = saved_data["high_level_classes"]
        instance.high_level_leaves = saved_data["high_level_leaves"]
        instance.leaf_image_counts = saved_data["leaf_image_counts"]
        instance.n_imagenet_class = len(instance.high_level_classes)

        # Rebuild the depth_image_distribution
        instance.depth_image_distribution = {}  # This will be populated if needed

        instance.logger.info(msg=f"Graph loaded from {path}", std_out=True)

        return instance
        
        
    def map_leaf_to_high_level(
        self,
        leaf_synset_name: Optional[str] = None,
        leaf_offset: Optional[str] = None,
        return_object: Literal["synset_name", "offset"] = "synset_name",
    ):
        """
        Given a leaf synset name or offset, return the high-level class it belongs to.
        """
        if leaf_synset_name is None and leaf_offset is not None:
            leaf_synset_name = wn.synset_from_pos_and_offset(
                "n", int(leaf_offset[1:])
            ).name()  # type: ignore

        for high_level_class, leaves in self.high_level_leaves.items():
            if leaf_synset_name in leaves or leaf_offset in leaves:
                if return_object == "synset_name":
                    return high_level_class
                elif return_object == "offset":
                    return f"n{wn.synset(high_level_class).offset():08d}"

        return None

    def get_high_level_leaves(self):
        return self.high_level_leaves

    def get_leaf_image_counts(self, return_offset: bool = False):
        if return_offset:
            return {self._get_offset_from_synset(leaf): count for leaf, count in self.leaf_image_counts.items()}
        return self.leaf_image_counts
    
    def print_detailed_distribution(self):
        for high_level_class, num_images in self.high_level_classes.items():
            print(f"\nHigh-level class: {high_level_class} (Total images: {num_images})")
            
            # Print leaf distribution
            print("\nLeaf distribution:")
            for leaf in self.high_level_leaves[high_level_class]:
                image_count = self.leaf_image_counts[leaf]
                print(f"  {leaf}: {image_count} images")
            
            high_level_synset = wn.synset(high_level_class)
            high_level_offset = f"n{high_level_synset.offset():08d}"
            
            # Calculate subgraph distribution
            depth_nodes = defaultdict(list)
            node_image_counts = {}
            
            for node in nx.dfs_preorder_nodes(self.graph, high_level_offset):
                node_synset = wn.synset_from_pos_and_offset('n', int(node[1:]))
                depth = len(node_synset.hypernym_paths()[0]) - len(high_level_synset.hypernym_paths()[0])
                depth_nodes[depth].append(node_synset)
                
                # Calculate image count for this node's subgraph
                subgraph_leaves = [leaf for leaf in nx.dfs_preorder_nodes(self.graph, node) if leaf in self.imagenet_synsets]
                node_image_counts[node] = sum(self.leaf_image_counts.get(wn.synset_from_pos_and_offset('n', int(leaf[1:])).name(), 0) for leaf in subgraph_leaves)
            
            print("\nSubgraph distribution:")
            for depth, nodes in sorted(depth_nodes.items()):
                print(f"\nDepth {depth}:")
                for node in nodes:
                    node_offset = f"n{node.offset():08d}"
                    image_count = node_image_counts[node_offset]
                    print(f"  {node.name()} ({image_count} images)")
                depth_total = sum(node_image_counts[f"n{node.offset():08d}"] for node in nodes)
                print(f"  Total at depth {depth}: {depth_total} images")
                
    