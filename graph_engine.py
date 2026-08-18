"""
graph_engine.py
The "Graph Engine" half of the dual-engine system.

Responsibilities:
  1. Load MetaQA's knowledge graph (a set of subject-relation-object
     triples) into an in-memory graph structure.
  2. Given a query, find relevant facts two ways:
       a) Graph traversal — find entities mentioned in the query,
          walk outward from them along edges (this is what makes
          multi-hop questions answerable, and is the core reason
          we're using a graph instead of plain text retrieval).
       b) Semantic similarity — embed the query and every triple,
          and rank by cosine similarity. This is the fallback for
          when entity matching in (a) fails, or when the phrasing
          doesn't map cleanly onto a graph edge.
  3. Combine both into one hybrid retriever, since we decided a
     hybrid (not graph-only, not text-only) is the strongest design.

--------------------------------------------------------------------
SETUP — MetaQA isn't on pip/HuggingFace as a clean one-line download.
Get it manually:
  1. Clone/download from https://github.com/yuyuz/MetaQA
  2. You need: kb.txt  (the knowledge graph triples)
  3. Place it at:  data/metaqa/kb.txt
  4. kb.txt format is pipe-separated, one triple per line, e.g.:
       Kismet|directed_by|Andrew Marton
       Kismet|written_by|John Balderston
--------------------------------------------------------------------
"""

import os
from collections import deque

import networkx as nx
from sentence_transformers import SentenceTransformer, util

from config import DEVICE


# ---------------------------------------------------------------------
# Step 1: Load raw triples from MetaQA's kb.txt
# ---------------------------------------------------------------------
def load_triples(kb_path: str) -> list[tuple[str, str, str]]:
    """
    Reads MetaQA's kb.txt and returns a list of (subject, relation, object)
    triples. Raises a clear error if the file is missing, instead of a
    confusing downstream crash.
    """
    if not os.path.exists(kb_path):
        raise FileNotFoundError(
            f"Couldn't find {kb_path}. Download MetaQA's kb.txt from "
            f"https://github.com/yuyuz/MetaQA and place it at this path."
        )

    triples = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue  # skip malformed lines rather than crashing
            subj, rel, obj = parts
            triples.append((subj.strip(), rel.strip(), obj.strip()))
    return triples


# ---------------------------------------------------------------------
# Step 2: Build a NetworkX graph from those triples
# ---------------------------------------------------------------------
def build_graph(triples: list[tuple[str, str, str]]) -> nx.MultiDiGraph:
    """
    Builds a directed multigraph (multiple edges between the same two
    nodes are allowed, since e.g. a movie can have multiple writers).

    We add BOTH the forward edge (subject -> object) and a reverse
    edge (object -> subject) labeled with an "_inverse" relation.
    This matters because MetaQA questions are phrased in both
    directions — e.g. "who directed X" (forward) vs "what did
    director Y direct" (reverse) — and a directed-only graph would
    miss half of these during traversal.
    """
    graph = nx.MultiDiGraph()
    for subj, rel, obj in triples:
        graph.add_edge(subj, obj, relation=rel)
        graph.add_edge(obj, subj, relation=f"{rel}_inverse")
    return graph


# ---------------------------------------------------------------------
# Step 3a: Entity linking — find graph nodes mentioned in a query
# ---------------------------------------------------------------------
def find_entities_in_query(query: str, graph: nx.MultiDiGraph) -> list[str]:
    """
    Naive but effective-enough-for-a-first-version entity linker:
    checks which node names appear as a substring of the query
    (case-insensitive). Sorted by length, longest first, so "The
    Dark Knight" matches before a shorter partial match like "Dark".

    This is the crudest part of the pipeline and a legitimate thing
    to improve later (e.g. proper NER) — flagging that honestly
    rather than pretending this is production-grade.
    """
    query_lower = query.lower()
    matches = [
        node for node in graph.nodes
        if isinstance(node, str) and node.lower() in query_lower
    ]
    matches.sort(key=len, reverse=True)
    return matches


# ---------------------------------------------------------------------
# Step 3b: Graph traversal — walk outward from seed entities
# ---------------------------------------------------------------------
def traverse_graph(
    graph: nx.MultiDiGraph,
    seed_entities: list[str],
    max_hops: int = 2,
) -> list[tuple[str, str, str]]:
    """
    Breadth-first traversal starting from each seed entity, up to
    max_hops edges away. Returns the triples encountered along the
    way. This is what makes multi-hop questions answerable — e.g.
    "capital of the country that signed a deal with Japan" needs
    2 hops, not 1.
    """
    visited_edges = set()
    result_triples = []

    for seed in seed_entities:
        if seed not in graph:
            continue
        frontier = deque([(seed, 0)])
        visited_nodes = {seed}

        while frontier:
            node, depth = frontier.popleft()
            if depth >= max_hops:
                continue
            for neighbor in graph.successors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                for _, data in edge_data.items():
                    relation = data["relation"]
                    edge_key = (node, relation, neighbor)
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        result_triples.append(edge_key)
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    frontier.append((neighbor, depth + 1))

    return result_triples


# ---------------------------------------------------------------------
# Step 4: Semantic similarity fallback
# ---------------------------------------------------------------------
class SemanticIndex:
    """
    Embeds every triple as a short natural-language sentence and
    supports similarity search against a query. This is the fallback
    for when entity linking misses (e.g. a misspelling, an alias,
    or phrasing that doesn't literally contain a node's name).
    """

    def __init__(self, triples: list[tuple[str, str, str]], model_name: str = "all-MiniLM-L6-v2"):
        self.triples = triples
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.sentences = [self._triple_to_sentence(t) for t in triples]
        self.embeddings = self.model.encode(
            self.sentences, convert_to_tensor=True, show_progress_bar=True
        )

    @staticmethod
    def _triple_to_sentence(triple: tuple[str, str, str]) -> str:
        subj, rel, obj = triple
        readable_rel = rel.replace("_", " ")
        return f"{subj} {readable_rel} {obj}"

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, str, str]]:
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)[0]  # remove this during testing , for parallel computing.
        return [self.triples[hit["corpus_id"]] for hit in hits]


# ---------------------------------------------------------------------
# Step 5: The combined hybrid retriever
# ---------------------------------------------------------------------
class HybridGraphRetriever:
    """
    Combines graph traversal + semantic similarity. Graph traversal
    is tried first (it's cheap and precise when entity linking
    succeeds); semantic search fills in when traversal comes up
    empty, or is added alongside as extra candidates.

    Returns results tagged with their source ("graph" or "semantic")
    so that evaluate.py can later run ablations — e.g. "how much
    worse is graph-only, or semantic-only, than the hybrid?" — which
    is exactly the kind of experiment a reviewer will want to see.
    """

    def __init__(self, kb_path: str, max_hops: int = 2, semantic_top_k: int = 5):
        self.triples = load_triples(kb_path)
        self.graph = build_graph(self.triples)
        self.semantic_index = SemanticIndex(self.triples)
        self.max_hops = max_hops
        self.semantic_top_k = semantic_top_k

    def retrieve(self, query: str) -> list[dict]:
        results = []

        seed_entities = find_entities_in_query(query, self.graph)
        graph_triples = traverse_graph(self.graph, seed_entities, self.max_hops)
        for t in graph_triples:
            results.append({"triple": t, "source": "graph"})

        semantic_triples = self.semantic_index.search(query, self.semantic_top_k)
        existing = {r["triple"] for r in results}
        for t in semantic_triples:
            if t not in existing:
                results.append({"triple": t, "source": "semantic"})

        return results

    def retrieve_as_facts(self, query: str) -> list[str]:
        """Convenience method: returns plain-text fact strings, ready
        to be injected into the text engine's context or KV cache."""
        return [
            SemanticIndex._triple_to_sentence(r["triple"])
            for r in self.retrieve(query)
        ]


# ---------------------------------------------------------------------
# Quick manual test — run this file directly to sanity-check things
# ---------------------------------------------------------------------
if __name__ == "__main__":
    KB_PATH = "data/metaqa/kb.txt"

    retriever = HybridGraphRetriever(KB_PATH, max_hops=2)

    test_query = "who directed the movie written by John Balderston"
    facts = retriever.retrieve(test_query)

    print(f"Query: {test_query}")
    print(f"Found {len(facts)} facts:")
    for f in facts[:10]:
        print(f"  [{f['source']}] {f['triple']}")