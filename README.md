# indexing-and-matching
Implementation of two very important and interesting papers : "Indexing Hierarchical Structures Using Graph Spectra" and "Indexing and Matching for View-Based 3-D Object Recognition Using Shock Graphs"

# Tasks 
- [ ] Implement a DAG class, with adequate attributes and methods (visualisation too : _print__).
- [ ] Calculate topological signature vector TSV of a subgraph.
- [ ] Calculate recusively TSV for all non-terminal nodes of DAG and store them in each node.
- [ ] Create an index database (of TSV space) with pointers to model object and model node.
- [ ] Using NN search (or other algorithm), and given a query implement a voting mechanism using index database. (vote weights and complex voting strategy for later).
- [ ] Return most voted models using vote bins/scores. 


accesing and changing nodes labels : 
import networkx as nx

G = nx.Graph()
G.add_nodes_from(['A', 'B', 'C'])

# Accessing node labels (default is node name)
print(G.nodes(data=True))  # Output: [('A', {}), ('B', {}), ('C', {})]

# Adding labels to nodes
G.nodes['A']['label'] = 'Node A'
G.nodes['B']['label'] = 'Node B'
G.nodes['C']['label'] = 'Node C'

print(G.nodes(data=True))  # Output: [('A', {'label': 'Node A'}), ('B', {'label': 'Node B'}), ('C', {'label': 'Node C'})]

# Accessing specific node's label
print(G.nodes['A']['label'])  # Output: Node A

level traversal of the graph : 
def level_order_traversal(graph, root):
    queue = deque([(root, 0)])
    levels = {}
    while queue:
        node, level = queue.popleft()
        levels.setdefault(level, []).append(node)
        for neighbor in graph.neighbors(node):
            queue.append((neighbor, level + 1))
    return levels
# Draw the graph
level_order_traversal(G, "A")

subgraph rooted at a node : 
subgraph = nx.ego_graph(G, 'A', radius=None)

children of a node: 
for neighbor in graph.neighbors(node):

DAG instance : 
G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D')])

import ast

my_list = [1, 2, 3, "hello", 4.5]

# Convert list to string
string_representation = str(my_list)
print(string_representation)  # Output: [1, 2, 3, 'hello', 4.5]

# Convert string back to list
original_list = ast.literal_eval(string_representation)
print(original_list)  # Output: [1, 2, 3, 'hello', 4.5]