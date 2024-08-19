import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
# DAG instanciation
class DAG(nx.DiGraph):
    def __init__(self, file_path=None, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr) # DiGraph 
        self.root = '#' # will always use # as a root 
        self.file_path = file_path # store a pointer to the object file path
    
    def level_order_traversal(self, root):
        queue = deque([(root, 0)])
        levels = {}
        while queue:
            node, level = queue.popleft()
            levels.setdefault(level, []).append(node)
            for neighbor in self.neighbors(node):
                queue.append((neighbor, level + 1))
        return levels

    def visualize_dag_by_level(self, root):
        levels = self.level_order_traversal(root)

        pos = {}
        x_offset = 0.5
        y_offset = 1.5

        for level, nodes in levels.items():
            for i, node in enumerate(nodes):
                pos[node] = (x_offset + i, -level * y_offset)

        nx.draw(self, pos, with_labels=True)
        plt.show()

    def TSV(self):
        for node in self.nodes():
            # Identifyinself non-terminal nodes 
            if self.out_degree(node) > 0:
                ## Calculate the topological signature vector of each node
                subgraph = nx.ego_graph(self, node, radius=None)
                # Calculate delta : the root degree
                delta = self.out_degree(node)
                # Adjacency matrix {0,-1,1}
                adj_matrix = nx.to_numpy_array(subgraph, nodelist=subgraph.nodes, dtype=int)
                adj_matrix = adj_matrix - adj_matrix.T
                # Compute eigenvalues
                eigenvalues = np.linalg.eigvals(adj_matrix)
                sorted_ev = np.sort(np.abs(eigenvalues))[::-1]
                S = np.sum(sorted_ev[:delta]) # summing the largest kv eigenvalues corresponding to the subgraph rooted at the node v
                self.nodes[node]['label'] = str(S)

        dimension = max([self.out_degree(node) for node in self.nodes]) + 1
        # Access all nodes and their labels
        for node in self.nodes:
            if self.out_degree(node) > 0:
                #print(f"Node: {node}, Label: {float(G.nodes[node]['label'])}")
                S = float(self.nodes[node]['label'])
                S_child = [float(self.nodes[child]['label']) for child in self.neighbors(node) if self.out_degree(child) > 0]
                S_child.sort(reverse=True)
                if S_child is None :
                    S_child = []
                padding = (dimension - 1) - len(S_child)
                S_child_ordered_padded = S_child + padding*[0]
                self.nodes[node]['TSV'] = [S]+S_child_ordered_padded
                #print(G.nodes[node]['TSV'])
  
    def is_acyclic(self):
        """Checks if a graph is acyclic.

        Args:
            G: The input graph.

        Returns:
            True if the graph is acyclic, False otherwise.
        """

        try:
            nx.find_cycle(G)
            return False  # Cycle found, not acyclic
        except nx.NetworkXNoCycle:
            return True  # No cycle found, acyclic
    
    
#assert(is_acyclic(G), "The provided graph is not DAG")






if __name__ == '__main__':
    #TSV(G)
    G = DAG()

    # More complex DAG structure
    G.add_edges_from([
        ('A', 'B'), ('A', 'C'),
        ('B', 'D'), ('B', 'E'),
        ('C', 'F'), ('C', 'G'),
        ('D', 'H'), ('E', 'I'),
        ('F', 'J'), ('G', 'K'),
        ('H', 'L'), ('I', 'M'),
        ('J', 'N'), ('K', 'O')
    ])
    G.visualize_dag_by_level("A")