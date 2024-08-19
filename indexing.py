from DAG import DAG, TSV
import numpy as np
from sklearn.neighbors import KDTree

class Point():
    def __init__(self, vector: list, model_object : DAG, node : str) -> None:
        self.vector = vector 
        self.dag = model_object
        self.node = node
    def __repr__(self):
        return f"Point(tsv={self.vector}, graph={self.dag.file_path}, node={self.node})"

def find_nearest_neighbors(point, database, k=5):
    """Finds the k nearest neighbors of a point in the database.

    Args:
        point: The query point.
        database: A list of Point objects.
        k: The number of nearest neighbors to find.

    Returns:
        A list of the k nearest neighbors.
    """

    # Extract TSV vectors
    X = np.array([p.vector for p in database])

    # Create a KD-Tree
    tree = KDTree(X)

    # Find nearest neighbors
    distances, indices = tree.query([point.vector], k=k)

    return [database[i] for i in indices[0]]


def W(Q, q, M, m, algo='basic', w=0.5, normalise=False):
    "q : query node, m : graph node"
    if algo=='basic':
        vote_weight = np.linalg.norm(np.array(Q.nodes[q]['TSV'])) / (1+ np.linalg.norm(np.array((M.nodes[m]['TSV']) - np.array(Q.nodes[q]['TSV']))))
    elif algo=="Occam_Razor":
        denum_query = np.sum(np.array([(1 + np.linalg.norm(np.array(M.nodes[m]['TSV']) - np.array(Q.nodes[query_node]['TSV']))) for query_node in Q.nodes if Q.out_degree(query_node) > 0]))
        denum_model = np.sum(np.array([(1 + np.linalg.norm(np.array(M.nodes[model_node]['TSV']) - np.array(Q.nodes[q]['TSV']))) for model_node in M.nodes if M.out_degree(model_node) > 0]))

        query_weight = (1-w) * np.linalg.norm(np.array(Q.nodes[q]['TSV'])) / denum_query
        model_weight = w * np.linalg.norm(np.array(M.nodes[m]['TSV'])) / denum_model

        vote_weight = query_weight + model_weight

        normalisation_factor = np.sum(
        np.fromiter((np.linalg.norm(np.array(Q.nodes[node]['TSV'])) for node in Q.nodes if Q.out_degree(node) > 0), dtype=float)
        ) + np.sum(
        np.fromiter((np.linalg.norm(np.array(M.nodes[node]['TSV'])) for node in M.nodes if M.out_degree(node) > 0), dtype=float)
        )

        if normalise:
            vote_weight=vote_weight/normalisation_factor

    return vote_weight


def get_top_n_keys(my_dict, n=5):
    """Returns the top n keys with the highest values from a dictionary.

    Args:
        my_dict: The input dictionary.
        n: The number of top keys to return.

    Returns:
        A list of the top n keys.
    """

    return sorted(my_dict, key=my_dict.get, reverse=True)[:n]

if __name__ == "__main__":
    Q = DAG(file_path="Query Graph")
    M1 = DAG(file_path="M1")
    M2 = DAG(file_path="M2")
    M3 = DAG(file_path="M3")
    

    # More complex DAG structure
    Q.add_edges_from([
        ('q1', 'q2'), ('q1', 'q3'),
        ('q2', 'q4'), ('q2', 'q5'),
        ('q3', 'q7'), ('q3', 'q9'),
        ('q5', 'q6'), 
        ('q7', 'q8')
    ])
    M1.add_edges_from([
        ('m1', 'm2'),
        ('m2', 'm3'), ('m2', 'm4'),
        ('m4', 'm5')
    ])
    M2.add_edges_from([
        ('m1', 'm2'), ('m1', 'm3'),
        ('m4', 'm3'), ('m4', 'm5'),
        ('m2', 'm6'), ('m2', 'm7'),
        ('m7', 'm8'), 
        ('m3', 'm9'), ('m3', 'm10'),
        ('m10', 'm11'),
        ('m5', 'm12'), ('m5', 'm14'),
        ('m12', 'm13')
    ])
    M3.add_edges_from([
        ('m1', 'm2'), ('m1', 'm3'),
        ('m2', 'm4'), ('m2', 'm5'),
        ('m3', 'm7'), ('m3', 'm9'),
        ('m5', 'm6'), 
        ('m7', 'm8')
    ])

    Q.TSV()
    models=[M1,M2,M3] # list of graph instances
    for model in models :
        model.TSV()
    index_database = [Point(G.nodes[node]['TSV'], G, node) for G in models for node in G.nodes if G.out_degree(node) > 0]
    accumulator = {dag.file_path : 0 for dag in models}
    for node in Q.nodes:
        if Q.out_degree(node) > 0:
            index = Point(Q.nodes[node]['TSV'], Q, node)
            # search in database for k-NN using point.vector
            result = find_nearest_neighbors(index, index_database, k=2)
            for point in result:
                # accumulate vote for the point.model 
                accumulator[point.dag.file_path] +=W(Q, node, point.dag, point.node,algo='basic', normalise=False)
    print(accumulator)
    print(get_top_n_keys(accumulator, 2))

