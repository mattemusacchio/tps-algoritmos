import networkx as nx
import random
from tqdm import tqdm
import pickle
from graph import Graph

"""
from typing import Optional, Any, List


class Graph:
    Graph class
    def __init__(self):
        self._graph = {}

    def add_vertex(self, vertex: str, data: Optional[Any]=None) -> None:

        Adds a vertex to the graph
        :param vertex: the vertex name
        :param data: data associated with the vertex

        if vertex not in self._graph:
            self._graph[vertex] = {'data': data, 'neighbors': {}}

    def add_edge(self, vertex1: str, vertex2: str, data: Optional[Any]=None) -> None:

        Adds an edge to the graph
        :param vertex1: vertex1 key
        :param vertex2: vertex2 key
        :param data: the data associated with the vertex

        if not vertex1 in self._graph or not vertex2 in self._graph:
            raise ValueError("The vertexes do not exist")
        self._graph[vertex1]['neighbors'][vertex2] = data

    def get_neighbors(self, vertex) -> List[str]:

        Get the list of vertex neighbors
        :param vertex: the vertex to query
        :return: the list of neighbor vertexes

        if vertex in self._graph:
            return list(self._graph[vertex]['neighbors'].keys())
        else:
            return []

    def get_vertex_data(self, vertex: str) -> Optional[Any]:

        Gets  vertex associated data
        :param vertex: the vertex name
        :return: the vertex data

        if self.vertex_exists(vertex):
            return self._graph[vertex]['data']
        else:
            return None

    def get_edge_data(self, vertex1: str, vertex2: str) -> Optional[Any]:

        Gets the vertexes edge data
        :param vertex1: the vertex1 name
        :param vertex2: the vertex2 name
        :return: vertexes edge data

        if self.edge_exists(vertex1, vertex2):
            return self._graph[vertex1]['neighbors'][vertex2]
        raise ValueError("The edge does not exist")

    def print_graph(self) -> None:

        Prints the graph

        n = 0
        for vertex, data in self._graph.items():
            n += 1
            print("Vertex:", vertex)
            print("Data:", data['data'])
            print("Neighbors:", data['neighbors'])
            print("")
            if n == 5:
                break
        

    def vertex_exists(self, vertex: str) -> bool:

        If contains a vertex
        :param vertex: the vertex name
        :return: boolean

        return vertex in self._graph

    def edge_exists(self, vertex1: str, vertex2: str) -> bool:

        If contains an edge
        :param vertex1: the vertex1 name
        :param vertex2: the vertex2 name
        :return: boolean

        return vertex1 in self._graph and vertex2 in self._graph[vertex1]['neighbors']
    
    def plot_graph(self):

        Plot the directed graph, only first 5 vertexes

        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()
        n = 0
        for vertex, data in self._graph.items():
            # in the neighbors data, the key is the vertex and the value is the edge data
            for neighbor, edge_data in data['neighbors'].items():
                G.add_edge(vertex, neighbor, weight=edge_data)
            n += 1
            if n == 5:
                break

        pos = nx.spring_layout(G)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

    def get_vertexes(self):

        Get the vertexes
        :return: the vertexes

        return list(self._graph.keys())
    
    def get_edges(self):

        Get the edges
        :return: the edges

        edges = []
        for vertex, data in self._graph.items():
            for neighbor, edge_data in data['neighbors'].items():
                edges.append((vertex, neighbor, edge_data))
        return edges
        
"""

# Cargar los productos y construir el grafo
with open('products.pickle', 'rb') as file:
    products = pickle.load(file)

grafo = Graph()

print("Loading")
for p in tqdm(products):
    grafo.add_vertex(str(p["id"]), data={'title': p['title'],
                                         'group': p['group'],
                                         'categories': p['categories']})
    for reviewer, score in p['reviewers']:
        if not grafo.vertex_exists(reviewer):
            grafo.add_vertex(reviewer)
        grafo.add_edge(reviewer, str(p["id"]), score)
        grafo.add_edge(str(p["id"]), reviewer, score)

# Convertir el grafo a un formato compatible con NetworkX
G = nx.Graph()
for vertex in grafo.get_vertexes():
    G.add_node(vertex, data=grafo.get_vertex_data(vertex))
for edge in grafo.get_edges():
    G.add_edge(edge[0], edge[1], weight=edge[2])


def find_long_cycle(graph, iterations=100):
    # """Finds a long cycle in the graph using random walks and local optimization."""
    # longest_cycle = []
    # longest_cycle_length = 0
    # for _ in tqdm(range(iterations)):
    #     print(f"Longest cycle length found so far: {longest_cycle_length}")
    #     # Elegir un nodo aleatorio
    #     node = random.choice(list(graph.nodes))
    #     cycle = [node]
    #     cycle_length = 0
    #     n = 0
    #     while True:
    #         n += 1
    #         if n % 1000:
    #             print(f"Cycle length: {cycle_length}")
    #         # Elegir un vecino aleatorio
    #         neighbors = list(graph.neighbors(node))
    #         if not neighbors:
    #             break
    #         neighbor = random.choice(neighbors)
    #         # Añadir el vecino al ciclo
    #         cycle.append(neighbor)
    #         cycle_length += graph[node][neighbor]['weight']
    #         # Si el ciclo es más largo que el ciclo más largo encontrado hasta ahora, almacenarlo
    #         if cycle_length > longest_cycle_length:
    #             longest_cycle = cycle.copy()
    #             longest_cycle_length = cycle_length
    #         # Si el vecino es el primer nodo del ciclo, terminar
    #         if neighbor == cycle[0]:
    #             break
    #         node = neighbor
    # return longest_cycle

    # Con el metodo de arriba, se hace infinito, posiblemente haya subciclos infinitos adentro del ciclo inicial por lo que buscar que el vecino sea el primer nodo no es una buena idea

    # Vamos a hacer un ciclo que no sea infinito
    longest_cycle = []
    longest_cycle_length = 0

    for _ in tqdm(range(iterations)):
        print(f"Longest cycle length found so far: {longest_cycle_length}")
        # Elegir un nodo aleatorio
        node = random.choice(list(graph.nodes))
        cycle = [node]
        cycle_length = 0
        n = 0
        while True:
            n += 1
            if n % 100000 == 0:
                print(f"Cycle length: {cycle_length}")
            # Elegir un vecino aleatorio
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                break
            neighbor = random.choice(neighbors)
            # Añadir el vecino al ciclo
            cycle.append(neighbor)
            cycle_length += graph[node][neighbor]['weight']
            # Si el ciclo es más largo que el ciclo más largo encontrado hasta ahora, almacenarlo
            if cycle_length > longest_cycle_length:
                longest_cycle = cycle.copy()
                longest_cycle_length = cycle_length
            # Si el vecino esta entre los recientes visitados, terminar
            if neighbor in cycle[:-1]:
                break
            

# Parámetros
iterations = 1000

# Buscar un ciclo largo
long_cycle = find_long_cycle(G, iterations)

# Calcular la longitud del ciclo
cycle_length = len(long_cycle) - 1  # Restamos 1 porque el último nodo es el mismo que el primero

print(f"Longitud estimada del ciclo más largo: {cycle_length}")
