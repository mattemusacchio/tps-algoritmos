from typing import Optional, Any, List, Dict, Tuple
from collections import deque
import random
import numpy as np
import time
from tqdm import tqdm

class Graph:
    """
    Graph class
    """
    def __init__(self):
        self._graph = {}

    def add_vertex(self, vertex: str, data: Optional[Any]=None) -> None:
        """
        Adds a vertex to the graph
        :param vertex: the vertex name
        :param data: data associated with the vertex
        """
        if vertex not in self._graph:
            self._graph[vertex] = {'data': data, 'neighbors': {}}

    def add_edge(self, vertex1: str, vertex2: str, data: Optional[Any]=None) -> None:
        """
        Adds an edge to the graph
        :param vertex1: vertex1 key
        :param vertex2: vertex2 key
        :param data: the data associated with the vertex
        """
        if not vertex1 in self._graph or not vertex2 in self._graph:
            raise ValueError("The vertexes do not exist")
        self._graph[vertex1]['neighbors'][vertex2] = data

    def get_neighbors(self, vertex) -> List[str]:
        """
        Get the list of vertex neighbors
        :param vertex: the vertex to query
        :return: the list of neighbor vertexes
        """
        if vertex in self._graph:
            return list(self._graph[vertex]['neighbors'].keys())
        else:
            return []

    def get_vertex_data(self, vertex: str) -> Optional[Any]:
        """
        Gets  vertex associated data
        :param vertex: the vertex name
        :return: the vertex data
        """
        if self.vertex_exists(vertex):
            return self._graph[vertex]['data']
        else:
            return None

    def get_edge_data(self, vertex1: str, vertex2: str) -> Optional[Any]:
        """
        Gets the vertexes edge data
        :param vertex1: the vertex1 name
        :param vertex2: the vertex2 name
        :return: vertexes edge data
        """
        if self.edge_exists(vertex1, vertex2):
            return self._graph[vertex1]['neighbors'][vertex2]
        raise ValueError("The edge does not exist")

    def print_graph(self) -> None:
        """
        Prints the graph
        """
        for vertex, data in self._graph.items():
            print("Vertex:", vertex)
            print("Data:", data['data'])
            print("Neighbors:", data['neighbors'])
            print("")

    def vertex_exists(self, vertex: str) -> bool:
        """
        If contains a vertex
        :param vertex: the vertex name
        :return: boolean
        """
        return vertex in self._graph

    def edge_exists(self, vertex1: str, vertex2: str) -> bool:
        """
        If contains an edge
        :param vertex1: the vertex1 name
        :param vertex2: the vertex2 name
        :return: boolean
        """
        return vertex1 in self._graph and vertex2 in self._graph[vertex1]['neighbors']
    
    def get_edges_count(self) -> int:
        """
        Gets the number of edges
        :return: the number of edges
        """
        count = 0
        for vertex in self._graph:
            count += len(self._graph[vertex]['neighbors'])
        return count
    
    def get_vertices(self) -> List[str]:
        """
        Gets the list of vertices
        :return: the list of vertices
        """
        return list(self._graph.keys())
    
    def dfs(self, start: str) -> Dict[str, Tuple[str, int]]:
        """
        Depth-first search
        :param start: the start vertex
        :return: the distances from the start vertex to all other vertices
        """
        distances = {start: (None, 0)}
        stack = [start]

        while stack:
            current_vertex = stack.pop()

            for neighbor in self.get_neighbors(current_vertex):
                if neighbor not in distances:
                    distances[neighbor] = (current_vertex, distances[current_vertex][1] + 1)
                    stack.append(neighbor)

        return distances
    
    def bfs(self, start: str) -> Dict[str, Tuple[str, int]]:
        """
        Breadth-first search
        :param start: the start vertex
        :return: the distances from the start vertex to all other vertices
        """
        distances = {start: (None, 0)}
        queue = deque([start])

        while queue:
            current_vertex = queue.popleft()

            for neighbor in self.get_neighbors(current_vertex):
                if neighbor not in distances:
                    distances[neighbor] = (current_vertex, distances[current_vertex][1] + 1)
                    queue.append(neighbor)

        return distances
    
    def tiempo_estimado_bfs(self,n):
        """
        estimar tiempo que tarda en encontrar todos los caminos mas cortos
        """
        np.random.seed(0)
        random_samples = np.random.choice(self.get_vertices(), n, replace=False)
        
        tiempo_promedio = []
        for vertex in random_samples:
            start = time.time()
            self.bfs(vertex)
            end = time.time()
            tiempo_promedio.append(end-start)
        return np.mean(tiempo_promedio) * len(self.get_vertices())
    
    
    def count_directed_triangles(self):
        triangle_count = 0

        # Iterar sobre cada vértice
        for vertex in self._graph:
            neighbors = self.get_neighbors(vertex)
            
            # Iterar sobre cada vecino
            for neighbor in neighbors:
                second_neighbors = self.get_neighbors(neighbor)
                
                # Verificar si hay una arista de second_neighbor de vuelta al vértice original
                for second_neighbor in second_neighbors:
                    if self.edge_exists(second_neighbor, vertex):
                        triangle_count += 1

        return triangle_count//3
        
    def diametroEstimado(self,m) -> List[str]:
        """
        estimar el diametro del grafo tomando m veces el camino mas corto de entre 100 caminos distintos entre 2 vertices. De esos m caminos, devolver el mas largo.
        """
        vertices = self.get_vertices()
        n = len(vertices)
        shortest_lengths = []
        for _ in range(m):
            v1,v2 = vertices[random.randint(0,n-1)],vertices[random.randint(0,n-1)]
            if v1 != v2:
                shortest_lengths.append(len(self.shortest_path_from_n_random_paths(v1,v2,100)))
            print(max(shortest_lengths))
        return max(shortest_lengths)
    
    def shortest_path_from_n_random_paths(self, start: str, end: str,n) -> List[str]:
        """
        Tomo el camino mas corto entre n caminos que encuentro entre 2 vertices
        :param start: the start vertex
        :param end: the end vertex
        :return: the shortest of the five distinct paths as a list of vertices
        """
        if start == end:
            return [start]

        queue = deque([(start, [start])])  # Queue of tuples (current_vertex, path_to_vertex)
        visited = {start: 0}  # Tracks the minimum distance of visiting each node
        paths = []  # Stores the found paths

        while queue and len(paths) < n:
            current_vertex, path = queue.popleft()
            current_distance = len(path)

            for neighbor in self.get_neighbors(current_vertex):
                if neighbor == end:
                    paths.append(path + [end])
                    if len(paths) >= n:
                        break
                if neighbor not in visited or visited[neighbor] > current_distance:
                    visited[neighbor] = current_distance
                    if neighbor not in path:  # Detect cycles
                        queue.append((neighbor, path + [neighbor]))
        # Return the shortest path among the found shortest paths
        return min(paths, key=len) if paths else []
    
    def pageRank(self, m, damping = 0.85, tolerance = 1e-8, max_iter = 100):
        """
        implementar PageRank y devolver los primeros m vertices con mayor PageRank
        """

        vertices = self.get_vertices()
        n = len(vertices)
        page_rank = {vertex: 1/n for vertex in vertices}
        out_degree = {vertex: len(self.get_neighbors(vertex)) for vertex in vertices}
        transpuesto = self.transponerGrafo()
        transpuesto_vecinos = {vertex: transpuesto.get_neighbors(vertex) for vertex in vertices}
        
        for _ in tqdm(range(max_iter)):
            new_page_rank = {}
            for vertex in vertices:
                new_page_rank[vertex] = (1-damping)/n + damping*sum([page_rank[neighbor]/out_degree[neighbor] for neighbor in transpuesto_vecinos[vertex]])
            if sum([abs(new_page_rank[vertex] - page_rank[vertex]) for vertex in vertices]) < tolerance:
                break
            page_rank = new_page_rank

        return dict(sorted(page_rank.items(), key=lambda x: x[1], reverse=True)[:m])

    def transponerGrafo(self):
        """
        Transponer el grafo
        """
        transposed = Graph()
        for vertex in self.get_vertices():
            transposed.add_vertex(vertex)
        for vertex in self.get_vertices():
            for neighbor in self.get_neighbors(vertex):
                transposed.add_edge(neighbor, vertex)
        return transposed

    def _dfs_buscar_ciclo(self, vertice, visitados, stack_path, longitud_ciclo):
        stack = [(vertice, [vertice])]
        while stack:
            current_node, path = stack.pop()
            if current_node not in visitados:
                visitados.add(current_node)
                stack_path[current_node] = path
                for neighbor in self.get_neighbors(current_node):
                    if neighbor not in visitados:
                        stack.append((neighbor, path + [neighbor]))
                    elif neighbor in stack_path and neighbor in path:
                        cycle = path
                        if len(cycle) - cycle.index(neighbor) == longitud_ciclo:
                            return cycle[cycle.index(neighbor):]
            elif current_node in stack_path and current_node in path:
                if len(path) - path.index(current_node) == longitud_ciclo:
                    return path[path.index(current_node):]
                else:
                    return []
        return []

    def buscar_ciclos(self, longitud_ciclo, max_time):
        start_time = time.time()
        bar = tqdm(total=max_time, desc=f"Buscando ciclos de longitud {longitud_ciclo}",leave=False)
        
        for node in self._graph:
            if time.time() - start_time > max_time:
                break
            visitados = set()
            path = {}
            cycle = self._dfs_buscar_ciclo(node, visitados, path, longitud_ciclo)
            if cycle:
                bar.close()
                print(f"Ciclo encontrado, longitud: {longitud_ciclo}")
                return cycle
            bar.update(time.time() - start_time - bar.n)
        
        bar.close()
        return []

    def circunferencia(self, max_time=5):
        vertices = list(self._graph.keys())
        min, max = 2, len(vertices)
        circunferencia = 0

        while min <= max:
            mitad = (min + max) // 2
            cycle = self.buscar_ciclos(mitad, max_time)
            if cycle:
                circunferencia = mitad
                min = mitad + 1
            else:
                max = mitad - 1

        return circunferencia
    
    def dfs_polygons(self, origen, n, k, longitud, visitados, contador):
        if longitud == k:
            if n == origen:
                contador[0] += 1
            return
        for vecino in self.get_neighbors(n):
            if (vecino, longitud+1) not in visitados:
                visitados.add((vecino, longitud+1))
                self.dfs_polygons(origen, vecino, k, longitud+1, visitados, contador)
                visitados.remove((vecino, longitud+1))

    def count_directed_polygons(self, k):
        contador = [0]
        for vertex in tqdm(self.get_vertices(), desc=f"Buscando poligonos de longitud {k}"):
            visitados = set()
            visitados.add((vertex, 0))
            self.dfs_polygons(vertex, vertex, k, 0, visitados, contador)
        return contador[0] // k
    
    def coeficiente_clustering(self):
        coefficients = []
        
        for node in self.get_vertices():
            neighbors = self.get_neighbors(node)
            if len(neighbors) < 2:
                coefficients.append(0.0)
                continue

            link_count = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in self.get_neighbors(neighbors[i]):
                        link_count += 1
                    if neighbors[i] in self.get_neighbors(neighbors[j]):
                        link_count += 1
            
            possible_links = len(neighbors) * (len(neighbors) - 1)
            coefficients.append(link_count / possible_links)

        return sum(coefficients) / len(coefficients)
    
    def bfs_shortest_path(self, start, goal):
        explored = set()
        queue = deque([[start]])
        
        if start == goal:
            return [start]
        
        while queue:
            path = queue.popleft()
            node = path[-1]
            
            if node not in explored:
                neighbors = self.get_neighbors(node)
                
                for neighbor in neighbors:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
                    
                    if neighbor == goal:
                        return new_path
                
                explored.add(node)
        
        return None

    def estimate_betweenness_centrality(self, sample_size):
        nodes = self.get_vertices()
        betweenness_estimate = {node: 0 for node in nodes}
        
        for _ in tqdm(range(sample_size), desc="Calculando betweenness centrality"):
            # Selecciona aleatoriamente un par de nodos
            source, target = random.sample(nodes, 2)
            shortest_path = self.bfs_shortest_path(source, target)
            if shortest_path:
                for node in shortest_path[1:-1]:
                    betweenness_estimate[node] += 1
        
        # Encuentra el nodo con la máxima centralidad de intermediación estimada
        max_betweenness_node = max(betweenness_estimate, key=betweenness_estimate.get)
        return max_betweenness_node, betweenness_estimate[max_betweenness_node]