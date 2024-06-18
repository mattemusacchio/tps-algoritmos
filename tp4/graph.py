from typing import Optional, Any, List, Dict, Tuple
from collections import deque
import random

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
    
    def shortest_paths(self, start: str, end: str,n) -> List[str]:
        """
        Gets the shortest path among the five distinct paths between two vertices.
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

        if len(paths) < n:
            print("Found fewer than n distinct paths")

        # Return the shortest path among the found shortest paths
        return min(paths, key=len) if paths else []
    
    def find_longest_cycle_length(self, sample_size: int = None) -> int:
        longest_cycle_length = 0
        vertices = list(self.get_vertices())

        if sample_size is not None and sample_size < len(vertices):
            sample_vertices = random.sample(vertices, sample_size)
        else:
            sample_vertices = vertices

        for start_vertex in sample_vertices:
            visited = set()
            queue = deque([(start_vertex, 0)]) 

            while queue:
                current, distance = queue.popleft()

                if current in visited:
                    longest_cycle_length = max(longest_cycle_length, distance)
                    continue

                visited.add(current)

                for neighbor in self.get_neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))

        if longest_cycle_length == 0:
            return float('inf')  
        else:
            return longest_cycle_length