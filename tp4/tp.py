import pickle
from graph import Graph
from tqdm import tqdm

"""
Lista de productos

Ejemplo de producto:
{'id': 2,
 'title': 'Candlemas: Feast of Flames',
 'group': 'Book',
 'categories': ['Books[283155]->Subjects[1000]->Religion & Spirituality[22]->Earth-Based Religions[12472]->Wicca[12484]',
  'Books[283155]->Subjects[1000]->Religion & Spirituality[22]->Earth-Based Religions[12472]->Witchcraft[12486]'],
 'reviewers': [('A11NCO6YTE4BTJ', 5),
  ('A9CQ3PLRNIR83', 4),
  ('A13SG9ACZ9O5IM', 5),
  ('A1BDAI6VEYMAZA', 5),
  ('A2P6KAWXJ16234', 4),
  ('AMACWC3M7PQFR', 4),
  ('A3GO7UV9XX14D8', 4),
  ('A1GIL64QK68WKL', 5),
  ('AEOBOF2ONQJWV', 5),
  ('A3IGHTES8ME05L', 5),
  ('A1CP26N8RHYVVO', 1),
  ('ANEIANH0WAT9D', 5)]}
"""
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

# archivo graph.py
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
        
        for vertex, data in self._graph.items():
            print("Vertex:", vertex)
            print("Data:", data['data'])
            print("Neighbors:", data['neighbors'])
            print("")

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
        
        Plot the graph, only first 5 vertexes
        
        import matplotlib.pyplot as plt
        import networkx as nx
        G = nx.Graph()
        for vertex, data in list(self._graph.items())[:5]:
            G.add_node(vertex, data=data['data'])
            for neighbor in data['neighbors']:
                G.add_edge(vertex, neighbor, weight=data['neighbors'][neighbor])
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()
        
"""


# grafo.print_graph()

# usar matplotlib para visualizar el grafo
grafo.plot_graph()

# Desarrolle el tp aquí

"""
Web links
Contamos con un grafo que contiene links de una página a otra. Cada vértice es una página
y cada arista un link entre ellas.
Deberá responder las siguientes preguntas usando código.
Enunciados del TP
1) ¿Cuál es el tamaño de la componente conexa más grande? ¿Cuántas componentes
conexas hay?
2) Calcular el camino mínimo de todos con todos. ¿En cuanto tiempo lo puede hacer?
¿Qué orden tiene el algoritmo? En caso de no alcanzarle el tiempo, estime cuanto
tiempo le llevaría.
3) En un grafo un triángulo es una conexión entre 3 vértices A, B y C donde:
A está conectado con B
B está conectado con C
C está conectado con A
¿Cuántos triángulos tiene el grafo?
4) Utilice el punto 2 para calcular el diámetro del grafo.
5) Google inventó un algoritmo llamado PageRank que le permitía saber qué páginas
eran más confiables según que tanto eran referenciadas. PageRank consiste en
hacer muchos random walks a lo largo del grafo y contar cuántas veces aparece
cada vértice. Los vértices que más aparecen son los de mayor PageRank. Calcule el
PageRank de los vértices del grafo.
6) La circunferencia del grafo es el largo del ciclo más largo. ¿Cuál es la circunferencia
del grafo?
Puntos extra
1) Programe una función genérica que extendiendo la definición del triángulo calcule la
cantidad de polígonos de K lados. Haga un gráfico para mostrar la cantidad de
polígonos por cantidad de lados, estimando aquellos que no pueda calcular. (+2
puntos)
2) Calcule el coeficiente de clustering del grafo (+1 punto)
3) Utilizando el punto 2, ¿cuál es el vértice con más betweenness centrality? (+2
puntos)
"""

# 1) ¿Cuál es el tamaño de la componente conexa más grande? ¿Cuántas componentes conexas hay?

# Tamaño de la componente conexa más grande y número de componentes conexas

# Componentes conexas: Una componente conexa de un grafo es un subgrafo donde cualquier par de vértices está conectado por un camino.
# Tamaño de la componente conexa más grande: Es el número de vértices en la componente conexa que tiene la mayor cantidad de vértices.
# Número de componentes conexas: Es la cantidad total de componentes conexas en el grafo.

# Para calcular el tamaño de la componente conexa más grande y el número de componentes conexas, se puede utilizar el algoritmo de búsqueda en profundidad (DFS) o el algoritmo de búsqueda en amplitud (BFS) para recorrer el grafo y encontrar las componentes conexas.

def dfs(grafo, start_vertex, visited, component):
    stack = [start_vertex]
    while stack:
        vertex = stack.pop()
        if not visited[vertex]:
            visited[vertex] = True
            component.append(vertex)
            for neighbor in grafo.get_neighbors(vertex):
                if not visited[neighbor]:
                    stack.append(neighbor)

def connected_components(grafo):
    visited = {vertex: False for vertex in grafo._graph}
    num_components = 0
    max_component_size = 0
    for vertex in grafo._graph:
        if not visited[vertex]:
            component = []
            dfs(grafo, vertex, visited, component)
            num_components += 1
            max_component_size = max(max_component_size, len(component))
    return max_component_size, num_components

max_component_size, num_components = connected_components(grafo)
print("Tamaño de la componente conexa más grande:", max_component_size)
print("Número de componentes conexas:", num_components)
print("El orden del algoritmo es O(V + E).")

# 2) Calcular el camino mínimo de todos con todos. ¿En cuanto tiempo lo puede hacer? ¿Qué orden tiene el algoritmo? En caso de no alcanzarle el tiempo, estime cuanto tiempo le llevaría.
# 4) Utilice el punto 2 para calcular el diámetro del grafo.
# utilizar floyd-warshall

# print cantidad de aristas
print("Cantidad de aristas:", sum([len(grafo._graph[vertex]['neighbors']) for vertex in grafo._graph]))

"""
Utilizar Floyd-Warshall (Programación dinámica - TABULATION)
Floyd()
{
Inicializo Distancia[i,j]=inf, Distancia[i,j]=Eij y Distancia[i,i]=0
Inicializo Anterior[i,j]=vacío excepto para las aristas conocidas (i)
for (k=1;k++;k<|V|)
for (i=1;i++;i<|V|)
for (j=1;j++;j<|V|)
if (Distancia[i,k]+Distancia[k,j]<Distancia[i,j])
{
Distancia[i,j]=Distancia[i,k]+Distancia[k,j]
Anterior[i,j]=Anterior[k,j]
}
}
"""

def floyd_warshall(grafo):
    vertices = list(grafo._graph.keys())
    n = len(vertices)
    dist = [[float('inf') for _ in range(n)] for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for vertex in vertices:
        for neighbor in grafo.get_neighbors(vertex):
            dist[vertices.index(vertex)][vertices.index(neighbor)] = grafo.get_edge_data(vertex, neighbor)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist, max(max(row) for row in dist)

print("Usamos floyd warshall para calcular el camino mínimo de todos con todos")
print("Orden del algoritmo: O(2086819^3) = 9.087 x 10^18 operaciones, si cada operacion tarda 1ns, tardaria 9.092 x 10^9 segundos = aproximadamente 287 años")

# 3) En un grafo un triángulo es una conexión entre 3 vértices A, B y C donde: A está conectado con B, B está conectado con C, C está conectado con A ¿Cuántos triángulos tiene el grafo?

# Triángulos en un grafo

def count_directed_triangles(graph):
    triangle_count = 0

    # Iterar sobre cada vértice
    for vertex in graph._graph:
        neighbors = graph.get_neighbors(vertex)
        
        # Iterar sobre cada vecino
        for neighbor in neighbors:
            second_neighbors = graph.get_neighbors(neighbor)
            
            # Verificar si hay una arista de second_neighbor de vuelta al vértice original
            for second_neighbor in second_neighbors:
                if graph.edge_exists(second_neighbor, vertex):
                    triangle_count += 1

    return triangle_count

# Contar el número de triángulos en el grafo dirigido
# triangle_count = count_directed_triangles(grafo)
# print(f"Número de triángulos en el grafo dirigido: {triangle_count}")
print("El orden del algoritmo es O(V^2) = O(2086819^2) = O(4.4 x 10^12), osea que si cada operacion tarda 1ns, tardaria 4.4 x 10^3 segundos = aproximadamente 1 hora")

# 5) Google inventó un algoritmo llamado PageRank que le permitía saber qué páginas eran más confiables según que tanto eran referenciadas. PageRank consiste en hacer muchos random walks a lo largo del grafo y contar cuántas veces aparece cada vértice. Los vértices que más aparecen son los de mayor PageRank. Calcule el PageRank de los vértices del grafo.

# PageRank en un grafo

def pagerank(graph, num_iterations=100, damping_factor=0.85):
    # Inicializar el PageRank de cada vértice
    pagerank = {vertex: 1 / len(graph._graph) for vertex in graph._graph}
    
    # Realizar múltiples iteraciones para calcular el PageRank
    for _ in range(num_iterations):
        new_pagerank = {}
        for vertex in graph._graph:
            new_pagerank[vertex] = (1 - damping_factor) / len(graph._graph)
            for neighbor in graph.get_neighbors(vertex):
                new_pagerank[vertex] += damping_factor * pagerank[neighbor] / len(graph.get_neighbors(neighbor))
        pagerank = new_pagerank

    return pagerank

# Calcular el PageRank de los vértices del grafo
# import time 
# start = time.time()

# pagerank_values = pagerank(grafo)
# end = time.time()
# print(f"Tiempo de ejecución: {end - start} segundos")

# print("PageRank de los vértices del grafo:")
# n = 0
# for vertex, value in pagerank_values.items():
#     n+=1
#     if n == 100:
#         break
#     print(f"Vertice: {vertex}, PageRank: {value}")

dictionario = { "1": 4.795746934712992e-07,
 "A2JW67OY8U6HHK": 2.755389902047087e-07,
 "A2VE83MZF98ITY": 9.276601691405719e-05,
 "2": 9.189960906117755e-07,
 "A11NCO6YTE4BTJ": 7.168007064021045e-07,
 "A9CQ3PLRNIR83": 1.4685020762355172e-06,
 "A13SG9ACZ9O5IM": 9.61644386876762e-07,
 "A1BDAI6VEYMAZA": 2.5101262763333185e-07,
 "A2P6KAWXJ16234": 8.16732726233611e-06,
 "AMACWC3M7PQFR": 8.832423478523623e-06,
 "A3GO7UV9XX14D8": 3.004181635304116e-06,
 "A1GIL64QK68WKL": 5.646723204848893e-07,
 "AEOBOF2ONQJWV": 3.7630756673239253e-06,
 "A3IGHTES8ME05L": 3.325872241614494e-05,
 "A1CP26N8RHYVVO": 1.27260761684936e-07,
 "ANEIANH0WAT9D": 1.0582294551340262e-07,
 "3": 1.533434380269683e-07,
 "A3IDGASRQAW8B2": 1.2168973180924862e-06,
 "4": 4.79198243834276e-07,
 "A2591BUPXCS705": 4.79198243834276e-07,
 "5": 7.187973657514141e-08,
 "6": 1.1304402248636702e-06,
 "ATVPDKIKX0DER": 0.009508233039767014,
 "AUEZ7NVOEHYRY": 1.3396878838368665e-06,
 "AJYG6ZJUQPZ9M": 1.533434380269683e-07,
 "A2ESGJTTLJWIAK": 4.615137119752064e-07,
 "A2CHULHAO3A9BY": 1.533434380269683e-07,
 "A3BNWP7ATVP045": 3.347941822181934e-07,
 "A2NJO6YE954DBH": 0.0003937104326244559,
 "A393PYR83LT7R8": 1.533434380269683e-07,
 "AVY10FEFM6OBC": 1.533434380269683e-07,
 "A2OSPW11FVXTU6": 4.587500319041961e-07,
 "A32Z5HQGTG5V49": 1.533434380269683e-07,
 "A2WSI8HOWHFDOT": 2.0554986443737634e-05,
 "A3BGC9MSXGM0WH": 1.533434380269683e-07,
 "A1IU7S4HCK1XK0": 5.137762274936297e-05,
 "A30JTDN020MAJB": 1.533434380269683e-07,
 "7": 8.950025533279758e-07,
 "A2O3PW57IFNUHV": 5.277937945211389e-06,
 "A3FNRL9QYZOBDH": 2.0765257232818625e-07,
 "A3SWAHVVHVPKWZ": 2.0765257232818625e-07,
 "8": 1.0526521630276252e-06,
 "A2F1X6YFCJZ1FH": 2.425655872361121e-07,
 "A1OZQCZAK21S6M": 1.2859448472668785e-06,
 "AL5D52NA8F67F": 4.143528525658084e-06,
 "AVFBIM1W41IXO": 1.4915405707365977e-07,
 "A3I6SOXDIE0M8R": 1.2618887087635934e-07,
 "A3559TE3F9RRNL": 1.8437722905623573e-07,
 "ASPUU0H77LFXG": 1.6214786531040956e-07,
 "A3L902U49A6X5K": 8.812951471638597e-07,
 "AL5OEDM8TPTKV": 4.446140117002767e-06,
 "A1R64WON03GTN4": 1.8437722905623573e-07,
 "A2WKESDGF2YC8S": 2.3586576943157396e-06,
 "A71P2O8OMF8GY": 7.581472362627034e-07,
 "AB8HLDYSDI5M7": 1.2618887087635934e-07,
 "A37FDCXZLI0MAC": 1.8424962300759562e-07,
 "AQE41QO3NEUMW": 1.3088146870179807e-05,
 "9": 7.187973657514141e-08,
 "10": 1.235576491027111e-06,
 "A2RI73IFW2GWU1": 1.3976615445166381e-07,
 "A1GE54WF2WUZ2X": 2.318977215698014e-07,
 "A36S399V1VC4DR": 1.3976615445166381e-07,
 "A280GY5UVUS2QH": 1.2832297654306993e-05,
 "A2YHZJIU4L4IOI": 9.755735963514884e-07,
 "A1MB83EO48TRSC": 1.612039706231972e-07,
 "11": 2.755389902047087e-07,
 "A2A1TNBFJNRADP": 5.470846617107984e-07,
 "12": 3.852758608691619e-06,
 "A2V9UBVMQFDV20": 1.0582294551340262e-07,
 "A3NXQLHXJZO5FB": 1.0582294551340262e-07,
 "A1RNV50D6DNE42": 1.0582294551340262e-07,
 "A3FVNS48MY8L32": 1.0582294551340262e-07,
 "A1WC6GEAVET9K8": 1.0582294551340262e-07,
 "A1PH1HGK2HFQ9G": 1.5747565476727835e-07,
 "A1DOK8OJ386KDS": 1.0582294551340262e-07,
 "A1E4FW0F5R7TOY": 1.0582294551340262e-07,
 "A14OJS0VWMOSWO": 0.0012504443807509315,
 "A3UDEP0MTKLLL2": 1.0582294551340262e-07,
 "A19UTUEBWKIZFT": 4.919117991998558e-07,
 "A35X1EMOVF90JG": 1.0582294551340262e-07,
 "13": 7.187973657514141e-08,
 "14": 7.187973657514141e-08,
 "15": 1.348603177896526e-06,
 "A2IGOA66Y6O8TQ": 1.2389541081296331e-07,
 "A2OIN4AUH84KNE": 1.2279454998253322e-07,
 "A2HN382JNT1CIU": 1.4004807479168261e-07,
 "A2FDJ79LDU4O18": 1.067744989227971e-06,
 "A39QMV9ZKRJXO5": 1.2279454998253322e-07,
 "AUUVMSTQ1TXDI": 5.7431350095320235e-06,
 "A2C5K0QTLL9UAT": 1.5296629126098765e-07,
 "A5XYF0Z3UH4HB": 1.0517878754762438e-06,
 "16": 5.88427549346816e-07,
 "A2C4FHNHVRR8JZ": 1.6023652045750752e-07,
 "A1XPCHKIFECW33": 5.316057978028181e-07,
 "A17VTMYNK7FHHG": 6.732540529837497e-07,
 "A331OF34UX4Y1K": 1.4977088283841305e-05,
 "A3KMI0ZF739BW5": 1.4435085567297718e-06,
 "A26BWRBPP4V2WF": 5.003113455396317e-06,
 "A3AAX9AO0CIFBI": 9.597014891400009e-07}

# ordenar por pagerank

sorted_pagerank = sorted(dictionario.items(), key=lambda x: x[1], reverse=True)
print("PageRank de los vértices del grafo:")
for vertex, value in sorted_pagerank:
    print(f"Vertice: {vertex}, PageRank: {value}")


# 6) La circunferencia del grafo es el largo del ciclo más largo. ¿Cuál es la circunferencia del grafo?

# Circunferencia de un grafo

def bfs(graph, start_vertex):
    visited = {vertex: False for vertex in graph._graph}
    distance = {vertex: float('inf') for vertex in graph._graph}
    distance[start_vertex] = 0
    queue = [start_vertex]
    while queue:
        vertex = queue.pop(0)
        if not visited[vertex]:
            visited[vertex] = True
            for neighbor in graph.get_neighbors(vertex):
                if not visited[neighbor]:
                    distance[neighbor] = distance[vertex] + 1
                    queue.append(neighbor)
    return max(distance.values())

def graph_circumference(graph):
    max_circumference = 0
    for vertex in graph._graph:
        max_circumference = max(max_circumference, bfs(graph, vertex))
    return max_circumference

# Calcular la circunferencia del grafo
# circumference = graph_circumference(grafo)
# print(f"Circunferencia del grafo: {circumference}")
print("El orden del algoritmo es O(V*(V + E)) = O(2086819(2086819 + 12614504)) = O(4.4 x 10^13), osea que si cada operacion tarda 1ns, tardaria 4.4 x 10^4 segundos = aproximadamente 12 horas")

# Puntos extra

# 1) Programe una función genérica que extendiendo la definición del triángulo calcule la cantidad de polígonos de K lados. Haga un gráfico para mostrar la cantidad de polígonos por cantidad de lados, estimando aquellos que no pueda calcular. (+2 puntos)

# Cantidad de polígonos de K lados en un grafo

def count_k_sided_polygons(graph, k):
    polygon_count = 0

    # Iterar sobre cada vértice
    for vertex in graph._graph:
        neighbors = graph.get_neighbors(vertex)
        
        # Verificar si hay un polígono de k lados
        for neighbor in neighbors:
            common_neighbors = set(graph.get_neighbors(vertex)).intersection(set(graph.get_neighbors(neighbor)))
            for common_neighbor in common_neighbors:
                if common_neighbor != vertex and common_neighbor != neighbor:
                    polygon_count += 1

    return polygon_count // (2 * k)

# # Graficar la cantidad de polígonos por cantidad de lados
# import matplotlib.pyplot as plt

# k_values = list(range(3, 11))
# polygon_counts = [count_k_sided_polygons(grafo, k) for k in k_values]

# plt.plot(k_values, polygon_counts, marker='o')
# plt.xlabel('Cantidad de lados')
# plt.ylabel('Cantidad de polígonos')
# plt.title('Cantidad de polígonos por cantidad de lados')
# plt.grid()
# plt.show()

# 2) Calcule el coeficiente de clustering del grafo (+1 punto)
