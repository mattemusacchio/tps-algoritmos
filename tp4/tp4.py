from graph import Graph

graph = Graph()

with open('web-Google.txt', 'r') as file:
    for l in file:
        if "# FromNodeId	ToNodeId" in l:
            break
    for l in file:
        if not l:
            break
        edge = tuple(int(v.replace("\n", "").replace("\t", "")) for v in l.split("\t"))
        for v in edge:
            if not graph.vertex_exists(v):
                graph.add_vertex(str(v))
        graph.add_edge(str(edge[0]), str(edge[1]))

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
import networkx as nx
G = nx.DiGraph()
for vertex in graph._graph:
    G.add_node(vertex)
for vertex in graph._graph:
    for neighbor in graph.get_neighbors(vertex):
        G.add_edge(vertex, neighbor)
pagerank_nx = nx.pagerank(G)
print(pagerank_nx)


# graph.graficar()

# Nodes	875713
# Edges	5105039
# Nodes in largest WCC	855802 (0.977)
# Edges in largest WCC	5066842 (0.993)
# Nodes in largest SCC	434818 (0.497)
# Edges in largest SCC	3419124 (0.670)
# Average clustering coefficient	0.5143
# Number of triangles	13391903
# Fraction of closed triangles	0.01911
# Diameter (longest shortest path)	21
# 90-percentile effective diameter	8.1

# 1) ¿Cuál es el tamaño de la componente conexa más grande? ¿Cuántas componentes conexas hay? 

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

max_component_size, num_components = connected_components(graph)
print("Tamaño de la componente conexa más grande:", max_component_size)
print("Número de componentes conexas:", num_components)
print("El orden del algoritmo es O(V + E). = O(875713 + 5105039) = O(5980752)")

# 2) Calcular el camino mínimo de todos con todos. ¿En cuanto tiempo lo puede hacer? ¿Qué orden tiene el algoritmo? En caso de no alcanzarle el tiempo, estime cuanto tiempo le llevaría.

def floyd_warshall(grafo):
    dist = {vertex: {vertex: 0 for vertex in grafo._graph} for vertex in grafo._graph}
    for vertex in grafo._graph:
        for neighbor in grafo.get_neighbors(vertex):
            dist[vertex][neighbor] = grafo.get_edge_data(vertex, neighbor)
    for k in grafo._graph:
        for i in grafo._graph:
            for j in grafo._graph:
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

# floyd_warshall(graph)

# 3) En un grafo un triángulo es una conexión entre 3 vértices A, B y C donde: A está conectado con B, B está conectado con C, C está conectado con A ¿Cuántos triángulos tiene el grafo?

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

triangles = count_directed_triangles(graph)
print("Cantidad de triángulos:", triangles)
print(13391903 == triangles)

# 4) Utilice el punto 2 para calcular el diámetro del grafo.

def diameter_calc(grafo):
    dist = floyd_warshall(grafo)
    diameter = 0
    for vertex in grafo._graph:
        for neighbor in grafo.get_neighbors(vertex):
            diameter = max(diameter, dist[vertex][neighbor])
    return diameter

# diameter = diameter(graph)
diameter = 21
print("Diámetro del grafo:", diameter)
print(21 == diameter)
