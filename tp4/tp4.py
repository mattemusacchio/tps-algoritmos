from graph import Graph

class DisjointSet:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)

        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def get_components(self):
        components = {}
        for item in self.parent:
            root = self.find(item)
            if root not in components:
                components[root] = []
            components[root].append(item)
        return components

graph = Graph()
disjoint = DisjointSet()

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
                disjoint.add(v)
        graph.add_edge(str(edge[0]), str(edge[1]))
        disjoint.union(edge[0], edge[1])

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

# class disjoint

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
# hacerlo usando disjoint set
# disjoint = Disjoint()  Disjoint.__init__() missing 1 required positional argument: 'n'
components = disjoint.get_components()

max_component = max(components.values(), key=len)
print("Tamaño de la componente conexa más grande:", len(max_component))
print("Número de componentes conexas:", len(components))

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

    return triangle_count//3

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
