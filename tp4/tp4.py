from graph import Graph
import random
from tqdm import tqdm

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

def inicializar_grafos():
    grafo = Graph()
    disjoint = DisjointSet()

    with open('web-Google.txt', 'r') as file:
        for l in file:
            if "# FromNodeId	ToNodeId" in l:
                break
        for l in tqdm(file, total=5105039, initial=0):
            if not l:
                break
            edge = tuple(int(v.replace("\n", "").replace("\t", "")) for v in l.split("\t"))
            for v in edge:
                if not grafo.vertex_exists(v):
                    grafo.add_vertex(str(v))
                    disjoint.add(v)
            grafo.add_edge(str(edge[0]), str(edge[1]))
            disjoint.union(edge[0], edge[1])
    return grafo, disjoint

def punto_1(disjoint):
    # 1) ¿Cuál es el tamaño de la componente conexa más grande? ¿Cuántas componentes conexas hay? 
    print("************************** Punto 1 **************************")
    components = disjoint.get_components()
    max_component = max(components.values(), key=len)
    print("Tamaño de la componente conexa más grande:", len(max_component))
    print("Número de componentes conexas:", len(components))
    
def punto_2(grafo: Graph):
    # 2) Calcular el camino mínimo de todos con todos. ¿En cuanto tiempo lo puede hacer? ¿Qué orden tiene el algoritmo? En caso de no alcanzarle el tiempo, estime cuanto tiempo le llevaría.
    print("************************** Punto 2 **************************")
    n = 100
    tiempo = grafo.tiempo_estimado_bfs(n)
    print(f"Estimación de tiempo para encontrar todos los caminos más cortos: {estimar_tiempo(tiempo)}")

def punto_3(grafo: Graph):
    # 3) En un grafo un triángulo es una conexión entre 3 vértices A, B y C donde: A está conectado con B, B está conectado con C, C está conectado con A ¿Cuántos triángulos tiene el grafo?
    print("************************** Punto 3 **************************")
    print("Cantidad de triángulos en el grafo:", grafo.count_directed_triangles())

def punto_4(grafo: Graph):
    # 4) Utilice el punto 2 para calcular el diámetro del grafo.
    print("************************** Punto 4 **************************")
    n=100
    print("Diámetro del grafo:", grafo.diametroEstimado(n))

def punto_5(grafo: Graph):
    # 5) Google inventó un algoritmo llamado PageRank que le permitía saber qué páginas eran más confiables según que tanto eran referenciadas. PageRank consiste en hacer muchos random walks a lo largo del grafo y contar cuántas veces aparece cada vértice. Los vértices que más aparecen son los de mayor PageRank. Calcule el PageRank de los vértices del grafo.
    print("************************** Punto 5 **************************")
    print("PageRank de los vértices del grafo:")
    n = 10
    for vertex, rank in grafo.pageRank(n).items():
        print(f"Vértice {vertex}: {rank}")

def punto_6(grafo: Graph):
    # 6) La circunferencia del grafo es el largo del ciclo más largo. ¿Cuál es la circunferencia del grafo?
    print("************************** Punto 6 **************************")
    print("Circunferencia del grafo:", grafo.circunferencia())

def punto_extra_1(graph: Graph):
    # 1) Programe una función genérica que extendiendo la definición del triángulo calcule la cantidad de polígonos de K lados. Haga un gráfico para mostrar la cantidad de polígonos por cantidad de lados, estimando aquellos que no pueda calcular. (+2 puntos)
    print("************************** Punto Extra 1 **************************")
    print("Cantidad de polígonos de K lados:")
    k = 4
    print(f"Cantidad de polígonos de {k} lados: {grafo.count_directed_polygons(4)}")

def punto_extra_2(graph: Graph):
    # 2) Calcule el coeficiente de clustering del grafo (+1 punto)
    print("************************** Punto Extra 2 **************************")
    print("Coeficiente de clustering del grafo:", grafo.coeficiente_clustering())

def punto_extra_3(graph: Graph):
    # 3) Utilizando el punto 2, ¿cuál es el vértice con más betweenness centrality? (+2 puntos)
    print("************************** Punto Extra 3 **************************")
    vertice, betweenness = grafo.estimate_betweenness_centrality(sample_size=100)
    print("Vértice con más betweenness centrality:", vertice, "con un valor de:", betweenness)

def estimar_tiempo(tiempo):
        return f"{tiempo/3600:.2f} horas"

if __name__ == "__main__":
    grafo , disjoint = inicializar_grafos()

    punto_1(disjoint)
    punto_2(grafo)
    punto_3(grafo)
    punto_4(grafo)
    punto_5(grafo)
    punto_6(grafo)
    punto_extra_1(grafo)
    punto_extra_2(grafo)
    punto_extra_3(grafo)

    

