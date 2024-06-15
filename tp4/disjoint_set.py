class Disjoint:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0 for i in range(n)]

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1

    def add(self, x):
        self.parent.append(x)
        self.rank.append(0)

    def get_components(self):
        components = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root in components:
                components[root].append(i)
            else:
                components[root] = [i]
        return list(components.values())

    def same_set(self, x, y):
        return self.find(x) == self.find(y)
