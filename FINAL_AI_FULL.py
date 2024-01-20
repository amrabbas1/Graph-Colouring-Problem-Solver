import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import random
import time

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super(MatplotlibCanvas, self).__init__(self.figure)
        self.setParent(parent)

    def draw_graph(self, graph, solution):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        node_colors = [solution['colors'][node - 1] for node in graph.nodes]

        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, ax=ax, with_labels=True, node_color=node_colors)

        self.draw_idle()


class GraphDrawer(QMainWindow):
    def __init__(self):
        super(GraphDrawer, self).__init__()

        self.graph = nx.Graph()
        self.solution_genetic = None
        self.solution_backtracking = None

        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Graph Drawer')

        self.num_nodes_label = QLabel("Number of Nodes:", self)
        self.num_nodes_input = QLineEdit(self)
        self.num_nodes_input.setText("2")

        self.num_edges_label = QLabel("Number of Edges:", self)
        self.num_edges_input = QLineEdit(self)
        self.num_edges_input.setText("1")

        self.edges_label = QLabel("Edges (comma-separated pairs):", self)
        self.edges_input = QLineEdit(self)
        self.edges_input.setText("1,2")

        self.population_size_label = QLabel("Population Size:", self)
        self.population_size_input = QLineEdit(self)
        self.population_size_input.setText("50")

        self.mutation_rate_label = QLabel("Mutation Rate:", self)
        self.mutation_rate_input = QLineEdit(self)
        self.mutation_rate_input.setText("0.1")

        self.draw_genetic_button = QPushButton("Draw Genetic Graph", self)
        self.draw_genetic_button.clicked.connect(self.draw_genetic_graph)

        self.draw_backtracking_button = QPushButton("Draw Backtracking Graph", self)
        self.draw_backtracking_button.clicked.connect(self.draw_backtracking_graph)

        self.chromatic_label = QLabel("Chromatic Number: ", self)
        self.time_label = QLabel("Computational Time:  * seconds", self)

        layout = QVBoxLayout()

        layout.addWidget(self.num_nodes_label)
        layout.addWidget(self.num_nodes_input)
        layout.addWidget(self.num_edges_label)
        layout.addWidget(self.num_edges_input)
        layout.addWidget(self.edges_label)
        layout.addWidget(self.edges_input)
        layout.addWidget(self.population_size_label)
        layout.addWidget(self.population_size_input)
        layout.addWidget(self.mutation_rate_label)
        layout.addWidget(self.mutation_rate_input)
        layout.addWidget(self.draw_genetic_button)
        layout.addWidget(self.draw_backtracking_button)
        layout.addWidget(self.chromatic_label)
        layout.addWidget(self.time_label)

        self.canvas = MatplotlibCanvas(self, width=5, height=4)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

    def initialize_population(self, population_size, num_nodes, max_colors, color_names):
        population = []
        for _ in range(population_size):
            num_colors = 1
            individual = {
                'colors': [random.choice(color_names) for _ in range(num_nodes)],
                'num_colors': num_colors
            }
            population.append(individual)
        return population

    def fitness(self, individual, adj_matrix):
        conflicts = 0
        for i in range(len(individual['colors'])):
            for j in range(i + 1, len(individual['colors'])):
                if adj_matrix[i][j] and individual['colors'][i] == individual['colors'][j]:
                    conflicts += 1
        return conflicts + len(set(individual['colors'])) 

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1['colors']) - 1)
        child_colors = parent1['colors'][:crossover_point] + parent2['colors'][crossover_point:]
        num_colors = max(parent1['num_colors'], parent2['num_colors'])
        child_num_colors = len(set(child_colors))
        return {'colors': child_colors, 'num_colors': child_num_colors}

    def mutate(self, individual, mutation_rate, max_colors, color_names):
        mutated_colors = []
        for color in individual['colors']:
            if random.uniform(0, 1) < mutation_rate:
                mutated_colors.append(random.choice(color_names))
            else:
                mutated_colors.append(color)
        num_colors = max(individual['num_colors'], len(set(mutated_colors)))
        return {'colors': mutated_colors, 'num_colors': num_colors}

    def genetic_algorithm(self, adj_matrix, population_size, num_generations, mutation_rate, max_colors, color_names):
        num_nodes = len(adj_matrix)
        population = self.initialize_population(population_size, num_nodes, max_colors, color_names)

        start_time = time.time()  # Start measuring execution time

        for generation in range(num_generations):
            fitness_values = [self.fitness(individual, adj_matrix) for individual in population]

            
            parents = []
            for _ in range(population_size):
                tournament = random.sample(range(population_size), 3)
                winner = min(tournament, key=lambda i: fitness_values[i])
                parents.append(population[winner])

            
            next_generation = []
            while len(next_generation) < population_size:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                child = self.mutate(self.crossover(parent1, parent2), mutation_rate, max_colors, color_names)
                next_generation.append(child)

            population = next_generation

        end_time = time.time()  # Stop measuring execution time

        best_solution = min(population, key=lambda x: self.fitness(x, adj_matrix))

        chromatic_number = len(set(best_solution['colors'])) 

        execution_time = end_time - start_time  

        return best_solution, chromatic_number, execution_time

    def draw_genetic_graph(self):
        try:
            num_nodes = int(self.num_nodes_input.text())
            num_edges = int(self.num_edges_input.text())
            edges_text = self.edges_input.text()
            edges = [tuple(map(int, edge.split(','))) for edge in edges_text.split(';')]

            if num_edges != len(edges):
                raise ValueError("Number of edges does not match the input.")

            adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]

            for edge in edges:
                u, v = edge
                adj_matrix[u - 1][v - 1] = 1
                adj_matrix[v - 1][u - 1] = 1

            population_size = int(self.population_size_input.text())
            num_generations = 100
            mutation_rate = float(self.mutation_rate_input.text())
            max_colors = 5

            nodes_color = [
                "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "cyan", "magenta",
                "teal", "lavender", "maroon", "olive", "navy", "aquamarine", "coral", "gold", "silver"
            ]

            solution_genetic, chromatic_number, execution_time = self.genetic_algorithm(adj_matrix, population_size, num_generations, mutation_rate, max_colors, nodes_color)

            self.solution_genetic = solution_genetic
            self.graph = nx.Graph()
            self.graph.add_nodes_from(range(1, num_nodes + 1))
            self.graph.add_edges_from(edges)

            self.canvas.draw_graph(self.graph, solution_genetic)

            self.chromatic_label.setText(f"Chromatic Number: {chromatic_number}")
            self.time_label.setText(f"Computational Time: {execution_time:.5f} seconds")

        except ValueError as e:
            print(f"Error: {e}")

    def ok(self, v, colors, adj, c):
        for u in adj[v]:
            if colors[u] == c:
                return False
        return True

    def graph_coloring(self, v, colors, adj, c):
        if v == len(colors):
            return True

        for i in range(1, c + 1):
            if self.ok(v, colors, adj, i):
                colors[v] = i
                if self.graph_coloring(v + 1, colors, adj, c):
                    return True
                colors[v] = 0  # Backtrack
        return False

    def backtracking_algorithm(self, num_nodes, edges):
        adj = [[] for i in range(num_nodes + 1)]

        for edge in edges:
            u, v = edge
            adj[u].append(v)
            adj[v].append(u)

        colors = [0] * (num_nodes + 1)
        c = 1
        while not self.graph_coloring(1, colors, adj, c):
            c += 1

        return colors

    def draw_backtracking_graph(self):
        try:
            num_nodes = int(self.num_nodes_input.text())
            edges_text = self.edges_input.text()
            edges = [tuple(map(int, edge.split(','))) for edge in edges_text.split(';')]

            adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]

            for edge in edges:
                u, v = edge
                adj_matrix[u - 1][v - 1] = 1
                adj_matrix[v - 1][u - 1] = 1

            start_time = time.time()  # Start measuring execution time

            solution_backtracking = self.backtracking_algorithm(num_nodes, edges)

            end_time = time.time()  # Stop measuring execution time

            self.solution_backtracking = {'colors': solution_backtracking[1:], 'num_colors': max(solution_backtracking)}

            self.graph = nx.Graph()
            self.graph.add_nodes_from(range(1, num_nodes + 1))
            self.graph.add_edges_from(edges)

            self.canvas.draw_graph(self.graph, self.solution_backtracking)

            chromatic_number = max(solution_backtracking)
            execution_time = end_time - start_time  # Total execution time

            self.chromatic_label.setText(f"Chromatic Number: {chromatic_number}")
            self.time_label.setText(f"Computational Time: {execution_time:.5f} seconds")

        except ValueError as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GraphDrawer()
    window.show()
    sys.exit(app.exec_())
