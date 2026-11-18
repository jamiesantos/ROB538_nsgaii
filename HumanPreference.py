# Simulates a human with fixed preference weightings
import numpy as np
import torch

class HumanSimulation:
    def __init__(self, num_objs, preference_weights=None, noise=0.0, seed=None):
        self.rng = np.random.default_rng(seed)
        self.num_objectives = num_objs
        self.noise = noise
        
        if preference_weights is None:
            self.weights = self.rng.dirichlet(np.ones(num_objs))  # Random weights that sum to 1
        else:
            self.weights = np.array(preference_weights, dtype=float)
            self.weights /= np.sum(self.weights)  # Normalize to sum to 1
    
    def compare(self, fitness_a, fitness_b):
        # Compare two fitness vectors based on the human's preference weights
        util_a = np.dot(self.weights, fitness_a)
        util_b = np.dot(self.weights, fitness_b)
        
        # Simulate slightly confused person
        if self.rng.random() < self.noise:
            return self.rng.integers(0, 2)
        
        return 0 if util_a > util_b else 1  # Return 0 if a is preferred, 1 if b is preferred
    
    def generate_comparisons(self, fitnesses, num_pairs):
        # Randomly select pairs of solutions and simulate human comparisons
        comparisons = []
        
        for pair in range(num_pairs):
            a, b = self.rng.choice(len(fitnesses), size=2, replace=False)
            winner = self.compare(fitnesses[a], fitnesses[b])
            comparisons.append((a, b, winner))
        return comparisons
    
class HumanPreference:
    def __init__(self, num_objs, learning_rate=0.01):
        self.num_objectives = num_objs
        self.learning_rate = learning_rate
        self.weights = torch.nn.Parameter(torch.zeros(num_objs))  # Start with equal weights
        self.optimizer = torch.optim.Adam([self.weights], lr=self.learning_rate)
    
    def utility(self, fitnesses):
        # Predict utility for each policy
        weights = torch.softmax(self.weights, dim=0)  # Ensure weights sum to 1
        return fitnesses @ weights
    
    def train(self, fitnesses, comparisons, epochs=200):
        # Learn objective preferences from pairwise comparisons
        # comparisons: (index_a, index_b, winner)
        # winner = 0 means a is preffered, 1 means b is preferred
        fitnesses = torch.tensor(fitnesses, dtype=torch.float32)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            a = torch.tensor([c[0] for c in comparisons])
            b = torch.tensor([c[1] for c in comparisons])
            winner = torch.tensor([c[2] for c in comparisons])
            
            util_a = self.utility(fitnesses[a])
            util_b = self.utility(fitnesses[b])
            
            # probability a is preferred over b
            logits = util_a - util_b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, 1 - winner.float())
            loss.backward()
            self.optimizer.step()
            
        return loss.item()
    
    def predict(self, fitnesses):
        # Predict utility for each policy
        fitnesses = torch.tensor(fitnesses, dtype=torch.float32)
        with torch.no_grad():
            return self.utility(fitnesses).numpy()
    
    def rank(self, fitnesses):
        # Rank policies based on predicted utility
        utilities = self.predict(fitnesses)
        return np.argsort(-utilities)  # Descending order