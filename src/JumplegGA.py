import numpy as np
import torch
import torch.nn as nn
import copy
import joblib


class JumplegGA():
    def __init__(self, obs_dim, action_dim, pop_number=10, layer_dim=128, w_restore = None):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pop_number = pop_number
        self.device = 'cpu'

        self.model = nn.Sequential(
            nn.Linear(obs_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, action_dim),
            nn.Tanh()
        )
        if w_restore is not None:
            self.pop_weights = joblib.load(w_restore)
            print("Population restored")
        else:
            self.pop_weights = self.create_population(pop_number)
        self.dna_size = self.pop_weights[0].shape[0]
        self.parent_n = 2

    def vectorize_weights(self, model):
        weights_vector = []

        for weights in model.state_dict().values():
            weights = weights.detach().numpy()
            weights_vector.extend(np.reshape(weights, newshape=(weights.size)))

        return np.array(weights_vector)

    def restore_weights(self, model, weights_vector):
        weights_dict = model.state_dict()

        index = 0
        for key in weights_dict:
            w = weights_dict[key].detach().numpy()
            w_shape = w.shape
            w_size = w.size

            layer_w_vector = weights_vector[index:index+w_size]
            layer_w_matrix = np.reshape(layer_w_vector, newshape=(w_shape))
            weights_dict[key] = torch.from_numpy(layer_w_matrix)

            index += w_size

        return weights_dict

    def predict(self, i, X):
        model = self.model
        model_w_dict = self.restore_weights(model, self.pop_weights[i])
        model.load_state_dict(model_w_dict)
        prediction = model(torch.Tensor(X))

        return prediction.cpu().data.numpy()

    def create_population(self, pop_number):
        initial_w_vector = self.vectorize_weights(self.model)
        pop_weights = []
        pop_weights.append(initial_w_vector)

        for i in range(pop_number-1):
            net_w = copy.deepcopy(initial_w_vector)
            net_w = np.random.permutation(net_w)
            # net_w = copy.deepcopy(initial_w_vector)
            # net_w = net_w + \
            #     np.random.uniform(-1, 1, size=initial_w_vector.size)
            # met_w = np.clip(net_w,-1,1)

            pop_weights.append(net_w)

        return pop_weights

    def steady_state_selection(self, score, num_parents):
        fitness_sorted = sorted(range(len(score)), key=lambda k: score[k])
        fitness_sorted.reverse()
        parents = np.empty((num_parents, self.dna_size))
        for i in range(num_parents):
            parents[i] = self.pop_weights[fitness_sorted[i]]

        return parents, fitness_sorted[:num_parents]

    def single_point_crossover(self, p1, p2):

        cross_pt = np.random.randint(1, len(p1)-1)

        c1 = np.zeros(len(p1))
        c2 = np.zeros(len(p2))

        c1[0:cross_pt] = p1[0:cross_pt]
        c1[cross_pt:] = p2[cross_pt:]

        c2[0:cross_pt] = p2[0:cross_pt]
        c2[cross_pt:] = p1[cross_pt:]

        return c1, c2

    def mutation_randomly(self, child, mutation_percentage=0.1):

        child_ = np.copy(child)
        random_indx = np.random.choice(range(self.dna_size), np.ceil(
            self.dna_size*mutation_percentage).astype(int))

        for indx in random_indx:
            child_[indx] = np.random.uniform(-1, 1)

        return child_

    def new_generation(self, score):
        parents, _ = self.steady_state_selection(score, self.parent_n)
        p1, p2 = parents

        joblib.dump(self.pop_weights, '/home/riccardo/pop_weights.joblib')
        torch.save(self.restore_weights(self.model, p1), 'best.pt')

        for i in range(0, self.pop_number, 2):
            c1, c2 = self.single_point_crossover(p1, p2)
            c1 = self.mutation_randomly(c1)
            c2 = self.mutation_randomly(c2)
            self.pop_weights[i] = c1.copy()
            self.pop_weights[i+1] = c2.copy()
