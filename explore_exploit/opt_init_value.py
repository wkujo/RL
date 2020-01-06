import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, chance, init_value):
        self.rel_win_chance = chance
        self.num_pulls = 0
        self.mean = init_value

    def pull(self):
        return np.random.randn() * self.rel_win_chance

    def update(self, x):
        self.num_pulls += 1
        self.mean = (1 - 1.0/self.num_pulls)*self.mean + 1.0/self.num_pulls*self.rel_win_chance


def model(chance1, chance2, chance3, num_runs, init_value):
    bandits = [Bandit(chance1, init_value), Bandit(chance2, init_value), Bandit(chance3, init_value)]
    best_choice = []
    act_choice = []

    for i in range(num_runs):
        j = np.argmax([b.mean for b in bandits]) 
            
        result = bandits[j].pull()
        bandits[j].update(result)
        
        best_choice.append(np.argmax([b.mean for b in bandits]) + 1)
        act_choice.append(j + 1)
    
    return best_choice, act_choice


if __name__ == '__main__':

    num_runs = 10000
    init_value = 100

    best_choice, act_choice = model(3, 2, 1, num_runs, init_value)

    # what was considered "best" at each pull
    plt.plot(best_choice)
    plt.xscale('log')
    plt.show()

    # non-log plot of "best" at each pull
    plt.plot(best_choice)
    plt.show()

    # what was actually chosen
    plt.scatter(range(num_runs), act_choice, s=1, marker='.')
    plt.show()

    