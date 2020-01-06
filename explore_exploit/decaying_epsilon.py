import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, chance):
        self.rel_win_chance = chance
        self.num_pulls = 0
        self.mean = 0

    def pull(self):
        return np.random.randn() * self.rel_win_chance

    def update(self, x):
        self.num_pulls += 1
        self.mean = (1 - 1.0/self.num_pulls)*self.mean + 1.0/self.num_pulls*self.rel_win_chance


def model(chance1, chance2, chance3, num_runs):
    bandits = [Bandit(chance1), Bandit(chance2), Bandit(chance3)]
    best_choice = []
    act_choice = []

    for i in range(num_runs):
        epsilon = 1/(i+1)
        roll = np.random.rand()
        if roll < epsilon: # explore
            j = np.random.choice(3)

        else: # exploit
            j = np.argmax([b.mean for b in bandits]) 
            
        result = bandits[j].pull()
        bandits[j].update(result)
        
        best_choice.append(np.argmax([b.mean for b in bandits]) + 1)
        act_choice.append(j + 1)
    
    return best_choice, act_choice


if __name__ == '__main__':

    num_runs = 10000

    best_choice, act_choice = model(1, 2, 10, num_runs)

    # what was considered "best" at each pull
    plt.plot(best_choice)
    plt.xscale('log')
    plt.show()

    # what was actually chosen
    plt.scatter(range(num_runs), act_choice, marker='.')
    plt.show()


    