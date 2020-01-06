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


def model(chance1, chance2, chance3, num_runs, epsilon):
    bandits = [Bandit(chance1), Bandit(chance2), Bandit(chance3)]
    best_choice = []
    act_choice = []

    for i in range(num_runs):
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
    eps1 = 0.3
    eps2 = 0.1
    eps3 = 0.05
    eps4 = 0.01

    best_choice1, act_choice1 = model(1, 2, 10, num_runs, eps1)
    best_choice2, act_choice2 = model(1, 2, 10, num_runs, eps2)
    best_choice3, act_choice3 = model(1, 2, 10, num_runs, eps3)
    best_choice4, act_choice4 = model(1, 2, 10, num_runs, eps4)

    # what was considered "best" at each pull
    plt.plot(best_choice1, label='eps 0.3')
    plt.plot(best_choice2, label='eps 0.1')
    plt.plot(best_choice3, label='eps 0.05')
    plt.plot(best_choice4, label='eps 0.01')

    plt.xscale('log')
    plt.legend()
    plt.show()

    # what was actually chosen
    plt.scatter(range(num_runs), act_choice1, s=1, label='eps 0.3', marker='.')
    plt.legend()
    plt.show()

    plt.scatter(range(num_runs), act_choice2, s=1, label='eps 0.1', marker='.')
    plt.legend()
    plt.show()

    plt.scatter(range(num_runs), act_choice3, s=1, label='eps 0.05', marker='.')
    plt.legend()
    plt.show()

    plt.scatter(range(num_runs), act_choice4, s=1, label='eps 0.01', marker='.')
    plt.legend()
    plt.show()
    