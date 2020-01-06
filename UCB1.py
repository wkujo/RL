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


def ucb(bandit, total_pulls):
    if bandit.num_pulls == 0:
        return float('inf')
    else:
        return bandit.mean + np.sqrt(2 * np.log(total_pulls) / bandit.num_pulls)


def model(chance1, chance2, chance3, num_runs):
    bandits = [Bandit(chance1), Bandit(chance2), Bandit(chance3)]
    act_choice = []

    for i in range(num_runs):
        j = np.argmax([ucb(b, i+1) for b in bandits]) 
            
        result = bandits[j].pull()
        bandits[j].update(result)

        act_choice.append(j + 1)
    
    return act_choice


if __name__ == '__main__':

    num_runs = 10000

    act_choice = model(1, 2, 3, num_runs)

    # what was actually chosen
    plt.plot(act_choice)
    plt.xscale('log')
    plt.show()

    plt.plot(act_choice)
    plt.show()