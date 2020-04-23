import numpy as np
class TransportationProblem(object):
    def __init__(self, I,J):
        # The grid is I by J
        self.I = I
        self.J = J
        self.grid=np.ones((I,J))
    def set_avoids(self,items):
        #needs list of lists to avoid ex. ((1,4),(3,4),(4,2))
        for item in items:
            i=item[0]
            j=item[1]
            self.grid[i,j]=1000
        return
    def random_populate(self):
        self.grid= np.random.rand(self.I,self.J)
        return
    def startState(self):
        return (0,0)
    def isEnd(self, state):
        return state == (self.I-1,self.J-1)
    def succAndCost(self, state):
        # return a list of (action, newState, cost) triples, takes a list of current potion (x,y)
        result = []
        if state[1]+1<self.J and state[0]>=0 and state[1]>=0:
            result.append(('right', (state[0],state[1]+1), self.grid[state[0],state[1]+1]))
        if state[0]+1<self.I  and state[0]>=0 and state[1]>=0:
            result.append(('down', (state[0]+1,state[1]), self.grid[state[0]+1,state[1]]))
        return result
def dynamicProgramming(problem):
    # state -> futureCost
    cache = {}
    def futureCost(state):
        # return best cost of reaching the end from state
        if problem.isEnd(state):
            return 0
        if state in cache:
            return cache[state]



        result = min([(cost+futureCost(newState),action,newState, cost) for action, newState, cost in problem.succAndCost(state)])

        cache[state] = result
        return result
    # recover total cost
    totalCost = futureCost(problem.startState())
    # recover history
    history = []
    state = problem.startState()
    while not problem.isEnd(state):
        print(cache)
        action, newState, cost = cache[state]
        history.append((action, newState, cost))
        state = newState
   
    return (totalCost, history)
def printSolution(solution):
    totalCost, history = solution
    print('Total cost: {}'.format(totalCost))
    for step in history:
        print(step)
