import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """A planner that plans for each robot sequentially."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.CPU_time = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        result = []
        constraints = []
        # Constraints for task 1
        # c = {'agent': 1,
        #             'loc': [(1,2)],
        #             'timestep': 2}
        # a = {'agent': 1,
        #             'loc': [(1,3)],
        #             'timestep': 2}
        # b = {'agent': 1,
        #             'loc': [(1,4)],
        #             'timestep': 2}
        # # d = {'agent': 0,
        # #             'loc': [(1,5)],
        # #             'timestep': 10}
        # constraints.append(c)
        # constraints.append(a)
        # constraints.append(b)
        # constraints.append(d)
        for i in range(self.num_of_agents):  # Find path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints)
            if path is None:
                raise BaseException('No solutions')
            result.append(path)

            ##############################
            # Task 2: Add constraints here
            #         Useful variables:
            #            * path contains the solution path of the current (i'th) agent, e.g., [(1,1),(1,2),(1,3)]
            #            * self.num_of_agents has the number of total agents
            #            * constraints: array of constraints to consider for future A* searches
            for constraint in constraints:
                constraint['agent'] = i+1
            timestep = 0
            for loc in path:
                c1 = {'agent': i+1,
                     'loc': [loc],
                     'timestep': timestep}
                if timestep < len(path)-1:
                    c2 = {'agent': i+1,
                        'loc': [path[timestep+1],loc],
                        'timestep': timestep +1
                    }
                if timestep == len(path)-1:
                    for j in range(10000):
                        c3 = {'agent': i+1,
                            'loc': [loc],
                            'timestep': timestep + j}
                        constraints.append(c3)
                constraints.append(c1)
                constraints.append(c2)
                timestep = timestep + 1 
            # breakpoint()

            ##############################

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
