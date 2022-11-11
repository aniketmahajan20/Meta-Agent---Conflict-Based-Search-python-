import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
import copy
from itertools import combinations
import numpy as np
from cbs import CBSSolver

def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst

def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    
    # finding which path is greater to determine the lenght of search for collision.
    if len(path1) > len(path2):
        path = path1
    else:
        path = path2
    for timestep in range(len(path)):
        loc1_agent1 = get_location(path1, timestep)
        loc2_agent1 = get_location(path1, timestep +1)
        loc1_agent2 = get_location(path2, timestep)
        loc2_agent2 = get_location(path2, timestep + 1)
        # print(loc1_agent1,loc1_agent2)
        # For vertex collision and also goal collisions
        if loc1_agent1 == loc1_agent2:
            collision = {'loc': [loc1_agent1],
                 'timestep': timestep
            }
            return collision
        # For edge collision
        elif [loc1_agent1,loc2_agent1] == [loc2_agent2,loc1_agent2]:
            collision = {'loc': [loc1_agent1,loc2_agent1],
                 'timestep': timestep + 1
            } 
            return collision
    return None


def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    collisions = []
    # Creating all possible pairs of agents in paths
    for agent1 in range(len(paths)):
        for agent2 in range((len(paths)-1),agent1,-1):
            c = detect_collision(paths[agent1],paths[agent2])
            if c:
                # print(agent1,agent2)
                collision = {'a1': agent1,
                            'a2': agent2,
                            'loc': c['loc'],
                            'timestep': c['timestep'] }
                collisions.append(collision)
    return collisions



def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
    
    # For Vertex Collisions. I have added 'positive'  key here as well for easier testing. It is always False for standard splitting
    if len(collision['loc']) == 1:
        c1 = {'agent1': collision['a1'],
              'agent2': collision['a2'],  
              'loc': collision['loc'],
              'timestep': collision['timestep'],
              'positive': False}
        c2 = {'agent1': collision['a2'],
              'agent2': collision['a1'],   
              'loc': collision['loc'],
              'timestep': collision['timestep'],
              'positive': False}
    # For edge collisions
    elif len(collision['loc']) == 2:
        c1 = {'agent1': collision['a1'],
              'agent2': collision['a2'],
              'loc': [collision['loc'][0],collision['loc'][1]],
              'timestep': collision['timestep'],
              'positive': False}
        c2 = {'agent1': collision['a2'],
              'agent2': collision['a1'],
              'loc': [collision['loc'][1],collision['loc'][0]],
              'timestep': collision['timestep'],
              'positive': False}
    else: 
        return None
    return [c1,c2]


def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep

    #Choose the agent randomly
    i = random.randint(0,1)
    if i == 0:
        agent = collision['a1']
    else:
        agent = collision['a2']
    # For vertex collisions
    if len(collision['loc']) == 1:
        c1 = {'agent': agent,
              'loc': collision['loc'],
              'timestep': collision['timestep'],
              'positive': True}
        c2 = {'agent': agent,
              'loc': collision['loc'],
              'timestep': collision['timestep'],
              'positive': False}
    # For edge collisions. First check the agent for including the correct constraint
    elif len(collision['loc']) == 2:
        if agent == collision['a1']:
            c1 = {'agent': agent,
                'loc': [collision['loc'][0],collision['loc'][1]],
                'timestep': collision['timestep'],
                'positive': True}
            c2 = {'agent': agent,
                'loc': [collision['loc'][0],collision['loc'][1]],
                'timestep': collision['timestep'],
                'positive': False}
        else:
            c1 = {'agent': agent,
                'loc': [collision['loc'][1],collision['loc'][0]],
                'timestep': collision['timestep'],
                'positive': True}
            c2 = {'agent': agent,
                'loc': [collision['loc'][1],collision['loc'][0]],
                'timestep': collision['timestep'],
                'positive': False}
    else: 
        return None
    return [c1,c2]

def should_merge(a1,a2,Matrix):
    B = 2
    if len(a1) > 1 and len(a2) > 1:
        for i in a1:
            for k in a2:
                if Matrix[i[0]][k[0]] < B:
                    return False
    elif len(a1) > 1:
        for i in a1:
            for k in a2:
                if Matrix[i[0]][k] < B:
                    return False
    elif len(a2) > 1:
        for i in a1:
            for k in a2:
                if Matrix[i][k[0]] < B:
                    return False
    else:
        if Matrix[a1[0]][a2[0]] < B:
            return False
    return True


class MACBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self,node):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'agents': [],
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['agents'].append([i])
            root['paths'].append(path)
        # breakpoint()
        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        if root['collisions'] == None:
            return self.num_of_generated, self.num_of_expanded, root['paths']
        self.push_node(root)
        root['Matrix'] = [[0 for x in range(self.num_of_agents)] for y in range(self.num_of_agents)]
        # Task 3.1: Testing
        # print(root['collisions'])

        # # Task 3.2: Testingq
        # for collision in root['collisions']:
        #     print(standard_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit
       
        while len(self.open_list) > 0:
            parent = self.pop_node(self.open_list)
            # breakpoint()
            if parent['collisions'] == 0:
                print(parent['agents'])
                return self.num_of_generated, self.num_of_expanded,parent['paths']
            # breakpoint()
            # detecting collision in paths of all agents in parent node
            collisions = detect_collisions(parent['paths'])
            if collisions == []:
                print(parent['agents'])
                return self.num_of_generated, self.num_of_expanded,parent['paths']
            # Conflict Matrix
            # Creates a list containing 5 lists, each of 8 items, all set to 0
            
            for collision in collisions:
                parent['Matrix'][collision['a1']][collision['a2']] += 1
                # print(Matrix)
                # breakpoint()

            # implementing merging ciriteria
            flag = 0
            for index1 in range(len(parent['agents'])):
                for index2 in range(len(parent['agents'])-1,index1,-1):
                    if should_merge(parent['agents'][index1],parent['agents'][index2],parent['Matrix']):
                        # merging agents
                        flag = 1
                        a1 = parent['agents'][index1]
                        a2 = parent['agents'][index2]
                        parent['agents'].pop(index2)
                        parent['agents'].pop(index1)
                        # breakpoint()
                        if len(a1) > 1 and len(a2) > 1:
                            merged_agent = a1 + a2
                            for constraint in parent['constraints']:
                                if [constraint['agent1']] in a1 and [constraint['agent2']] in a2:
                                    parent['constraints'].remove(constraint)
                                elif [constraint['agent1']] in a2 and [constraint['agent2']] in a1:
                                    parent['constraints'].remove(constraint)

                        elif len(a1) > 1:
                            merged_agent = a1.append(a2)
                            for constraint in parent['constraints']:
                                if [constraint['agent1']] in a1 and [constraint['agent2']] == a2:
                                    parent['constraints'].remove(constraint)
                                elif [constraint['agent1']] == a2 and [constraint['agent2']] in a1:
                                    parent['constraints'].remove(constraint)

                        elif len(a2) > 1:
                            merged_agent = a2.append(a1)
                            for constraint in parent['constraints']:
                                if [constraint['agent1']] == a1 and [constraint['agent2']] in a2:
                                    parent['constraints'].remove(constraint)
                                elif [constraint['agent1']] in a2 and [constraint['agent2']] == a1:
                                    parent['constraints'].remove(constraint)
                        else:
                            merged_agent = [a1,a2]
                            for constraint in parent['constraints']:
                                if [constraint['agent1']] == a1 and [constraint['agent2']] == a2:
                                    parent['constraints'].remove(constraint)
                                elif [constraint['agent1']] == a2 and [constraint['agent2']] == a1:
                                    parent['constraints'].remove(constraint)
                        agents = []
                        # print(merged_agent)
                        # print(parent['Matrix'])
                        
                        
                        for agent in merged_agent:
                            agents.append(agent[0])
                        parent['agents'].insert(0 , merged_agent)
                        # print(parent['agents'])
                        starts = []
                        goals = []
                        lower_constraints = []
                        i = 0
                        for agent in agents:
                            starts.append(self.starts[agent])
                            goals.append(self.goals[agent])
                            for constraint in parent['constraints']:
                                    if constraint['agent1'] == agent:
                                        lower_constraint = copy.deepcopy(constraint)
                                        lower_constraint['agent1'] = i
                                        lower_constraints.append(lower_constraint)
                            i = i + 1
                        # breakpoint()
                        # print(parent['agents'])
                        # calling the low level cbs solver
                        cbs = CBSSolver(self.my_map, starts, goals,lower_constraints)
                        paths = cbs.find_solution()
                        if paths:
                            i = 0
                            for agent in agents:
                                parent['paths'][agent] =  paths[i]
                                i = i+1
                            parent['collisions'] = detect_collisions(parent['paths'])
                            if parent['collisions'] is None:
                                print(parent['agents'])
                                return self.num_of_generated, self.num_of_expanded,parent['paths']
                            parent['cost'] = get_sum_of_cost(parent['paths'])
                            if parent['cost']:
                                self.push_node(parent)
            if flag == 1:
                continue

            # breakpoint()
            
            
            # converting collisions in to constraints
            for collision in collisions:
                # constraints = disjoint_splitting(collision)
                constraints = standard_splitting(collision)
            for constraint in constraints:
                # making a deep copy so that changes are not made to the original dictionary
                child = copy.deepcopy(parent)
                child['constraints'].append(constraint)
                # breakpoint()
                a = [constraint['agent1']]
                for index in range(len(child['agents'])):
                    a1 = child['agents'][index]
                    if len(a1) > 1:
                        if a in a1:
                            agents = []
                            for agent in a1:
                                agents.append(agent[0])
                            starts = []
                            goals = []
                            lower_constraints = []
                            i = 0
                            for agent in agents:
                                starts.append(self.starts[agent])
                                goals.append(self.goals[agent])
                                # breakpoint()
                                for constraint in child['constraints']:
                                    if constraint['agent1'] == agent:
                                        lower_constraint = copy.deepcopy(constraint)
                                        lower_constraint['agent1'] = i
                                        lower_constraints.append(lower_constraint)
                                i = i + 1
                                # breakpoint()
                            # print(parent['agents'])
                            # calling the low level cbs solver
                            # breakpoint()
                            cbs = CBSSolver(self.my_map, starts, goals,lower_constraints)
                            paths = cbs.find_solution()
                            # print(paths)
                            # breakpoint()
                            if paths:
                                i = 0
                                for agent in agents:
                                    # breakpoint()
                                    child['paths'][agent] =  paths[i]
                                    i = i+1
                                child['collisions'] = detect_collisions(child['paths'])
                                if child['collisions'] is None:
                                    print(child['agents'])
                                    return self.num_of_generated, self.num_of_expanded,child['paths']
                                child['cost'] = get_sum_of_cost(child['paths'])
                                # breakpoint()
                                # print("...............................Upper node running")
                                self.push_node(child)
                    else:
                        if a == a1:
                            path = a_star(self.my_map, self.starts[a[0]], self.goals[a[0]], self.heuristics[a[0]],a[0], child['constraints'])
                            if path:
                                child['paths'][a[0]] = path
                                child['collisions'] = detect_collisions(child['paths'])
                                if child['collisions'] is None:
                                    print(child['agents'])
                                    return self.num_of_generated, self.num_of_expanded,child['paths']
                                child['cost'] = get_sum_of_cost(child['paths'])
                                # print(child['Matrix'])
                                # print(child['agents'])
                                # breakpoint()
                                # print("...............................Lower Node running")
                                self.push_node(child)      
                                    
                    
                    # flag = 0
                    # For positive constraints, all agents which violate the constraint need to find a new shortest path, while using disjoint splitting
                    # if constraint['positive'] is True:
                    #     agents =  paths_violate_constraint(constraint,parent['paths'])
                    #     for agent in agents:
                    #         new_constraint = copy.deepcopy(constraint)
                    #         new_constraint["agent1"] = agent
                    #         new_constraint["positive"] = False
                    #         child["constraints"].append(new_constraint)
                    #         path = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent],
                    #             agent, child['constraints'])
                    #         if path:
                    #             child['paths'][agent] = path
                    #         else:
                    #             # breakpoint()
                    #             flag = 1
                    # if flag == 1:
                    #     continue 
                    # print(child['agents'])
        return None


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
