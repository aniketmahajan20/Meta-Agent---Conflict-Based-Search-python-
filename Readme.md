# About
This is a project that builds on top of Conflict based Search (CBS) by implementing Meta Agent Conflict Based Search (MA-CBS) for Multi Agent Path Planning. MA-CBS improves the performance of CBS by finding pairs of agents that have high number of collisions. It then plans paths for these two agents first before advancing the search space to contains all the agents, thus reducing the search space spruning trees with probable non-optimal outcomes beforehand. This algorithm is very helpful in solving complex problems with high number of agents where multiple pairs might be strongly coupled. 

Due to these limitations G. Sharon et al.[1] have presented Meta Agent Conflict Based Search. The
MA-CBS algorithm automatically identifies strongly coupled agents that have multiple conflicts and
merges them to create a meta-agent. Then the meta-agent is treated as single agent and search proceeds
normally using CBS. This makes MA-CBS dynamic in nature that merges the agents as it proceeds
with the search. Meta-Agent requires a complete and optimal Multi Agent Path Finding algorithm to
run at a lower level and return optimal paths for merged agents. CBS is chosen as the low-level MAPF
algorithm. 

For further detail about the implementation details, please refer: https://drive.google.com/file/d/1Sw0I1BTVbQ5_uSocXCe0NktbJP_QOEeL/view

# Testing
To run and test this code, clone the repository. Use the following command by opening a terminal in the project directory:
```
python run_experiments.py --instance instances/test_1.txt --solver MACBS
```

If you wnat to test performance for all the test instances without seeing the animation, use the folowing command:

```
python run_experiments.py --instance "instances/test_*" --solver MACBS --batch
```

> The total time taken will be printed along with the final paths for each agent. The results.csv will contain (num_of_agentss,nodes_generated,nodes_expanded) for each test case. No animation will be run in this case.

Have fun experimenting with the project :-D

