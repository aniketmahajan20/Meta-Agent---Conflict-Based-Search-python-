import csv
from matplotlib import pyplot as plt 
import numpy as np
csv_filename = 'results.csv'
num_of_agents = []
nodes_generated = []
nodes_expanded = []
num_of_agents_cbs = []
nodes_generated_cbs = []
nodes_expanded_cbs = []
x1 = []
i = 0
with open(csv_filename) as f:
    reader = csv.DictReader(f)
    for row in reader:
        x1.append(int(i))
        i = i+1
        num_of_agents.append(int(row['num_of_agentss']))
        nodes_generated.append(int(row['nodes_generated']))
        nodes_expanded.append(int(row['nodes_expanded']))
with open('results_cbs.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        num_of_agents_cbs.append(int(row['num_of_agentss']))
        nodes_generated_cbs.append(int(row['nodes_generated']))
        nodes_expanded_cbs.append(int(row['nodes_expanded']))

average_nodes_generated_macbs = sum(nodes_generated)/len(nodes_generated)
average_nodes_generated_cbs = sum(nodes_generated_cbs)/len(nodes_generated_cbs)
print("Average nodes generated MACBS (B = 5)" ,average_nodes_generated_macbs)
print("Average nodes generated CBS" ,average_nodes_generated_cbs)
average_nodes_expanded_macbs = sum(nodes_expanded)/len(nodes_expanded)
average_nodes_expanded_cbs = sum(nodes_expanded_cbs)/len(nodes_expanded_cbs)
print("Average nodes expanded MACBS (B = 5)" ,average_nodes_expanded_macbs)
print("Average nodes expanded CBS" ,average_nodes_expanded_cbs)


agents_vs_nodesgenerated_macbs, = plt.plot(x1, nodes_generated, label='MA-CBS with B = 1')
agents_vs_nodesgenerated_cbs, = plt.plot(x1, nodes_generated_cbs,label='CBS')

# agents_vs_nodesexpanded_macbs, = plt.plot(x1, nodes_expanded,label='MA-CBS with B = 1')
# agents_vs_nodesexpanded_cbs, = plt.plot(x1, nodes_expanded_cbs,label='CBS')
print(len(x1))
print(x1)
print(len(nodes_generated_cbs))
plt.legend(handles=[agents_vs_nodesgenerated_macbs, agents_vs_nodesgenerated_cbs])
plt.xlabel("Problem Instance")
plt.ylabel("Number of Nodes Generated")
plt.title("Number of Nodes Generated by CBS and MA-CBS with B=1")
plt.show() 
quit()
