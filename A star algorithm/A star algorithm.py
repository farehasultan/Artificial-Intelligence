#Fareha Sultan
#Student Number: 100968491
#Assignment 1 - Introduction to Artificial Intelligence

'''
PROBLEM: 
Implement an algorithm using A* search to find the optimal path from the start state to a goal state.
In this problem, assume we have an agent that starts at a start state on an m x n grid. The agent tries to find a path
from the start state to the goal state by moving to adjacent horizontal or vertical squares on the grid (cannot move
diagonally). The cost of moving to an adjacent square is 1. There may also be hazards on the grid which cannot be
traversed nor can any square horizontally or vertically adjacent to it be traversed. Assume the environment is fully
observable; assume there always exists a path from the start state to the goal state.

Resources used to solve the problem:
I used the pseudo-code we wrote in class as a basis for this implementation
also used these sources for learning more about the A* algorithm :
https://www.redblobgames.com/pathfinding/a-star/implementation.html
https://www.youtube.com/watch?v=aKYlikFAV4k

'''


''' This function will be called when finding neighbours of the current node being explored. 
    It returns an array of neighbours that are within the m x n grid '''
def findNeighbours(curr,rows,cols):
  directions=[[1, 0], [0, 1], [-1, 0], [0, -1]] #takes into account North, South, East, West directions
  neighboursList =[]
  for dir in directions:
    neighbour = (curr[0] + dir[0], curr[1] + dir[1])
    if (0<=neighbour[0] < rows and 0<=neighbour[1]<cols): #making sure they are within the grid dimensions
      neighboursList.append(neighbour)
  return neighboursList


''' This function calculates the heuristic of the node . Point_b is always the goalState state for this implementation'''
def heuristic(point_a, point_b):
  x1=point_a[0]
  y1=point_a[1]
  x2=point_b[0]
  y2=point_b[1]
  return abs(x1 - x2) + abs(y1 - y2)

''' The actual A* algortithm which takes in a file path.
 When I tested , I made sure that assignment1.py was in the same folder as the input file.'''
 
def pathfinding(input_filepath):

  ''' Part 1: Reading the file and converting it into coordinates (x,y)'''
  results = [] #Saving the graph from text file 
  with open(input_filepath) as f: 
      for line in f.readlines():
        results.append(line[:-1].split(',')) #gets rid of \n
  f.close()
  graph = [] #here we will have the coordinates of the graph
  startState = () #startState coordinate 
  goalState = () #goalState coordinate
  obstacles = [] #all the obstacles : X and H including the squares adjacent to H
  rows = len(results) # length of results , here we are assuming that our input file is m * n
  cols = len(results[0]) # number of columns
  directions=[[1, 0], [0, 1], [-1, 0], [0, -1]] #all the directions aroung a coordinate , North, South , East, West
 
  ''' creating the graph with [x,y] coordinates + adding all the obstacles, goalState, startState as tuples'''
  for x in range(rows):
    for y in range(cols):
        if (results[x][y] == 'S'):
          startState=(x,y)
        if(results[x][y]=='X'):
          obstacles.append((x,y))
        if(results[x][y]=='H'): #recall that neighbours adjacent to H are not accessible to traverse
          obstacles.append((x,y))
          #append its neighbours
          for dir in directions:
            neighbour = (x + dir[0], y + dir[1])
            if (0<=neighbour[0] < rows and 0<=neighbour[1]<cols):
              obstacles.append(neighbour) 
        if(results[x][y]== 'G'):
          goalState=(x,y)
        graph.append((x, y))

  '''Part 2 : Implementation of the A* Algorithm'''

  frontier = dict()  #needs to be evaluates, when empty algo is finished, takes in a node coordinate and f(n)
  frontier[startState]=0 
  parent = dict() #node and its parent as value
  parent[startState] = None
  cost = dict() #to keep track of the total movement cost from the startState location. 
  cost[startState] = 0
  explored=[] #no need to revisit thsee
 
  while frontier: # a bool checks if the dict() is empty or not
    
    x= min(frontier.values()) #using it as a priority queue

    current = ()
    
    for key, value in frontier.items():
        if x == value:
            current= key
 
    
    if current == goalState: 
        explored.append(current)
        break

    del frontier[current] 
    explored.append(current)
 
    #what should I add to Frontier?
    for next in findNeighbours(current,rows,cols):
      if(next not in obstacles and next not in explored ):
        new_cost = cost[current] + 1 #cost of moving to adjacent square is 1
        if ((next not in cost) or (new_cost < cost[next])): #if a cost for that node already exists, and the new cost it lower use that as its cost
          cost[next] = new_cost
          priority = new_cost + heuristic(next,goalState)
          frontier[next] = priority
          parent[next] = current
        
  
  #reconstructing paths is by moving backwards using the parent dictionary
  end = goalState
  cost = cost[goalState]
  path = []
  while end != startState: 
    path.append(end)
    end = parent[end]
  path.append(startState) 
  path.reverse() #reversing to have it in order from start to goal


  print("This is the explored list", explored)
  print("This is the optimal path", path)
  print("this is the optimal path cost", cost)

  # optimal_path is a list of tuples indicated the optimal path from start to goal
  # explored_list is the list of nodes explored during search
  # optimal_path_cost is the cost of the optimal path from the start state to the goal state

  optimal_path = open('optimal_path.txt', 'w')
  optimal_path.write(str(path))
  optimal_path.close()

  explored_list = open('explored_list.txt', 'w')
  explored_list.write(str(explored))
  explored_list.close()

  optimal_path_cost = open('optimal_path_cost.txt', 'w')
  optimal_path_cost.write(str(cost))
  optimal_path_cost.close()

 
''' Calling the function , I put the name of the input file in quotations'''
pathfinding("input4.txt")




