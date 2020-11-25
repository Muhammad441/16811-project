# 16811-project
The directory contains a map file map1.txt. Here is an example of running the test from Matlab when planning
for a 5-DOF arm:
To compile the C code:
1. mex planner.cpp
To run the planner:
2. startQ = [pi/2 pi/4 pi/2 pi/4 pi/2];
3. goalQ = [pi/8 3*pi/4 pi 0.9*pi 1.5*pi];
4. planner_id = 0 % placeholder for now
5. runtest('map1.txt',startQ, goalQ, planner_id);
