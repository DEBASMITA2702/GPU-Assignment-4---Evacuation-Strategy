# GPU Assignment 4 – Evacuation Strategy Simulation

This repository contains the solution for my **GPU Assignment 4**, where the task is to simulate **civilian evacuation planning** during a Titan invasion using CUDA.  
The objective is to minimize evacuation time while maximizing the number of survivors.

---

## Problem Overview
- Cities are connected via roads with **limited capacity** and **travel time**.  
- Civilians are categorized as:
  - **Prime-age** (can travel any distance)  
  - **Elderly** (can only travel up to a maximum distance)  
- **Shelters** have limited capacity. Overcrowding causes penalties.  
- Roads must be used **sequentially**, with priority given to cities having larger evacuee populations.  

---

## Input
- Number of cities, roads, and shelters  
- Road details: `[u, v, length, capacity]`  
- Shelter info: `[city, capacity]`  
- Populated cities with `[city, prime-age, elderly]`  
- `max_distance_elderly`  

---

## Output
- ```path_sizes``` → number of cities in each evacuation path  
- ```paths``` → actual evacuation routes  
- ```num_drops``` → number of drop points along each path  
- ```drops``` → detailed drop info:  
[ City/Shelter, Prime-age dropped, Elderly dropped ]

---

## Sample Example
**Input Setup**  
Populated Cities:  
City 0 -> 400 prime, 80 elderly  
City 2 -> 300 prime, 60 elderly  
City 5 -> 500 prime, 90 elderly  

Shelters:  
S1 at City 4 -> capacity 500  
S2 at City 6 -> capacity 600  
S3 at City 9 -> capacity 400  

max_distance_elderly = 10 km  

**One possible evacuation strategy:**  
paths  
0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 7 -> 8 -> 9  
2 -> 3 -> 4 -> 5 -> 6  
5 -> 4 -> 5 -> 6  

drops  
[1, 0, 80], [9, 400, 0]  
[3, 0, 60], [6, 300, 0]  
[4, 410, 90], [6, 90, 0]  


---

## How to Run
Compile and run using **nvcc**:
```bash
nvcc Evacuation.cu -o evac
./evac < input.txt > output.txt
