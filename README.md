# Project 2 - Parallelism 

## How to run ?

```
mpicc -std=c99 -o defranoux defranoux.c -lm -fopenmp
```
```
mpirun -np 4 ./defranoux a_4 b_4
```
## Matrix representation

This project use a struct of table of table of int : 
```
struct Matrix {
	int** data;
	int x;
	int y;
};
```

## Communications

The project respect the virtual ring constraint. It use :
* Broadcast
* Scatter
* Gather

## Structures

1. Parse file a in parallel with file b. b is parsing upside down.
2. Broadcast the size x and y of the sub matrix.
3. Scatter the sub matrix a and b. P0 keep the first one of both.
4. P0 calculate and gather the result of each processor.
5. Rotation the sub matrix b.
6. Print the result :+1: 

## Features 

|                           Work 				              |      State       |
| ----------------------------------------------------------- | ---------------- |
| Various matrices size with P divisor of N                   | Done             |
| Very large matrices                                         | Done             |
| More processor than necessary (P not divisor of N and P>N)  | Done             |
| Unbalanced computations (P not divisor of N and P<N)        | Work in progress |

