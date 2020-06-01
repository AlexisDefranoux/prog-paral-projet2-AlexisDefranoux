#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>

struct Matrix {
  int** data;
  int x;
  int y;
};

typedef struct Matrix Matrix;

Matrix* allocateMatrix(int x, int y) {
    struct Matrix* matrix = malloc(sizeof(struct Matrix));
    matrix->x = x;
    matrix->y = y;
    matrix->data = (int**)malloc(sizeof(int*) * y);
    #pragma omp parallel for
    for(int yi = 0; yi < y; yi++)
      matrix->data[yi] = (int*)calloc(x, sizeof(int));
    return matrix;
}

void freeMatrix(Matrix* matrix) {
  #pragma omp parallel for
  for(int y = 0; y < matrix->y; y++)
    free(matrix->data[y]);
  free(matrix->data);
  free(matrix);
}

void printMatrix(Matrix* matrix) {
  for(int y = 0; y < matrix->y; y++) {
    for(int x = 0; x < matrix->x; x++) {
      printf("%i ", matrix->data[y][x]);
    }
    printf("\n");
  }
}

Matrix* parseFile(char* filePath, bool isInversed) {
  FILE* file;
  char c;

  if ((file = fopen(filePath, "r")) == NULL){
    printf("Erreur sur l'ouverture du fichier");
    exit(1);
  }

  int cols = 1;
  for (c = getc(file); c != '\n'; c = getc(file)) 
    if (c == ' ') cols++;

  struct Matrix* matrix = allocateMatrix(cols, cols);
  rewind(file);

  int data = 0;
  fscanf(file, "%d", &data);
  for(int y=0; !feof (file); ++y){
    for(int x=0; x<cols; ++x) {
      if(isInversed)
        matrix->data[x][y] = data; 
      else
        matrix->data[y][x] = data; 
      fscanf(file, "%d", &data);
    }
  }
  fclose(file);
  return matrix;
}

void produitMatricielSequentielEsclave(Matrix* a, Matrix* b, Matrix* result, int nbrTab) {
  #pragma omp parallel for
  for(int z = 0; z < nbrTab; z++){
    #pragma omp parallel for
    for(int y = 0; y < nbrTab; y++) {
      result->data[z][y] = 0;
      for(int x = 0; x < a->x; x++) { 
        result->data[z][y] = result->data[z][y] + (a->data[z][x] * b->data[y][x]);
      }
    }
  }
}

void produitMatricielSequentielLeader(Matrix* a, Matrix* b, Matrix* result, int nbrTab, int startX){
  #pragma omp parallel for
  for(int z = startX; z < (nbrTab+startX); z++){
    #pragma omp parallel for
    for(int y = 0; y < nbrTab; y++) {
      result->data[y][z] = 0;
      for(int x = 0; x < a->x; x++) { 
        result->data[y][z] = result->data[y][z] + (a->data[y][x] * b->data[z][x]);

      }
    }
  }
}

void scatterInit(Matrix* a, Matrix* b, int dest, int startY, int endY, int sizeTab){
  for(int y = startY; y < endY; y++) {
    MPI_Send(a->data[y], sizeTab, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(b->data[y], sizeTab, MPI_INT, dest, 0, MPI_COMM_WORLD);
  }
}

void scatter(Matrix* a, int dest1, int dest2, int endY, int sizeTab){
  for(int y = 0; y < endY; y++) {
    MPI_Send(a->data[y], sizeTab, MPI_INT, dest1, 0, MPI_COMM_WORLD);
    MPI_Recv(a->data[y], sizeTab, MPI_INT, dest2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void scatterInverse(Matrix* a, int dest1, int dest2, int endY, int sizeTab){
  for(int y = 0; y < endY; y++) {
    MPI_Recv(a->data[y], sizeTab, MPI_INT, dest2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(a->data[y], sizeTab, MPI_INT, dest1, 0, MPI_COMM_WORLD);
  }
}

void scatterEsclave(Matrix* ax, Matrix* bx, int previous, int next, int nbrTab, int sizeTab, int nbrProcsUse, int rank){
  for(int y = 0; y < nbrTab; y++) {
    MPI_Recv(ax->data[y], sizeTab, MPI_INT, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(bx->data[y], sizeTab, MPI_INT, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  int tmp[sizeTab];
  for(int i = rank; i < nbrProcsUse -1; i++) {
    for(int y = 0; y < nbrTab; y++) {
      MPI_Recv(&tmp, sizeTab, MPI_INT, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&tmp, sizeTab, MPI_INT, next, 0, MPI_COMM_WORLD);
      MPI_Recv(&tmp, sizeTab, MPI_INT, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&tmp, sizeTab, MPI_INT, next, 0, MPI_COMM_WORLD);
    }
  }
}

void gatherEsclave(Matrix* result, int previous, int next, int nbrTab, int rank, int nbrProcsUse){
  if(rank<nbrProcsUse){
    for(int y = 0; y < nbrTab; y++) {
      for(int x = 0; x < nbrTab; x++) {
        MPI_Send(&result->data[y][x], 1, MPI_INT, previous, 0, MPI_COMM_WORLD);
      }
    }
  }
  for(int i = 0; i < ((nbrProcsUse-1)-rank); i++) {
    for(int y = 0; y < nbrTab; y++) {
      for(int x = 0; x < nbrTab; x++) {
        MPI_Recv(&result->data[y][x], 1, MPI_INT, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&result->data[y][x], 1, MPI_INT, previous, 0, MPI_COMM_WORLD);
      }
    }
  }
}

void gatherLeader(Matrix* result, int next, int nbrTab, int j, int k){
  for(int y = 0; y < nbrTab; y++) {
    for(int x = 0; x < nbrTab; x++) {
      MPI_Recv(&result->data[(j*nbrTab)+y][(k*nbrTab)+x], 1, MPI_INT, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

int main(int argc, char* argv[]) {

  int rank, nbrProcs, sizeTab, nbrTab, nbrProcsUse;
  
  omp_set_nested(1);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nbrProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int previous = ((rank-1+nbrProcs)%nbrProcs);
  int next = ((rank+1)%nbrProcs);

  struct Matrix* a;
  struct Matrix* b;
  struct Matrix* result;

  bool divisible = false;

  if (rank == 0) {

    #pragma omp parallel sections
    {
      #pragma omp section
      {
        a = parseFile(argv[1], false);
      }
      #pragma omp section
      {
        b = parseFile(argv[2], true);
      }
    }

    result = allocateMatrix(a->x, a->x);
    sizeTab = a->y;

    if((sizeTab / nbrProcs) < 1){
      nbrTab = 1;
      nbrProcsUse = sizeTab;
    }
    else{
      nbrTab = (int)(sizeTab / nbrProcs);
      nbrProcsUse = nbrProcs;

      if((sizeTab % nbrProcs) == 0)
        divisible = true; 
    }

    MPI_Send(&sizeTab, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    MPI_Send(&nbrTab, 1, MPI_INT, next, 0, MPI_COMM_WORLD);

    for(int z = 1; z < nbrProcsUse; z++) {
      scatterInit(a, b, next, (z*nbrTab), ((z*nbrTab)+nbrTab), sizeTab);
    }

    int k;
    for(int i = 0; i < nbrProcsUse; i++) {
      k=i;

      for(int j = 0; j < nbrProcsUse; j++) {

        if(j==0)
          produitMatricielSequentielLeader(a, b, result, nbrTab, (k*nbrTab));
        else
          gatherLeader(result, next, nbrTab, j, k);

        k++;
        if(k > (nbrProcsUse-1))
          k=0;
      }

      if(i != nbrProcsUse-1)
        scatter(b, previous, next, nbrTab, sizeTab);
    }

    printMatrix(result);

    freeMatrix(a);
    freeMatrix(b);
    freeMatrix(result);

  } else {

    MPI_Recv(&sizeTab, 1, MPI_INT, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&nbrTab, 1, MPI_INT, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if(rank != (nbrProcs-1)) {
      MPI_Send(&sizeTab, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
      MPI_Send(&nbrTab, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    if((sizeTab / nbrProcs) < 1)
      nbrProcsUse = sizeTab;
    else
      nbrProcsUse = nbrProcs;
    
    b = allocateMatrix(sizeTab, nbrTab);
    if(rank<nbrProcsUse){
      a = allocateMatrix(sizeTab, nbrTab);
      result = allocateMatrix(nbrTab, nbrTab);
      scatterEsclave(a, b, previous, next, nbrTab, sizeTab, nbrProcsUse, rank);
    }

    for(int z = 0; z < nbrProcsUse; z++) {

      if(z != 0){
        if(rank<nbrProcsUse)
          scatter(b, previous, next, nbrTab, sizeTab);
        else
          scatterInverse(b, previous, next, nbrTab, sizeTab);
      }

      if(rank<nbrProcsUse)
        produitMatricielSequentielEsclave(a, b, result, nbrTab);
        
      if(rank<nbrProcsUse)
        gatherEsclave(result, previous, next, nbrTab, rank, nbrProcsUse);
      else
        gatherEsclave(b, previous, next, nbrTab, rank, nbrProcsUse);
      
    }

    if(rank<nbrProcsUse){
      freeMatrix(a);
      freeMatrix(result);
    }
    freeMatrix(b);
  }

  MPI_Finalize();

  return 0;
}