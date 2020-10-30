/* Author - Satyam Ramawat - 19210520
** satyamramawat@gmail.com
** Efficient Algorithm for Large Multiplication
** Test and Comparison between ikj-Algorithm, ijk-Algorithm and Traditional Approach
** Created by 11th April 2020
** Modified by 18th April 2002
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>

typedef double TYPE;
/* Large memory allocation can be done upto 46000 x 46000 matrix*/
#define MAX_DIM 46000*46000
#define MAX_VAL 10
#define MIN_VAL 1

// 1 Dimensional matrix on stack
TYPE OneD_A[MAX_DIM];
TYPE OneD_B[MAX_DIM];

/* Below function generate 2D random TYPE matrix. */
TYPE** randomSquareMatrix(int dimension){
    
    //Memory allocation according demanded dimensions
    TYPE** matrix = malloc(dimension * sizeof(TYPE*));

    for(int i=0; i<dimension; i++){
        matrix[i] = malloc(dimension * sizeof(TYPE));
    }

    //Random seed
    srandom(time(0)+clock()+random());

    #pragma omp parallel for
    for(int i=0; i<dimension; i++){
        for(int j=0; j<dimension; j++){
            matrix[i][j] = rand() % MAX_VAL + MIN_VAL;
        }
    }

    return matrix;
}

/* Below function generates 2D zero TYPE matrix. */
TYPE** zeroSquareMatrix(int dimension){

    TYPE** matrix = malloc(dimension * sizeof(TYPE*));

    for(int i=0; i<dimension; i++){
        matrix[i] = malloc(dimension * sizeof(TYPE));
    }

    //Random seed
    srandom(time(0)+clock()+random());
    for(int i=0; i<dimension; i++){
        for(int j=0; j<dimension; j++){
            matrix[i][j] = 0;
        }
    }

    return matrix;
}

/*Function to convert 2D matrices into 1D in order to reduce cache misses */
void convert(TYPE** matrixA, TYPE** matrixB, int dimension){
    #pragma omp parallel for
    for(int i=0; i<dimension; i++){
        for(int j=0; j<dimension; j++){
            OneD_A[i * dimension + j] = matrixA[i][j];
            OneD_B[j * dimension + i] = matrixB[i][j];
        }
    }
}

/* Traditional Multiplication Function, input matrices and return resultant matrix */
double TraditionalMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension){

	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	float Tota_Sum=0.0;

	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			for(int k=0; k<dimension; k++){
				matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
				//Below line calculate sumation of matrix multiplication values
				//Tota_Sum+=matrixC[i][j];
			}
		}
	}
	// Below line prints the Sumation Result of Matrix Multiplication
	//printf("Resutl %f ", Tota_Sum);

	/* This will generate time taken by algorithm in order to multiply N dimension */
	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;
	return elapsed;
}

//Test function to generate N dimension matrix to test efficiency in Traditional-algorithm
void TraditionalMultiplyTest(int dimension, int iterations){
    FILE* fp;
    fp = fopen("Results/TraditionalMultiplyTest.txt", "a+");

    // Console write
    printf("..................................\n");
    printf("Dimension : %d\n", dimension);
    
    // File write
    fprintf(fp, "----------------------------------\n");
    fprintf(fp, "Test : Traditional Multiplication         \n");
    fprintf(fp, "----------------------------------\n");
    fprintf(fp, "Dimension : %d\n", dimension);
    fprintf(fp, "..................................\n");

    double* opmLatency = malloc(iterations * sizeof(double));
    TYPE** matrixA = randomSquareMatrix(dimension);
    TYPE** matrixB = randomSquareMatrix(dimension);
    
    // Iterate and measure performance
    for(int i=0; i<iterations; i++){
        TYPE** matrixResult = zeroSquareMatrix(dimension);
        opmLatency[i] = TraditionalMultiply(matrixA, matrixB, matrixResult, dimension);
        free(matrixResult);

        // Console write
        printf("%d.\t%f\n", i+1, opmLatency[i]);

        // File write
        fprintf(fp, "%d.\t%f\n", i+1, opmLatency[i]);
    }

    // Releasing memory
    fclose(fp);
    free(opmLatency);
    free(matrixA);
    free(matrixB);
}

/* ijk-Algorithm for large Matrix Multiplication Function, input matrices and return resultant matrix */
double ijk_algorithm(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension){

	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	float Tota_Sum=0.0;

	#pragma omp parallel for
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			for(int k=0; k<dimension; k++){
				matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
				//This calculate sumation of matrix multiplication values
				//Tota_Sum+=matrixC[i][j];
			}
		}
	}
	// Below line prints the Sumation Result of Matrix Multiplication
	//printf("Resutl %f ", Tota_Sum);

	/* This will generate time taken by algorithm in order to multiply N dimension */
	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;
	return elapsed;
}

//Test function to generate N dimension matrix to test efficiency in ijk-algorithm
void ijk_algorithmTest(int dimension, int iterations){
    FILE* fp;
    fp = fopen("Results/ijk_algorithmTest.txt", "a+");

    // Console write
    printf("..................................\n");
    printf("Dimension : %d\n", dimension);
    
    // File write
    fprintf(fp, "----------------------------------\n");
    fprintf(fp, "Test : ijk-Algorithm Matrix Multiplication\n");
    fprintf(fp, "----------------------------------\n");
    fprintf(fp, "Dimension : %d\n", dimension);
    fprintf(fp, "..................................\n");

    double* opmLatency = malloc(iterations * sizeof(double));
    TYPE** matrixA = randomSquareMatrix(dimension);
    TYPE** matrixB = randomSquareMatrix(dimension);
    
    // Iterate and measure performance
    for(int i=0; i<iterations; i++){
        TYPE** matrixResult = zeroSquareMatrix(dimension);
        opmLatency[i] = ijk_algorithm(matrixA, matrixB, matrixResult, dimension);
        free(matrixResult);

        // Console write
        printf("%d.\t%f\n", i+1, opmLatency[i]);

        // File write
        fprintf(fp, "%d.\t%f\n", i+1, opmLatency[i]);
    }

    // Releasing memory
    fclose(fp);
    free(opmLatency);
    free(matrixA);
    free(matrixB);
}

/* ikj-Algorithm for efficient large matrix multiplication Function, input matrices and return resultant matrix */
double ikj_algorithm(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension){

	int i, j, k, iOff, jOff;
	TYPE tot;

	struct timeval t0, t1;
	gettimeofday(&t0, 0);

	convert(matrixA, matrixB, dimension);
	#pragma omp parallel shared(matrixC) private(i, j, k, iOff, jOff, tot) num_threads(50)
	{
		#pragma omp for schedule(static)
		for(i=0; i<dimension; i++){
			iOff = i * dimension;
			for(j=0; j<dimension; j++){
				jOff = j * dimension;
				tot = 0;
				for(k=0; k<dimension; k++){
					tot += OneD_A[iOff + k] * OneD_B[jOff + k];
				}
				matrixC[i][j] = tot;
			}
		}
        //Below line prints the result of Multiplication of Matrix
        //printf("%f ",tot);
	}

	/* This will generate time taken by algorithm in order to multiply N dimension */
	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;
	return elapsed;
}

//Test function to generate N dimension matrix to test efficiency in ikj-algorithm
void ikj_algorithmTest(int dimension, int iterations){
	FILE* fp;
	fp = fopen("Results/ikj_algorithmTest.txt", "a+");

	// Console write
    printf("..................................\n");
	printf("Dimension : %d\n", dimension);
	
	// File write
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Test : ikj-Algorithm Matrix Multiplication\n");
	fprintf(fp, "----------------------------------\n");
	fprintf(fp, "Dimension : %d\n", dimension);
	fprintf(fp, "..................................\n");

	double* opmLatency = malloc(iterations * sizeof(double));
	TYPE** matrixA = randomSquareMatrix(dimension);
	TYPE** matrixB = randomSquareMatrix(dimension);
	
	// Iterate and measure performance
	for(int i=0; i<iterations; i++){
		TYPE** matrixResult = zeroSquareMatrix(dimension);
		opmLatency[i] = ikj_algorithm(matrixA, matrixB, matrixResult, dimension);
		free(matrixResult);

		// Console write
		printf("%d.\t%f\n", i+1, opmLatency[i]);

		// File write
		fprintf(fp, "%d.\t%f\n", i+1, opmLatency[i]);
	}

	// Releasing memory
	fclose(fp);
	free(opmLatency);
	free(matrixA);
	free(matrixB);
}
//Main Function Start point of program
int main(int argc, char* argv[]){
    int iterations = strtol(argv[1], NULL, 10);
    int start_2D = strtol(argv[2], NULL, 10);
    int end_2D = strtol(argv[3], NULL, 10);
    int interval_2D = strtol(argv[4], NULL, 10);
    
    // Create Traditional Multiplication test log
    FILE* fp;
    fp = fopen("Results/TraditionalMultiplyTest.txt", "w+");
    fclose(fp);

    // Create ijk algorithm  test log
    fp = fopen("Results/ijk_algorithmTest.txt", "w+");
    fclose(fp);

    // Create ikj algorithm test log file
    fp = fopen("Results/ikj_algorithmTest.txt", "w+");
    fclose(fp);

    printf("----------------------------------\n");
    printf("Test : ikj-Algorithm Matrix Multiplication\n");
    printf("----------------------------------\n");
    for(int dimension=start_2D; dimension<=end_2D; dimension+=interval_2D){
        ikj_algorithmTest(dimension, iterations);
    }
    
    printf("----------------------------------\n");
    printf("Test : ijk-Algorithm Matrix Multiplication\n");
    printf("----------------------------------\n");
    for(int dimension=start_2D; dimension<=end_2D; dimension+=interval_2D){
        ijk_algorithmTest(dimension, iterations);
    }
    
    printf("----------------------------------\n");
    printf("Test : Traditional Multiplication\n");
    printf("----------------------------------\n");
    for(int dimension=start_2D; dimension<=end_2D; dimension+=interval_2D){
        TraditionalMultiplyTest(dimension, iterations);
    }

    return 0;
}
