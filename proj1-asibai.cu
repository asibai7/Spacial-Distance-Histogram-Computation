/* ==================================================================
    Programmer: Ahmad Sibai (asibai@usf.edu)
    Project 1: Spacial Distance Histogram Computation (CUDA)
    To compile: nvcc SDH.c -o SDH in the GAIVI machines
   ==================================================================*/
// For test where atoms = 200,000 and bucket range = 2000, CPU time: 540.8, GPU time: 19.3
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>             //header for cudaMalloc, cudaFree, cudaMemcpy
#include <device_launch_parameters.h> //header for blockIdx, blockDim, threadIdx
// cuda_runtime.h and device_launch_parameters.h are not needed because of nvcc, i'm keeping them for practice
#define BOX_SIZE 23000 // size of the 3d data box

typedef struct atomdesc // struct for single atom
{
    double x_pos;
    double y_pos;
    double z_pos;
} atom;

typedef struct hist_entry // struct for histogram bucket
{
    unsigned long long d_cnt; // need a long long type as the count might be huge
} bucket;

// Host(CPU) variables
bucket *histogram; // pointer to array of all buckets in the histogram
long long PDH_acnt; // total number of atoms (data points)
int num_buckets; // total number of buckets in the histogram
double PDH_res; // range of each bucket in histogram
atom *atom_list; // list of all atoms

// Device(GPU) variables
bucket *GPUhistogram; // device memory for histogram buckets (GPU)
bucket *GPUhistogramOnHost; // host memory for histogram buckets (GPU)
double *GPUpositionX, *GPUpositionY, *GPUpositionZ; // pointer to 3 array of all atoms each of which contains one of the values (x, y, z)

// These are for an old way of tracking time
struct timezone Idunno;
struct timeval startTime, endTime;

// calculates distance of two points in the atom_list on Host(CPU)
double p2p_distance(int ind1, int ind2)
{

    double x1 = atom_list[ind1].x_pos;
    double x2 = atom_list[ind2].x_pos;
    double y1 = atom_list[ind1].y_pos;
    double y2 = atom_list[ind2].y_pos;
    double z1 = atom_list[ind1].z_pos;
    double z2 = atom_list[ind2].z_pos;
    return sqrt((x1 - x2) * (x1 - x2) +
                (y1 - y2) * (y1 - y2) +
                (z1 - z2) * (z1 - z2));
}

int PDH_baseline() // brute-force SDH solution in a single CPU thread on Host(CPU)
{
    int i, j, h_pos;
    double dist;

    for (i = 0; i < PDH_acnt; i++)
    {
        for (j = i + 1; j < PDH_acnt; j++)
        {
            dist = p2p_distance(i, j);
            h_pos = (int)(dist / PDH_res);
            histogram[h_pos].d_cnt++;
        }
    }
    return 0;
}

// CUDA kernel to calculate distances and update histogram
__global__ void p2p_distancekernel(double *GPUpositionX, double *GPUpositionY, double *GPUpositionZ, bucket *histogram, long long PDH_acnt, double PDH_res, int num_buckets)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // calculates thread index
    if (i < PDH_acnt) // checks atom index is within bounds
    {
        for (int j = i + 1; j < PDH_acnt; j++) // loops through all atoms that come after current atom
        {
            double x1 = GPUpositionX[i]; // sets all x, y, z values
            double y1 = GPUpositionY[i];
            double z1 = GPUpositionZ[i];
            double x2 = GPUpositionX[j];
            double y2 = GPUpositionY[j];
            double z2 = GPUpositionZ[j];
            double distance = sqrt((x1 - x2) * (x1 - x2) + // calculates distance
                                   (y1 - y2) * (y1 - y2) +
                                   (z1 - z2) * (z1 - z2));
            int h_pos = (int)(distance / PDH_res); // determines which bucket distance value falls into
            if (h_pos < num_buckets) // checks bucket index is within bounds
            { // updates bucket count atomically (ensures concurrent environment because multiple threads might try to update same value simultaneously)
                unsigned long long increment = 1;
                atomicAdd(&histogram[h_pos].d_cnt, increment);
            }
        }
    }
}

void PDH_baselineGPU() // function to compute histogram using GPU
{
    cudaError_t err; // holds error status
    int blockSize = 1024;
    int gridSize = (PDH_acnt + blockSize - 1) / blockSize; // formula to find number of blocks needed
    // allocate memory on the Device(GPU)
    err = cudaMalloc(&GPUhistogram, sizeof(bucket) * num_buckets);
    if (err != cudaSuccess) // check for error
    {
        printf("CUDA error after 1st cudaMalloc: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE); // exits on error
    }
    err = cudaMalloc(&GPUpositionX, sizeof(double) * PDH_acnt);
    if (err != cudaSuccess) {
        printf("CUDA error after 2nd cudaMalloc (X): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&GPUpositionY, sizeof(double) * PDH_acnt);
    if (err != cudaSuccess) {
        printf("CUDA error after 3rd cudaMalloc (Y): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&GPUpositionZ, sizeof(double) * PDH_acnt);
    if (err != cudaSuccess) {
        printf("CUDA error after 4th cudaMalloc (Z): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    double *hostPositionX = (double *)malloc(sizeof(double) * PDH_acnt);
    double *hostPositionY = (double *)malloc(sizeof(double) * PDH_acnt);
    double *hostPositionZ = (double *)malloc(sizeof(double) * PDH_acnt);
    // copy x, y, z positions from atom_list
    for (int i = 0; i < PDH_acnt; i++) {
        hostPositionX[i] = atom_list[i].x_pos;
        hostPositionY[i] = atom_list[i].y_pos;
        hostPositionZ[i] = atom_list[i].z_pos;
    }
    // copy data from Host(CPU) to Device(GPU)
    err = cudaMemcpy(GPUpositionX, hostPositionX, sizeof(double) * PDH_acnt, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error after 1st cudaMemcpy (X): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(GPUpositionY, hostPositionY, sizeof(double) * PDH_acnt, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error after 2nd cudaMemcpy (Y): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(GPUpositionZ, hostPositionZ, sizeof(double) * PDH_acnt, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error after 3rd cudaMemcpy (Z): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemset(GPUhistogram, 0, sizeof(bucket) * num_buckets); // initalize to 0 so when we increment we aren't adding to random values
    if (err != cudaSuccess)
    {
        printf("CUDA error after 4th cudaMemcpy: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // launch kernel
    p2p_distancekernel<<<gridSize, blockSize>>>(GPUpositionX, GPUpositionY, GPUpositionZ, GPUhistogram, PDH_acnt, PDH_res, num_buckets);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // copy results from Device(GPU) to Host(CPU)
    err = cudaMemcpy(GPUhistogramOnHost, GPUhistogram, sizeof(bucket) * num_buckets, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("CUDA error after Memcpy from Device to Host: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // free Device(GPU) memory
    err = cudaFree(GPUhistogram);
    if (err != cudaSuccess)
    {
        printf("CUDA error after 1st cudaFree: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(GPUpositionX);
    if (err != cudaSuccess)
    {
        printf("CUDA error after 2nd cudaFree: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(GPUpositionY);
    if (err != cudaSuccess)
    {
        printf("CUDA error after 3rd cudaFree: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(GPUpositionZ);
    if (err != cudaSuccess)
    {
        printf("CUDA error after 4th cudaFree: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// set a checkpoint and show the (natural) running time in seconds
double report_running_time(const char *label)
{
    long sec_diff, usec_diff;
    gettimeofday(&endTime, &Idunno);
    sec_diff = endTime.tv_sec - startTime.tv_sec;
    usec_diff = endTime.tv_usec - startTime.tv_usec;
    if (usec_diff < 0)
    {
        sec_diff--;
        usec_diff += 1000000;
    }
    printf("Running time for %s version: %ld.%06ld\n", label, sec_diff, usec_diff);
    return (double)(sec_diff * 1.0 + usec_diff / 1000000.0);
}

// print the counts in all buckets of the histogram
void output_histogram(bucket *histogram, const char *label)
{
    int i;
    long long total_cnt = 0;
    for (i = 0; i < num_buckets; i++)
    {
        if (i % 5 == 0) // we print 5 buckets in a row 
            printf("\n%02d: ", i);
        printf("%15lld ", histogram[i].d_cnt);
        total_cnt += histogram[i].d_cnt;
        // we also want to make sure the total distance count is correct
        if (i < num_buckets - 1)
            printf("| ");
    }
    if (strcmp(label, "Compute Difference") != 0)
    {
        printf("\nTotal number of distances calculated: %lld for %s\n", total_cnt, label);
    }
}

//compute differences between histogram (CPU) and GPUhistogramOnHost
int computeDifference(bucket *histogram, bucket *GPUhistogramOnHost, bucket *differenceHistogram, int num_buckets) {
    int hasDifference = 0; // used track if there's any difference greater than 1
    for (int i = 0; i < num_buckets; i++) {
        differenceHistogram[i].d_cnt = llabs(histogram[i].d_cnt - GPUhistogramOnHost[i].d_cnt); // calculate the absolute difference
        if (differenceHistogram[i].d_cnt > 1) { // check if the difference is greater than 1
            hasDifference = 1;
        }
    }
    return hasDifference; // return 0 if no differences, or 1 if any difference exists
}

int main(int argc, char **argv)
{
    int i;
    if (argc < 3) // check command line arguments are provided
    {
        printf("Usage: %s <atomCount> <bucketRange>\n", argv[0]);
        return 1;
    }
    PDH_acnt = atoi(argv[1]); // number of atoms
    PDH_res = atof(argv[2]);  // range of each bucket
    printf("args are Atom Count: %d and Bucket Range: %f\n", PDH_acnt, PDH_res);
    num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
    histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
    atom_list = (atom *)malloc(sizeof(atom) * PDH_acnt);
    GPUhistogramOnHost = (bucket *)malloc(sizeof(bucket) * num_buckets); // host memory to store GPU histogram
    bucket *differenceHistogram = (bucket *)malloc(sizeof(bucket) * num_buckets); // host memory to store GPU histogram
    // allocate memory on Host(CPU)
    srand(1);
    // generate data following a uniform distribution
    for (i = 0; i < PDH_acnt; i++)
    {
        atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
        atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
        atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
    }
    // start counting Host(CPU) time
    gettimeofday(&startTime, &Idunno);
    // call Host(CPU) single thread version to compute the histogram
    PDH_baseline();
    // print out the Host(CPU) histogram
    output_histogram(histogram, "CPU");
    // check the total running time of Host(CPU) code
    report_running_time("CPU");

    // start counting Device(GPU) time
    gettimeofday(&startTime, &Idunno);
    // Call Device(GPU) multithreaded version to compute the histogram
    PDH_baselineGPU();
    // print out the Device(GPU) histogram
    output_histogram(GPUhistogramOnHost, "GPU");
    // check the total running time of Device(GPU) code
    report_running_time("GPU");
    // compute difference
    int isdifferent = computeDifference(histogram, GPUhistogramOnHost, differenceHistogram, num_buckets);
    if(isdifferent == 0) // if there are no differences
    {
        printf("The differenceHistogram was empty, meaning the CPU histogram and GPU histogram were identical!\n");
    }
    else // if there are differences then we print out the computeDifference histogram to show the buckets where their are differences
    {
        printf("The CPU and GPU histograms had differences in their bucket values. The following printed differenceHistogram shows these differences:\n");
        output_histogram(differenceHistogram, "Compute Difference");
    }
    free(atom_list); //free all dynamically allocated memory
    free(histogram);
    free(GPUhistogramOnHost);
    free(differenceHistogram);
    return 0;
}