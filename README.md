# Spacial-Distance-Histogram-Computation

This project implements a CUDA-accelerated computation of a histogram that categorizes distances between atoms in a 3D space. The program computes distances between each pair of atoms and increments the appropriate histogram bucket, leveraging parallel processing on the GPU to improve performance over the CPU-only version. It measures and compares the execution times of the CPU and GPU versions, highlighting the performance difference. Additionally, it checks for any discrepancies between the CPU and GPU-generated histograms.

Requirements
To compile and run the program, you need access to a CUDA-enabled machine. Ensure you have nvcc installed to compile CUDA code.

To install and use:
1. Clone the repository:
```
git clone https://github.com/asibai7/Spacial-Distance-Histogram-Computation.git
cd Spacial-Distance-Histogram-Computation
```
2. Compile the code using nvcc on a CUDA-enabled machine:
```
nvcc SDH.c -o SDH
```
3. Run program:
```
./SDH <atomCount> <bucketRange>
```
Example: 
```
./SDH 200000 2000
```
With 200,000 atoms and a bucket range of 2,000:
- **CPU time**: 540.8 seconds
- **GPU time**: 19.3 seconds
-**<atomCount>**: The number of atoms to generate in 3D space.  
-**<bucketRange>**: The range for each bucket in the histogram.



