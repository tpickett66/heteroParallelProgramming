// MP 1
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include	<wb.h>

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if(idx < len) {
    out[idx] = in1[idx] + in2[idx];
  }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    wbTime_start(GPU, "Allocating GPU memory.");

    cudaMalloc((void **) &deviceInput1, &inputLength);
    cudaMalloc((void **) &deviceInput2, &inputLength);
    cudaMalloc((void **) &deviceOutput, &inputLength);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceInput1, hostInput1, inputLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    int blockSize = 256;
    struct dim3 DimGrid((inputLength - 1)/blockSize + 1, 1, 1);
    struct dim3 DimBlock(blockSize, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    vecAdd<<DimGrid, DimBlock>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostOutput, deviceOutput, inputLength, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}


