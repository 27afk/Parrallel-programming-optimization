#define TILE_SIZE 16

__kernel void matrixMultiply(__global const float* A,  __global const float* B, __global float* C, const int N) 
{
    __local float boxA[2][TILE_SIZE][TILE_SIZE]; // 2 SRAM buffers for A
    __local float boxB[2][TILE_SIZE][TILE_SIZE]; // 2 SRAM buffers for B
    float sum = 0.0f;

    int row = get_global_id(1); 
    int col = get_global_id(0);

    int localRow = get_local_id(1);
    int localCol = get_local_id(0);
    int numTiles = N / TILE_SIZE;

    bool currBox = 0; // Read buffer index

    boxA[0][localRow][localCol] = A[row * N + (0 * TILE_SIZE + localCol)];
    boxB[0][localRow][localCol] = B[(0 * TILE_SIZE + localRow) * N + col];

    barrier(CLK_LOCAL_MEM_FENCE);

    // main loop for matrix multiplcaiton
    for (int tileStep = 0; tileStep < numTiles; ++tileStep) {
        bool currWriteBox = currBox ^ 1; // Opposite buffer index for fetching
        // copy form ram to sram
        if (tileStep < numTiles - 1) {
            boxA[currWriteBox][localRow][localCol] = A[row * N + ((tileStep + 1) * TILE_SIZE + localCol)];
            boxB[currWriteBox][localRow][localCol] = B[((tileStep + 1) * TILE_SIZE + localRow) * N + col];
        }
        #pragma unroll
        // compute
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += boxA[currBox][localRow][k] * boxB[currBox][k][localCol];
        }
        // wait for threads to finish
        barrier(CLK_LOCAL_MEM_FENCE);

        currBox ^= 1;
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}