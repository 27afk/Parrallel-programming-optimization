#define TILE_SIZE 16

__kernel void matrixMultiply(__global const float* A, __global const float* B, __global float* C, const int N) 
{
    // Double SRAM buffers for double buffering (boxA: 32x16, boxB: 16x32)
    __local float boxA[2][2 * TILE_SIZE][TILE_SIZE]; 
    __local float boxB[2][TILE_SIZE][2 * TILE_SIZE]; 

    // 4 register accumulators per thread (for a 2x2 output block)
    float c00 = 0.0f;
    float c01 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;
    int groupRow = get_group_id(1) * 32;
    int groupCol = get_group_id(0) * 32;

    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    // Primary row/col coordinate for this thread's top-left cell
    int row = groupRow + localRow;
    int col = groupCol + localCol;

    int numTiles = N / TILE_SIZE;
    bool currBox = 0; // Active read buffer index

    // setup first tile in SRAM
    boxA[0][localRow][localCol]      = A[row * N + localCol];
    boxA[0][localRow + 16][localCol] = A[(row + 16) * N + localCol];

    boxB[0][localRow][localCol]      = B[localRow * N + col];
    boxB[0][localRow][localCol + 16] = B[localRow * N + (col + 16)];

    barrier(CLK_LOCAL_MEM_FENCE);

    // fetch and compute loop
    for (int tileStep = 0; tileStep < numTiles; ++tileStep) {
        bool currWriteBox = currBox ^ 1; // Buffer index for pre-fetching next tile

        if (tileStep < numTiles - 1) {
            int nextTileOffset = (tileStep + 1) * TILE_SIZE;

            boxA[currWriteBox][localRow][localCol]      = A[row * N + (nextTileOffset + localCol)];
            boxA[currWriteBox][localRow + 16][localCol] = A[(row + 16) * N + (nextTileOffset + localCol)];

            boxB[currWriteBox][localRow][localCol]      = B[(nextTileOffset + localRow) * N + col];
            boxB[currWriteBox][localRow][localCol + 16] = B[(nextTileOffset + localRow) * N + (col + 16)];
        }

        // COMPUTE: Register-tiled 2x2 outer-product calculations
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a0 = boxA[currBox][localRow][k];
            float a1 = boxA[currBox][localRow + 16][k];

            float b0 = boxB[currBox][k][localCol];
            float b1 = boxB[currBox][k][localCol + 16];

            c00 += a0 * b0; // Top-Left
            c01 += a0 * b1; // Top-Right
            c10 += a1 * b0; // Bottom-Left
            c11 += a1 * b1; // Bottom-Right
        }

        // wait for pre-fetch to complete before next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
        currBox ^= 1;
    }

    // write accumulators back
    if (row < N && col < N) {
        C[row * N + col]               = c00;
        C[row * N + (col + 16)]        = c01;
        C[(row + 16) * N + col]        = c10;
        C[(row + 16) * N + (col + 16)] = c11;
    }
}