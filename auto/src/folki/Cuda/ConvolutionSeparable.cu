/*
      This file is part of FolkiGpu.

    FolkiGpu is free software: you can redistribute it and/or modify
    it under the terms of the GNU Leeser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Leeser General Public License
    along with FolkiGpu.  If not, see <http://www.gnu.org/licenses/>.

*/

/*
      FolkiGpu is a demonstration software developed by Aurelien Plyer during
    his phd at Onera (2008-2011). For more information please visit :
      - http://www.onera.fr/dtim-en/gpu-for-image/folkigpu.php
      - http://www.plyer.fr (author homepage)
*/

#include "ConvolutionSeparable.hpp"

__device__ __constant__ float kernel[MAX_KERNEL_RADIUS*2+1];

template<int i> __device__ float convolutionRow(float *data){
    return
        data[MAX_KERNEL_RADIUS - i] * kernel[i]
        + convolutionRow<i - 1>(data);
}

template<> __device__ float convolutionRow<-1>(float *data){
    return 0;
}

template<int i> __device__ float convolutionColumn(float *data){
    return 
        data[(MAX_KERNEL_RADIUS - i) * COL_W] * kernel[i]
        + convolutionColumn<i - 1>(data);
}

template<> __device__ float convolutionColumn<-1>(float *data){
    return 0;
}

__global__ void convolutionRowGPU(
	float *d_Result,
	float *d_Data,
		unsigned int kernelRadius,    
		int3 imSize)
{
	extern __shared__ float data[];
	const int         tileStart = IMUL(blockIdx.x, ROW_W);
	const int           tileEnd = tileStart + ROW_W - 1;
	const int        apronStart = tileStart - kernelRadius;
	const int          apronEnd = tileEnd   + kernelRadius;
	const int    tileEndClamped = min(tileEnd, imSize.x - 1);
	const int apronStartClamped = max(apronStart, 0);
	const int   apronEndClamped = min(apronEnd, imSize.x - 1);
	const int          rowStart = IMUL(blockIdx.y, imSize.z);
	const int apronStartAligned = tileStart - iAlignUp(kernelRadius,ALLIGN_ROW);
	const int           loadPos = apronStartAligned + threadIdx.x;
	const int           smemPos = loadPos - apronStart;
	const int          writePos = tileStart + threadIdx.x;

#ifndef UNROLL_INNER
	float sum ;
#endif

	if(loadPos >= apronStart){
		data[smemPos] = 
			((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
			d_Data[rowStart + loadPos] : 0;
	}
	__syncthreads();

	if(writePos <= tileEndClamped){
		const int smemPos = writePos - apronStart;
#ifdef UNROLL_INNER
		d_Result[rowStart + writePos] = convolutionRow<2 * MAX_KERNEL_RADIUS>(data + smemPos);
#else
		sum=0;
		for(int k = 0; k < kernelRadius*2+1; k++){
			sum += data[smemPos - kernelRadius + k] * kernel[k];
				}
		d_Result[rowStart + writePos] = sum;
#endif
	}
}



__global__ void convolutionColumnGPU(
    float *d_Result,
    float *d_Data,
    int3 imSize,
		unsigned int kernelRadius,
    int smemStride,
    int gmemStride
){
    extern __shared__ float data[];
    const int         tileStart = IMUL(blockIdx.y, COL_H);
    const int           tileEnd = tileStart + COL_H - 1;
    const int        apronStart = tileStart - kernelRadius;
    const int          apronEnd = tileEnd   + kernelRadius;
    const int    tileEndClamped = min(tileEnd, imSize.y - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, imSize.y - 1);
    const int       columnStart = IMUL(blockIdx.x, COL_W) + threadIdx.x;
    int smemPos = IMUL(threadIdx.y, COL_W) + threadIdx.x;
    int gmemPos = IMUL(apronStart + threadIdx.y, imSize.z) + columnStart;
#ifndef UNROLL_INNER
	float sum ;
#endif

    for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y){
        data[smemPos] = 
        ((y >= apronStartClamped) && (y <= apronEndClamped)) ? 
        d_Data[gmemPos] : 0;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }
    __syncthreads();
	smemPos = IMUL(threadIdx.y + kernelRadius, COL_W) + threadIdx.x;
	gmemPos = IMUL(tileStart + threadIdx.y , imSize.z) + columnStart;
	for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y){
#ifdef UNROLL_INNER
		d_Result[gmemPos] = convolutionColumn<2 * MAX_KERNEL_RADIUS>(data + smemPos);
#else
	sum = 0;
	for(int k = 0; k <2*kernelRadius+1; k++){
		sum += 
			data[smemPos + IMUL(k- kernelRadius, COL_W)] *
			kernel[k];
		}
		d_Result[gmemPos] = sum;
#endif
		smemPos += smemStride;
		gmemPos += gmemStride;
	}

}



void
convolutionSeparable(	float *src , float *dest, float *buff, int3 imSize,
											float * kernelRow, float *kernelCol,unsigned int kernelRowRadius, unsigned int kernelColRadius )
{
	dim3 blockGridRows(iDivUp(imSize.x, ROW_W), imSize.y);
	unsigned int  sizeSharedRow;
	unsigned int  kernelSizeByte;
	dim3 threadBlockRows(ROW_W + kernelRowRadius+iAlignUp(kernelRowRadius,ALLIGN_ROW));
	kernelSizeByte = sizeof(float)*(2*kernelRowRadius+1);

	CUDA_SAFE_CALL( cudaMemcpyToSymbol(kernel, kernelRow, kernelSizeByte) );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	sizeSharedRow =	(kernelRowRadius+iAlignUp(kernelRowRadius,ALLIGN_ROW) + ROW_W)*sizeof(float);
	convolutionRowGPU<<<blockGridRows, threadBlockRows,sizeSharedRow>>>(
			buff,
			src,
			kernelRowRadius,
			imSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	dim3 blockGridColumns(iDivUp(imSize.x, COL_W), iDivUp(imSize.y, COL_H));
	size_t sizeSharedCol;
	dim3 threadBlockColumns(COL_W, kernelColRadius);
	kernelSizeByte = sizeof(float)*(2*kernelColRadius+1);
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(kernel, kernelCol, kernelSizeByte) );
	sizeSharedCol=COL_W * (kernelColRadius+COL_H+kernelColRadius)*sizeof(float);
	convolutionColumnGPU<<<blockGridColumns, threadBlockColumns, sizeSharedCol>>>(
			dest,
			buff,
			imSize,
			kernelColRadius,
			COL_W * threadBlockColumns.y,
			imSize.z * threadBlockColumns.y);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

}

