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

#include "Morpho.hpp"



template<int i> __device__ float sumRowf(float *data){
	return
		data[U_KERNEL_RADIUS - i]
		+ sumRowf<i - 1>(data);
}

template<> __device__ float sumRowf<0>(float *data){
	return data[U_KERNEL_RADIUS];
}


template<int i> __device__ float sumColumnf(float *data){
	return 
		data[(U_KERNEL_RADIUS - i) * COL_W] 
		+ sumColumnf<i - 1>(data);
}

template<> __device__ float sumColumnf<0>(float *data){
	return data[U_KERNEL_RADIUS* COL_W];
}


template<int i> __device__ float maxRowf(float *data){
	return
		fmaxf(data[U_KERNEL_RADIUS - i]
		, maxRowf<i - 1>(data));
}

template<> __device__ float maxRowf<0>(float *data){
	return data[U_KERNEL_RADIUS];
}


template<int i> __device__ float maxColumnf(float *data){
	return 
		fmaxf( data[(U_KERNEL_RADIUS - i) * COL_W] 
		, maxColumnf<i - 1>(data));
}

template<> __device__ float maxColumnf<0>(float *data){
	return data[U_KERNEL_RADIUS* COL_W];
}


template<int i> __device__ float prodRowf(float *data){
	return
		data[U_KERNEL_RADIUS - i]
		* prodRowf<i - 1>(data);
}

template<> __device__ float prodRowf<0>(float *data){
	return data[U_KERNEL_RADIUS];
}


template<int i> __device__ float prodColumnf(float *data){
	return 
		data[(U_KERNEL_RADIUS - i) * COL_W] 
		* prodColumnf<i - 1>(data);
}

template<> __device__ float prodColumnf<0>(float *data){
	return data[U_KERNEL_RADIUS* COL_W];
}

__global__ 
void 
binDilatefKerX(
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
	const int          writePos = tileStart + threadIdx.x;

	if(loadPos >= apronStart){
		const int smemPos = loadPos - apronStart;
		data[smemPos] = 
			((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
			d_Data[rowStart + loadPos] : 0;
	}
	__syncthreads();
	if(writePos <= tileEndClamped){
		const int smemPos = writePos - apronStart;
		d_Result[rowStart + writePos]  = (sumRowf<2 * U_KERNEL_RADIUS>(data + smemPos)>0.0f)?1.0f:0.0f;
	}
}



__global__ 
void 
binDilatefKerY(
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
	int 				smemPos = IMUL(threadIdx.y, COL_W) + threadIdx.x;
	int 				gmemPos = IMUL(apronStart + threadIdx.y, imSize.z) + columnStart;
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
		d_Result[gmemPos] = (sumColumnf<2 *U_KERNEL_RADIUS>(data + smemPos)!=0.0f)?1.0f:0.0f;
		smemPos += smemStride;
		gmemPos += gmemStride;
	}

}


__global__ 
void 
dilatefKerX(
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
	const int          writePos = tileStart + threadIdx.x;

	if(loadPos >= apronStart){
		const int smemPos = loadPos - apronStart;
		data[smemPos] = 
			((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
			d_Data[rowStart + loadPos] : 0;
	}
	__syncthreads();
	if(writePos <= tileEndClamped){
		const int smemPos = writePos - apronStart;
		d_Result[rowStart + writePos]  = maxRowf<2 * U_KERNEL_RADIUS>(data + smemPos);
	}
}



__global__ 
void 
dilatefKerY(
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
	int 				smemPos = IMUL(threadIdx.y, COL_W) + threadIdx.x;
	int 				gmemPos = IMUL(apronStart + threadIdx.y, imSize.z) + columnStart;
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
		d_Result[gmemPos] = maxColumnf<2 *U_KERNEL_RADIUS>(data + smemPos);
		smemPos += smemStride;
		gmemPos += gmemStride;
	}

}



/* ======================================== fonction appelantes ======================================== */



void
binDilatationSeparablef(	float *src , float *dest, float *buff, int3 imSize,
						unsigned int kernelRowRadius, unsigned int kernelColRadius )
{
	dim3 blockGridRows(iDivUp(imSize.x, ROW_W), imSize.y);
	dim3 threadBlockRows(ROW_W + kernelRowRadius+iAlignUp(kernelRowRadius,ALLIGN_ROW));
	unsigned int  sizeSharedRow;

	sizeSharedRow =	(kernelRowRadius*2 + ROW_W)*sizeof(float);
	binDilatefKerX<<<blockGridRows, threadBlockRows,sizeSharedRow>>>(
            buff,
            src,
			kernelRowRadius,
            imSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );


	dim3 blockGridColumns(iDivUp(imSize.x, COL_W), iDivUp(imSize.y, COL_H));
	dim3 threadBlockColumns(COL_W, kernelColRadius);
	size_t sizeSharedCol;
	sizeSharedCol=COL_W * (kernelColRadius+COL_H+kernelColRadius)*sizeof(float);
	binDilatefKerY<<<blockGridColumns, threadBlockColumns, sizeSharedCol>>>(
            dest,
            buff,
			imSize,
			kernelColRadius,
            COL_W * threadBlockColumns.y,
            imSize.z * threadBlockColumns.y);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

}

void
dilatationSeparablef(float *src, float *dest, float *buff, int3 imSize,
					unsigned int kernelRowRadius, unsigned int kernelColRadius )
{
	dim3 blockGridRows(iDivUp(imSize.x, ROW_W), imSize.y);
	dim3 threadBlockRows(ROW_W + kernelRowRadius+iAlignUp(kernelRowRadius,ALLIGN_ROW));
	unsigned int  sizeSharedRow;

	sizeSharedRow =	(kernelRowRadius*2 + ROW_W)*sizeof(float);
	dilatefKerX<<<blockGridRows, threadBlockRows,sizeSharedRow>>>(
            buff,
            src,
			kernelRowRadius,
            imSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );


	dim3 blockGridColumns(iDivUp(imSize.x, COL_W), iDivUp(imSize.y, COL_H));
	dim3 threadBlockColumns(COL_W, kernelColRadius);
	size_t sizeSharedCol;
	sizeSharedCol=COL_W * (kernelColRadius+COL_H+kernelColRadius)*sizeof(float);
	dilatefKerY<<<blockGridColumns, threadBlockColumns, sizeSharedCol>>>(
            dest,
            buff,
			imSize,
			kernelColRadius,
            COL_W * threadBlockColumns.y,
            imSize.z * threadBlockColumns.y);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}