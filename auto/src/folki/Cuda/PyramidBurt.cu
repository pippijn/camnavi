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

#include "PyramidBurt.hpp"


#define convolutionColumnBurt(data) (data)[-2* COL_W]*1./16. + \
	(data)[-1* COL_W]*1./4. + \
	(data)[0]*3./8. + \
	(data)[1* COL_W]*1./4. + \
	(data)[2* COL_W]*1./16.

#define convolutionRowBurt(data) *(data-2)*1./16. + \
	*(data-1)* 1./4. + \
	*(data)* 3./8. + \
	*(data+1)* 1./4. + \
	*(data+2)* 1./16.

__global__ void convolutionRowGPUBurt(
    float *d_Result,
    float *d_Data,
	int3 imSize)
{
	__shared__ float data[(2+iAlignUp(2,ALLIGN_ROW)+ ROW_W)*sizeof(float)];
    const int         tileStart = IMUL(blockIdx.x, ROW_W);
    const int           tileEnd = tileStart + ROW_W - 1;
    const int        apronStart = tileStart - 2;
    const int          apronEnd = tileEnd   + 2;
    const int    tileEndClamped = min(tileEnd, imSize.x - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, imSize.x - 1);
    const int          rowStart = IMUL(blockIdx.y, imSize.z);
	const int apronStartAligned = tileStart - iAlignUp(2,ALLIGN_ROW);
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
        d_Result[rowStart + writePos] = convolutionRowBurt(data + smemPos);
    }
}

__global__ void convolutionColumnGPUBurt(
    float *d_Result,
    float *d_Data,
    int3 imSize,
    int smemStride,
	int gmemStride
){
	__shared__ float data[COL_W * (4+COL_H)*sizeof(float)];
	const int         tileStart = IMUL(blockIdx.y, COL_H);
	const int           tileEnd = tileStart + COL_H - 1;
	const int        apronStart = tileStart - 2;
	const int          apronEnd = tileEnd   + 2;
	const int    tileEndClamped = min(tileEnd, imSize.y - 1);
	const int apronStartClamped = max(apronStart, 0);
	const int   apronEndClamped = min(apronEnd, imSize.y - 1);
	const int       columnStart = IMUL(blockIdx.x, COL_W) + threadIdx.x;
	int smemPos = IMUL(threadIdx.y, COL_W) + threadIdx.x;
	int gmemPos = IMUL(apronStart + threadIdx.y, imSize.z) + columnStart;
	for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y){
		data[smemPos] = 
		((y >= apronStartClamped) && (y <= apronEndClamped)) ? 
		d_Data[gmemPos] : 0;
		smemPos += smemStride;
		gmemPos += gmemStride;
	}
	__syncthreads();
	smemPos = IMUL(threadIdx.y + 2, COL_W) + threadIdx.x;
	gmemPos = IMUL(tileStart + threadIdx.y , imSize.z) + columnStart;
	for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y){
		d_Result[gmemPos] = convolutionColumnBurt(data+smemPos);
		smemPos += smemStride;
		gmemPos += gmemStride;
	}

}


__global__
void
decimationKer(float *src, float *dest, int3 srcSize, int3 destSize)
{
	int2 addr;
	int2 srcAddr;
	unsigned int soffset;
	unsigned int offset;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	float a,b,c,d;

	if(addr.x < destSize.x && addr.y < destSize.y){
		srcAddr.x=2*addr.x;
		srcAddr.y=2*addr.y;
		soffset=srcAddr.x+srcSize.z *srcAddr.y;
		offset = addr.x+destSize.z*addr.y;
		a=src[soffset];
		b=src[soffset+1];
		c=src[soffset+srcSize.z];
		d=src[soffset+srcSize.z+1];
		dest[offset]=(a+b+c+d)/4.;
	}
}


/* ======================================== fonction appelantes ======================================== */
void
PyramidDownSampleBurt(float *src, float *dest, float *buff1, float *buff2,int3 srcSize, int3 destSize)
{

	dim3 blockGridRows(iDivUp(srcSize.x, ROW_W), srcSize.y);
	dim3 threadBlockRows(ROW_W +2+iAlignUp(2,ALLIGN_ROW));
	convolutionRowGPUBurt<<<blockGridRows, threadBlockRows>>>(buff1,src,srcSize);


	dim3 blockGridColumns(iDivUp(srcSize.x, COL_W), iDivUp(srcSize.y, COL_H));
	dim3 threadBlockColumns(COL_W,2);
	convolutionColumnGPUBurt<<<blockGridColumns, threadBlockColumns>>>(buff2,buff1,srcSize,
									COL_W * threadBlockColumns.y,
            								srcSize.z * threadBlockColumns.y);

	dim3 gridDec(iDivUp(destSize.x, CB_TILE_W), iDivUp(destSize.y, CB_TILE_H));
	dim3 threadsDec(CB_TILE_W, CB_TILE_H);
	decimationKer<<<gridDec,threadsDec>>>(buff2,dest,srcSize,destSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}
