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

#include "DeriveGradient.hpp"

__global__
void
gradientKerX(float *I, float *Ix, int3 imSize)
{
	int2 addr;
	unsigned int offset;
	unsigned int moffset;
	addr.x = blockIdx.x * (blockDim.x-2) + threadIdx.x-1;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	__shared__ float data[ROW_W+2];

	
	offset = (addr.x) + imSize.z*(addr.y);
	moffset = threadIdx.x;
	data[moffset]=(addr.x>=0&&addr.x<imSize.x)?I[offset]:0.;
	__syncthreads();
	if(addr.x < imSize.x && addr.y < imSize.y){
		if(threadIdx.x >0 &&  threadIdx.x<blockDim.x-1)
		{
			Ix[offset]=(data[moffset+1]-data[moffset-1])/2;
		}else{
			if(addr.x == 0 ) 
				Ix[offset]=(data[moffset+1]-data[moffset]);
			else if (addr.x == imSize.x-1)
				Ix[offset]=(data[moffset]-data[moffset-1]);
		}
	}
}


__global__
void
gradientKerY(float *I, float *Iy, int3 imSize)
{
	int2 addr;
	unsigned int offset;
	unsigned int moffset;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * (blockDim.y-2) + threadIdx.y-1;
	__shared__ float data[CB_TILE_W* (CB_TILE_H+2)];

	offset = (addr.x) + imSize.z*(addr.y);
	moffset = threadIdx.x+blockDim.x * threadIdx.y;
	data[moffset]=(addr.y>=0&&addr.y<imSize.y)?I[offset]:0.;
	__syncthreads();

	if(addr.x < imSize.x && addr.y < imSize.y){
		if(threadIdx.y >0 &&  threadIdx.y<blockDim.y-1)
		{
			Iy[offset]=(data[moffset+blockDim.x]-data[moffset-blockDim.x])/2;
		}else{
			if(addr.y == 0)
				Iy[offset]=data[moffset+blockDim.x]-data[moffset];
			else if (addr.y == imSize.y-1)
				Iy[offset]=data[moffset]-data[moffset-blockDim.x];
		}
	}
}



void
gradient(	float *I, float* Idx, float * Idy,int3 imSize)
{
	dim3 gridRows(iDivUp(imSize.x, ROW_W), imSize.y);
	dim3 threadsRows(ROW_W+2);
	dim3 gridCol(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
	dim3 threadsCol(CB_TILE_W,CB_TILE_H+2);

	gradientKerX<<<gridRows,threadsRows>>>(I,Idx,imSize);
	gradientKerY<<<gridCol,threadsCol>>>(I,Idy,imSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

}


