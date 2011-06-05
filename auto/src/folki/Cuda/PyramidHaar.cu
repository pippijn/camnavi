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

#include "PyramidHaar.hpp"


__global__
void
decimationHaarKer(float *src, float *dest, int3 srcSize, int3 destSize)
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

__global__
void
upSampleKer(float *Dest,float *Src, int3 destSize, int3 srcSize)
{
	int2 addr;
	int2 daddr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
  addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	float a;
	
	unsigned int offset;
	unsigned int offa;
	unsigned int offb;
	unsigned int offc;
	unsigned int offd;


	if(addr.x < srcSize.x && addr.y < srcSize.y){
		offset = addr.x+addr.y*srcSize.z;
		daddr.x=2*addr.x;
		daddr.y=2*addr.y;
		if(daddr.x < destSize.x && daddr.y < destSize.y){
			offa = daddr.x+daddr.y*destSize.z;
			offb = offa+1;
			offc = offa +destSize.z;
			offd = offb +destSize.z;

			a=Src[offset]*2.f;

			Dest[offa]=a;
			Dest[offb]=a;
			Dest[offc]=a;
			Dest[offd]=a;
		}
	}
}


/* ======================================== fonction appelantes ======================================== */




void
PyramidDownSampleHaar(float *src , float *dest, int3 srcSize, int3 destSize)
{
	dim3 gridDec(iDivUp(destSize.x, CB_TILE_W), iDivUp(destSize.y, CB_TILE_H));
	dim3 threadsDec(CB_TILE_W, CB_TILE_H);
	decimationHaarKer<<<gridDec,threadsDec>>>(src,dest,srcSize,destSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void
PyramidUpSampleHaar(float *Dest, float *Src, int3 destSize, int3 srcSize)
{
	dim3 grid(iDivUp(srcSize.z, CB_TILE_W), iDivUp(srcSize.y, CB_TILE_H));
	dim3 threads(CB_TILE_W, CB_TILE_H);

	upSampleKer<<<grid,threads>>>(Dest,Src,destSize,srcSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}


