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

#include "DivRot.hpp"




__global__
void
divRotKerX(float *u, float *v,float *Div, float *Rot, int3 imSize,int ordre)
{
	int2 addr;
	unsigned int offset;
	unsigned int moffset;
	addr.x = blockIdx.x * (blockDim.x-2*ordre) + threadIdx.x-ordre;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	extern __shared__ float data[];
	float *dataRot = (float *)data;
	float *dataDiv = (float *) &data[blockDim.x];
	
	offset = (addr.x) + imSize.z*(addr.y);
	moffset = threadIdx.x;
	dataDiv[moffset]=(addr.x>=0&&addr.x<imSize.x)?u[offset]:0.;
	dataRot[moffset]=(addr.x>=0&&addr.x<imSize.x)?v[offset]:0.;
	__syncthreads();
	if(addr.x < imSize.x && addr.y < imSize.y){
		if(threadIdx.x >= ordre &&  threadIdx.x<blockDim.x-ordre){
			Div[offset]=(dataDiv[moffset+ordre]-dataDiv[moffset-ordre])/(2*ordre);
			Rot[offset]=(dataRot[moffset+ordre]-dataRot[moffset-ordre])/(2*ordre);
		}
	}
}


__global__
void
divRotKerRotY(float *u, float *Rot, int3 imSize, int ordre)
{
	int2 addr;
	unsigned int offset;
	unsigned int moffset;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * (blockDim.y-2*ordre) + threadIdx.y-ordre;
	extern __shared__ float dataRot[];
	
	offset = (addr.x) + imSize.z*(addr.y);
	moffset = threadIdx.x+blockDim.x * threadIdx.y;
	dataRot[moffset]=(addr.y>=0&&addr.y<imSize.y)?u[offset]:0.;
	__syncthreads();

	if(addr.x < imSize.x && addr.y < imSize.y){
		if(threadIdx.y >= ordre&&  threadIdx.y<blockDim.y-ordre)
		{
			Rot[offset]-=(dataRot[moffset+blockDim.x*ordre]-dataRot[moffset-blockDim.x*ordre])/(2*ordre);
		}
	}
}

__global__
void
divRotKerDivY(float *v,float *Div,int3 imSize,int ordre)
{
	int2 addr;
	unsigned int offset;
	unsigned int moffset;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * (blockDim.y-2*ordre) + threadIdx.y-ordre;
	extern __shared__ float dataDiv[];

	
	offset = (addr.x) + imSize.z*(addr.y);
	moffset = threadIdx.x+blockDim.x * threadIdx.y;
	dataDiv[moffset]=(addr.y>=0&&addr.y<imSize.y)?v[offset]:0.;
	__syncthreads();

	if(addr.x < imSize.x && addr.y < imSize.y){
		if(threadIdx.y >= ordre &&  threadIdx.y<blockDim.y-ordre)
		{
			Div[offset]+=(dataDiv[moffset+blockDim.x*ordre]-dataDiv[moffset-blockDim.x*ordre])/(2*ordre);
		}
	}
}



__global__
void
seuilNormKer(float *u, float *v, int3 imSize, float seuil)
{
	int2 addr;
	unsigned int offset;
	float tu,tv,n;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z*addr.y+addr.x;
		tu = u[offset];
		tv = v[offset];
		n = sqrtf(tu*tu+tv*tv);
		if(n < seuil){
			u[offset] = 0.0f;
			v[offset] = 0.0f;
		}
	}
}


__global__
void
seuilNormMaskKer(float *u, float *v,float *mask, int3 imSize, float seuil)
{
	int2 addr;
	unsigned int offset;
	float tu,tv,n;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z*addr.y+addr.x;
		tu = u[offset];
		tv = v[offset];
		n = sqrtf(tu*tu+tv*tv);
		if(n < seuil){
			mask[offset] = 0.0f;
		}else{
			mask[offset] = 1.0f;
		}
	}
}


__global__
void
rotMaskKerY(float *u,float *v, float *Rot,float *maskX, int3 imSize, int ordre, float seuilN)
{
	int2 addr;
	unsigned int offset,offseti,offsets;
	float ti, ts, mx,r;
	float tvi,tvs,tni,tns;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x+ordre;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y+ordre;	
	if(addr.x < imSize.x-ordre && addr.y < imSize.y-ordre){
		offseti = (addr.x) + imSize.z*(addr.y-ordre);
		offsets = (addr.x) + imSize.z*(addr.y+ordre);
		offset = (addr.x) + imSize.z*(addr.y);
		ti = u[offseti];
		ts = u[offsets];
		tvi = v[offseti];
		tvs = v[offsets];
		tni = sqrtf(ti*ti+tvi*tvi);
		tns = sqrtf(ts*ts+tvs*tvs);
		mx = maskX[offset];
		if(tni > seuilN && tns > seuilN && mx !=0.0f){
			r = Rot[offset];
			Rot[offset] =  r +  ( ts - ti ) / (2*ordre);
		}else{
			Rot[offset] = 0.0f;
		}
	}
}


__global__
void
rotMaskKerX(float *v, float *u, float *Rot, float *maskX, int3 imSize, int ordre,float seuilN)
{
	int2 addr;
	unsigned int offset,offseti,offsets;
	float ti, ts, mi, ms;
	float tui,tus,tni,tns;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x+ordre;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y+ordre;	
	if(addr.x < imSize.x-ordre && addr.y < imSize.y-ordre){
		offseti = (addr.x) + imSize.z*addr.y-ordre;
		offsets = (addr.x) + imSize.z*addr.y+ordre;
		offset = (addr.x) + imSize.z*(addr.y);
		ti = v[offseti];
		ts = v[offsets];
		tui = u[offseti];
		tus = u[offsets];
		tni = sqrtf(ti*ti+tui*tui);
		tns = sqrtf(ts*ts+tus*tus);
		if(tni > seuilN && tns > seuilN ){
			Rot[offset]= (ti-ts)/(2*ordre);
			maskX[offset] = 1.0f;
		}else{
			maskX[offset] = 0.0f;
		}
	}
}


/*              APPELANTES                     */


void
calculRotGPU(float *u, float *v, float *Rot, float *buff1, int3 imSize, int ordre, float nSeuil)
{

	// calcul de la convolution en dx
	dim3 gridX(iDivUp(imSize.x-(2*ordre), ROW_W), imSize.y-(2*ordre));
	dim3 threadsX(ROW_W);
	rotMaskKerX<<<gridX,threadsX>>>(v,u,Rot,buff1,imSize,ordre,nSeuil);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );


	// calcul de a confolution en dy
	dim3 gridY(iDivUp(imSize.x-(2*ordre), CB_TILE_W), iDivUp(imSize.y-2*ordre, CB_TILE_H));
	dim3 threadsY(CB_TILE_W,CB_TILE_H);
	rotMaskKerY<<<gridY,threadsY>>>(u,v,Rot,buff1,imSize,ordre,nSeuil);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	

}














void
seuilNorm(float *u, float *v,int3 imSize,float nSeuil)
{
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
	dim3 threads(CB_TILE_W, CB_TILE_H);
	seuilNormKer<<<grid,threads>>>(u,v,imSize,nSeuil);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void
DivRot(float *u, float *v,float *Div, float *Rot,int3 imSize, int ordre, float nSeuil)
{
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
	dim3 threads(CB_TILE_W, CB_TILE_H);
	seuilNormKer<<<grid,threads>>>(u,v,imSize,nSeuil);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );


	dim3 gridRows(iDivUp(imSize.x, ROW_W), imSize.y);
	dim3 threadsRows(ROW_W+(2*ordre));
	size_t sizeRow =( ROW_W+(2*ordre));
	sizeRow *= sizeof(float);
	sizeRow *=2;
	divRotKerX<<<gridRows,threadsRows,sizeRow>>>(u,v,Div,Rot,imSize,ordre);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );


	dim3 gridCol(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
	dim3 threadsCol(CB_TILE_W,CB_TILE_H+(2*ordre));
	size_t sizeCol = (CB_TILE_W) * (CB_TILE_H+2*ordre);
	sizeCol *= sizeof(float);
	divRotKerDivY<<<gridCol,threadsCol,sizeCol>>>(v,Div,imSize,ordre);
	divRotKerRotY<<<gridCol,threadsCol,sizeCol>>>(u,Rot,imSize,ordre);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

}

