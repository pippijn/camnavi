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

#include "Reduction.hpp"


__global__
void
sumGPUfKer(float *a,int size, float *res)
{
	__shared__ float mem[MAXTH];
	uint offset;
	offset = blockIdx.x * blockDim.x + threadIdx.x;

	
	// lecture des donnes threads
	mem[threadIdx.x] = (offset < size)?a[offset]:0.0f;
	__syncthreads();

	//reduction du block en log2(BlockSize.x)
	for(int modulo = 1; modulo < blockDim.x ; modulo *= 2)
		if(threadIdx.x % (modulo*2) == 0)
			mem[threadIdx.x] += mem[threadIdx.x+modulo];

	//eriture resultat
	if(threadIdx.x == 0)
		res[blockIdx.x] = mem[0];
}

__global__
void
sumGPUf2Ker(float2 *a,int size, float2 *res)
{
	__shared__ float2 mem[MAXTH];
	uint offset;
	offset = blockIdx.x * blockDim.x + threadIdx.x;

	
	// lecture des donnes threads
	mem[threadIdx.x] = (offset < size)?a[offset]:make_float2(0.0f,0.0f);
	__syncthreads();

	//reduction du block en log2(BlockSize.x)
	for(int modulo = 1; modulo < blockDim.x ; modulo *= 2)
		if(threadIdx.x % (modulo*2) == 0)
			mem[threadIdx.x] = make_float2(mem[threadIdx.x].x+mem[threadIdx.x+modulo].x,
										mem[threadIdx.x].y+mem[threadIdx.x+modulo].y);

	//eriture resultat
	if(threadIdx.x == 0)
		res[blockIdx.x] = mem[0];
}

__global__
void
stdGPUf2Ker(float2 *a,int size, float *res, float2 mu)
{
	__shared__ float mem[MAXTH];
	uint offset;
	offset = blockIdx.x * blockDim.x + threadIdx.x;
	float d,e;
	float2 ta;

	// lecture des donnes threads
	if(offset < size){
		ta =a[offset];
		d=(ta.x-mu.x);
		e=(ta.y-mu.y);
		d*=d;
		e*=e;
		mem[threadIdx.x] = sqrt(e+d);
	}
	__syncthreads();

	//reduction du block en log2(BlockSize.x)
	for(int modulo = 1; modulo < blockDim.x ; modulo *= 2)
		if(threadIdx.x % (modulo*2) == 0)
			mem[threadIdx.x] += mem[threadIdx.x+modulo];

	//eriture resultat
	if(threadIdx.x == 0)
		res[blockIdx.x] = mem[0];
}


__global__
void
sum36GPUfKer(float *a,int size, float *res)
{
	__shared__ float mem[MAXTH];
	uint offset;
	offset = blockIdx.x * blockDim.x + threadIdx.x;
	offset *=36;
	
	// lecture des donnes threads
	#pragma unroll 36
	for(int i=0;i<36;i++)
		mem[threadIdx.x+i] = (offset < size)?a[offset+i]:0.0f;
	__syncthreads();

	//reduction du block en log2(BlockSize.x)
	for(int modulo = 1; modulo < blockDim.x ; modulo *= 2)
		if(threadIdx.x % (modulo*2) == 0){
			#pragma unroll 36
			for(int i=0;i<36;i++)
				mem[threadIdx.x+i] += mem[threadIdx.x+modulo+i];
		}

	//eriture resultat
	if(threadIdx.x <36)
		res[blockIdx.x+threadIdx.x] = mem[threadIdx.x];
}

__global__
void
sum36pGPUfKer(float *a, float *poids, int size, float *res)
{
	__shared__ float mem[MAXTH];
	uint offset;
	offset = blockIdx.x * blockDim.x + threadIdx.x;
	float tp;

	if(offset < size)
		tp = poids[offset];

	offset *=36;
	// lecture des donnes threads
	#pragma unroll 36
	for(int i=0;i<36;i++)
		mem[threadIdx.x+i] = (offset < size)?tp*a[offset+i]:0.0f;
	__syncthreads();

	//reduction du block en log2(BlockSize.x)
	for(int modulo = 1; modulo < blockDim.x ; modulo *= 2)
		if(threadIdx.x % (modulo*2) == 0){
			#pragma unroll 36
			for(int i=0;i<36;i++)
				mem[threadIdx.x+i] += mem[threadIdx.x+modulo+i];
		}

	//eriture resultat
	if(threadIdx.x <36)
		res[blockIdx.x+threadIdx.x] = mem[threadIdx.x];
}

__global__
void
distGPUKer(float *a, float *buff, int size,float mean)
{
	uint offset = blockIdx.x * blockDim.x + threadIdx.x;
	float ta;
	if(offset < size){
		ta = a[offset];
		buff[offset] = fabsf(ta-mean);
	}
}


__global__
void
dist2DGPUKer(float *u, float *v, float *buff, int3 imSize, float meanx, float meany)
{
	uint offset;
	uint2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y;

	float tu,tv;
	float a,b;

	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = addr.x + imSize.z * addr.y;
		tu = u[offset];
		tv = v[offset];
		a = tu-meanx;
		b = tv-meany;
		buff[offset] = sqrtf(a*a+b*b);
	}
}

__global__
void
deleteMeanKer(float *u,float *v,float mu,float mv,int3 imSize)
{
	unsigned int offset;
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	float tu,tv;
	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z*addr.y+addr.x;
		tu = u[offset];
		tv = v[offset];
		u[offset] = tu-mu;
		v[offset] = tv-mv;
	}
}


float
sumGPU(float *a, int size)
{
	float *resPtr1 = 0;
	float *tmpData = a;
	int tmpSize = size;
	float res;

	while(tmpSize != 0){
		dim3 grid(iDivUp(tmpSize, MAXTH));
		dim3 threads(MAXTH);
		CUDA_SAFE_CALL( cudaMalloc( (void **)&resPtr1,sizeof(float)*grid.x) );
		sumGPUfKer<<<grid,threads>>>(tmpData,tmpSize, resPtr1);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		if(a!=tmpData)
			CUDA_SAFE_CALL( cudaFree(tmpData) );
		tmpData = resPtr1;
		tmpSize = (grid.x != 1)?grid.x:0;
		resPtr1 = 0;
	}

	CUDA_SAFE_CALL( cudaMemcpy(&res,tmpData,sizeof(float),cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaFree( tmpData ) );
	return res;
}


float2
meanGPUf2(float2 *a,int size)
{
	float2 *resPtr1 = 0;
	float2 *tmpData = a;
	int tmpSize = size;
	float2 res;

	while(tmpSize != 0){
		dim3 grid(iDivUp(tmpSize, MAXTH));
		dim3 threads(MAXTH);
		CUDA_SAFE_CALL( cudaMalloc( (void **)&resPtr1,sizeof(float2)*grid.x) );
		sumGPUf2Ker<<<grid,threads>>>(tmpData,tmpSize, resPtr1);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		if(a!=tmpData)
			CUDA_SAFE_CALL( cudaFree(tmpData) );
		tmpData = resPtr1;
		tmpSize = (grid.x != 1)?grid.x:0;
		resPtr1 = 0;
	}

	CUDA_SAFE_CALL( cudaMemcpy(&res,tmpData,sizeof(float2),cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaFree( tmpData ) );
	return make_float2(res.x/size,res.y/size);
}



float
stdGPUf2(float2 *a,int size,float2 mu)
{
	float *resPtr1 = 0;
	float2 *tmpData = a;
	float *tmpData2;
	int tmpSize = size;
	float res;

	dim3 grid(iDivUp(tmpSize, MAXTH));
	dim3 threads(MAXTH);
	CUDA_SAFE_CALL( cudaMalloc( (void **)&resPtr1,sizeof(float)*grid.x) );
	stdGPUf2Ker<<<grid,threads>>>(tmpData,tmpSize, resPtr1,mu);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	tmpData2 = resPtr1;
	tmpSize = (grid.x != 1)?grid.x:0;
	resPtr1 = 0;

	while(tmpSize != 0){
		dim3 grid(iDivUp(tmpSize, MAXTH));
		dim3 threads(MAXTH);
		CUDA_SAFE_CALL( cudaMalloc( (void **)&resPtr1,sizeof(float)*grid.x) );
		sumGPUfKer<<<grid,threads>>>(tmpData2,tmpSize, resPtr1);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		CUDA_SAFE_CALL( cudaFree(tmpData2) );
		tmpData2 = resPtr1;
		tmpSize = (grid.x != 1)?grid.x:0;
		resPtr1 = 0;
	}

	CUDA_SAFE_CALL( cudaMemcpy(&res,tmpData,sizeof(float),cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaFree( tmpData ) );
	return res;
}

float
meanGPU(float *a, int size)
{
	return sumGPU(a,size)/size;
}

float
stdGPU(float *a, float *buff, int size, float mean)
{
	dim3 grid(iDivUp(size, MAXTH));
	dim3 threads(MAXTH);
	distGPUKer<<<grid,threads>>>(a,buff,size,mean);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	return meanGPU(buff,size);
}

void
sum36pGPU(float *a, float *poids,int size, float *res)
{
	float *resPtr1 = 0;
	float *tmpData = a;
	int tmpSize = size;

	dim3 grid(iDivUp(tmpSize, MAXTH/36));
	dim3 threads(MAXTH/36);
	CUDA_SAFE_CALL( cudaMalloc( (void **)&resPtr1,sizeof(float)*grid.x*36) );
	sum36pGPUfKer<<<grid,threads>>>(tmpData,poids,tmpSize, resPtr1);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	tmpData = resPtr1;
	tmpSize = (grid.x != 1)?grid.x:0;
	resPtr1 = 0;

	while(tmpSize != 0){
		dim3 grid(iDivUp(tmpSize, MAXTH/36));
		dim3 threads(MAXTH/36);
		CUDA_SAFE_CALL( cudaMalloc( (void **)&resPtr1,sizeof(float)*grid.x*36) );
		sum36GPUfKer<<<grid,threads>>>(tmpData,tmpSize, resPtr1);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		CUDA_SAFE_CALL( cudaFree(tmpData) );
		tmpData = resPtr1;
		tmpSize = (grid.x != 1)?grid.x:0;
		resPtr1 = 0;
	}

	CUDA_SAFE_CALL( cudaMemcpy(res,tmpData,sizeof(float),cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaFree( tmpData ) );
}

float
mean2DGPU(float *a, int3 imSize)
{
	float acc =0;
	for(int i = 0; i< imSize.y;i ++)
		acc += meanGPU(&a[i*imSize.z],imSize.x);
	return acc/(float)imSize.y;
}

float
sum2DGPU(float *a, int3 imSize)
{
	float acc =0;
	for(int i = 0; i< imSize.y;i ++)
		acc += sumGPU(&a[i*imSize.z],imSize.x);
	return acc;
}

void
deleteMean(float *u, float *v, float mu, float mv, int3 imSize)
{
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
	dim3 threads(CB_TILE_W, CB_TILE_H);
	deleteMeanKer<<<grid,threads>>>(u,v,mu,mv,imSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}


float
std2DGPU(float *u,float *v, float *buff, int3 size, float meanx, float meany)
{
	float res;
	dim3 grid(iDivUp(size.x, MAXTH),size.y);
	dim3 threads(MAXTH);
	dist2DGPUKer<<<grid,threads>>>(u,v,buff,size,meanx,meany);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	res=sum2DGPU(buff,size);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	return res/(float)(size.x*size.y);
}

