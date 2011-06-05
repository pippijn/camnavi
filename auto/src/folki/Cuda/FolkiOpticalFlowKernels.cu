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

#include "FolkiOpticalFlowKernels.hpp"

static texture<float,2> texI;


__global__
void
calculTensorKer( float *Ix,float *Iy, /*INPUT*/
							float *A,float *B,float *C,/*OUTPOUT*/
							int3 imSize)
{
	int2 addr;
	float tx,ty;
	unsigned int offset;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
 	addr.y = blockIdx.y * blockDim.y + threadIdx.y;

	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z*addr.y+addr.x;
		tx=Ix[offset];
		ty=Iy[offset];
		A[offset]=tx*tx;
		B[offset]=ty*ty;
		C[offset]=tx*ty;

	}
}

__global__
void
calculDenominateurKer( float *A,float *B,float *C, /*INPUT*/
							float *D, float * lMin,/*OUTPOUT*/
							int3 imSize)
{
	int2 addr;
	float a,b,c;
	unsigned int offset;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;

	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z*addr.y+addr.x;
		c=C[offset];
		a=A[offset];
		b=B[offset];
		lMin[offset] = ((a+b)-__fsqrt_rn((a-b)*(a-b)+4.0f*c*c))/2.0f ;
		D[offset]=(a)*(b)-c*c;
	}
}

__global__
void
resoudSystemKer( float *A,float *B,float *C,float *D,float *G,float *H, /*INPUT*/
				float *u,float *v, /*OUTPOUT*/
				float *lMin, float talon, bool useLmin, float slmin,
				int3 imSize)
{
	int2 addr;
	float d;
	unsigned int offset;
	float tu,tv;
	float a,b,c,g,h;
	float tlmin;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;

	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z*addr.y+addr.x;
		d=D[offset];
		a=A[offset];
		b=B[offset];
		c=C[offset];
		g=G[offset];
		h=H[offset];
		if(useLmin){
			tlmin = lMin[offset];
			tu=(tlmin>slmin)?(g*b-c*h)/d:0.0f;
			tv=(tlmin>slmin)?(a*h-c*g)/d:0.0f;
		}else{
			a+=talon;
			b+=talon;
			tu=(g*b-c*h)/d;
			tv=(a*h-c*g)/d;
		}
		u[offset]=(isnan(tu)||isinf(tu))?0.:tu;
		v[offset]=(isnan(tv)||isinf(tv))?0.:tv;
	}
}


__global__
void
calculDroiteKer(float *I1,float *u,float *v,float *Ix,float *Iy,
				 float *G, float *H,//float *I2w,
				 int3 imSize)
{
	int2 addr;
	float ix,iy;
	float2 pts;
	float du,dv;
	float i1;
	float i2w;
	float it;
	unsigned int offset;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;

	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z*addr.y+addr.x;
		ix=Ix[offset];
		iy=Iy[offset];
		du=u[offset];
		dv=v[offset];
		i1=I1[offset];
		pts.x=(float)(addr.x)+du;
		pts.y=(float)(addr.y)+dv;
		pts.x=MAX(0,MIN(pts.x+0.5,(float)imSize.x-1.));
		pts.y=MAX(0,MIN(pts.y+0.5,(float)imSize.y-1.));
		i2w=tex2D(texI,pts.x,pts.y);
		it=i1-i2w+du*ix+dv*iy;
		G[offset]=ix*it;
		H[offset]=iy*it;
	}
}


__global__
void
swapKer(float *a,float *b, int3 imSize)
{
	int2 addr;
	unsigned int offset;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	float ta,tb;
	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z*addr.y+addr.x;
		ta = a[offset];
		tb  = b[offset];
		b[offset] = ta;
		a[offset] = tb;
	}
}

__global__
void
calculItKer(float *I1,float *u,float *v,float *Ix,float *Iy,
				 float *It,//float *I2w,
				 int3 imSize)
{
	int2 addr;
	float ix,iy;
	float2 pts;
	float du,dv;
	float i1;
	float i2w;
	float it;
	unsigned int offset;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;

	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z*addr.y+addr.x;
		ix=Ix[offset];
		iy=Iy[offset];
		du=u[offset];
		dv=v[offset];
		i1=I1[offset];
		pts.x=(float)(addr.x)+du;
		pts.y=(float)(addr.y)+dv;
		pts.x=MAX(0,MIN(pts.x+0.5,(float)imSize.x-1.));
		pts.y=MAX(0,MIN(pts.y+0.5,(float)imSize.y-1.));
		i2w=tex2D(texI,pts.x,pts.y);
		it=i1-i2w+du*ix+dv*iy;
		It[offset]=it;
	}
}

__global__ void gestionBordW(float * A,  int3 imSize, unsigned int bords,float val)
{
	int2 addr;
	unsigned int offset;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	if(addr.x < imSize.x && addr.y < imSize.y ){
		if(addr.x < bords ){
			offset = imSize.z*addr.y+addr.x;
			A[offset]=val;
			offset = imSize.z*addr.y+imSize.x-addr.x-1;
			A[offset]=val;
		}
	}
}


__global__ void gestionBordH(float * A, int3 imSize, int bords,float val)
{
	int2 addr;
	unsigned int offset;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
  addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	if(addr.x < imSize.x && addr.y < imSize.y ){
		if(addr.y < bords ){
			offset = imSize.z*addr.y+addr.x;
			A[offset]=val;
			offset = imSize.z*(imSize.y-addr.y-1)+addr.x;
			A[offset]=val;
		}
	}
}


/* ============================================================================================================*/
/* calcul du tenseur de structure */
void
calculTensor( float *Ix,float *Iy, /*INPUT*/
	      float *A,float *B,float *C, /*OUTPOUT*/
	      int3 imSize, int bords)
{
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
	dim3 threads(CB_TILE_W, CB_TILE_H);

	calculTensorKer<<<grid,threads>>>(Ix,Iy,A,B,C,imSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	if(bords >0){
		dim3 grid(iDivUp(bords, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
		dim3 threads(CB_TILE_W, CB_TILE_H);
		gestionBordW<<<grid,threads>>>(A,imSize, bords,0.);
		gestionBordW<<<grid,threads>>>(B,imSize, bords,0.);
		gestionBordW<<<grid,threads>>>(C,imSize, bords,0.);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		dim3 grid2(iDivUp(imSize.x, CB_TILE_W), iDivUp(bords, CB_TILE_H));
		dim3 threads2(CB_TILE_W, CB_TILE_H);
		gestionBordH<<<grid2,threads2>>>(A,imSize, bords,0.);
		gestionBordH<<<grid2,threads2>>>(B,imSize, bords,0.);
		gestionBordH<<<grid2,threads2>>>(C,imSize, bords,0.);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
	}

		
}

/* calcul le determinant du tenseur */
void
calculDenominateur( float *A,float *B,float *C, /*INPUT*/
			float *D,float *lMin, /*OUTPOUT*/
			int3 imSize, int bords)
{
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
	dim3 threads(CB_TILE_W, CB_TILE_H);

	calculDenominateurKer<<<grid,threads>>>(A,B,C,D,lMin,imSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	if(bords >0){
		dim3 grid(iDivUp(bords, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
		dim3 threads(CB_TILE_W, CB_TILE_H);
		gestionBordW<<<grid,threads>>>(D,imSize, bords,1.);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		dim3 grid2(iDivUp(imSize.x, CB_TILE_W), iDivUp(bords, CB_TILE_H));
		dim3 threads2(CB_TILE_W, CB_TILE_H);
		gestionBordH<<<grid2,threads2>>>(D,imSize, bords,1.);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
	}
}


/*inversion du systeme 2 inconnus*/
void
resoudSystem( float *A,float *B,float *C,float *D,float *G,float *H, /*INPUT*/
							float *u,float *v, /*OUTPOUT*/
							float *lMin, float talon, bool useLmin, float slmin,
							int3 imSize, int bords)
{
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
	dim3 threads(CB_TILE_W, CB_TILE_H);

	resoudSystemKer<<<grid,threads>>>(A,B,C,D,G,H,u,v,lMin,talon,useLmin,slmin,imSize);

	if(bords >0){
		dim3 grid(iDivUp(bords, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
		dim3 threads(CB_TILE_W, CB_TILE_H);
		gestionBordW<<<grid,threads>>>(u,imSize, bords,0.);
		gestionBordW<<<grid,threads>>>(v,imSize, bords,0.);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		dim3 grid2(iDivUp(imSize.x, CB_TILE_W), iDivUp(bords, CB_TILE_H));
		dim3 threads2(CB_TILE_W, CB_TILE_H);
		gestionBordH<<<grid2,threads2>>>(u,imSize, bords,0.);
		gestionBordH<<<grid2,threads2>>>(v,imSize, bords,0.);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
	}
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}


void
calculDroite(	float *I1,float *I2,float *u,float *v,float *Ix,float *Iy,
		float *G, float *H, 
		int3 imSize, int bords)
{
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
  	dim3 threads(CB_TILE_W, CB_TILE_H);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	texI.normalized = false;
  	texI.filterMode = cudaFilterModeLinear;
  	texI.addressMode[0] = cudaAddressModeClamp;
  	texI.addressMode[1] = cudaAddressModeClamp;


	CUDA_SAFE_CALL( cudaBindTexture2D(NULL,&texI, I2, &channelDesc , imSize.x, imSize.y,imSize.z*sizeof(float)) );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	calculDroiteKer<<<grid,threads>>>(I1,u,v,Ix,Iy,G,H,imSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	CUDA_SAFE_CALL( cudaUnbindTexture(texI) );

	if(bords >0){
		dim3 grid(iDivUp(bords, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
		dim3 threads(CB_TILE_W, CB_TILE_H);
		gestionBordW<<<grid,threads>>>(G,imSize, bords,0.);
		gestionBordW<<<grid,threads>>>(H,imSize, bords,0.);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		dim3 grid2(iDivUp(imSize.x, CB_TILE_W), iDivUp(bords, CB_TILE_H));
		dim3 threads2(CB_TILE_W, CB_TILE_H);
		gestionBordH<<<grid2,threads2>>>(G,imSize, bords,0.);
		gestionBordH<<<grid2,threads2>>>(H,imSize, bords,0.);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
	}
}

void
calculIt(float *I1,cudaArray *aI2,float *u,float *v,float *Ix,float *Iy,
		float *It, 
		int3 imSize, int bords)
{
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
  	dim3 threads(CB_TILE_W, CB_TILE_H);

	texI.normalized = false;
  	texI.filterMode = cudaFilterModeLinear;
  	texI.addressMode[0] = cudaAddressModeClamp;
  	texI.addressMode[1] = cudaAddressModeClamp;


	CUDA_SAFE_CALL( cudaBindTextureToArray(texI, aI2) );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	calculItKer<<<grid,threads>>>(I1,u,v,Ix,Iy,It,imSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	CUDA_SAFE_CALL( cudaUnbindTexture(texI) );

	if(bords >0){
		dim3 grid(iDivUp(bords, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
		dim3 threads(CB_TILE_W, CB_TILE_H);
		gestionBordW<<<grid,threads>>>(It,imSize, bords,0.);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		dim3 grid2(iDivUp(imSize.x, CB_TILE_W), iDivUp(bords, CB_TILE_H));
		dim3 threads2(CB_TILE_W, CB_TILE_H);
		gestionBordH<<<grid2,threads2>>>(It,imSize, bords,0.);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
	}
}



void
swapMat(float* a, float *b, int3 imSize)
{
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
  	dim3 threads(CB_TILE_W, CB_TILE_H);
	swapKer<<<grid,threads>>>( a,b, imSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}


