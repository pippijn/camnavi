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

#include "RenderGL.hpp"
// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}


__device__ int rgbToInt(float b, float g, float r)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

__device__
float4 blueColormap(float val, float max)
{
	float4 color;
	float n = (3.0f*max/8.0f);
	/* colormap maison (trop de la bombe de balle) */
	color.y=255*((val<2*n)?0.:val/(val-2.*n));
	color.x=255*((val<n)?val/n:1.);
	color.z=255*((val<n)?0.:(val>2*n)?1.:(float)(val-n)/n);
	color.w = 1.0;
	return color;
}


__device__
float4 hsv2rgb(float h, float s, float v)
{
	h = MAX(h, 0);
	h = fmodf(h, 360.0);
	
	s = MAX(0, MIN(s, 1.0));
	v = MAX(0, MIN(v, 1.0));
	
	int hi = int(h/60)%6;
	float f = h/60.0 - hi;
	float p = v*(1 - s);
	float q = v*(1 - f*s);
	float t = v*(1 - (1 - f)*s);	
	v *= 255;  
	q *= 255;
	p *= 255;
	t *= 255;
	switch(hi) {
		case 0: return make_float4(v,t,p,1.0);
		case 1: return make_float4(q,v,p,1.0);
		case 2: return make_float4(p,v,t,1.0);
		case 3: return make_float4(p,q,v,1.0);
		case 4: return make_float4(t,p,v,1.0);
		case 5: return make_float4(v,p,q,1.0);
		default: return make_float4(0,0,0,1.0);
	}
}

__device__
float4 middleburryColormap(float u, float v,float n, float min, float max)
{
	float ch; // [0, 360]
	float cs; // [0, 1]
	float cv = 1.0f;// [0, 1]
	float4 col;
	cs = (n-min)/(max-min);
	ch = (atan2f(u/n, v/n)+3.14159265)*360/(2*3.14159265);
	
	col = hsv2rgb(ch,cs,cv);
	return col;
}


__device__
float4 middleburryColormapLmin(float u, float v,float lmin, float n, float min, float max,float clmin)
{
	float ch; // [0, 360]
	float cs; // [0, 1]
	float cv = 1.0f;// [0, 1]
	float4 col;
	cs = (n-min)/(max-min);
	ch = (atan2f(u/n, v/n)+3.14159265)*360/(2*3.14159265);
	cv=(lmin < clmin)?0.0f:1.0f;
	col = hsv2rgb(ch,cs,cv);
	return col;
}


__device__
float4 middleburryColormapLmin2(float u, float v,float lmin, float n, float min, float max,float clmin)
{
	float ch; // [0, 360]
	float cs; // [0, 1]
	float cv = 1.0f;// [0, 1]
	float4 col;
	cs = (n-min)/(max-min);
	ch = (atan2f(u/n, v/n)+3.14159265)*360/(2*3.14159265);
	cv=(lmin > clmin)?1.0f:lmin/clmin;
	col = hsv2rgb(ch,cs,cv);
	return col;
}


__device__
float4
HSVsoft(float val, float max)
{
	float4 color;
	if(val > max || val < -max)
	{ 
		color = make_float4(0, 0, 0, 0);
	}else{
			if(val>0.0f){
				float n = sqrt(val) / max;
				float t = 153-(153*sqrt(n));
				color = hsv2rgb(t,sqrt(n),1.0f);
			}else{
				float n = -sqrt(-val) / max;
				float t = 153+(sqrt(-n)*100);
				color = hsv2rgb(t,sqrt(-n),1.0f);
			}
	}
	return color;
}

__device__
float4 hotColormap(float val, float max)
{
	float4 color;
	float n = (3.0f*max/8.0f);
	/* colormap maison (trop de la bombe de balle) */
	color.x=255*((val<2*n)?0.:val/(val-2.*n));
	color.z=255*((val<n)?val/n:1.);
	color.y=255*((val<n)?0.:(val>2*n)?1.:(float)(val-n)/n);
	color.w = 1.0;
	return color;
}


__device__
float4 catVis(float val,float im, float max,float coeff)
{
	float4 color;
	float n = (3.0f*max/8.0f);
	color.z=255*((val<2*n)?0.:val/(val-2.*n));
	color.x=255*((val<n)?val/n:1.);
	color.y=255*((val<n)?0.:(val>2*n)?1.:(float)(val-n)/n);

	color.z=(1.0f-coeff)*color.z+coeff*255*im;
	color.x=(1.0f-coeff)*color.x+coeff*255*im;
	color.y=(1.0f-coeff)*color.y+coeff*255*im;
	color.w = 1.0;
	return color;
}


__device__
float4 grayColormap(float val, float max)
{
	float4 color;
	color.z=255*(val/max);
	color.x=255*(val/max);
	color.y=255*(val/max);
	color.w = 1.0;
	return color;
}

__device__
float4 blueColormapLmin2(float u, float v,float lmin, float p, float min, float max,float clmin)
{

	float4 color;
	float n = (3.0f*max/8.0f);
	float val = logf(sqrt(u*u+v*v));
	val -= min;
	if(lmin == clmin)
		val = min;
	color.x=255*((val<2*n)?0.:val/(val-2.*n));
	color.z=255*((val<n)?val/n:1.);
	color.y=255*((val<n)?0.:(val>2*n)?1.:(float)(val-n)/n);
	color.w = 1.0;
	return color;
}



__global__
void
drawNormKernel(float *u, float *v, float *lmin,int *dest, int3 imSize, int cmap, float max, float min,float *im, float coeff,float clmin)
{
	unsigned int offset, pboOffset;
	float4 col;
	float tu,tv,n,tim;
	float tlmin;
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z*addr.y+addr.x;
		pboOffset = imSize.x*(addr.y)+addr.x;
		tu = u[offset];
		tv = v[offset];
		n = sqrtf(tu*tu+tv*tv)-min;
		switch (cmap){
			case 1:
					col = blueColormap(n,max-min);
					break;
			case 0:
					col = hotColormap(n,max-min);
					break;
			case 2:
					col = grayColormap(n,max-min);
					break;
			case 4:
					col = blueColormap(log2f(n),max-min);
					break;
			case 3:
					col = hotColormap(log2f(n),max-min);
					break;
			case 5:
					col = grayColormap(log2f(n),max-min);
					break;
			case 6:
					col = HSVsoft(n, max-min);
					break;
			case 8:
					col = HSVsoft(log2f(n), max-min);
					break;
			case 7:
					tim = im[offset];
					col = catVis(n,tim,max-min,coeff);
					break;
			case 9:
					col = middleburryColormap(tu,tv,n,min,max);
					break;
			case 10:
					tim = im[offset];
					col = middleburryColormap(tu,tv,n,min,max);
					tim *= coeff*255;
					col.x *= (1-coeff);
					col.y *= (1-coeff);
					col.z *= (1-coeff);

					col.x += tim;
					col.y += tim;
					col.z += tim;
					break;
			case 11:
					tlmin = lmin[offset];
					col = middleburryColormapLmin(tu,tv,tlmin,n,min,max,clmin);
					break;
			case 12:
					tlmin = lmin[offset];
					col = middleburryColormapLmin2(tu,tv,tlmin,n,min,max,clmin);
					break;
			case 13:
					tlmin = lmin[offset];
					col = middleburryColormapLmin2(tu,tv,tlmin,n,min,max,clmin);
					tim = im[offset];
					tim *= coeff*255;
					col.x *= (1-coeff);
					col.y *= (1-coeff);
					col.z *= (1-coeff);

					col.x += tim;
					col.y += tim;
					col.z += tim;

					break;
			case 14:
				tlmin = lmin[offset];
				col = blueColormapLmin2(tu,tv,tlmin,n,min,max,clmin);
					tim = im[offset];
					tim *= coeff*255;
					col.x *= (1-coeff);
					col.y *= (1-coeff);
					col.z *= (1-coeff);

					col.x += tim;
					col.y += tim;
					col.z += tim;
					break;
			default:
					col = hotColormap(n,max-min);
					break;
		}
		dest[pboOffset] = rgbToInt(col.x,col.y,col.z);
	}
}

__global__
void
drawImKernel(float *I, int *dest, int3 imSize)
{
	unsigned int offset, pboOffset;
	float tg;
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z*addr.y+addr.x;
		pboOffset = imSize.x*(addr.y)+addr.x;
		tg = I[offset];
		dest[pboOffset] = rgbToInt(255.0f * tg,255.0f * tg,255.0f * tg);
	}
}

__global__
void
drawScalaireKernel(float *s, int *dest, int3 imSize, int cmap,float max, int bord)
{
	unsigned int offset, pboOffset;
	float4 col;
	float ts;
	int2 addr;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z*addr.y+addr.x;
		pboOffset = imSize.x*(addr.y)+addr.x;
		if(addr.x >=bord && addr.y >= bord && addr.x < imSize.x-bord && addr.y < imSize.y-bord){
			ts = s[offset];
			switch (cmap){
				case 1:
						col = blueColormap(ts,max);
						break;
				case 0:
						col = hotColormap(ts,max);
						break;
				case 2:
						col = grayColormap(ts,max);
						break;
				case 4:
						col = blueColormap(log2f(ts),max);
						break;
				case 3:
						col = hotColormap(log2f(ts),max);
						break;
				case 5:
						col = grayColormap(log2f(ts),max);
						break;
				case 6:
						col = HSVsoft(ts, max);
						break;
				case 7:
						col = HSVsoft(log2f(fabs(ts))*ts/fabs(ts), max);
						break;
				default:
						col = hotColormap(ts,max);
						break;
			}
			dest[pboOffset] = rgbToInt(col.x,col.y,col.z);
		}else{
			dest[pboOffset] = 0;
		}
	}
}


__global__
void
drawVectKernel(float *u, float *v, float2 *dest, int3 imSize, int spaceVect,float scale)
{
	const float alpha = 0.33f;
	const float beta = 0.33f;
	int2 addr;
	int2 daddr;
	float2 vertex[6];
	uint offset;
	uint doffset;
	addr.x = (blockIdx.x * blockDim.x + threadIdx.x)*(spaceVect+1);
	addr.y = (blockIdx.y * blockDim.y + threadIdx.y)*(spaceVect+1);
	daddr.x = blockIdx.x * blockDim.x+ threadIdx.x;
	daddr.y = blockIdx.y * blockDim.y+ threadIdx.y;
	float2 p2;
	float tu,tv;
	if(addr.x < imSize.x && addr.y < imSize.y){
		if(daddr.x < imSize.x/(spaceVect+1) && daddr.y < imSize.y/(spaceVect+1)){
		offset =  imSize.z*addr.y+addr.x;
		doffset = (imSize.x/(spaceVect+1))*daddr.y+daddr.x;
		doffset = doffset*6;
		tu = u[offset];
		tv = v[offset];
		vertex[0] = make_float2(addr.x,addr.y);
		tu *= scale;
		tv *= scale;
		p2 = make_float2((float)addr.x+tu,(float)addr.y+tv);
		vertex[1] = make_float2(p2.x,p2.y);
		vertex[2] = vertex[1];
		vertex[4] = vertex[1];
		vertex[3] = make_float2(p2.x-alpha*(tu+beta*(tv)),p2.y-alpha*(tv+beta*(tu)));
		vertex[5] = make_float2(p2.x-alpha*(tu-beta*(tv)),p2.y-alpha*(tv-beta*(tu)));
 		dest[doffset] = vertex[0];
 		dest[doffset+1] = vertex[1];
 		dest[doffset+2] = vertex[2];
 		dest[doffset+3] = vertex[3];
 		dest[doffset+4] = vertex[4];
 		dest[doffset+5] = vertex[5];
		}
	}
}


__global__
void
applyMaskKernel(float *u, float *v,unsigned char *m, int3 imSize)
{
	unsigned int offset;
	int2 addr;
	float tu,tv;
	unsigned char tm;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z*addr.y+addr.x;
		tu = u[offset];
		tv = v[offset];
		tm = m[offset];
		u[offset] = (tm!=0)?tu:0.0f;
		v[offset] = (tm!=0)?tv:0.0f;
	}
}

void
drawNormGPU(float *u, float *v,float *lMin, int *dest, int3 imSize, int cmap, float max,float min, float *im, float coeff,float clmin)
{
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
	dim3 threads(CB_TILE_W, CB_TILE_H);
	drawNormKernel<<<grid,threads>>>(u, v,lMin, dest, imSize,cmap, max,min,im, coeff,clmin);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void
applyMask(float *u, float *v, unsigned char *m, int3 imSize)
{
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
	dim3 threads(CB_TILE_W, CB_TILE_H);
	applyMaskKernel<<<grid,threads>>>(u, v, m, imSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );


}

void
drawScalaireGPU(float *s, int *dest, int3 imSize, int cmap,float max,int bord)
{
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
	dim3 threads(CB_TILE_W, CB_TILE_H);
	drawScalaireKernel<<<grid,threads>>>(s, dest, imSize,cmap, max,bord);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void
drawflotGPU(float *u, float *v, float2 *dest, int3 imSize, int spaceVect,float scale)
{
	dim3 grid(iDivUp(imSize.x/(spaceVect+1), CB_TILE_W), iDivUp(imSize.y/(spaceVect+1), CB_TILE_H));
	dim3 threads(CB_TILE_W, CB_TILE_H);
	drawVectKernel<<<grid,threads>>>(u,v, dest, imSize,spaceVect, scale);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}


void
drawImGPU(float *I, int *dest, int3 imSize)
{
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
	dim3 threads(CB_TILE_W, CB_TILE_H);
	drawImKernel<<<grid,threads>>>(I, dest, imSize);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

