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

#include "Propage.hpp"



static texture<float,2> texI;




__global__
void
propageMaskKer(float *dest, float *u, float *v, int3 imSize)
{
	int2 addr;
	uint offset;
	addr.x = blockIdx.x * blockDim.x + threadIdx.x;
	addr.y = blockIdx.y * blockDim.y + threadIdx.y;
	float tu,tv;
	float td;
	float x,y;
	if(addr.x < imSize.x && addr.y < imSize.y){
		offset = imSize.z * addr.y + addr.x;
		tu = u[offset];
		tv = v[offset];
		x = (float) addr.x + tu;
		y = (float) addr.y + tv;
		td = tex2D(texI,x+0.5f,y+0.5f);
		dest[offset] = floor(td+0.5f);
	}

}




void
propageMask(float *dest , float *src, float *u, float *v, int3 imSize)
{
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	texI.normalized = false;
 	texI.filterMode = cudaFilterModeLinear;
 	texI.addressMode[0] = cudaAddressModeClamp;
 	texI.addressMode[1] = cudaAddressModeClamp;
	cudaBindTexture2D(0,&texI, src, &channelDesc , imSize.x, imSize.y, imSize.z * sizeof(float));
	dim3 grid(iDivUp(imSize.x, CB_TILE_W), iDivUp(imSize.y, CB_TILE_H));
 	dim3 threads(CB_TILE_W, CB_TILE_H);
	propageMaskKer<<<grid,threads>>>(dest, u, v, imSize);

	cudaThreadSynchronize();
	cudaUnbindTexture(texI);
}





