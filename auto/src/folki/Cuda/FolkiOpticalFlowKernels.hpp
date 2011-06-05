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

#ifndef __LI_FOLKI_OPTICAL_FLOW_KERNELS_HPP__
#define __LI_FOLKI_OPTICAL_FLOW_KERNELS_HPP__

#include "LiCuda.hpp"

extern "C"
void
resoudSystem( float *A,float *B,float *C,float *D,float *G,float *H, /*INPUT*/
							float *u,float *v, /*OUTPOUT*/
							float *lMin, float talon, bool useLmin, float slmin,
							int3 imSize, int bords);

extern "C"
void
calculTensor( float *Ix,float *Iy, /*INPUT*/
			float *A,float *B,float *C, /*OUTPOUT*/
			int3 imSize, int bords);

extern "C"
void
calculDenominateur( float *A,float *B,float *C, /*INPUT*/
					float *D,  float *lMin, /*OUTPOUT*/
					int3 imSize, int bords);

extern "C"
void	
calculDroite(float *I1,float *I2,float *u,float *v,float *Ix,float *Iy, /* INPOUT */
						 float *G, float *H, /* OUTPOUT */
						 int3 imSize, int bords);


extern "C"
void
calculIt(float *I1,cudaArray *aI2,float *u,float *v,float *Ix,float *Iy,
		float *It, 
		int3 imSize, int bords);

extern "C"
void
swapMat(float* a, float *b, int3 imSize);

#endif
