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

#ifndef __LI_CU_REDUCTION_HPP
#define __LI_CU_REDUCTION_HPP


#include "LiCuda.hpp"


extern "C"
float
sumGPU(float *a, int size);

extern "C"
void
sum36pGPU(float *a, float *poids,int size, float *res);

extern "C"
float2
meanGPUf2(float2 *a,int size);

extern "C"
float
stdGPUf2(float2 *a,int size,float2 mu);

extern "C"
float
meanGPU(float *a, int size);

extern "C"
float
stdGPU(float *a, float *buff, int size, float mean);

extern "C"
float
sum2DGPU(float *a, int3 size);


extern "C"
float
mean2DGPU(float *a, int3 imSize);

extern "C"
float
std2DGPU(float *u,float *v, float *buff, int3 size, float meanx, float meany);

extern "C"
void
deleteMean(float *u, float *v, float mu, float mv, int3 imSize);

#endif

