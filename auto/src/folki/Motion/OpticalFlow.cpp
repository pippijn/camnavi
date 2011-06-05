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

#include "OpticalFlow.hpp"

#include "Cuda/PyramidBurt.hpp"
#include "Cuda/PyramidHaar.hpp"
#include "Cuda/DeriveGradient.hpp"

namespace LIVA
{
OpticalFlow::OpticalFlow()
{
}

OpticalFlow::~OpticalFlow()
{
}


void
OpticalFlow::_pyrAlloc(float ***Pyr)
{
	CUDA_SAFE_CALL( cudaMallocHost((void **) Pyr,(1+_nLevels)*sizeof(float *)) );
	for(int i=_nLevels;i>=0;--i){
		CUDA_SAFE_CALL( cudaMalloc((void **) &((*Pyr)[i]),_levelSize[i].z*_levelSize[i].y*sizeof(float)) );
	}
}


void 
OpticalFlow::_pyrFree(float **Pyr)
{
	for(unsigned int i=0;i>=_nLevels;i++){	
		CUDA_SAFE_CALL( cudaFree(Pyr[i]) );
	}
	CUDA_SAFE_CALL( cudaFreeHost(Pyr));
}


void
OpticalFlow::_pyrReAlloc(float ***Pyr,unsigned int oldLevels)
{
	if(oldLevels > _nLevels){
		for(unsigned int i=_nLevels+1; i<=oldLevels;i++)
			CUDA_SAFE_CALL( cudaFree( (*Pyr)[i] ) );
	}else{
		float **old = *Pyr;
		CUDA_SAFE_CALL( cudaMallocHost((void **)Pyr,(1+_nLevels)*sizeof(float *)) );
		for(unsigned int i=0;i<= oldLevels;i++)
			(*Pyr)[i]=old[i];
		for(unsigned int i=oldLevels+1;i<=_nLevels;i++)
			CUDA_SAFE_CALL( cudaMalloc((void **)&((*Pyr)[i]),_levelSize[i].z*_levelSize[i].y*sizeof(float)) );
		CUDA_SAFE_CALL( cudaFreeHost( old ) );
	}
}



void
OpticalFlow::_updatenLevels(unsigned int nLevels, int3 imSize)
{
	unsigned int s;
	s=MAX(imSize.x,imSize.z);
	unsigned int maxLevel = (unsigned int)(log2f((float)s)-1);
	_nLevels = MIN(nLevels,maxLevel);
	_levelSize=(int3 *)malloc((1+_nLevels)*sizeof(int3));
	unsigned int acc=1;
	for(unsigned int i=0;i<=_nLevels;i++){
		_levelSize[i].x = (imSize.x/acc);
		_levelSize[i].y = (imSize.y/acc);
		_levelSize[i].z = iAlignUp(_levelSize[i].x,ALLIGN_ROW);
		acc *=2;
	}
}



int3
OpticalFlow::getImSize()
{
	return _levelSize[0];
}

void 
OpticalFlow::setPyrType(int pyr)
{
	_pyrType=pyr;
}

void
OpticalFlow::_resample(float **Pyr, unsigned int level)
{
	PyramidUpSampleHaar(Pyr[level-1], Pyr[level] , _levelSize[level-1], _levelSize[level]);
}

void
OpticalFlow::_copyDeviceToHost(float **d_ptr,float *h_ptr,unsigned int level, int3 h_imSize)
{
	CUDA_SAFE_CALL( cudaMemcpy2D((void*)h_ptr,h_imSize.z*sizeof(float), (void*)d_ptr[level], _levelSize[level].z*sizeof(float),_levelSize[level].x* sizeof(float),_levelSize[level].y,cudaMemcpyDeviceToHost) );
}

void
OpticalFlow::_copyHostToDevice(float **d_ptr,float *h_ptr,unsigned int level, int3 h_imSize)
{
	CUDA_SAFE_CALL( cudaMemcpy2D((void*)d_ptr[level],_levelSize[level].z*sizeof(float), (void*)h_ptr,h_imSize.x*sizeof(float), h_imSize.x* sizeof(float),h_imSize.y,cudaMemcpyHostToDevice) );
}

void
OpticalFlow::_fillPyramid(float **Pyr,float *buff1, float *buff2)
{
	for(unsigned int i=0; i<_nLevels; i++){
		if(_pyrType == 0)
			PyramidDownSampleHaar(Pyr[i] ,Pyr[i+1],_levelSize[i],_levelSize[i+1]);
		else
			PyramidDownSampleBurt(Pyr[i] ,Pyr[i+1],buff1,buff2,_levelSize[i],_levelSize[i+1]);
	}
}

void
OpticalFlow::_depPyram(float **Pyr,float *buff1, float *buff2,  int lsrc, int ldest)
{
	if(lsrc < ldest){
		for(int i=lsrc; i<ldest; i++){
			if(_pyrType == 0)
				PyramidDownSampleHaar(Pyr[i] ,Pyr[i+1],_levelSize[i],_levelSize[i+1]);
			else
				PyramidDownSampleBurt(Pyr[i] ,Pyr[i+1],buff1,buff2,_levelSize[i],_levelSize[i+1]);
		}
	}else if(lsrc > ldest){
		//std::cout << "monte niveau :\n";
		for(int i=lsrc; i>ldest; i--){
			//std::cout << "niveau " <<i << "\n";
			_resample(Pyr, (unsigned int)i);
		}
	}
}

void 
OpticalFlow::_gradient(float *I,float *Ix,float *Iy,int3 imSize)
{
	gradient(I,Ix,Iy,imSize);
}


}//namespace LIVA
