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

#include "FolkiOpticalFlow.hpp"

#include "FolkiOpticalFlowKernels.hpp"
#include "ConvolutionSeparable.hpp"
#include "ConvolutionSeparableFlat.hpp"
#include "ConvolutionSeparableU.hpp"
#include "ConvolutionSeparableFlatU.hpp"
#include "DivRot.hpp"
#include "RenderGL.hpp"
#include "Reduction.hpp"
#include "Propage.hpp"

namespace LIVA
{

FolkiOpticalFlow::FolkiOpticalFlow(QWidget *parent):QWidget(parent)
{
	// initialisations aveugles 
	_nImages=2;
	_nItter = 1;
	_algoBords = 1;
	_typeKer = 0;
	_pyrType = 0;
	_talon = 0.000001;
	_lminTalon = true;
	_kernelRadius = 5;
	_unRolling = false;
	_mgrid = false;
	_tMgrid = 0;
	_mGridStrategie = NULL;
	_bMask = false;
	lMinCoeff = 0.005;
	divRotOrdre = 1;
	bMarker = false;
	m_flushFlow = true;
}

FolkiOpticalFlow::~FolkiOpticalFlow()
{
	_nettoieStruct();
}

void
FolkiOpticalFlow::setMarker(float *data)
{
	cudaMemcpy2D((void *)_marker[0],_levelSize[0].z*sizeof(float),(void *)data,_levelSize[0].x*sizeof(float),_levelSize[0].x*sizeof(float),_levelSize[0].y,cudaMemcpyHostToDevice);

}

void
FolkiOpticalFlow::setFlushFlow(bool val)
{
	m_flushFlow = val;
}

void
FolkiOpticalFlow::activateMarker(bool val)
{
	bMarker = val;
}

void 
FolkiOpticalFlow::setMask(uchar * data)
{
	cudaMemcpy2D((void *)_Mask,_levelSize[0].z*sizeof(uchar),(void *)data,_levelSize[0].x*sizeof(uchar),_levelSize[0].x*sizeof(uchar),_levelSize[0].y,cudaMemcpyHostToDevice);
}

void
FolkiOpticalFlow::activateMask(bool val)
{
	_bMask = val;
}

void
FolkiOpticalFlow::useLminTalon(bool val)
{
	_lminTalon = val;
}

void
FolkiOpticalFlow::_nettoieStruct()
{
	for(unsigned int k=0;k<_nImages;k++){
		_pyrFree(_I[k]);
	}
	CUDA_SAFE_CALL( cudaFreeHost(_I) );
	// liberation des flots
	_pyrFree(_u);
	_pyrFree(_v);
	_pyrFree(_Idx);
	_pyrFree(_Idy);
	_pyrFree(_Mxx);
	_pyrFree(_Mxy);
	_pyrFree(_Myy);
	_pyrFree(_D);

	if(_mGridStrategie != NULL)
		free(_mGridStrategie);



	// liberation des buffers
	for(int i=0; i < NB_GPU_BUFF;i++){
		CUDA_SAFE_CALL( cudaFree(_buff[i]) );
	}
}



void
FolkiOpticalFlow::init(int3 imSize,unsigned int nLevels)
{
	_updatenLevels(nLevels,imSize);
	_upMgridStrategie();
	_allouStruct();
}

void
FolkiOpticalFlow::setMgrid(bool val)
{
	if(_mgrid != val){
		_mgrid = val;
		_upMgridStrategie();
	}
}


void
FolkiOpticalFlow::setTMgrid(int val)
{
	if(_tMgrid != val){
		_tMgrid = val;
		_upMgridStrategie();
	}
}


void
FolkiOpticalFlow::setLminCoef(double val)
{
	lMinCoeff = (float)val;
}



void
FolkiOpticalFlow::_upMgridStrategie()
{
	if(_mGridStrategie != NULL)
		free(_mGridStrategie);

	if(_mgrid){
		switch (_tMgrid){
			case 0: // strategie V
				_sMgrid = 2*_nLevels +1;
				_mGridStrategie = (unsigned int *) malloc(sizeof(unsigned int)*(_sMgrid));
				for(unsigned int i=0;i < _nLevels;i++){
					_mGridStrategie[i] = i;
					_mGridStrategie[_sMgrid-i-1] = i;
				}
				_mGridStrategie[_nLevels] = _nLevels;
				break;
			case 1: // scie ascendante
				break;

			case 2: // W cycle
				_sMgrid = 4*_nLevels+1;
				_mGridStrategie = (unsigned int *) malloc(sizeof(unsigned int)*(_sMgrid));
				for(unsigned int i=0;i < _nLevels;i++){
					_mGridStrategie[i] = i;
					_mGridStrategie[i+_nLevels+1] =_nLevels-i-1;
					_mGridStrategie[i+(_nLevels)*2+1] = i+1;
					_mGridStrategie[i+(_nLevels)*3+1] = _nLevels-i-1;
				}
				_mGridStrategie[_nLevels] = _nLevels;
				break;

			case 3: // N cycle
				_sMgrid = 3*_nLevels+1;
				_mGridStrategie = (unsigned int *) malloc(sizeof(unsigned int)*(_sMgrid));
				for(unsigned int i=0;i < _nLevels;i++){
					_mGridStrategie[i] = _nLevels-i;
					_mGridStrategie[i+_nLevels+1] = i+1;
					_mGridStrategie[_sMgrid-i-1] = i;
				}
				_mGridStrategie[_nLevels] = 0;
				break;

		}
		
	}else{
		_sMgrid = _nLevels+1;
		_mGridStrategie = (unsigned int *) malloc(sizeof(unsigned int)*(_sMgrid));
		for(unsigned int i=0;i <= _nLevels;i++){
			_mGridStrategie[i] = _nLevels-i;
			//std::cout << _mGridStrategie[i] << "\n";
		}
	}
	std::cout << "strategie : \n";
	for(int i = 0; i < _sMgrid;i++)
		std::cout << _mGridStrategie[i] << " ";
	std::cout << "\n --------------------- \n";
}

void
FolkiOpticalFlow::drawNorm(uint pbo, int cmap, float max,float min, float catCoeff, float seuil)
{
	int *dptr;
	if(pbo != 0){
		seuilNorm(_u[0], _v[0], _levelSize[0],seuil);
		CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dptr, pbo));
		if(!bMarker)
			drawNormGPU(_u[0], _v[0],_lMin, dptr, _levelSize[0], cmap, max,min,_I[0][0], catCoeff,lMinCoeff);
		else
			drawNormGPU(_marker[0], _marker[0],_lMin, dptr, _levelSize[0], cmap, max,min,_I[0][0], catCoeff,lMinCoeff);


		CUDA_SAFE_CALL(cudaGLUnmapBufferObject( pbo));
		emit updateNormPbo();
	}
}

void
FolkiOpticalFlow::drawSrc(uint pbo)
{
	int *dptr;
	if(pbo != 0){
		std::cout << "draw src\n";
		CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dptr, pbo));
		drawImGPU(_I[0][0], dptr, _levelSize[0]);
			
		CUDA_SAFE_CALL(cudaGLUnmapBufferObject( pbo));
		emit updateSrcPbo();
	}
}

void 
FolkiOpticalFlow::drawDiv(uint pbo,int colormap, float max, float seuil)
{
	if(pbo != 0){
		int *dptr;
		DivRot(_u[0],_v[0], _buff[0], _buff[1],_levelSize[0],divRotOrdre,seuil);
		CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dptr, pbo));
		drawScalaireGPU(_buff[0], dptr, _levelSize[0], colormap, max,divRotOrdre);
		CUDA_SAFE_CALL(cudaGLUnmapBufferObject( pbo));
		emit updateDivPbo();
	}
}


void 
FolkiOpticalFlow::drawRot(uint pbo, int colormap, float max, float seuil)
{
	if(pbo != 0){
		int *dptr;
		calculRotGPU(_u[0],_v[0], _buff[0], _buff[1], _levelSize[0],divRotOrdre,seuil);
		CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dptr, pbo));
		drawScalaireGPU(_buff[0], dptr, _levelSize[0], colormap, max,divRotOrdre);
		CUDA_SAFE_CALL(cudaGLUnmapBufferObject( pbo));
		emit updateRotPbo();
	}
}


void
FolkiOpticalFlow::drawDivRot(uint pbo, int colormap, float max, uint pbo2, int colormap2, float max2, float seuil)
{
	if(pbo!=0 && pbo2 != 0){
		int *dptrDiv, *dptrRot;
		DivRot(_u[0],_v[0], _buff[0], _buff[1],_levelSize[0],divRotOrdre,seuil);

		CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dptrDiv, pbo));
		drawScalaireGPU(_buff[0], dptrDiv, _levelSize[0], colormap, max,divRotOrdre);
		CUDA_SAFE_CALL(cudaGLUnmapBufferObject( pbo));
		emit updateDivPbo();
		CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dptrRot, pbo2));
		drawScalaireGPU(_buff[1], dptrRot, _levelSize[0], colormap2, max2,divRotOrdre);
		CUDA_SAFE_CALL(cudaGLUnmapBufferObject( pbo2));
		emit updateRotPbo();
	}
}

void 
FolkiOpticalFlow::drawVect(uint vbo, int spaceVect, float scale)
{
	if(vbo !=0){
		float2 *ptr;
		CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&ptr, vbo));
		drawflotGPU(_u[0],_v[0], ptr,_levelSize[0], spaceVect, scale);
		CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vbo));
		emit updtateVectVbo();
	}
}


void
FolkiOpticalFlow::copyMat(float *a, float *b,int3 imSize)
{
	CUDA_SAFE_CALL( cudaMemcpy((void*)b ,(void*)a,imSize.z*imSize.y*sizeof(float),cudaMemcpyDeviceToDevice));
}

void
FolkiOpticalFlow::setTypePyr(int typePyr)
{
	setPyrType(typePyr);
}

void
FolkiOpticalFlow::setUnrolling(bool un)
{
	_unRolling = un;
	if(_unRolling){
		_kernelRadius = U_KERNEL_RADIUS;
		_setKernel();
	}
}

void
FolkiOpticalFlow::init(int w, int h)
{
	unsigned int nLevels = 0;
	int3 imSize;
	imSize.x = w;
	imSize.y = h;
	imSize.z= w;
	_updatenLevels(nLevels,imSize);
	_upMgridStrategie();
	_allouStruct();
#ifdef EXPERIMENT
	CUDA_SAFE_CALL( cudaMalloc( (void **)&distH,_levelSize[0].z*_levelSize[0].y*sizeof(float)));
#endif
}

FolkiOpticalFlow::FolkiOpticalFlow(int3 imSize,unsigned int nLevels)
{
	init(imSize,nLevels);
}

void
FolkiOpticalFlow::setImage(float *image, int3 imSize, unsigned int nImage)
{
	if(imSize.x == _levelSize[0].x && imSize.y == _levelSize[0].y)
	{
		float **pyI=_I[nImage];
		_copyHostToDevice(pyI,image,0,imSize);
	}
}

void
FolkiOpticalFlow::setTypeKer(int type)
{
	_typeKer = type;
}

void 
FolkiOpticalFlow::setImage(float *image, unsigned int nImage)
{
	float **pyI=_I[nImage];
	_copyHostToDevice(pyI,image,0,_levelSize[0]);
}

void
FolkiOpticalFlow::_allouStruct()
{
	std::cout << "taille pyramide : \n";
	for(unsigned int i=0; i <= _nLevels;i++){
		std::cout << "niveau " << i << " : " << _levelSize[i].x << "x" << _levelSize[i].y << " : " << _levelSize[i].z << "\n";
	}
	/* allocation du tableau d'images*/
	CUDA_SAFE_CALL( cudaMallocHost((void **)&_I,(_nImages)*sizeof(float **)) );

	/* allocation des pyramides */
	for(unsigned int i=0;i<_nImages;i++){
		_pyrAlloc(&(_I[i]));
	}
	std::cout << "images allouees \n";
	_pyrAlloc(&_u);
	_pyrAlloc(&_v);
	std::cout << "vecteurs allouees \n";
	_pyrAlloc(&_Idx);
	_pyrAlloc(&_Idy);
	std::cout << "gradient allouees \n";
	_pyrAlloc(&_Mxx);
	_pyrAlloc(&_Mxy);
	_pyrAlloc(&_Myy);
	_pyrAlloc(&_D);
	std::cout << "tenseur allouee \n";

	float *tmpPtr;

//		==== our version avec Mask ====
		CUDA_SAFE_CALL( cudaMalloc(&tmpPtr,_levelSize[0].z*_levelSize[0].y*sizeof(unsigned char)) );
		_Mask=(unsigned char *)tmpPtr;
		CUDA_SAFE_CALL( cudaMalloc((void **)&tmpPtr,_levelSize[0].z*sizeof(float)*_levelSize[0].y) );
		
		_lMin = (float *) tmpPtr;	
		CUDA_SAFE_CALL( cudaMalloc((void **)&(_marker[0]),_levelSize[0].z*_levelSize[0].y*sizeof(float)) );
		CUDA_SAFE_CALL( cudaMalloc((void **)&(_marker[1]),_levelSize[0].z*_levelSize[0].y*sizeof(float)) );
	
	for(int i=0; i < NB_GPU_BUFF;i++){
		CUDA_SAFE_CALL( cudaMalloc((void **)&tmpPtr,_levelSize[0].z*sizeof(float)*_levelSize[0].y) );
		_buff[i]=(float *)tmpPtr;
	}
	std::cout << "buffers allouees\n";
}


void
FolkiOpticalFlow::_setKernel()
{
	float kernelSum = 0;
	int kernelSize= 2*_kernelRadius+1;
	float sigma = (_kernelRadius+3)/3;
	sigma *= sigma;
	for(unsigned int i=0;i< kernelSize;i++){
		float dist = ((float)i - (float)_kernelRadius);
		_kernel[i] = expf(- (dist * dist) / (2.f*sigma));
		kernelSum += _kernel[i];
	}
}


void 
FolkiOpticalFlow::calculFolki()
{
	if(m_flushFlow)
		_flushFlow();
	_computeFlowFolki();
#ifdef EXPERIMENT
	if(locHomog){
		localHomog(_u[0],_v[0], 0, distH,locHomogOrdre, _levelSize[0]);
	}
#endif
	emit newRes(_I[0][0], _u[0], _v[0]);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void
FolkiOpticalFlow::swapImages(){
	_swapImages(0,1);
}


void 
FolkiOpticalFlow::_swapImages(int a, int b)
{
	float **tmp;
	tmp=_I[a];
	_I[a]=_I[b];
	_I[b]=tmp;
}


void
FolkiOpticalFlow::getFlow(float* dx, float *dy, int level)
{
	_copyDeviceToHost(_u,dx,level,_levelSize[level]);
	_copyDeviceToHost(_v,dy,level,_levelSize[level]);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}


void 
FolkiOpticalFlow::_computeFlowFolki()
{
	float *G, *H;
	float *A,*B,*C,*D;
	float **pyI1,**pyI2;
	float *Ix,*Iy;
	float *I1, *I2;
	int bords;

	pyI1 = _I[0];
	pyI2 = _I[1];

	_fillPyramid(_I[0], _buff[0], _buff[1]);
	_fillPyramid(_I[1], _buff[0], _buff[1]);
	for(unsigned int i=0;i<_sMgrid;i++){
		int n = _mGridStrategie[i];
		//std::cout << "niveau " << n << " \n";
		I1=pyI1[n];
		I2=pyI2[n];
		Ix=_Idx[n];
		Iy=_Idy[n];
		A=_Mxx[n];
		B=_Myy[n];
		C=_Mxy[n];
		D=_D[n];
		G=_buff[0];
		H=_buff[1];
		if(_algoBords > 0){
			bords = MAX(1,_algoBords/pow(2,n));
		}else{
			bords = _algoBords;
		}
		for(unsigned int itt=0;itt<_nItter;itt++){
			_gradient(I1,_Idx[n],_Idy[n],_levelSize[n]);

			// partie de gauche
			calculTensor(_Idx[n],_Idy[n],_Mxx[n],_Myy[n],_Mxy[n],_levelSize[n],bords);
			if(_unRolling){
				if(_typeKer == 1){
					_setKernel();
					convolutionSeparableU( _Mxx[n] ,_Mxx[n] , _buff[0], _levelSize[n],_kernel ,_kernel);
					convolutionSeparableU(	_Mxy[n] ,_Mxy[n] , _buff[0], _levelSize[n],_kernel,_kernel);
					convolutionSeparableU(	_Myy[n] ,_Myy[n] , _buff[0], _levelSize[n],_kernel,_kernel);
				}else{
					convolutionSeparableFlatU( _Mxx[n] ,_Mxx[n] , _buff[0], _levelSize[n]);
					convolutionSeparableFlatU(	_Mxy[n] ,_Mxy[n] , _buff[0], _levelSize[n]);
					convolutionSeparableFlatU(	_Myy[n] ,_Myy[n] , _buff[0], _levelSize[n]);
				}
			}else{
				if(_typeKer == 1){
					_setKernel();
					convolutionSeparable( _Mxx[n] ,_Mxx[n] , _buff[0], _levelSize[n],_kernel ,_kernel ,_kernelRadius,_kernelRadius);
					convolutionSeparable(	_Mxy[n] ,_Mxy[n] , _buff[0], _levelSize[n],_kernel,_kernel,_kernelRadius,_kernelRadius);
					convolutionSeparable(	_Myy[n] ,_Myy[n] , _buff[0], _levelSize[n],_kernel,_kernel,_kernelRadius,_kernelRadius);
				}else{
					convolutionSeparableFlat( _Mxx[n] ,_Mxx[n] , _buff[0], _levelSize[n],_kernelRadius,_kernelRadius);
					convolutionSeparableFlat(	_Mxy[n] ,_Mxy[n] , _buff[0], _levelSize[n],_kernelRadius,_kernelRadius);
					convolutionSeparableFlat(	_Myy[n] ,_Myy[n] , _buff[0], _levelSize[n],_kernelRadius,_kernelRadius);
				}
			}

			// calcul du denominateur
			calculDenominateur(_Mxx[n],_Myy[n],_Mxy[n], _D[n],_lMin, _levelSize[n],bords);

			// partie de droite
			calculDroite(I1, I2,_u[n],_v[n], Ix, Iy ,G,H , _levelSize[n],bords);
			if(_unRolling){
				if(_typeKer == 1){
					convolutionSeparableU(G,G, _buff[2] ,_levelSize[n], _kernel,_kernel);
					convolutionSeparableU(H,H , _buff[2],_levelSize[n], _kernel,_kernel);
				}else{
					convolutionSeparableFlatU(G,G, _buff[2] ,_levelSize[n]);
					convolutionSeparableFlatU(H,H , _buff[2],_levelSize[n]);
				}
			}else{
				if(_typeKer == 1){
					convolutionSeparable(G,G, _buff[2] ,_levelSize[n], _kernel,_kernel,_kernelRadius,_kernelRadius);
					convolutionSeparable(H,H , _buff[2],_levelSize[n], _kernel,_kernel,_kernelRadius,_kernelRadius);
				}else{
					convolutionSeparableFlat(G,G, _buff[2] ,_levelSize[n],_kernelRadius,_kernelRadius);
					convolutionSeparableFlat(H,H , _buff[2],_levelSize[n],_kernelRadius,_kernelRadius);
				}
			}

			resoudSystem(A,B,C,D, G, H,_u[n],_v[n],_lMin, _talon, _lminTalon,lMinCoeff,_levelSize[n],bords);
		}
			if(i<_sMgrid-1){
				//std::cout << n <<" vers "<< _mGridStrategie[i+1] <<"\n";
				_depPyram(_u,_buff[0],_buff[1],(int)n,(int)_mGridStrategie[i+1]);
				_depPyram(_v,_buff[0],_buff[1],(int)n,(int)_mGridStrategie[i+1]);
			}
	}
	if(_bMask)
		applyMask(_u[0],_v[0],_Mask,_levelSize[0]);

	if(bMarker){
		propageMask(_marker[1],_marker[0],_u[0],_v[0],_levelSize[0]);
		_swapMarker();
	}

//	float mu = mean2DGPU(_u[0],_levelSize[0]);
//	float mv = mean2DGPU(_v[0],_levelSize[0]);
//	deleteMean(_u[0],_v[0],mu,mv,_levelSize[0]);

}




void
FolkiOpticalFlow::_flushFlow()
{
	CUDA_SAFE_CALL(cudaMemset((void *)_u[_mGridStrategie[0]], 0, _levelSize[_mGridStrategie[0]].y*_levelSize[_mGridStrategie[0]].z*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemset((void *)_v[_mGridStrategie[0]], 0, _levelSize[_mGridStrategie[0]].y*_levelSize[_mGridStrategie[0]].z*sizeof(float)));
}

void
FolkiOpticalFlow::getDivRot(float *div, float *rot)
{
	DivRot(_u[0],_v[0], _buff[0], _buff[1],_levelSize[0],divRotOrdre,0.0f);
	_copyDeviceToHost(&(_buff[1]),rot,0,_levelSize[0]);
	_copyDeviceToHost(_buff,div,0,_levelSize[0]);
}

void
FolkiOpticalFlow::getRot(float *rot)
{

	DivRot(_u[0],_v[0], _buff[0], _buff[1],_levelSize[0],divRotOrdre,0.0f);
	_copyDeviceToHost(&(_buff[1]),rot,0,_levelSize[0]);
}

void
FolkiOpticalFlow::getDiv(float *div)
{
	//convolutionSeparableFlatU( _u[0] ,_u[0] , _buff[0], _levelSize[0]);
	//convolutionSeparableFlatU( _v[0] ,_v[0] , _buff[0], _levelSize[0]);
	DivRot(_u[0],_v[0], _buff[0], _buff[1],_levelSize[0],divRotOrdre,0.0f);
	_copyDeviceToHost(_buff,div,0,_levelSize[0]);
}


void
FolkiOpticalFlow::setBords(int bords)
{
	if(bords != _algoBords)
		if(bords != -1) // on passera pas setMask pour utiliser un mask
			_algoBords=bords;
}

void 
FolkiOpticalFlow::setnItter(int nItter)
{
	_nItter =(unsigned int) MAX(1,nItter);
}

void 
FolkiOpticalFlow::setTalon(double talon)
{
	_talon = powf(10,-(float)talon);
}

void 
FolkiOpticalFlow::setDivRotOrdre(int val)
{
	divRotOrdre = val;
std::cout << "div ordre :" << divRotOrdre << "\n";
}


void 
FolkiOpticalFlow::setKernelRadius( int kernelRadius)
{
	_kernelRadius = (unsigned int)MIN(MAX(1,kernelRadius),MAX_KERNEL_RADIUS);
	_setKernel();
}

//void setMask(unsigned char *data, int2 size);


void FolkiOpticalFlow::setnLevel(int level)
{
	unsigned int oldLevel;
	oldLevel = _nLevels;
	unsigned int nlevel = (unsigned int)level;
	unsigned int s;
	s=MAX(_levelSize[0].x,_levelSize[0].z);
	unsigned int maxLevel = (unsigned int)(log2f((float)s)-1);
	nlevel = MIN(nlevel,maxLevel);

	if(nlevel<_nLevels){
		_nLevels = nlevel;
		_pyrReAlloc(&_u,oldLevel);
		_pyrReAlloc(&_v,oldLevel);
		_pyrReAlloc(&_Idx,oldLevel);
		_pyrReAlloc(&_Idy,oldLevel);
		_pyrReAlloc(&_Mxx,oldLevel);
		_pyrReAlloc(&_Mxy,oldLevel);
		_pyrReAlloc(&_Myy,oldLevel);
		_pyrReAlloc(&_D,oldLevel);
		for(int k=0;k<_nImages;k++){
				_pyrReAlloc(&(_I[k]),oldLevel);
		}
	}else if(nlevel>_nLevels){
		int3 *oldLevelSize = _levelSize;
		_levelSize=(int3 *)malloc((1+nlevel)*sizeof(int3));
		for(unsigned int i=0;i<=_nLevels;i++){
			_levelSize[i] = oldLevelSize[i];
		}
		free( oldLevelSize);
		for(unsigned int i=_nLevels+1;i<=nlevel;i++){
			_levelSize[i].x = (int)_levelSize[i-1].x/2;
			_levelSize[i].y = (int)_levelSize[i-1].y/2;
			_levelSize[i].z = iAlignUp(_levelSize[i].x,ALLIGN_ROW);
		}
		_nLevels = nlevel;

		for(unsigned int k=0;k<_nImages;k++)
			_pyrReAlloc(&(_I[k]),oldLevel);
		_pyrReAlloc(&_u,oldLevel);
		_pyrReAlloc(&_v,oldLevel);
		_pyrReAlloc(&_Idx,oldLevel);
		_pyrReAlloc(&_Idy,oldLevel);
		_pyrReAlloc(&_Mxx,oldLevel);
		_pyrReAlloc(&_Mxy,oldLevel);
		_pyrReAlloc(&_Myy,oldLevel);
		_pyrReAlloc(&_D,oldLevel);
	}
	_upMgridStrategie();
}

} // namespace LIVA
