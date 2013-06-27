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

#include "FlotRender.hpp"
#include <cuda_runtime.h>
#include <cutil.h>
#include <cuda.h>
#include "endianness.h"
#include <QDir>

namespace LIVA
{

FlotRender::FlotRender(QWidget * parent):QWidget(parent){
}

FlotRender::~FlotRender(){};

void
FlotRender::init(int w,int h)
{
	imSize=cvSize(w,h);
	cmap = 0;
	gridSpace = 10;
	flip = false;
	normMax = 10;
	normMin = 0;
	divMax = 10;
	rotMax = 10;
	cmapDiv = 0;
	cmapRot = 0;
	nSeuil = 0.0f;
	distMax = 5;
	distCmap = 0;
	waitTime = 10;
	distH = false;
	saveFlot = false;
	saveCurl = false;
	saveDiv = false;
	saveNorm = false;
	vectCmap = false;
	affFlot = false;
	affNorm = false;
	flotNorm = false;
	affSR = false;
	fondFlot = 0;
	affDiv = false;
	flotVal = 0;
	catCoeff = 0.5;
	flotCmap = 0;
	affRot = false;
	affSrc = false;
	gridMultFactor = 1.;

	image = NULL;
	save = false;
	urlSave = QDir::currentPath();
	
}

void
FlotRender::nettoie()
{
	/*CUDA_SAFE_CALL( cudaFreeHost( (void *) _u) );
	CUDA_SAFE_CALL( cudaFreeHost( (void *) _v) );
	delete image;*/
}


void
FlotRender::setSR(bool val)
{
	affSR = val;
}

void
FlotRender::renderFlow(){
	if(save){
		emit getFlow(_u,_v);
		ecritFlot();
	}

	if(affNorm){
		emit drawPboNorm(pbo,cmap,normMax,normMin,catCoeff,nSeuil);
	}

	if(affFlot){
		emit drawFlotVect((vbo),gridSpace,gridMultFactor);
	}

	if(affDiv){
		emit drawPboDiv(pbo,cmapDiv,divMax,nSeuil);
	}
	if(affRot){
		emit drawPboRot(pbo,cmapRot,rotMax,nSeuil);
	}
	
	if(affSrc)
		emit drawPboSrc(pbo);

	if(saveNorm){
		emit saveTheNorm(urlSaveNorm);
	}
	cvWaitKey(waitTime);
}


void
FlotRender::drawSR()
{
	if(affSR){
		emit drawPboSR(pbo);
	}
}


void
FlotRender::setPbo(uint val)
{
	pbo = val;
}

void
FlotRender::setnSeuil(double val)
{
	nSeuil = (double)val;
}

void
FlotRender::setCat(double val)
{
	catCoeff = (float)val;
}

void 
FlotRender::setFlip(bool val)
{
	flip = val;
}


void 
FlotRender::setFlotNorm(bool val)
{
	flotNorm = val;
}

void 
FlotRender::setFondFlot(int val)
{
	fondFlot = val;
}


void 
FlotRender::setSaveFlot(bool val)
{
	saveFlot = val;
}


void 
FlotRender::actualiseFlow()
{
	
}

void
FlotRender::setAffFlot(bool val)
{
	affFlot = val;
}

void
FlotRender::setAffNorm(bool val)
{
	affNorm = val;
}

void
FlotRender::setAffSrc(bool val)
{
	affSrc = val;
}

void
FlotRender::setSave(bool val)
{
	save = val;
}

void
FlotRender::setDistMax(double val)
{
	distMax = (float)val;
}

void
FlotRender::setUrlSave(QString val)
{
	std::cout << "sauvegarde dans :\n";
	std::cout << val.toStdString().c_str();
	std::cout << "\n";
	urlSave = val;
}

void
FlotRender::setUrlSaveFlotIm(QString val)
{
	std::cout << "sauvegarde dans :\n";
	std::cout << val.toStdString().c_str();
	std::cout << "\n";
	urlSaveFlotIm = val;
}

void 
FlotRender::setdelay(int delay)
{
	waitTime = delay;
}

void 
FlotRender::setScalVect(double scale)
{
	gridMultFactor =(float) scale;
}

void 
FlotRender::setColormap(int map)
{
	cmap = map;
}


void 
FlotRender::setColormapDist(int val)
{
	distCmap = val;
}

void 
FlotRender::setDistH(bool val)
{
	distH = val;
}
void
FlotRender::setSpaceVect(int space)
{
	gridSpace = space;
}

void
FlotRender::setVbo(uint val)
{
	vbo = val;
}

void 
FlotRender::setNormMin(double max)
{
	normMin = (float)max;
}




void 
FlotRender::setNormMax(double max)
{
	normMax = (float)max;
}


void
FlotRender::setUrlSaveNorm(QString val)
{
	std::cout << "sauvegarde dans :\n";
	std::cout << val.toStdString().c_str();
	std::cout << "\n";
	urlSaveNorm = val;
}

void
FlotRender::setUrlSaveDiv(QString val)
{
	std::cout << "sauvegarde dans :\n";
	std::cout << val.toStdString().c_str();
	std::cout << "\n";
	urlSaveDiv = val;
}

void
FlotRender::setUrlSaveCurl(QString val)
{
	std::cout << "sauvegarde dans :\n";
	std::cout << val.toStdString().c_str();
	std::cout << "\n";
	urlSaveCurl = val;
}

void 
FlotRender::setSaveCurl(bool val)
{
	saveCurl = val;
}

void 
FlotRender::setSaveNorm(bool val)
{
	saveNorm = val;
}

void 
FlotRender::setSaveDiv(bool val)
{
	saveDiv = val;
}

CvScalar
FlotRender::hsv2rgb(float h, float s, float v)
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
		case 0: return CV_RGB(v,t,p);
		case 1: return CV_RGB(q,v,p);
		case 2: return CV_RGB(p,v,t);
		case 3: return CV_RGB(p,q,v);
		case 4: return CV_RGB(t,p,v);
		case 5: return CV_RGB(v,p,q);
	}
}

void
FlotRender::setFlotVal(int val)
{
	flotVal = val;
}

void
FlotRender::setFlotCmap(int val)
{
	flotCmap = val;
}

void
FlotRender::drawOpticalFlow()
{
	CvSize gridSize=cvSize(gridSpace,gridSpace);
	unsigned int offset;
	CvSize size = _displayImage.size();
	CvScalar color = CV_RGB(255, 0, 0);
	float alpha = 0.33f;
	float beta = 0.33f;

	if(gridMultFactor > 0) {
		srand(0);
		for(int i = gridSize.height; i < (size.height - gridSize.height); i+= gridSize.height) {
		for(int j = gridSize.width; j < (size.width - gridSize.width); j+= gridSize.width) {
					CvPoint p1 = cvPoint(j, i);
					CvPoint p2;
					offset = j+i*imSize.width;
					float x = _u[offset];
					float y = _v[offset];
					float norm = sqrt(x * x + y * y);
					x *= gridMultFactor;
					y *= gridMultFactor;
					if(flotNorm){
						float toto=0;
						float totoMax=0;
						if(!vectCmap){
							color = vectColor;
						}else{
						switch (flotVal){
							case 0:
								flotCmap = 42;
								break;
							case 1: // norm
								toto =norm;
								totoMax = normMax;
								break;
						}
						switch (flotCmap){
							case 0:
								colormapBlue(color,toto,totoMax);
								break;
							case 3:
								colormapLogBlue(color,toto,totoMax);
								break;
							case 1:
								colormapHot(color,toto,totoMax);
								break;
							case 4:
								colormapLogHot(color,toto,totoMax);
								break;
							case 2:
								colormapGray(color,toto,totoMax);
								break;
							case 5:
								colormapLogGray(color,toto,totoMax);
								break;
							case 6:
								color = hsv2rgb((int)(toto*totoMax*36)%360, 1, 1);
								break;
							default:
								color = CV_RGB(0, 0, 0);
						}
						}
					}
					p2.x = p1.x + (int)x;
					p2.y = p1.y + (int)y;
					cvLine(_displayImage, p1, p2, color, 1);

					CvPoint delta;
					delta = cvPoint(p2.x-alpha*(x+beta*(y)),p2.y-alpha*(y+beta*(x)));
					cvLine(_displayImage,p2,delta, color, 1);
					delta = cvPoint(p2.x-alpha*(x-beta*(y)),p2.y-alpha*(y-beta*(x)));
					cvLine(_displayImage,p2,delta, color, 1);
		}
		}
	}
}

void
FlotRender::drawDist()
{
	unsigned int offset;
	CvSize size = _displayDist.size();
	CvScalar color = CV_RGB(255, 0, 0);
	for(int i = 0; i < (imSize.height); ++i) {
		for(int j = 0; j < (imSize.width); ++j) {
			offset = j+i*imSize.width;
			float norm = fabs(_d[offset]);

			switch (distCmap) {
				case 0:
					colormapBlue(color,norm,distMax);
					break;
				case 3:
					colormapLogBlue(color,norm,distMax);
					break;
				case 1:
					colormapHot(color,norm,distMax);
					break;
				case 4:
					colormapLogHot(color,norm,distMax);
					break;
				case 2:
					colormapGray(color,norm,distMax);
					break;
				case 5:
					colormapLogGray(color,norm,distMax);
					break;
				case 6:
					color = hsv2rgb((int)(norm*distMax*36)%360, 1, 1);
					break;
				default:
					colormapBlue(color,norm,distMax);
			}
			cvSet2D(_displayDist,i,j, color);
		}
	}
}



void 
FlotRender::colormapBlue(CvScalar &color, float val, float max)
{
			float n = (3.0f*max/8.0f);
			/* colormap maison (trop de la bombe de balle) */
			color.val[2]=255*((val<2*n)?0.:val/(val-2.*n));
			color.val[0]=255*((val<n)?val/n:1.);
			color.val[1]=255*((val<n)?0.:(val>2*n)?1.:(float)(val-n)/n);
}

void 
FlotRender::colormapHSVsoft(CvScalar &color, float val, float max)
{
	if(val > max || val < -max)
	{ 
		color = CV_RGB(0, 0, 0);
	}else{
			float n = val / max;
			if(val>0.0f){
				float t = 153-(153*sqrt(n));
				color = hsv2rgb(t,(t<145)?1.0f:(t-153)*(t-153)/(64),1.0f);
			}else{
				float t = 153+(sqrt(-n)*100);
				color = hsv2rgb(t,(t<160)?1.0f:(t-153)*(t-153)/(49),1.0f);
			}
	}
}

void 
FlotRender::colormapLogBlue(CvScalar &color, float val, float max)
{
			val = (log2f(val))*10;
			float n = (3.0f*max/8.0f);
			/* colormap maison (trop de la bombe de balle) */
			color.val[2]=255*((val<2*n)?0.:val/(val-2.*n));
			color.val[0]=255*((val<n)?val/n:1.);
			color.val[1]=255*((val<n)?0.:(val>2*n)?1.:(float)(val-n)/n);
}


void 
FlotRender::colormapGray(CvScalar &color, float val, float max)
{
			/* colormap maison (trop de la bombe de balle) */
			color.val[2]=255*(val/max);
			color.val[0]=255*(val/max);
			color.val[1]=255*(val/max);
}

void 
FlotRender::colormapLogGray(CvScalar &color, float val, float max)
{
			val = (log2f(val))*10;
			/* colormap maison (trop de la bombe de balle) */
			color.val[2]=255*(val/max);
			color.val[0]=255*(val/max);
			color.val[1]=255*(val/max);
}

void 
FlotRender::colormapHot(CvScalar &color, float val, float max)
{
			float n = (3.0f*max/8.0f);
			/* colormap maison (trop de la bombe de balle) */
			color.val[0]=255*((val<2*n)?0.:val/(val-2.*n));
			color.val[2]=255*((val<n)?val/n:1.);
			color.val[1]=255*((val<n)?0.:(val>2*n)?1.:(float)(val-n)/n);
}

void 
FlotRender::colormapLogHot(CvScalar &color, float val, float max)
{
			val = (log2f(val))*10;
			float n = (3.0f*max/8.0f);
			/* colormap maison (trop de la bombe de balle) */
			color.val[0]=255*((val<2*n)?0.:val/(val-2.*n));
			color.val[2]=255*((val<n)?val/n:1.);
			color.val[1]=255*((val<n)?0.:(val>2*n)?1.:(float)(val-n)/n);
}


void
FlotRender::fli()
{
	CvImage *old;
	emit getOld(&old);
	while(!(cvWaitKey(waitTime)==' ')){
		cvShowImage("source",*image);
		if(cvWaitKey(waitTime)==' ')
			break;
		cvShowImage("source",*old);
	}
}

void
FlotRender::ecritFlotIm()
{
	char nom[256];
	sprintf(nom,"%s%08d_FlotIm.png",urlSaveFlotIm.toStdString().c_str(),_index);
	if(!cvSaveImage(nom,_displayImage))
				std::cout << "impossible de sauvegarder image la : " << nom <<"\n";
}

void
FlotRender::ecritFlot()
{
	int h = imSize.height;
	int w = imSize.width;

	char nom[256];

	
	sprintf(nom,"%s%08d.inf",urlSave.toStdString().c_str(),_index);
	FILE *f=fopen(nom,"w");
	fprintf(f,"%d\n%d\n",w,h);
	fprintf(f,"%d %c",2,'f');  
	fclose(f);

	sprintf(nom,"%s%08d.1",urlSave.toStdString().c_str(),_index);
	f = fopen(nom,"w");
	befwrite(_u,sizeof(float),h*w,f);
	fclose(f);
	
	// Ecriture du fichier raw : champ V 
	sprintf(nom,"%s%08d.2",urlSave.toStdString().c_str(),_index);
	f = fopen(nom,"w");
	befwrite(_v,sizeof(float),h*w,f);
	fclose(f);
}

void
FlotRender::setIndex(int val)
{
	_index = val;
}

void 
FlotRender::setVectCmap(bool val)
{
	vectCmap = val;
}

void 
FlotRender::setVectColor(int val)
{
	switch (val){
		case 0:
			vectColor = CV_RGB(0, 0, 0);
			break;
		case 1:
			vectColor = CV_RGB(255, 0, 0);
			break;
		case 2:
			vectColor = CV_RGB(0, 0, 255);
			break;
		case 3:
			vectColor = CV_RGB(0, 255, 0);
			break;
		case 4:
			vectColor = CV_RGB(255, 255, 255);
			break;
		default:
			vectColor = CV_RGB(255, 0, 0);
	}
}




void
FlotRender:: setColormapDiv(int cmap)
{
	cmapDiv = cmap;
}


void
FlotRender:: setDivMax(double max)
{
	divMax = max;
}

void
FlotRender:: setDiv(bool val)
{
	affDiv = val;
}



void
FlotRender:: setColormapRot(int cmap)
{
	cmapRot = cmap;
}

void
FlotRender:: setRotMax(double max)
{
	rotMax = (float)max;
}


void
FlotRender:: setRot(bool val)
{
	affRot = val;
}

}

