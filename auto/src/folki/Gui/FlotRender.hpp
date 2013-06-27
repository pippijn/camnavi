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

#ifndef __LI_FLOT_RENDER_HPP
#define __LI_FLOT_RENDER_HPP

#include "GestionBuffer.hpp"

#include <opencv2/legacy/legacy.hpp>
#include <opencv/highgui.h>
#include <QWidget>
#include <iostream>

namespace LIVA
{
class FlotRender : public QWidget
{
	Q_OBJECT
	public:
	FlotRender(QWidget * parent =0);
	~FlotRender();

	public slots:
	void init(int w,int h);
	void renderFlow();
	void actualiseFlow();
	void setNormMax(double max);
	void setNormMin(double);
	void setDistMax(double max);
	void setColormapDist(int cmap);
	void setDistH(bool);
	void setScalVect(double scale);
	void setColormap(int cmap);
	void setSpaceVect(int space);
	void setdelay(int delay);
	void setSave(bool);
	void setUrlSave(QString);
	void setUrlSaveFlotIm(QString);
	void setIndex(int);
	void nettoie();
	void setAffFlot(bool);
	void setAffNorm(bool);
	void setAffSrc(bool);
	void setFlip(bool);
	void setColormapDiv(int cmap);
	void setDivMax(double max);
	void setDiv(bool);
	void setColormapRot(int cmap);
	void setRotMax(double max);
	void setRot(bool);
	void setFlotNorm(bool);
	void setFlotVal(int);
	void setFlotCmap(int);
	void setSaveFlot(bool);
	void fli();
	void setFondFlot(int);

	void setUrlSaveNorm(QString);
	void setSaveNorm(bool);

	void setUrlSaveDiv(QString);
	void setSaveDiv(bool);

	void setUrlSaveCurl(QString);
	void setSaveCurl(bool);

	void setVectCmap(bool);
	void setVectColor(int);

	void setPbo(uint);
	void setVbo(uint);

	void setCat(double);
	void setSR(bool);
	void drawSR();
	void setnSeuil(double);

	signals:
	void getFlow(float*u,float*v);
	void getDist(float *d);
	void getCvImage(CvImage **im);
	void getOld(CvImage **im);
	void nMax(double);
	void getDiv(float *div);
	void getRot(float *rot);
	void getDivRot(float *div, float *rot);
	void drawPboNorm(uint,int cmap, float max, float min,float coeff,float seuil);
	void drawPboRot(uint,int cmap, float max,float seuil);
	void drawPboDiv(uint,int cmap, float max,float seuil);
	void drawPboDivRot(uint,int cmap, float max,uint,int cmap2, float max2,float seuil);
	void drawFlotVect(uint,int,float);
	void  drawPboSR(uint);
	void drawPboSrc(uint);
	void saveTheNorm(QString);

	private:
	bool save;
	QString urlSave;
	QString urlSaveFlotIm;
	QString urlSaveNorm, urlSaveDiv, urlSaveCurl;

	CvSize imSize;
	CvImage *image;
	CvImage _displayImage;
	CvImage _displayDist;
	float nSeuil;
	int _index;
	int cmap;
	int distCmap;
	float gridMultFactor;
	int gridSpace;
	float normMax;
	float normMin;
	bool pboNorm, pboDiv, pboRot;

	float distMax;
	bool distH;

	bool affRot, affDiv;
	float divMax, rotMax;
	int cmapRot;
	int cmapDiv;
	uint pbo;
	uint vbo;
	float catCoeff;
	float *_u,*_v;	
	float *_d;
	CvScalar vectColor;
	bool vectCmap;
	bool flotNorm;
	int fondFlot;
	int flotVal;
	int flotCmap;
	bool saveFlot;
	bool saveNorm,saveDiv,saveCurl;

	int waitTime;
	float maxNorm;
	bool flip;
	bool affFlot,affNorm,affSrc,affSR;

	CvScalar hsv2rgb(float h, float s, float v);
	void drawOpticalFlow();
	void drawOpticalFlowNorm();
	void drawDist();
	void drawDiv();
	void drawRot();
 
	void colormapHSV(CvScalar &color, float val,float coef);
	void colormapHSVsoft(CvScalar &color, float val,float coef);
	void colormapBlue(CvScalar &color, float val, float max);
	void colormapLogBlue(CvScalar &color, float val, float max);
	void colormapHot(CvScalar &color, float val, float max);
	void colormapLogHot(CvScalar &color, float val, float max);
	void colormapGray(CvScalar &color, float val, float max);
	void colormapLogGray(CvScalar &color, float val, float max);
	void ecritFlot();
	void ecritFlotIm();
	void ecritNorm();
	void ecritCurl();
	void ecritDiv();
};
}

#endif

