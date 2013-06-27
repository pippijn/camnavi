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

#ifndef __LI_FOLKI_OPTICAL_FLOW_HPP__
#define __LI_FOLKI_OPTICAL_FLOW_HPP__

#include "OpticalFlow.hpp"
#include <QWidget>



namespace LIVA 
{

class FolkiOpticalFlow :public QWidget, public OpticalFlow
{
Q_OBJECT
public:
	~FolkiOpticalFlow();
	FolkiOpticalFlow(QWidget *parent =0);
	/* INPUT */
	// nouvelle image -> calcul du flot
	// initialisateurs des pyramides
	FolkiOpticalFlow(int3 imSize,unsigned int nLevels);

	/* fonction pour regler les parametres d'algo qu'on change peut */
public slots:
	void setFlushFlow(bool);
	void setMask(uchar *data);
	void setMarker(float *);
	void activateMarker(bool);
	void activateMask(bool);
	//void setHomog(floar *H);
	void init(int w, int h);
	void init(int3 imSize,unsigned	int nLevels);
	void setImage(float *image,int3 imSize, unsigned int nImage );
	void setImage(float *image, unsigned int nImage = 1);
	void setnItter(int nItter);
	void setTalon(double talon);
	void setKernelRadius(int kernelRadius);
	void setBords(int bords);
	void setnLevel(int nlevel);
	void setTypeKer(int typeKer);
	void setTypePyr(int typePyr);
	// calcul du flot
	void calculFolki();
	void swapImages();
	void setUnrolling(bool un);
	/* OUTPOUT*/
	/* recuperation du flot optique */
	void getFlow(float *u, float *v, int level= 0);	

	void setMgrid(bool val);	
	void setTMgrid(int val);

	void getDivRot(float *div, float *rot);
	void getRot(float *rot);
	void getDiv(float *div);
	void setDivRotOrdre(int val);
	void setLminCoef(double);

	// pour le rendu sans retour sur CPU
	void drawNorm(uint pbo, int colormap, float max,float min, float catCoeff, float seuil);
	void drawDiv(uint pbo,int colormap, float max, float seuil);
	void drawRot(uint pbo,int colormap, float max, float seuil);
	void drawDivRot(uint pbo,int colormap, float max,uint pbo2,int colormap2, float max2, float seuil);
	void drawVect(uint vbo, int spaceVect, float scale);
	void drawSrc(uint pbo);


	void useLminTalon(bool);
signals:
	void updateNormPbo();
	void updateDivPbo();
	void updateRotPbo();
	void updateSrcPbo();
	void updtateVectVbo();
	void newRes(float *I, float *u, float *v);

protected:
	/* buffers pour les calculs de It, I2w, G, H */
	float * _buff[NB_GPU_BUFF];
	/* données flot optique */
	float ***_I;
	float ** _Mxx, ** _Myy, ** _Mxy,** _D;
	float * _lMin;
	float **_Idy, **_Idx;
	float ** _u, ** _v;
	uchar * _Mask;
	float *_marker[2];
//	float H[9];
	bool bMarker;
	bool m_flushFlow;
	/* parametres */
	unsigned int _nItter;
#ifndef UNROLL_INNER
	unsigned int _kernelRadius;
#else
	unsigned int _kernelRadius=MAX_KERNEL_RADIUS;
#endif
	float _talon;
	unsigned int _nImages; // nombre d'image en meme temps dans la strutc (2 en général)

	int _algoBords;
	int _typeKer;
	bool _unRolling;

	bool _bMask;
	bool _lminTalon;
	bool _mgrid;
	int _tMgrid;
	int _sMgrid;
	unsigned int *_mGridStrategie;

	int divRotOrdre;

	float lMinCoeff;

	/* noyau de convolution pour la fenetre */
	float _kernel[MAX_KERNEL_RADIUS*2+1];

	/* ===== METHODES ==== */

	void _upMgridStrategie();

	/* mise a jour du noyau de la fenetre */
	void _setKernel();
	/*allocation des structures */
	void _allouStruct();

	/*fais le menage */
	void _nettoieStruct();

	/* calcul du flot */
	void _computeFlowFolki();
	/* propage le flot pour initialiser le temps suivant*/
//	void _propageFlow();
	void _flushFlow();
	void copyMat(float *a, float *b,int3 imSize);
//	void _initHomog();

	/* effectue un switche entre l'image 1 et 2, cela evite de recalculer la pyramide de I2 quand I2 devient I1*/
	void _swapImages(int a, int b);
	void _swapMarker(){
		float *tmp = _marker[0];
		_marker[0] = _marker[1];
		_marker[1] = tmp;
	}
};

} // namespace LIVA

#endif
