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

 #ifndef LI_MAINWINDOW_H
 #define LI_MAINWINDOW_H
#include "GestionBuffer.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil.h>



#include <QtGui>
#include <QMainWindow>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "paramsFolki.hpp"
#include "FlotRender.hpp"
#include "LkCvWidget.hpp"
#include "LiGL2Dwidget.hpp"
#include "FolkiOpticalFlow.hpp"
#include "MediaPlayer.hpp"


class QAction;
class QListWidget;
class QMenu;
class QTextEdit;

using namespace LIVA;

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow()
	{

		setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);

		DockParams = new QDockWidget(this);
		params = new LiParams;
		DockParams->setWidget(params);
		DockParams->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);
		addDockWidget(Qt::RightDockWidgetArea,DockParams);
		player = new MediaPlayer;
		render = new FlotRender(this);
		lkwidget = new LkCvWidget;
		wFolki = new FolkiOpticalFlow;
		glwidget = new LiGL2D(this);
		glwidget->setMinimumSize(512,512);
		glwidget->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
		glwidget->resize(512,512);
		setCentralWidget (glwidget);

		connect(params, SIGNAL(cat(double)), render, SLOT(setCat(double)));
		connect(this , SIGNAL(destroyed()), this,SLOT(closeBrutal()));
		connect(params, SIGNAL(quit()),this,SLOT(closeBrutal()));
		connect(params, SIGNAL(quit()),lkwidget,SLOT(stop()));
		connect(params, SIGNAL(pause()), lkwidget,SLOT(pause()));
		connect(params, SIGNAL(refresh()), render, SLOT(actualiseFlow()));
		connect(player, SIGNAL(end()),params,SIGNAL(stop()));

		connect(params, SIGNAL(flushFlow(bool)), wFolki, SLOT(setFlushFlow(bool)));
		connect(params, SIGNAL(pas(int)), player, SLOT(setPas(int)));
		connect(lkwidget, SIGNAL(fps(double)), params, SIGNAL(fps(double)));
		connect(lkwidget, SIGNAL(fpsGlobal(double)), params, SIGNAL(fpsGlobal(double)));
		connect(params, SIGNAL( repeat(bool)), player, SLOT( setRepeat(bool)));
		connect(params, SIGNAL(pivMode(bool)) , lkwidget,SLOT(setPiv(bool)));

		// gestion du slider
		connect(params, SIGNAL(slider(int)), player, SLOT(setIndex(int)));
		connect(player, SIGNAL(indexUpdate(int)), params, SIGNAL(setSliderIndex(int)));
		connect(player, SIGNAL(indexUpdate(int)), render, SLOT(setIndex(int)));
		connect(player, SIGNAL(setSliderMax(int)), params, SLOT(setSliderMax(int)));
		connect(player, SIGNAL(transfert(int)), params, SIGNAL(transfert(int)));
		connect(render, SIGNAL(nMax(double)), params, SIGNAL(normMax(double)));
		// initiaisation
		connect(params, SIGNAL(load()),lkwidget,SLOT(load()));
		connect(params, SIGNAL(cam()), lkwidget,SLOT(cam()));
		connect(lkwidget, SIGNAL(initS()), player, SLOT(init()));
		connect(lkwidget, SIGNAL(enable(bool)), params, SIGNAL(enable(bool)));
		connect(lkwidget, SIGNAL(initS(QStringList)), player, SLOT(init(QStringList)));
		connect(player, SIGNAL(initOut(int,int)),render,SLOT(init(int,int)));
		connect(player, SIGNAL(initOut(int,int)),wFolki,SLOT(init(int,int)));
		connect(player, SIGNAL(initOut(int,int)),glwidget, SLOT(init(int,int)));
		connect(params,SIGNAL(delay(int)),render,SLOT(setdelay(int)));
		connect(params, SIGNAL(play()),lkwidget,SLOT(start()));
		connect(params, SIGNAL(stop()),lkwidget,SLOT(stop()));
		connect(glwidget, SIGNAL(sendPbo(uint)),render, SLOT( setPbo(uint)));
		connect(render,SIGNAL(drawPboNorm(uint,int,float,float,float,float)), wFolki, SLOT(drawNorm(uint,int,float,float,float,float)));
		connect(render,SIGNAL(drawPboRot(uint,int,float,float)), wFolki, SLOT(drawRot(uint,int,float,float)));
		connect(render,SIGNAL(drawPboDiv(uint,int,float,float)), wFolki, SLOT(drawDiv(uint,int,float,float)));
		connect(render,SIGNAL(drawFlotVect(uint,int,float)),wFolki, SLOT(drawVect(uint,int,float)));
		// params Folki
		connect(params, SIGNAL(nItter(int)), wFolki, SLOT(setnItter(int)));
		connect(params, SIGNAL(typePyr(int)), wFolki, SLOT(setTypePyr(int)));
		connect(params, SIGNAL(typeKernel(int)), wFolki, SLOT(setTypeKer(int)));
		connect(params, SIGNAL(nLevels(int)), wFolki, SLOT(setnLevel(int)));
		connect(params, SIGNAL(kernelRadius(int)), wFolki, SLOT(setKernelRadius(int)));
		connect(params, SIGNAL(bords(int)), wFolki, SLOT(setBords(int)));
		connect(params, SIGNAL(talon(double)), wFolki, SLOT(setTalon(double)));
		connect(params, SIGNAL(unRolling(bool)), wFolki, SLOT(setUnrolling(bool)));
		connect(params, SIGNAL(affFlot(bool)), glwidget, SLOT( setAffFlot(bool)));
		connect(params, SIGNAL(nSeuil(double)), render, SLOT( setnSeuil(double)));

		// render
		connect(lkwidget, SIGNAL(renderFlot()), render,SLOT(renderFlow()));
		connect(params, SIGNAL(normeMax(double)), render, SLOT( setNormMax(double)));
		connect(params, SIGNAL( setMinNorm(double)), render, SLOT( setNormMin(double)));
		
		connect(params, SIGNAL(vecteurSpace(int)), render, SLOT(setSpaceVect(int)));
		connect(params, SIGNAL(cmap(int)), render, SLOT(setColormap(int)));
		connect(params, SIGNAL(mulVecteur(double)), render, SLOT(setScalVect(double)));
		connect(params, SIGNAL( setSave(bool)), render, SLOT( setSave(bool)));
		connect(params, SIGNAL( urlChange(QString)), render, SLOT(setUrlSave(QString)));
		connect(params, SIGNAL(vecteurSpace(int) ), glwidget, SLOT(setVbo(int)));
		connect(render, SIGNAL(getCvImage(CvImage **)), player, SLOT(getCvImage(CvImage **)));
		connect(render, SIGNAL(getFlow(float*,float*)), wFolki, SLOT(getFlow(float*,float*)));
		connect(wFolki, SIGNAL(updateNormPbo()), glwidget, SLOT(draw()));
		connect(wFolki, SIGNAL(updateDivPbo()), glwidget, SLOT(draw()));
		connect(wFolki, SIGNAL(updateRotPbo()), glwidget, SLOT(draw()));
		connect(wFolki, SIGNAL(updtateVectVbo()), glwidget, SLOT(draw()));
		connect(glwidget, SIGNAL( sendVbo(uint)), render, SLOT( setVbo(uint)));
		// comput Flow
		connect(lkwidget, SIGNAL(grabIm()), player, SLOT(sendImage()));
		connect(lkwidget, SIGNAL(grabPaireIm()), player, SLOT(sendPaireImage()));
		connect(lkwidget, SIGNAL(grabImpaireIm()), player, SLOT(sendImpaireImage()));
		connect(lkwidget, SIGNAL(calculFlot()), wFolki, SLOT(calculFolki()));
		connect(lkwidget, SIGNAL( swapIm()), wFolki, SLOT( swapImages()));
		connect(player, SIGNAL( newImage(float *)), wFolki, SLOT( setImage(float*)));
		connect(params, SIGNAL(affFlot(bool)), render, SLOT( setAffFlot(bool)));
		connect(params, SIGNAL(affNorm(bool)), render, SLOT( setAffNorm(bool)));
		connect(params, SIGNAL(affSrc(bool)), render, SLOT( setAffSrc(bool)));
		connect( params, SIGNAL( mGrid(bool)), wFolki,SLOT( setMgrid(bool)));
		connect(params, SIGNAL( stratMGrid(int)),wFolki,SLOT( setTMgrid(int)));
		connect(params, SIGNAL(normalize(bool)), player, SLOT(setNormalize(bool)));
		connect(params, SIGNAL(flip(bool)), render, SLOT(setFlip(bool)));
		connect(params, SIGNAL(affDiv(bool)), render, SLOT(setDiv(bool)));
		connect(params, SIGNAL(affRot(bool)), render, SLOT(setRot(bool)));
		connect(params, SIGNAL(affNorm(bool)), glwidget, SLOT(setDrawIm(bool)));
		connect(params, SIGNAL(affSrc(bool)), glwidget, SLOT(setDrawIm(bool)));
		connect(params, SIGNAL(affRot(bool)), glwidget, SLOT(setDrawIm(bool)));
		connect(params, SIGNAL(affDiv(bool)), glwidget, SLOT(setDrawIm(bool)));
		connect(params, SIGNAL(divMax(double)), render, SLOT(setDivMax(double)));
		connect(params, SIGNAL(divCmap(int)), render, SLOT(setColormapDiv(int)));
		connect(params, SIGNAL(rotMax(double)), render, SLOT(setRotMax(double)));
		connect(params, SIGNAL(rotCmap(int)), render, SLOT(setColormapRot(int)));
		connect(render, SIGNAL(getDiv(float *)), wFolki, SLOT(getDiv(float *)));
		connect(render, SIGNAL(getDivRot(float *,float *)), wFolki, SLOT(getDivRot(float *,float *)));
		connect(render, SIGNAL(getRot(float *)), wFolki, SLOT(getRot(float *)));
		connect(params, SIGNAL(flotNorm(bool)), render, SLOT( setFlotNorm(bool)));
		connect(params, SIGNAL(divCurlOrdre(int)), wFolki, SLOT(setDivRotOrdre(int)));
		connect(params, SIGNAL(fliX11()), render, SLOT(fli()));
		connect(render, SIGNAL(getOld(CvImage **)), player, SLOT( giveOld(CvImage **)));
		connect(params, SIGNAL(flotAff(int)), render, SLOT(setFlotVal(int)));
		connect(params, SIGNAL(flotCmap(int)),render, SLOT(setFlotCmap(int)));

		connect(params, SIGNAL(saveNorm(bool)), render, SLOT(setSaveNorm(bool)));
		connect(params, SIGNAL( updateLineSaveNorm(QString)),render, SLOT(setUrlSaveNorm(QString)));
		connect(params, SIGNAL(fondFlot(int)), render, SLOT( setFondFlot(int)));
		connect(params, SIGNAL(vectCmap(bool)), render, SLOT(setVectCmap(bool)));
		connect(params, SIGNAL(vectColor(int)), render, SLOT(setVectColor(int)));
		connect(render, SIGNAL(saveTheNorm(QString)), glwidget, SLOT(save(QString)));
		connect(render, SIGNAL(drawPboSrc(uint)), wFolki, SLOT(drawSrc(uint)));
		connect(wFolki, SIGNAL(updateSrcPbo()), glwidget, SLOT(draw()));
		connect(params, SIGNAL(openMarker(QString)), player, SLOT(openMarker(QString)));
		connect(params, SIGNAL(activateMarker(bool)), wFolki, SLOT(activateMarker(bool)));
		connect(params, SIGNAL(backward(bool)),lkwidget,SLOT(setBackward(bool)));
		connect(player, SIGNAL(newMarker( float *)), wFolki, SLOT(setMarker( float *)));


	
		// gestion du mask 
		connect(params,SIGNAL(clickMask()),lkwidget,SLOT(exploMask()));
		connect(params,SIGNAL(activateMask(bool)),player, SLOT(activateMask(bool)));
		connect(params,SIGNAL(activateMask(bool)),wFolki, SLOT(activateMask(bool)));
		connect(lkwidget,SIGNAL(maskList(QStringList)),player,SLOT(setMaskList(QStringList)));
		connect(player,SIGNAL(newMask(uchar *)), wFolki, SLOT(setMask(uchar *)));
		connect(params,SIGNAL(setLmin(double)),wFolki,SLOT(setLminCoef(double)));
		connect(params,SIGNAL(talonLmin(bool)),wFolki,SLOT(useLminTalon(bool)));
	}

public slots:
	void closeBrutal(){ exit(0); }


private:
	LkCvWidget *lkwidget;
	FolkiOpticalFlow *wFolki;
	MediaPlayer *player;
	FlotRender *render;
	QDockWidget *DockParams;
	LiParams *params;
	QTimer *timout;
	LiGL2D *glwidget;
};
#endif
