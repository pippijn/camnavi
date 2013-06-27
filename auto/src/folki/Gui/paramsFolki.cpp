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

#include "paramsFolki.hpp"

namespace LIVA
{

LiParams::LiParams(QWidget *parent):QTabWidget(parent){
	setupUi(this);
	connect( radiolminTalon , SIGNAL(toggled(bool)), this, SIGNAL(talonLmin(bool)));
	connect( pushMask , SIGNAL(clicked()), this, SIGNAL(clickMask()));
	connect( maskBox , SIGNAL(toggled(bool)), this, SIGNAL(activateMask(bool)));
	connect( LminSeuil,  SIGNAL(valueChanged(double)), this, SIGNAL(setLmin(double)));
	connect( playButton ,SIGNAL(clicked()),this,SIGNAL(play()));
	connect( stopButton ,SIGNAL(clicked()),this,SIGNAL(stop()));
	connect( loadButton , SIGNAL(clicked()),this,SIGNAL(load()));
	connect( videoButton , SIGNAL(clicked()),this,SIGNAL(cam()));
	connect( quitButton ,SIGNAL(clicked()),this,SIGNAL(quit()));
	connect( comboPyr ,SIGNAL(currentIndexChanged(int)),this,SIGNAL(typePyr(int)));
	connect( comboFlat ,SIGNAL(currentIndexChanged(int)),this,SIGNAL(typeKernel(int)));
	connect( spinItter, SIGNAL(valueChanged(int)),this,SIGNAL(nItter(int)));
	connect( spinLevel, SIGNAL(valueChanged(int)),this,SIGNAL(nLevels(int)));
	connect( checkUnRolling, SIGNAL(toggled(bool)),this,SIGNAL(unRolling(bool)));
	connect( spinKernelRadius, SIGNAL(valueChanged(int)), this,SIGNAL(kernelRadius(int)));
	connect( spinBords, SIGNAL(valueChanged(int)), this, SIGNAL(bords(int)));
	connect( talonSpin, SIGNAL(valueChanged(double)), this, SIGNAL(talon(double)));
	connect( comboGradient, SIGNAL(currentIndexChanged(int)),this,SIGNAL(gradient(int)));
	connect( comboInitFlot, SIGNAL(currentIndexChanged(int)),this,SIGNAL(initFlot(int)));
	connect( comboCmap, SIGNAL(currentIndexChanged(int)), this, SIGNAL(cmap(int)));
	connect( nomrMaxSpin, SIGNAL(valueChanged(double)), this, SIGNAL(normeMax(double)));
	connect( nomrMinSpin, SIGNAL(valueChanged(double)), this, SIGNAL(setMinNorm(double)));	
	connect( spaceSpin, SIGNAL(valueChanged(int)), this, SIGNAL(vecteurSpace(int)));
	connect( spinVectMul, SIGNAL(valueChanged(double)), this, SIGNAL(mulVecteur(double)));
	connect( pasSpin, SIGNAL(valueChanged(int)), this, SIGNAL(pas(int)));
	connect( delaySpin, SIGNAL(valueChanged(int)), this, SIGNAL(delay(int)));
	connect( pauseButton ,SIGNAL(clicked()),this,SIGNAL(pause()));
	connect( refreshButton ,SIGNAL(clicked()),this,SIGNAL(refresh()));
	connect( hSlider ,SIGNAL(sliderMoved(int)),this,SIGNAL(slider(int)));
	connect( this, SIGNAL(setSliderIndex(int)),hSlider,SLOT(setValue(int)));

	connect(affFlotBox , SIGNAL(toggled(bool)), this, SIGNAL(affFlot(bool)));
	connect(affNormBox , SIGNAL(toggled(bool)), this, SIGNAL(affNorm(bool)));
	connect(affSrcBox , SIGNAL(toggled(bool)), this, SIGNAL(affSrc(bool)));
	connect( comboCmap, SIGNAL(currentIndexChanged(int)), this, SIGNAL(cmap(int)));

	connect(comboDivCmap, SIGNAL(currentIndexChanged(int)), this, SIGNAL(divCmap(int)));
	connect(comboCurlCmap, SIGNAL(currentIndexChanged(int)), this, SIGNAL(rotCmap(int)));
	connect( divMaxSpin, SIGNAL(valueChanged(double)), this, SIGNAL(divMax(double)));
	connect( curlMaxSpin, SIGNAL(valueChanged(double)), this, SIGNAL(rotMax(double)));
	connect(divBox ,SIGNAL(toggled(bool)), this, SIGNAL(affDiv(bool)));
	connect(curlBox ,SIGNAL(toggled(bool)), this, SIGNAL(affRot(bool)));

	connect( normalizeBox, SIGNAL(toggled(bool)), this, SIGNAL(normalize(bool)));

	connect( flipBox, SIGNAL(toggled(bool)), this, SIGNAL(flip(bool)));
	connect(flushFlowBox, SIGNAL(toggled(bool)), this,SIGNAL( flushFlow(bool)));

	connect(this, SIGNAL(enable(bool)), LminSeuil ,SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), flushFlowBox, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), delaySpin ,SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), pasSpin ,SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), refreshButton ,SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), pauseButton ,SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), playButton ,SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), stopButton ,SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), comboPyr ,SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), comboFlat ,SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), spinItter, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), spinLevel, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), checkUnRolling, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), spinKernelRadius, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), spinBords, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), talonSpin, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), comboGradient, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), comboInitFlot, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), comboCmap, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), nomrMaxSpin, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), nomrMinSpin, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), maskBox, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)),pushMask, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), spaceSpin, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), spinVectMul, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), pivBox, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), checkSave, SLOT(setEnabled(bool)));
	connect(this,SIGNAL(enable(bool)), affFlotBox , SLOT(setEnabled(bool)));
	connect(this,SIGNAL(enable(bool)), affNormBox , SLOT(setEnabled(bool)));
	connect(this,SIGNAL(enable(bool)), affSrcBox , SLOT(setEnabled(bool)));
	connect(this,SIGNAL(enable(bool)), mGridBox , SLOT(setEnabled(bool)));
	connect(this,SIGNAL(enable(bool)), normalizeBox , SLOT(setEnabled(bool)));
	connect(this,SIGNAL(enable(bool)), flipBox , SLOT(setEnabled(bool)));
	connect(this,SIGNAL(enable(bool)), divBox , SLOT(setEnabled(bool)));
	connect(this,SIGNAL(enable(bool)), curlBox , SLOT(setEnabled(bool)));
	connect(this,SIGNAL(enable(bool)), nSeuilBox,  SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), loadButton, SLOT(setDisabled(bool)));
	connect(this, SIGNAL(enable(bool)), videoButton, SLOT(setDisabled(bool)));

	connect(this, SIGNAL(enable(bool)),backwardBox, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)),openMarkerButton, SLOT(setEnabled(bool)));
	connect(this, SIGNAL(enable(bool)), activeMarkerBox, SLOT(setEnabled(bool)));

	connect(backwardBox, SIGNAL(toggled(bool)), this, SIGNAL(backward(bool)));
	connect(activeMarkerBox, SIGNAL(toggled(bool)), this, SIGNAL(activateMarker(bool)));
	connect(openMarkerButton,SIGNAL(clicked()), this, SLOT(markerButton()));

	connect(this, SIGNAL(fps(double)), fpsLcd, SLOT(display(double)));
	connect(this, SIGNAL(fpsGlobal(double)), fpsGlobalLcd, SLOT(display(double)));
	connect(this, SIGNAL(normMax(double)), normLcd, SLOT(display(double)));
	connect(this, SIGNAL(transfert(int)), transfertLcd, SLOT(display(int)));
	connect(this, SIGNAL(fps(int)), fpsLcd, SLOT(display(int)));
	connect(saveButton, SIGNAL(clicked()), this, SLOT(urlButton()));
	connect(this, SIGNAL(updateLineSave(QString)), lineSave, SLOT( setText (QString)));
	connect(this, SIGNAL(updateLineSave(QString)), this, SIGNAL( urlChange(QString)));
	connect(pivBox, SIGNAL(toggled(bool)),this,SIGNAL(pivMode(bool)));
	connect(repeatBox, SIGNAL(toggled(bool)), this, SIGNAL(repeat(bool)));
	connect(checkSave, SIGNAL( toggled(bool)), this, SIGNAL(setSave(bool)));
	connect(mGridBox, SIGNAL(toggled(bool)), this, SIGNAL( mGrid(bool)));
	connect(stratMgridBox, SIGNAL(currentIndexChanged(int)), this, SIGNAL(stratMGrid(int)));
	connect(flotNormBox, SIGNAL(toggled(bool)), this, SIGNAL(flotNorm(bool)));
	connect(curlOrdreBox, SIGNAL(valueChanged(int)),this,SIGNAL(divCurlOrdre(int)));
	connect(fliButton, SIGNAL(clicked()),this,SIGNAL(fliX11()));
	connect(comboFlotVal, SIGNAL(currentIndexChanged(int)),this,SIGNAL(flotAff(int)));
	connect(flotCmapBox, SIGNAL(currentIndexChanged(int)),this,SIGNAL(flotCmap(int)));


	connect(saveNormButton, SIGNAL(clicked()), this, SLOT(urlSaveNormButton()));
	connect(this, SIGNAL(updateLineSaveNorm(QString)), lineSaveNorm ,SLOT(setText (QString)));
	connect(checkSave,SIGNAL(toggled(bool)), this,SIGNAL(setSave(bool)));
	connect(saveNormBox,SIGNAL(toggled(bool)), this,SIGNAL(saveNorm(bool)));
	connect(fondFlotBox, SIGNAL(currentIndexChanged(int)), this, SIGNAL(fondFlot(int)));
	connect( vectCmapBox,SIGNAL(toggled(bool)), this, SIGNAL(vectCmap(bool)));
	connect( vectColorBox,SIGNAL(currentIndexChanged(int)), this, SIGNAL(vectColor(int)));
	connect( catBox, SIGNAL( valueChanged(double)),this,SIGNAL(cat(double)));
	connect( nSeuilBox, SIGNAL( valueChanged(double)),this,SIGNAL(nSeuil(double)));

	lineSave->setReadOnly(true);
	groupNormBox->hide();
	groupCurlBox->hide();
	groupDivBox->hide();
	groupFlotBox->hide();
	fli->hide();
	
	// pas encore implemente
	label_15->hide();
	comboGradient->hide();
	label_16->hide();
	comboInitFlot->hide();
	flotWidget->hide();

	comboFlotVal->hide();
}

void
LiParams::setSliderMax(int max)
{
	hSlider->setEnabled(true);
	repeatBox->setEnabled(true);
	hSlider->setMaximum(max);
}

void
LiParams::urlButton()
{
	_url = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
											 QDir::currentPath(),
											QFileDialog::ShowDirsOnly
											| QFileDialog::DontResolveSymlinks);
	emit setSave(true);
	emit updateLineSave(_url);
}

void
LiParams::markerButton()
{
	QStringList toto;
	toto =QFileDialog::getOpenFileNames(0,tr("Accouche les Masques!!!"), QDir::currentPath(),"*"); 	emit openMarker(toto.at(0));
}

void
LiParams::urlSaveNormButton()
{
	_urlNorm = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
											 QDir::currentPath(),
											QFileDialog::ShowDirsOnly
											| QFileDialog::DontResolveSymlinks);
	emit saveNorm(true);
	emit updateLineSaveNorm(_urlNorm);
}

}//namespace LIVA
