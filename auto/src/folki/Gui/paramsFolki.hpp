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

#ifndef __LI_PARAMSFOLKI_H
#define __LI_PARAMSFOLKI_H

#include <QtGui>
#include "ui_paramsFolki.h"

namespace LIVA {
    class LiParams: public QTabWidget,  private Ui::LiParams {
	Q_OBJECT
public:
	LiParams(QWidget *parent = 0);

public slots:
	void setSliderMax(int);
	void urlButton();
	void urlSaveNormButton();
	void markerButton();


signals:
	void openMarker(QString);
	void activateMarker(bool);
	void backward(bool);
	void flushFlow(bool);

	void talonLmin(bool);
	void clickMask();
	void activateMask(bool);
	void setLmin(double);
	void setMinNorm(double);
	void setSliderIndex(int);
	void enable(bool);
	void nItter(int);
	void typePyr(int);
	void typeKernel(int);
	void nLevels(int);
	void kernelRadius(int);
	void bords(int);
	void unRolling(bool);
	void masque(bool);
	void talon(double);
	void gradient(int);
	void initFlot(int);
	void cmap(int);
	void distCmap(int);
	void distMax(double);
	void normeMax(double);
	void vecteurSpace(int);
	void mulVecteur(double);
	void play();
	void stop();
	void pause();
	void delay(int);
	void slider(int);
	void pas(int);
	void refresh();
	void load();
	void cam();
	void fps(int);
	void transfert(int);
	void normMax(double);
	void fps(double);
	void quit();
	void updateLineSave(QString);
	void urlChange(QString);
	void setSave(bool);
	void pivMode(bool);
	void repeat(bool);
	void affFlot(bool);
	void affNorm(bool);
	void affSrc(bool);
	void fpsGlobal(double);
	void mGrid(bool);
	void stratMGrid(int);
	void normalize(bool);
	void bibtex();
	void flip(bool);
	void affDiv(bool);
	void divMax(double);
	void divCmap(int);
	void affRot(bool);
	void rotMax(double);
	void rotCmap(int);
	void flotNorm(bool);
	void radiusFosi(int);
	void divCurlOrdre(int);
	void fliX11();
	void flotAff(int);
	void flotCmap(int);
	void fondFlot(int);
	void cat(double);
	void nSeuil(double);
	void saveFlot(bool);
	void updateLineSaveFlotIm(QString);
	void saveNorm(bool);
	void updateLineSaveNorm(QString);
	void saveDiv(bool);
	void updateLineSaveDiv(QString);
	void saveCurl(bool);
	void updateLineSaveCurl(QString);

	void flotRes(bool);
	void flotResInf(int);
	void flotResSup(int);
	void locHomogOrdre(int);
	void vectCmap(bool);
	void vectColor(int);
	void pboNorm(bool);
	void pboDiv(bool);
	void pboRot(bool);
	void visible(int);


	private:
	QString _url;
	QString _urlNorm;
};
} // namespace LIVA

#endif // __LI_PARAMSFOKI_H
