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

#include <GL/glew.h>
#include <GL/glut.h>
#include "LkCvWidget.hpp"
#include <cuda_runtime.h>
#include <cutil.h>
#include <cuda.h>
#include <QTime>

namespace LIVA
{
LkCvWidget::LkCvWidget(QWidget * parent):QThread(parent)
{
	piv = false;
	sr = false;
	bBacward = false;
}

void
LkCvWidget::init(QStringList files)
{
	emit enable(true);
	emit initS(files);
}

void
LkCvWidget::setPiv(bool val)
{
	piv = val;
}


void
LkCvWidget::load()
{
	init(QFileDialog::getOpenFileNames(0,tr("Open File"), QDir::currentPath(),"*"));
}

void
LkCvWidget::exploMask()
{
	emit maskList(QFileDialog::getOpenFileNames(0,tr("Accouche les Masques!!!"), QDir::currentPath(),"*"));
}

void 
LkCvWidget::cam()
{
	emit enable(true);
	emit initS();
}

void
LkCvWidget::run()
{
	QTime tic,toc;
	int m;
	while(isPlaying){
		toc.restart();
		
		if(piv){
			emit grabPaireIm();
			emit swapIm();
			emit grabImpaireIm();
		}else{
			if(sr){
				emit swapIm();
				emit grabIm();
				emit swapIm();
			}else{
				emit swapIm();
				emit grabIm();
			}
		}
		tic.restart();
		if(bBacward)
			emit swapIm();
		emit calculFlot();
		m = tic.restart();
		emit fps((double)m);
		emit renderFlot();
		emit fpsGlobal((double)toc.restart());
		if(bBacward)
			emit swapIm();

	}
	isPlaying = false;
}

void
LkCvWidget::stop()
{
	isPlaying = false;
	emit nettoie();
}

void
LkCvWidget::setBackward(bool val)
{
	bBacward = val;
}


void
LkCvWidget::setSR(bool val)
{
	sr = val;
}

void
LkCvWidget::pause()
{
	isPlaying = false;
}

void
LkCvWidget::start()
{
	isPlaying = true;
	run();
}


}
