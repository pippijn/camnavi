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

#ifndef __LI_LK_WIDGET_HPP
#define __LI_LK_WIDGET_HPP

#include <QThread>
#include <QFileDialog>

namespace LIVA
{
class LkCvWidget : public QThread
 {
	Q_OBJECT
	public:
	LkCvWidget(QWidget * parent =0);

	public slots:
	void init(QStringList files);
	void load();
	void stop();
	void pause();
	void start();
	void cam();
	void setPiv(bool);
	void setSR(bool);
	void exploMask();
	void setBackward(bool);

	signals:
	void enable(bool);
	void calculFlot();
	void renderFlot();
	void initS(QStringList files);
	void maskList(QStringList);
	void initS();
	void nettoie();
	void grabIm();
	void grabPaireIm();
	void grabImpaireIm();
	void swapIm();
	void fps(double);
	void fpsGlobal(double);

	private:
	void run();
	bool isPlaying;
	bool piv;
	bool sr;
	bool bBacward;
	int visible;
};
}
#endif
