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

#ifndef __LI_MEDIA_PLAYER_HPP
#define __LI_MEDIA_PLAYER_HPP

#include <opencv2/legacy/legacy.hpp>
#include <opencv/highgui.h>
#include <QStringList>
#include <QtGui>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cuda.h>

namespace LIVA
{
class MediaPlayer: public QWidget {
    Q_OBJECT

	public:
	MediaPlayer(QWidget * parent = 0);
	~MediaPlayer();

	signals:
	void initOut(int w, int h);
	void newImage(float *);
	void newMask(uchar *);
	void newMarker(float *);	
	void end();
	void indexUpdate(int);
	void setSliderMax(int);
	void transfert(int);


	public slots:
	void init(QStringList files);
	void init();
	void getCvImage(CvImage **im);
	void giveOld(CvImage **val);
	void sendImage();
	void setPas(int);
	void setIndex(int);
	void setRepeat(bool);
	void setNormalize(bool);
	void sendImpaireImage();
	void sendPaireImage();
	void setMaskList(QStringList);
	void openMarker(QString);
	void activateMask(bool);

	private:
	bool normalize;
	int3 size;
	int pas;
	bool repeat;
	bool _activeMask;
	QStringList _files;
	QStringList _maskList;
	int _index;
	bool seekable;
	CvImage image[2];
	bool boolMask;
	CvImage mask;
	CvImage marker;
	IplImage *framePtr;
	CvImage  aux;
	CvCapture *capture;
	bool isCapture;
};
}// namespace LIVA
#endif
