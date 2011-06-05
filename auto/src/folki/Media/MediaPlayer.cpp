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

#include "MediaPlayer.hpp"
#include <opencv2/legacy/compat.hpp>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <QtGui>

namespace LIVA
{
	MediaPlayer::MediaPlayer(QWidget * parent) : QWidget(parent)
	{
		boolMask = false;
		_activeMask = false;
	}

	void 
	MediaPlayer::activateMask(bool val)
	{
			_activeMask = val;
	}

	void
	MediaPlayer::init()
	{
		cvInitSystem(0,NULL);
		pas = 1;
		normalize = false;
		CvSize frameSize;
		capture = NULL;
		capture = cvCaptureFromCAM(CV_CAP_ANY);
		cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH,320.0);
	 	cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT,240.0);

		framePtr = cvQueryFrame(capture);
		frameSize = cvGetSize(framePtr);
		std::cout << "capture ok\n";
		aux.create(frameSize, IPL_DEPTH_8U, 1);
		image[0].create(frameSize, IPL_DEPTH_32F, 1);
		image[1].create(frameSize, IPL_DEPTH_32F, 1);
		mask.create(frameSize, IPL_DEPTH_8U, 1);
		marker.create(frameSize, IPL_DEPTH_32F, 1);
		cvConvertImage(framePtr, aux);
		cvConvertScale(aux, image[1], 1/255.0);
		_index = -1;
		seekable = false;
		repeat = false;
		isCapture = true;
		size.x = image[0].size().width;
		size.y = image[0].size().height;
		size.z = image[0].step()/sizeof(float);
		emit initOut(size.x,size.y);
	}

	void
	MediaPlayer::setRepeat(bool val)
	{
		repeat = val;
	}



	void
	MediaPlayer::init(QStringList file)
	{
		CvSize frameSize;
		_files = file;
		_files.sort();
		if(_files.size() == 1){ // video file
			capture = NULL;
			capture = cvCaptureFromAVI( _files.at(0).toStdString().c_str() );
			framePtr = cvQueryFrame(capture);
			frameSize = cvGetSize(framePtr);
			std::cout << "capture ok\n";
			aux.create(frameSize, IPL_DEPTH_8U, 1);
			image[0].create(frameSize, IPL_DEPTH_32F, 1);
			image[1].create(frameSize, IPL_DEPTH_32F, 1);
			mask.create(frameSize, IPL_DEPTH_8U, 1);
			marker.create(frameSize, IPL_DEPTH_32F, 1);
			cvConvertImage(framePtr, aux);
			cvConvertScale(aux, image[0], 1/255.0);
			_index = -1;
			seekable = false;
			isCapture = true;
			size.x = image[0].size().width;
			size.y = image[0].size().height;
			size.z = image[0].step()/sizeof(float);
			emit initOut(size.x,size.y);
		}else{ // images
			_index = 0;
			pas = 1;
			emit setSliderMax(_files.size());
			aux=cvLoadImage(_files.at(_index).toStdString().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			frameSize = aux.size();
			image[0].create(frameSize, IPL_DEPTH_32F, 1);
			image[1].create(frameSize, IPL_DEPTH_32F, 1);
			mask.create(frameSize, IPL_DEPTH_8U, 1);
			marker.create(frameSize, IPL_DEPTH_32F, 1);
			cvConvertScale(aux, image[0], 1/255.0);
			isCapture = false;
			size.x = image[0].size().width;
			size.y = image[0].size().height;
			size.z = image[0].step()/sizeof(float);
			emit initOut(size.x,size.y);
		}
	}


	void
	MediaPlayer::setMaskList(QStringList val)
	{
		_maskList = val;
		aux=cvLoadImage(_maskList.at(0).toStdString().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		cvConvert(aux,mask);
		if(val.size() == _files.size()){
			boolMask = true;
		}else{
			emit newMask((uchar *)mask.data());
		}

	}
	void
	MediaPlayer::openMarker(QString str)
	{
		aux=cvLoadImage(str.toStdString().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		cvConvert(aux,marker);		
		emit newMarker((float *)marker.data());
	}


	MediaPlayer::~MediaPlayer()
	{
	}

	void 
	MediaPlayer::setPas(int n)
	{
		pas = n;
	}

	void 
	MediaPlayer::setIndex(int n)
	{
		_index = n;
	}

	void 
	MediaPlayer::giveOld(CvImage **val)
	{
		*val = &image[0];
	}

	void 
	MediaPlayer::setNormalize(bool val)
	{
		normalize = val;
	}


	void
	MediaPlayer::getCvImage(CvImage **im)
	{
		*im = &image[1];
	}

	void
	MediaPlayer::sendImage()
	{
		QTime tic;
		cvCopyImage(image[1],image[0]);
		if(isCapture){
			framePtr = cvQueryFrame(capture);

			std::cout << "pouet\n";
			if(framePtr != NULL){
				cvConvertImage(framePtr, aux);
				cvConvertScale(aux, image[1], 1/255.0);
				if(normalize)
					cvNormalize(image[1], image[1], 1, 0, CV_MINMAX);
				tic.restart();
				emit newImage((float*)image[1].data());
				emit transfert(tic.restart());
			}else{
				emit end();
			}
		}else{
			if(_index<_files.size()){
				emit indexUpdate(_index);
				aux=cvLoadImage(_files.at(_index).toStdString().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
				cvConvertScale(aux, image[1], 1/255.0);
				if(normalize)
					cvNormalize(image[1], image[1], 1, 0, CV_MINMAX);
				tic.restart();
				emit newImage((float*)image[1].data());
				emit transfert(tic.restart());
				_index+=pas;
				if(boolMask && _activeMask){
					aux=cvLoadImage(_maskList.at(_index).toStdString().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
					cvConvert(aux,mask);
					emit newMask((uchar *)mask.data());
				}
			}else{
				if(repeat){
					_index=0;
					emit indexUpdate(_index);
					aux=cvLoadImage(_files.at(_index).toStdString().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
					cvConvertScale(aux, image[1], 1/255.0);
					if(normalize)
						cvNormalize(image[1], image[1], 1, 0, CV_MINMAX);
					tic.restart();
					emit newImage((float*)image[1].data());
					emit transfert(tic.restart());
				}else{
					emit end();
					_index=1;
					emit indexUpdate(_index);
				}
			}
		}
	}



void
MediaPlayer::sendPaireImage()
{
	QTime tic;
	if(_index<_files.size()/2){
		emit indexUpdate(_index);
		aux=cvLoadImage(_files.at(2*_index).toStdString().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		cvConvertScale(aux, image[0], 1/255.0);
		if(normalize)
			cvNormalize(image[0], image[0], 1, 0, CV_MINMAX);
		tic.restart();
		emit newImage((float*)image[0].data());
		emit transfert(tic.restart());
	}else{
		if(repeat){
			_index=0;
			emit indexUpdate(_index);
			aux=cvLoadImage(_files.at(2*_index).toStdString().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			cvConvertScale(aux, image[0], 1/255.0);
			if(normalize)
				cvNormalize(image[0], image[0], 1, 0, CV_MINMAX);
			tic.restart();
			emit newImage((float*)image[0].data());
			emit transfert(tic.restart());
		}else{
			emit end();
			_index=0;
			emit indexUpdate(_index);
		}
	}
}

void
MediaPlayer::sendImpaireImage()
{
	QTime tic;
	if(_index<_files.size()/2){
		emit indexUpdate(_index);
		aux=cvLoadImage(_files.at(2*_index+1).toStdString().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		cvConvertScale(aux, image[1], 1/255.0);
		if(normalize)
			cvNormalize(image[1], image[1], 1, 0, CV_MINMAX);
		tic.restart();
		emit newImage((float*)image[1].data());
		emit transfert(tic.restart());
		_index+=pas;
	}else{
		_index=0;
		if(!repeat){
			emit end();
		}
	}
}


} //namespace LIVA
