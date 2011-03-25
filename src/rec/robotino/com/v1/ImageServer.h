#ifndef _REC_ROBOTINO_COM_V1_IMAGESERVER_H_
#define _REC_ROBOTINO_COM_V1_IMAGESERVER_H_

#include <QObject>
#include <QUdpSocket>

#include "rec/robotino/com/ComImpl.hh"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			namespace v1
			{
				class ImageServer : public QUdpSocket
				{
					Q_OBJECT
				public:
					ImageServer( QObject* parent );

					void update();

				Q_SIGNALS:
					void imageReceived( const QByteArray& imageData,
						                  int type,
															unsigned int width,
															unsigned int height,
															unsigned int numChannels,
															unsigned int bitsPerChannel,
															unsigned int step );

				private:
					void processDatagram( const QByteArray& data );
					bool decodeHeader( const QByteArray& data );
					bool findStopSequence( const QByteArray& data );

					static const unsigned int _startSequenceSize;
					static const quint8 _startSequence[10];
					
					static const unsigned int _stopSequenceSize;
					static const quint8 _stopSequence[10];

					static const unsigned int _headerSize;

					QByteArray _imageData;

					rec::robotino::com::ComImpl::ImageType_t _imageType;
					unsigned int _width;
					unsigned int _height;
					unsigned int _numColorChannels;
					unsigned int _bitsPerChannel;

					int _offset;
					int _nparts;
					int _startOffset;
				};
			}
		}
	}
}

#endif //_REC_ROBOTINO_COM_V1_IMAGESERVER_H_