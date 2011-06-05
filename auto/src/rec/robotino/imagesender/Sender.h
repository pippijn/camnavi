#ifndef _REC_ROBOTINO_IMAGESENDER_SENDER_H_
#define _REC_ROBOTINO_IMAGESENDER_SENDER_H_

#include "rec/robotino/imagesender/defines.h"

#include <QObject>
#include <QUdpSocket>
#include <QByteArray>
#include <QTimer>

namespace rec
{
	namespace robotino
	{
		namespace imagesender
		{
			class REC_ROBOTINO_IMAGESENDER_EXPORT Sender : public QUdpSocket
			{
				Q_OBJECT
			public:
				typedef enum { IdleState, SendingState } State;

				Sender( QObject* parent );

				State state() const { return _state; }

				/**
				Stop sending current image data. state() will return IdleState after calling stop().
				*/
				void stop();

#ifdef QT_NO_KEYWORDS
			public Q_SLOTS:
#else
			public slots:
#endif
				void setRawImageData( const QByteArray& data,
																unsigned int width,
																unsigned int height,
																unsigned int numChannels,
																unsigned int bitsPerChannel,
																unsigned int step );

				void setJpgImageData( const QByteArray& data );

				void setReceiver( quint32 address, quint16 port );

#ifdef QT_NO_KEYWORDS
			Q_SIGNALS:
#else
			signals:
#endif
				void imageSendingCompleted();

#ifdef QT_NO_KEYWORDS
			private Q_SLOTS:
#else
			private slots:
#endif
				void on_timer_timeout();

			private:
				static const int _startSequenceSize;
				static const qint8 _startSequence[10];

				static const int _stopSequenceSize;
				static const qint8 _stopSequence[10];

				void setImageData( bool isJpg,
					const QByteArray& data,
					unsigned int width,
					unsigned int height,
					unsigned int numChannels,
					unsigned int bitsPerChannel,
					unsigned int step );

				QHostAddress _receiver;
				quint16 _receiverPort;

				QByteArray _imageData;
				qint32 _bytesWritten;

				qint32 _partSize;

				State _state;

				QByteArray _header;

				QTimer* _timer;
			};
		}
	}
}

#endif //_REC_ROBOTINO_IMAGESENDER_SENDER_H_
