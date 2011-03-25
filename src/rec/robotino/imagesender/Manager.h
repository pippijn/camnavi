#ifndef _REC_ROBOTINO_IMAGESENDER_MANAGER_H_
#define _REC_ROBOTINO_IMAGESENDER_MANAGER_H_

#include "rec/robotino/imagesender/defines.h"
#include "rec/robotino/imagesender/Sender.h"
#include "rec/robotino/imagesender/ImageMemory.h"

#include <QObject>
#include <QMap>
#include <QSharedMemory>

namespace rec
{
	namespace robotino
	{
		namespace imagesender
		{
			class Receiver
			{
			public:
				Receiver( quint32 addr, quint16 port_ )
					: address( addr ),
					port( port_ )
				{
				}

				Receiver()
					: address( 0 ),
					port( 0 )
				{
				}

				quint32 address;
				quint16 port;
			};

			class REC_ROBOTINO_IMAGESENDER_EXPORT Manager : public QObject
			{
				Q_OBJECT
			public:
				Manager( QObject* parent );

				~Manager();

				int numReceivers() const { return _senders.size(); }

				int numLocalReceivers() const { return _numLocalReceivers; }

				Sender::State state() const;

				/**
				Stop sending current image data. state() will return IdleState after calling stop().
				*/
				void stop();

				void createSharedMemory( quint16 serverPort );
				void releaseSharedMemory();

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

				void setLocalRawImageData( const QByteArray& data,
																unsigned int width,
																unsigned int height,
																unsigned int numChannels,
																unsigned int bitsPerChannel,
																unsigned int step );

				void setLocalJpgImageData( const QByteArray& data );

				void addReceiver( quint32 address, quint16 port );
				
				void removeReceiver( quint32 address, quint16 port );

				//remove all receivers
				void reset();

#ifdef QT_NO_KEYWORDS
			Q_SIGNALS:
#else
			signals:
#endif
				void imageSendingCompleted();

			private:
				QMap< Receiver, Sender* > _senders;
				int _numLocalReceivers;
				int _numSendesOnSetImageData;

				QSharedMemory _sharedMemory;
				ImageMemory* _imageMemory;

#ifdef QT_NO_KEYWORDS
			private Q_SLOTS:
#else
			private slots:
#endif
				void on_sender_imageSendingCompleted();
			};
		}
	}
}

QT_BEGIN_NAMESPACE
inline bool operator<( const rec::robotino::imagesender::Receiver& r1, const rec::robotino::imagesender::Receiver& r2 )
{
	if( r1.address < r2.address )
	{
		return true;
	}
	else if( r1.address == r2.address )
	{
		return ( r1.port < r2.port );
	}
	else
	{
		return false;
	}
}
QT_END_NAMESPACE


#endif //_REC_ROBOTINO_IMAGESENDER_MANAGER_H_
