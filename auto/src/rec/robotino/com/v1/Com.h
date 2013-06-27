#ifndef _REC_ROBOTINO_COM_V1_COM_H_
#define _REC_ROBOTINO_COM_V1_COM_H_

#include <QObject>
#include <QTcpSocket>

#include "rec/robotino/com/Com.h"
#include "rec/iocontrol/remotestate/SetState.h"
#include "rec/iocontrol/remotestate/SensorState.h"
#include "rec/robotino/com/v1/Gripper.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			namespace v1
			{
				class ImageServer;

				class Com : public QTcpSocket
				{
					Q_OBJECT
				public:
					Com( QObject* parent );

					void sendIOControl( rec::iocontrol::remotestate::SetState* setState );

					void sendCameraControl( unsigned int width, unsigned int height );

					bool update( rec::iocontrol::remotestate::SensorState* sensorState );

					rec::robotino::com::Com::Error comError() const;

					unsigned int imageServerPort() const;

				Q_SIGNALS:
					void infoReceived( const QString& text, bool isPassiveMode );
					void imageReceived( const QByteArray& imageData,
						                  int type,
															unsigned int width,
															unsigned int height,
															unsigned int numChannels,
															unsigned int bitsPerChannel,
															unsigned int step );

				private:
					ImageServer* _imageServer;
					int _readTimeout;
					Gripper _gripper;

				private Q_SLOTS:
					void on_stateChanged( QAbstractSocket::SocketState );
					void on_bytesWritten( qint64 bytes );
					void on_readyRead();
				};
			}
		}
	}
}

#endif //_REC_ROBOTINO_COM_V1_COM_H_
