#ifndef _REC_ROBOTINO_COM_V1_MESSAGES_IOSTATUS_H_
#define _REC_ROBOTINO_COM_V1_MESSAGES_IOSTATUS_H_

#include <QByteArray>

#include "rec/iocontrol/remotestate/SensorState.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			namespace v1
			{
				namespace messages
				{
					class IOStatus
					{
					public:
						static void decode( const QByteArray& data, rec::iocontrol::remotestate::SensorState* sensorState );
					};
				}
			}
		}
	}
}

#endif //_REC_ROBOTINO_COM_V1_MESSAGES_IOSTATUS_H_
