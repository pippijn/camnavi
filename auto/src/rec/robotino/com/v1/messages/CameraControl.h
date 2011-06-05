#ifndef _REC_ROBOTINO_COM_V1_MESSAGES_CAMERACONTROL_H_
#define _REC_ROBOTINO_COM_V1_MESSAGES_CAMERACONTROL_H_

#include <QByteArray>

#include "rec/iocontrol/remotestate/SetState.h"

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
					class CameraControl
					{
					public:
						static QByteArray encode( unsigned int width, unsigned int height );
					};
				}
			}
		}
	}
}

#endif //_REC_ROBOTINO_COM_V1_MESSAGES_CAMERACONTROL_H_
