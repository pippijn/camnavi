#ifndef _REC_ROBOTINO_COM_V1_MESSAGES_IOCONTROL_H_
#define _REC_ROBOTINO_COM_V1_MESSAGES_IOCONTROL_H_

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
					class IOControl
					{
					public:
						static QByteArray encode( const rec::iocontrol::remotestate::SetState& setState );
					};
				}
			}
		}
	}
}

#endif //_REC_ROBOTINO_COM_V1_MESSAGES_IOCONTROL_H_
