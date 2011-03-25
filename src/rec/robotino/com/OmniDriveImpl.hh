//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_OMNIRIVEIMPL_H_
#define _REC_ROBOTINO_COM_OMNIRIVEIMPL_H_

#include "rec/iocontrol/robotstate/State.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class OmniDriveImpl
			{
			public:
				OmniDriveImpl()
				{
				}

				rec::iocontrol::robotstate::DriveLayout layout;
			};
		}
	}
}

#endif
