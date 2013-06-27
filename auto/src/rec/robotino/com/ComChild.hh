//  Copyright (C) 2004-2010, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_COMCHILD_HH_
#define _REC_ROBOTINO_COM_COMCHILD_HH_

#include <string>

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			class ComChild
			{
			public:
				ComChild()
				{
				}

				virtual ~ComChild()
				{
				}

				virtual void connectToServer() = 0;

				virtual void disconnectFromServer() = 0;

				virtual void setAddress( const char* address ) = 0;
			};
		}
	}
}

#endif //_REC_ROBOTINO_COM_COMCHILD_HH_
