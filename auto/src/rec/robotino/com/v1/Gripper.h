#ifndef _REC_ROBOTINO_COM_V1_GRIPPER_H_
#define _REC_ROBOTINO_COM_V1_GRIPPER_H_

#include <QTime>

#include "rec/robotino/com/v1/MeanBuffer.h"
#include "rec/iocontrol/remotestate/SetState.h"
#include "rec/iocontrol/remotestate/SensorState.h"

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			namespace v1
			{
				class Gripper
				{
				public:
					Gripper();

					void set( rec::iocontrol::remotestate::SetState* setState );

					void set( rec::iocontrol::remotestate::SensorState* sensorState );

					void reset();

				private:
					MeanBuffer _current;
					bool _isEnabled;
					bool _isGripperClosed;

					enum { Gripper_Open, Gripper_Close, Gripper_NoCommand } _gripperSetAction;
					enum { GripperCur_MovingClose, GripperCur_MovingOpen, GripperCur_Idle, GripperCur_Starting, GripperCur_Undefined } _gripperCurAction;

					QTime _firstTimeCurrentToHigh;

					bool _isNewPowerOutputControlPoint;
					int _powerOutputControlPoint;
				};
			}
		}
	}
}

#endif //_REC_ROBOTINO_COM_V1_GRIPPER_H_