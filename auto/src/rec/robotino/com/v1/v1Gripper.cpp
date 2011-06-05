#include "rec/robotino/com/v1/Gripper.h"

using namespace rec::robotino::com::v1;

Gripper::Gripper()
{
	reset();
}

void Gripper::set( rec::iocontrol::remotestate::SetState* setState )
{
	if( false == setState->gripper_isEnabled )
	{
		reset();
		return;
	}

	_isEnabled = true;

	if( setState->gripper_close )
  {
	  _gripperSetAction = Gripper_Close;
  }
  else
  {
	  _gripperSetAction = Gripper_Open;
  }

	if( _isNewPowerOutputControlPoint )
	{
		_isNewPowerOutputControlPoint = false;
		setState->powerOutputControlPoint = _powerOutputControlPoint ;
	}

	// action requested
	if( _gripperSetAction == Gripper_Close || _gripperSetAction == Gripper_Open )
	{
		if( _gripperSetAction == Gripper_Close )
		{
			if( _gripperCurAction == GripperCur_Idle && _isGripperClosed )
			{
				// nothing to do
			}
			else
			{
				setState->powerOutputControlPoint = 255 ;
				if( _gripperCurAction == GripperCur_Idle || _gripperCurAction == GripperCur_Undefined )
				{
					_gripperCurAction = GripperCur_Starting;
				}
				_isGripperClosed = false;
			}
		}
		else if( _gripperSetAction == Gripper_Open )
		{
			if( _gripperCurAction == GripperCur_Idle && ! _isGripperClosed )
			{
				// nothing to do
			}
			else
			{
				setState->powerOutputControlPoint = -255;
				if( _gripperCurAction == GripperCur_Idle || _gripperCurAction == GripperCur_Undefined )
				{
					_gripperCurAction = GripperCur_Starting;
				}
				_isGripperClosed = false;
			}
		}
	}
}

void Gripper::set( rec::iocontrol::remotestate::SensorState* sensorState )
{
	if( false == _isEnabled )
	{
		return;
	}

	if( GripperCur_Idle != _gripperCurAction )
	{
		sensorState->isGripperClosed = false;
		sensorState->isGripperOpened = false;
	}

	_current.add( sensorState->powerOutputRawCurrent );

	if(  _current.mean() > 25.0f )
	{
		// action requested
		bool alreadyHandled = false;
		if( _gripperSetAction == Gripper_Close || _gripperSetAction == Gripper_Open )
		{
			alreadyHandled = true;
			if( _gripperCurAction == GripperCur_MovingClose && _gripperSetAction == Gripper_Close )
			{
				// already in action
				alreadyHandled = false;
			}
			if( _gripperCurAction == GripperCur_MovingOpen && _gripperSetAction == Gripper_Open )
			{
				// already in action
				alreadyHandled = false;
			}
			if( alreadyHandled )
			{
				_firstTimeCurrentToHigh.start();
				if( _gripperSetAction == Gripper_Close )
				{
					_gripperCurAction = GripperCur_MovingClose;
				}
				else
				{
					_gripperCurAction = GripperCur_MovingOpen;
				}
				_gripperSetAction = Gripper_NoCommand;
			}
		}
		if( ! alreadyHandled && (_gripperCurAction == GripperCur_MovingClose || _gripperCurAction == GripperCur_MovingOpen )  )
		{
			if( _gripperCurAction == GripperCur_MovingClose )
			{
				_isNewPowerOutputControlPoint = true;
				_powerOutputControlPoint = 255;
			}
			else
			{
				_isNewPowerOutputControlPoint = true;
				_powerOutputControlPoint = -255;
			}
			if( _firstTimeCurrentToHigh.elapsed() > 400 )
			{
				// gripper reached its position
				_isGripperClosed = ( _gripperCurAction == GripperCur_MovingClose );

				sensorState->isGripperClosed = _isGripperClosed;
				sensorState->isGripperOpened = !_isGripperClosed;

				_gripperSetAction = Gripper_NoCommand;
				_gripperCurAction = GripperCur_Idle;
				_current.reset();

				_isNewPowerOutputControlPoint = true;
				_powerOutputControlPoint = 0;
				_firstTimeCurrentToHigh = QTime();
			}
		}
	}
}

void Gripper::reset()
{
	_current.reset();
	_isEnabled = false;
	_isGripperClosed = false;
	_gripperSetAction = Gripper_NoCommand;
	_gripperCurAction = GripperCur_Undefined ;
	_isNewPowerOutputControlPoint = false;
	_powerOutputControlPoint = 0;
}