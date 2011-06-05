//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/robotino/com/ComId.h"
#include "rec/robotino/com/Com.h"
#include "rec/robotino/com/Camera.h"
#include "rec/robotino/com/AnalogInput.h"
#include "rec/robotino/com/DigitalInput.h"
#include "rec/robotino/com/DigitalOutput.h"
#include "rec/robotino/com/DistanceSensor.h"
#include "rec/robotino/com/Bumper.h"
#include "rec/robotino/com/Motor.h"
#include "rec/robotino/com/NorthStar.h"
#include "rec/robotino/com/OmniDrive.h"
#include "rec/robotino/com/PowerManagement.h"
#include "rec/robotino/com/Relay.h"
#include "rec/robotino/com/Info.h"
#include "rec/robotino/com/Gripper.h"
#include "rec/robotino/com/Odometry.h"
#include "rec/robotino/com/PowerOutput.h"
#include "rec/robotino/com/EncoderInput.h"
#include "rec/robotino/com/RobotinoException.h"
#include "rec/robotino/com/Manipulator.h"
#include "rec/robotino/com/LaserRangeFinder.h"

/**  \mainpage rec::robotino::com API documentation

The application programming interface (API) for Robotino(r) from Festo Didactic permits full
access to Robotino's sensors and actors. Communication between the control program and Robotino
is handled via TCP and UDP and is therefor fully network transparent. It does not matter whether the
control program runs direcly on Robotino or on a remote system.

The API is available in binary form for Windows and Linux and <A HREF="http://svn.openrobotino.org/trunk">source code</A>
via svn.

\section install Installation
Install the API either from binary or from source. The Windows binary installer will set the environment variable OPENROBOTINOAPI_DIR.
With a deafult installation OPENROBOTINOAPI_DIR is "C:\{ProgramFiles}\REC GmbH\OpenRobotinoAPI".

To build your own program you need to
-# add \$(OPENROBOTINOAPI_DIR)/1/include to your compilers include search path
-# if you want to use the convenience library rec_core_lt also add \$(OPENROBOTINOAPI_DIR)/share/include to your compilers include search path
-# use \#include "rec/robotino/com/all.h" in your program to use the rec::robotino::com API
-# add \$(OPENROBOTINOAPI_DIR)/1/lib/win32 or \$(OPENROBOTINOAPI_DIR)/1/lib/linux to your linkers library search path
-# link against rec_robotino_com.lib on win32 and librec_robotino_com.so on linux systems
-# for rec_core_lt link against rec_core_lt.lib on win32 and librec_core_lt.so on linux systems

If you are familiar with <A HREF="http://www.cmake.org">cmake</A> you might prefer using \$(OPENROBOTINOAPI_DIR)/1/tools/FindOpenRobotino1.cmake.

\section usage_sec Usage
In the following you will find a simple example on how to drive Robotino on a circle.

\subsection includes Including headers
You need to include at least "rec/robotino/com/all.h". <cmath> is for cos and sin functions.
<iostream> defines std::cout and std::cerr.
<PRE>
#include "rec/robotino/com/all.h"
#include "rec/core_lt/Timer.h"
#include <cmath>
#include <iostream>
</PRE>

\subsection declare Declarations
All classes of this API are in the namespace rec::robotino::com. To simplify usage we use this namespace as default.
So we can write Com instead of rec::robotino::com::Com. We define all objects used at global scope. There is no need to do so but it makes
life simple for this example.

<PRE>
using namespace rec::robotino::com;

class MyCom : public Com
{
  public:
    MyCom()
    {
    }

    void errorEvent( Error error, const char* errorString )
    {
      std::cerr << "Error: " << errorString << std::endl;
    }

    void connectedEvent()
    {
      std::cout << "Connected." << std::endl;
    }

    void connectionClosedEvent()
    {
      std::cout << "Connection closed." << std::endl;
    }
};

MyCom com;
OmniDrive omniDrive;
</PRE>

Notice the class MyCom which has rec::robotino::com::Com as base class. MyCom overwrites the three virtual funtions
to handle different events.

\subsection init Initialisation
In this function we register the handlers to the Com object. We have to tell the onmiDrive where it belongs to. This is done
by setComId. Notice that this enables us to have multiple Com objects at the same time each of which connecting to a different Robotino.

<PRE>
void init()
{
  // Tell omniDrive which Robotino to drive
  // It is possible to have multiple Com objects and multiple OmniDrive objects.
  // By this you can drive multiple Robotinos from one program.
  omniDrive.setComId( com.id() );

  // Connect
  std::cout << "Connecting..." << std::endl;
  com.setAddress( "127.0.0.1" );
  com.connect();
  std::cout << std::endl << "Connected" << std::endl;
}
</PRE>

\subsection drive Driving Robotino
Driving Robotino is done by contiunously sending set point values for the motors to Robotino. The motors set point values are
computed by omniDrive when calling its setVelocity function. Communication to Robotino is performed in a seperate thread. You can synchronize
your thread with the communication thread by calling rec::robotino::Com::waitForUpdate. This function blocks until set values are
trnasmitted to Robotino and sensor values are received.
<PRE>
void drive()
{
  rec::core_lt::Timer timer;
  timer.start();

  const float speed = 200.0f;
  const float rotationSpeed = 36.0f;
  
  while( com.isConnected() )
  {
    float rot = rotationSpeed * ( 2.0f * (float)M_PI / 360.0f ) * ( timer.msecsElapsed() / 1000.0f );

    omniDrive.setVelocity( cos(rot) * speed, sin(rot) * speed, 5.0f );

    com.waitForUpdate(); //wait until actor set values are transmitted and new sensor readings are available
  }
}
</PRE>

\subsection destroy Closing communication
We do not really need a seperate function here. But it looks nice in the main function.
<PRE>
void destroy()
{
  com.disconnect();
}
</PRE>

\subsection main The main program
This is pretty clear now, isn't it? Make sure to catch any exception which might be thrown from the functions within the try block.
<PRE>
int main()
{
  try
  {
    init();
    drive();
    destroy();
  }
  catch( const std::exception& e )
  {
    std::cerr << "Error: " << errorString << std::endl;
  }

  std::cout << "Press any key to exit..." << std::endl;
  rec::core_lt::waitForKey();
}
</PRE>
*/
