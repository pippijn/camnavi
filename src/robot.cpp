#include <iostream>

#include <opencv/highgui.h>

#include <rec/core_lt/Timer.h>

#include <rec/robotino/com/Com.h>
#include <rec/robotino/com/OmniDrive.h>

#include "robot.h"
#include "imagereceiver.h"

using namespace rec::robotino::com;

struct Robot::pimpl
  : Com
{
  pimpl (std::string const &host);

  void run ();

  void errorEvent (Error error, const char *errorString)
  {
    std::cerr << "Error: " << errorString << std::endl;
  }

  void connectedEvent ()
  {
    std::cout << "Connected." << std::endl;
  }

  void connectionClosedEvent ()
  {
    std::cout << "Connection closed." << std::endl;
  }

  ImageReceiver cam;
  OmniDrive drive;
};


Robot::pimpl::pimpl (std::string const &host)
{
  setAddress (host.c_str ());
  connect ();

  drive.setComId (id ());
  cam.setComId (id ());
}

void
Robot::pimpl::run ()
{
  rec::core_lt::Timer timer;
  timer.start ();

  const float speed = 200.0f;
  const float rotationSpeed = 36.0f;

  while (isConnected ())
    {
      float rot = rotationSpeed * (2.0f * (float)M_PI / 360.0f) * (timer.msecsElapsed () / 1000.0f);
      drive.setVelocity (cos (rot) * speed, sin (rot) * speed, 5.0f);
      waitForUpdate ();      //wait until actor set values are transmitted and new sensor readings are available

      char c = cvWaitKey (100);

      switch (c)
        {
        case 'q':
          return;
        }
    }
}


Robot::Robot (std::string const &host)
  : impl (new pimpl (host))
{
}

void
Robot::run ()
{
  impl->run ();
}
