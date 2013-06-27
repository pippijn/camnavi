%module(directors="1") robotinowrapper

%{
/* Includes the header in the wrapper code */
#include "rec/robotino/com/all.h"
%}

/* Include some definitions from the swig standard library */
%include "typemaps.i"
%include "std_string.i"

/* Include handling for Exceptions */
%include "exception.i"

%exception
{
    try
    {
      $action
    }
    catch(rec::robotino::com::RobotinoException &e)
    {
      SWIG_exception(SWIG_RuntimeError, e.what());
    }
}

/* Wrap all float and double pointers as scalar output parameters by default */
%apply float *OUTPUT { float* };
%apply double *OUTPUT { double* };

%feature("director") Com;
%feature("director") Camera; 

/* Include language specific definitions */
#ifdef SWIGJAVA
%include "java.i"
#elif SWIGCSHARP
%include "dotnet.i"
#endif

%ignore rec::robotino::com::ComId::null;

%include "rec/robotino/com/RobotinoException.h"

%include "rec/robotino/com/ComId.h"
%include "rec/robotino/com/Com.h"
%include "rec/robotino/com/Actor.h"
%include "rec/robotino/com/Motor.h"
%include "rec/robotino/com/Camera.h"
%include "rec/robotino/com/AnalogInput.h"
%include "rec/robotino/com/DigitalInput.h"
%include "rec/robotino/com/DigitalOutput.h"
%include "rec/robotino/com/DistanceSensor.h"
%include "rec/robotino/com/Bumper.h"

%include "rec/robotino/com/NorthStar.h"

%include "rec/robotino/com/OmniDrive.h"
%include "rec/robotino/com/PowerManagement.h"
%include "rec/robotino/com/Relay.h"
%include "rec/robotino/com/Info.h"
%include "rec/robotino/com/Gripper.h"
%include "rec/robotino/com/Odometry.h"
%include "rec/robotino/com/PowerOutput.h"
%include "rec/robotino/com/EncoderInput.h"
%include "rec/robotino/com/RobotinoException.h"
