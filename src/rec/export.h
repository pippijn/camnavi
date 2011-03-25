//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH


#ifdef WIN32
  #ifdef DLL_EXPORT 
    #pragma warning ( disable : 4251 ) 
    #define REC_EXPORT __declspec(dllexport)
  #else
    #define REC_EXPORT 
  #endif
#else
  #define REC_EXPORT
#endif
