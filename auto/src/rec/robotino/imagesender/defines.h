//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_IMAGESENDER_DEFINES_H_
#define _REC_ROBOTINO_IMAGESENDER_DEFINES_H_

/******************************************************************************/
/***   Defines                                                              ***/
/******************************************************************************/
/*
 * Defines that help to correctly specify dllexport and dllimport on
 * Windows platforms.
 */
#ifdef WIN32
#  ifdef rec_robotino_imagesender_EXPORTS
#    define REC_ROBOTINO_IMAGESENDER_EXPORT __declspec(dllexport)
#  else
#    define REC_ROBOTINO_IMAGESENDER_EXPORT __declspec(dllimport)
#  endif // rec_robotino_imagesender_EXPORTS
#else // WIN32
#  define REC_ROBOTINO_IMAGESENDER_EXPORT
#endif // WIN32

#endif // _REC_ROBOTINO_IMAGESENDER_DEFINES_H_
