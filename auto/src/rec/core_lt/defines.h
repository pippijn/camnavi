//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_DEFINES_H_
#define _REC_CORE_LT_DEFINES_H_

/******************************************************************************/
/***   Defines                                                              ***/
/******************************************************************************/
/*
 * Defines that help to correctly specify dllexport and dllimport on
 * Windows platforms.
 */
#ifdef WIN32
#  ifdef rec_core_lt_EXPORTS
#    define REC_CORE_LT_EXPORT __declspec(dllexport)
#  else
#    define REC_CORE_LT_EXPORT __declspec(dllimport)
#  endif // rec_core_EXPORTS
#else // WIN32
#  define REC_CORE_LT_EXPORT
#endif // WIN32

#endif // _REC_CORE_LT_DEFINES_H_
