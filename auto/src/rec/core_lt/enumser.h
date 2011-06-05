#ifdef HAVE_ATL

/*
Module : enumser.h
Purpose: Defines the interface for a class to enumerate the serial ports installed on a PC
         using a number of different approaches
Created: PJN / 03-11-1998

Copyright (c) 1998 - 2009 by PJ Naughter (Web: www.naughter.com, Email: pjna@naughter.com)

All rights reserved.

Copyright / Usage Details:

You are allowed to include the source code in any product (commercial, shareware, freeware or otherwise) 
when your product is released in binary form. You are allowed to modify the source code in any way you want 
except you cannot modify the copyright details at the top of each module. If you want to distribute source 
code with your application, then you are only allowed to distribute versions released by the author. This is 
to maintain a single distribution point for the source code. 

*/
#include "rec/core_lt/defines.h"


///////////////////////// Macros / Structs etc ////////////////////////////////

#pragma once

#ifndef __ENUMSER_H__
#define __ENUMSER_H__

#ifndef CENUMERATESERIAL_EXT_CLASS
#define CENUMERATESERIAL_EXT_CLASS
#endif


///////////////////////// Includes ////////////////////////////////////////////                      

#ifdef _AFX
  #ifndef __AFXTEMPL_H__
    #include <afxtempl.h> 
    #pragma message("To avoid this message, please put afxtempl.h in your pre compiled header (normally stdafx.h)")
  #endif
#else
  #ifndef __ATLSTR_H__
    #include <atlstr.h>
    //#pragma message("To avoid this message, please put atlstr.h in your pre compiled header (normally stdafx.h). Note non MFC mode is not supported on VC 6")
  #endif  
#endif


///////////////////////// Classes /////////////////////////////////////////////

class REC_CORE_LT_EXPORT CENUMERATESERIAL_EXT_CLASS CEnumerateSerial
{
public:
//Methods
#ifndef NO_ENUMSERIAL_USING_CREATEFILE
  #ifdef _AFX
    static BOOL UsingCreateFile(CUIntArray& ports);
  #else
    static BOOL UsingCreateFile(CSimpleArray<UINT>& ports);
  #endif
#endif

#ifndef NO_ENUMSERIAL_USING_QUERYDOSDEVICE
  #ifdef _AFX
    static BOOL UsingQueryDosDevice(CUIntArray& ports);
  #else
    static BOOL UsingQueryDosDevice(CSimpleArray<UINT>& ports);
  #endif
#endif

#ifndef NO_ENUMSERIAL_USING_GETDEFAULTCOMMCONFIG
  #ifdef _AFX
    static BOOL UsingGetDefaultCommConfig(CUIntArray& ports);
  #else
    static BOOL UsingGetDefaultCommConfig(CSimpleArray<UINT>& ports);
  #endif
#endif

#ifndef NO_ENUMSERIAL_USING_SETUPAPI1
  #ifdef _AFX
    static BOOL UsingSetupAPI1(CUIntArray& ports, CStringArray& sFriendlyNames);
  #else
    static BOOL UsingSetupAPI1(CSimpleArray<UINT>& ports, CSimpleArray<CString>& sFriendlyNames);
  #endif
#endif

#ifndef NO_ENUMSERIAL_USING_SETUPAPI2
  #ifdef _AFX
    static BOOL UsingSetupAPI2(CUIntArray& ports, CStringArray& sFriendlyNames);
  #else
    static BOOL UsingSetupAPI2(CSimpleArray<UINT>& ports, CSimpleArray<CString>& sFriendlyNames);
  #endif
#endif

#ifndef NO_ENUMSERIAL_USING_ENUMPORTS
  #ifdef _AFX
    static BOOL UsingEnumPorts(CUIntArray& ports);
  #else
    static BOOL UsingEnumPorts(CSimpleArray<UINT>& ports);
  #endif
#endif

#ifndef NO_ENUMSERIAL_USING_WMI
  #ifdef _AFX
    static BOOL UsingWMI(CUIntArray& ports, CStringArray& sFriendlyNames);
  #else
    static BOOL UsingWMI(CSimpleArray<UINT>& ports, CSimpleArray<CString>& sFriendlyNames);
  #endif
#endif

#ifndef NO_ENUMSERIAL_USING_COMDB
  #ifdef _AFX
    static BOOL UsingComDB(CUIntArray& ports);
  #else
    static BOOL UsingComDB(CSimpleArray<UINT>& ports);
  #endif
#endif

#ifndef NO_ENUMSERIAL_USING_REGISTRY
  #ifdef _AFX
    static BOOL UsingRegistry(CStringArray& ports);
  #else
    static BOOL UsingRegistry(CSimpleArray<CString>& ports);
  #endif
#endif

    static bool unloadSetupapiLibrary();
	static bool loadSetupapiLibrary();

protected:
//Methods
  static BOOL IsNumeric(LPCTSTR pszString, BOOL bIgnoreColon);
};

#endif //__ENUMSER_H__

#endif //HAVE_ATL