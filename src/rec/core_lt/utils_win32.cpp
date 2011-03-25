//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifdef WIN32

#include "rec/core_lt/utils_win32.h"

std::string rec::core_lt::fromWideChar( LPCWSTR wide )
{
	const DWORD cchBuffer = 1000;
	CHAR lpMultiByteStr[cchBuffer];
	BOOL usedDefaultChar = FALSE;

	int ret = WideCharToMultiByte( CP_ACP, WC_NO_BEST_FIT_CHARS, wide, -1, lpMultiByteStr, cchBuffer, "_", &usedDefaultChar );

	if( 0 == ret )
	{
		return "";
	}
	if( usedDefaultChar )
	{
		return "";
	}

	std::string str( lpMultiByteStr );
	return str;
}

std::string rec::core_lt::getShortPathName( LPCTSTR fileName )
{
	const DWORD cchBuffer = 1000;
	TCHAR lpszShortPath[cchBuffer];

	DWORD ret = GetShortPathName(	fileName, lpszShortPath, cchBuffer );
	if( 0 == ret )
	{
		DWORD err = GetLastError();
		return "";
	}

	return rec::core_lt::fromWideChar( lpszShortPath );
}

#endif // #ifdef WIN32
