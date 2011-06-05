//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_EXCEPTION_H_
#define _REC_CORE_LT_EXCEPTION_H_

#include <exception>
#include <string>

namespace rec
{
	namespace core_lt
	{
		class Exception : public std::exception
		{
		protected:
			std::string msg;

		public:
			Exception( const std::string &message )
				: msg( message )
			{
			}

			virtual ~Exception() throw ()
			{
			}

			virtual const std::string& getMessage() const
			{
				return msg;
			}

			//Compatibility functions for std::exception
			virtual const char* what() const throw ()
			{
				return msg.c_str();
			}
		};
	}
}

#endif //_REC_CORE_LT_EXCEPTION_H_
