#ifndef _REC_MATH_EXCEPTION_H_
#define _REC_MATH_EXCEPTION_H_

#include "rec/core/Exception.h"
#include <string>

namespace rec
{
	namespace math
	{
		class Exception : public rec::core::Exception
		{
		public:
			typedef enum {
				UNDEFINED_ERROR		= 0,
				DETAILS_GIVEN		= 1,
				WRONG_PARAMETER		= 2,
				USER				= 100,
			} Type;

			Exception( Type t )
				: rec::core::Exception( "" )
				,_t( t )
			{
			}

			Exception( const std::string& errorDetails )
				: rec::core::Exception( errorDetails )
				,_t( DETAILS_GIVEN )
			{
			}

			virtual ~Exception() throw ()
			{
			}

			char const* what() const throw ()
			{
				switch( _t )
				{
				case UNDEFINED_ERROR:
					return "Undefined error";
				case WRONG_PARAMETER:
					return "Wrong parameter";
				case DETAILS_GIVEN:
					return msg.c_str();
				default: 
					return "undefined";
				}
			}

		protected:
			Type _t;
		};
	}
}

#endif
