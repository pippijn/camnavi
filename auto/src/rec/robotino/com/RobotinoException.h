//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_ROBOTINO_COM_ROBOTINOEXCEPTION_H_
#define _REC_ROBOTINO_COM_ROBOTINOEXCEPTION_H_

#include <exception>
#include <string>

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			/**
			* @brief	An extended exception class used in all rec::robotino::com classes.
			*
			* Usually gets thrown if a used communication object is invalid,
			* but there are other reasons, too, so check the error message.
			*/
			class RobotinoException : public std::exception
			{
			public:
				/**
				* @brief	Creates a new Exception with a given text message.
				*
				* @param message	A text message describing the reason of the exception.
				* @see				what
				*/
				RobotinoException( const char* message )
					: _message( message )
				{
				}

				RobotinoException( const std::exception &cause )
					: _message( cause.what() )
				{
				}

				virtual ~RobotinoException() throw ()
				{
				}

				/**
				* @brief	Return the descriptive text message.
				*
				* Overloaded from std::exception.
				*
				* @return	The description of the reason, why this exception was thrown (in English).
				*/
				virtual const char* what() const throw ()
				{
					return _message.c_str();
				}

			private:
				const std::string _message;
			};
		}
	}
}

#endif
