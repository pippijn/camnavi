//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_LOG_H_
#define _REC_CORE_LT_LOG_H_

#include "rec/core_lt/defines.h"

#include <string>
#include <map>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/function.hpp>

#ifdef NDEBUG
#	define AT "" /* no output in release mode */
#else
#	define AT rec::core_lt::Log::at(__PRETTY_FUNCTION__, __FILE__, __LINE__)
#endif /* NDEBUG */

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4251 )
#endif


namespace rec
{
	namespace core_lt
	{
		static const char endl = '\n';
		
		typedef boost::function<void( const std::string& )> logCb_t;
		
		class REC_CORE_LT_EXPORT Log {
		public:
			typedef struct {
				const char *function;
				const char *file;
				size_t line;
			} LogMessageAtType;
			
			static Log &singleton (void);
			
			~Log();
			
			/// Initialization of singleton (call this method before
			/// using any other method of this class)
			void init (void);
			
			/// Singletons cleanup method (call this method when you no
			/// use the singleton, at the latest before your application terminates)
			void done (void);
			
			Log &operator<< (char c);
			Log &operator<< (int i);
			Log &operator<< (unsigned int i);
			Log &operator<< (double d);
			Log &operator<< (const std::string &str);
			Log &operator<< (const char *str);
			Log &operator<< (LogMessageAtType& at);
			
			/// Function for translating the name of the calling procedure, the source file
			/// it is defined in and the line number into a struct that can then be further
			/// processed by the \b Log class.
			/// @param function Name of the calling function/procedure.
			/// @param file Name of the source file where \a function was declared.
			/// @param line Line within \a file where the call to \a function was invoked.
			/// @return Structure of type \b LogMessageType for passing the collected
			/// 	information to the \b Log singleton.
			static LogMessageAtType at (const char *function, const char *file, size_t line);
			
			void setReceiver( logCb_t f );
			
		protected:
			Log ();
			
			boost::recursive_mutex _mutex;
			
			std::map<unsigned long, std::string> _messages;
			
		private:
			void initCurrent (unsigned long threadId);
			
			logCb_t _receiver;
			
		};
		
	}
}

#ifdef WIN32
#pragma warning( pop )
#endif

#endif //_REC_CORE_LT_LOG_H_
