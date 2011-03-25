//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#include "rec/core_lt/Log.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <sstream>

#ifdef WIN32
#include <windows.h>
#endif

using namespace rec::core_lt;
using namespace boost::posix_time;

Log& Log::singleton() {
	static Log log;
	return log;
}

void Log::init (void) {
}

void Log::done (void) {
}

Log::Log ()
{
}

Log::~Log () {
}

void Log::initCurrent (unsigned long threadId) {
	if (0 == _messages[threadId].size ()) {
		ptime t = microsec_clock::local_time ();
		std::ostringstream os;
		os << threadId << "> ";
		_messages[threadId] = os.str ();
		_messages[threadId] += to_simple_string (t);
		_messages[threadId] += " : ";
	}
}

Log &Log::operator<< (char c) {
	boost::recursive_mutex::scoped_lock l (_mutex);
#ifdef WIN32
	unsigned long threadId = GetCurrentThreadId ();
#else
	std::size_t threadId = (std::size_t)(pthread_self ());
#endif
	
	if (endl == c)
	{
		if( _receiver )
		{
			_receiver(_messages[threadId]);
			_messages.erase(threadId);
		}
		/* TODO:
		 * What should happen when the log receives an endl while no receiver is set?
		 * Suggestion: Insert newline and insert timestamp header into message buffer.
		 */
	}
	else
	{
		initCurrent (threadId);
		_messages[threadId] += c;
	}
	
	return *this;
}

Log &Log::operator<< (int i) {
	boost::recursive_mutex::scoped_lock l (_mutex);
#ifdef WIN32
	unsigned long threadId = GetCurrentThreadId ();
#else
	std::size_t threadId = (std::size_t)(pthread_self ());
#endif
	
	std::ostringstream os;
	initCurrent (threadId);
	os << i;
	_messages[threadId] += os.str ();
	
	return *this;
}

Log &Log::operator<< (unsigned int i) {
	boost::recursive_mutex::scoped_lock l (_mutex);
#ifdef WIN32
	unsigned long threadId = GetCurrentThreadId ();
#else
	std::size_t threadId = (std::size_t)(pthread_self ());
#endif
	
	std::ostringstream os;
	initCurrent (threadId);
	os << i;
	_messages[threadId] += os.str ();
	
	return *this;
}

Log &Log::operator<< (double d) {
	boost::recursive_mutex::scoped_lock l (_mutex);
#ifdef WIN32
	unsigned long threadId = GetCurrentThreadId();
#else
	std::size_t threadId = (std::size_t)(pthread_self ());
#endif
	
	std::ostringstream os;
	initCurrent(threadId);
	os << d;
	_messages[threadId] += os.str ();
	
	return *this;
}

Log &Log::operator<< (const std::string &str) {
	boost::recursive_mutex::scoped_lock l (_mutex);
#ifdef WIN32
	unsigned long threadId = GetCurrentThreadId ();
#else
	std::size_t threadId = (std::size_t)(pthread_self ());
#endif
	
	initCurrent	(threadId);
	_messages[threadId] += str;
	
	return *this;
}

Log &Log::operator<< (const char *str) {
	return operator<< (std::string (str));
}

//Log &Log::operator<< (const util::XmlString &str) {
//	return operator<< (str.toStdString ());
//}

//Log &Log::operator<< (const util::path &path) {
//	return operator<< (util::pathToStdString (path));
//}

Log &Log::operator<< (LogMessageAtType& at) {
	boost::recursive_mutex::scoped_lock l( _mutex );
#ifdef WIN32
	unsigned long threadId = GetCurrentThreadId ();
#else
	std::size_t threadId = (std::size_t)(pthread_self ());
#endif
	
	initCurrent (threadId);
	
	std::stringstream s;
	s << "(at: " << at.function << " in " << at.file << ":" << at.line << "): ";
	_messages[threadId] += s.str();
	
	return *this;
}

Log::LogMessageAtType Log::at (const char *function, const char *file, size_t line)
{
	LogMessageAtType a;
	
	a.function = function;
	a.file = file;
	a.line = line;
	
	return a;
}

/**The receiver needs to have a public slot on_received( QString )*/
void Log::setReceiver( logCb_t receiver )
{
	boost::recursive_mutex::scoped_lock l( _mutex );
	_receiver = receiver;
}
