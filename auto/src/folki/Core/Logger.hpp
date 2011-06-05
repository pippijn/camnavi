/*
      This file is part of FolkiGpu.

    FolkiGpu is free software: you can redistribute it and/or modify
    it under the terms of the GNU Leeser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Leeser General Public License
    along with FolkiGpu.  If not, see <http://www.gnu.org/licenses/>.

*/

/*
      FolkiGpu is a demonstration software developed by Aurelien Plyer during
    his phd at Onera (2008-2011). For more information please visit :
      - http://www.onera.fr/dtim-en/gpu-for-image/folkigpu.php
      - http://www.plyer.fr (author homepage)
*/

#ifndef __LI_LOGGER_HPP
#define __LI_LOGGER_HPP

#include "Singleton.hpp"

#include <sstream>
#include <string>
#include <fstream>
#include <assert.h>

namespace LIVA
{

class Logger: public Singleton<Logger>
{
	friend class Singleton<Logger>;

private:
	Logger(const std::string& Filename = "Output.log"):
	m_File(Filename.c_str()){
		m_File << "  ===========================================" << std::endl;
		m_File << "   LIVA v0.42 - Event log - " << CurrentDate() << std::endl;
		m_File << "  ===========================================" << std::endl << std::endl;
	}
	~Logger(){}

public:
	template <class T> Logger& operator <<(const T& ToLog){
		std::ostringstream Stream;
		Stream << ToLog;
		Write(Stream.str());
		return Log();
	}
	
	static Logger& Log(){
		make();
		return *_singleton;
	}

private:
	std::ofstream m_File;

	void Write(const std::string& Message){
		assert(m_File.is_open());
		m_File << CurrentTime()  << " >> ";
		m_File << Message << "\n";
	}

	std::string CurrentTime() const
	{
		// Récupération et formatage de la date
		char sTime[24];
		time_t CurrentTime = time(NULL);
		strftime(sTime, sizeof(sTime), "%H:%M:%S", localtime(&CurrentTime));
		
		return sTime;
	}

	std::string CurrentDate() const
	{
		// Récupération et formatage de la date
		char sTime[24];
		time_t CurrentTime = time(NULL);
		strftime(sTime, sizeof(sTime), "%d/%m/%Y", localtime(&CurrentTime));
		
		return sTime;
	}
};

}// namespace LIVA

#endif
