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

#ifndef __LI_SINGLETON_HPP
#define __LI_SINGLETON_HPP

#include <cstdlib>

namespace LIVA
{

template <typename T>
class Singleton
{
protected:
	//constructeurs
	Singleton(){}
	~Singleton(){}
	static T * _singleton;
public:
	static T* getInstance(){
		make();
		return (static_cast<T*>(_singleton));
	}

	static void make(){
		if( NULL == _singleton)
			_singleton = new T;
	}

	static void kill(){
		if(_singleton != NULL){
			delete _singleton;
			_singleton = NULL;
		}
	}

};

template <typename T>
T *Singleton<T>::_singleton = NULL;


} // namespace LIVA
#endif
