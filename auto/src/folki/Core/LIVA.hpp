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

#ifndef __LI_LIVA_HPP
#define __LI_LIVA_HPP
#ifndef __CUDACC__
	#include <assert.h>
	#include <stdlib.h>
	#include <iostream>
	#include "Logger.hpp"
#endif
#include <math.h>

// fonction utiles 
#define iDivUp(a,b) (((int)a % (int)b != 0) ? (((int)a / (int)b) + 1) : ((int)a / (int)b))
#define iDivDown(a,b) ((int)a/(int)b)
#define iAlignUp(a,b) (((int)a % (int)b != 0) ?  ((int)a - (int)a % (int)b + (int)b) : (int)a)
#define iAlignDown(a,b) ((int)a - (int)a % (int)b)
#define IMUL(a, b) __mul24(a, b)
#ifndef MAX
	#define MAX(a,b) ((a>b)?a:b)
#endif
#ifndef MIN
	#define MIN(a,b) ((a<b)?a:b)
#endif

#endif //__LI_LIVA_HPP
