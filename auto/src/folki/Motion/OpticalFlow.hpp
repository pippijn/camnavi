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

#ifndef __LI_OPTICAL_FLOW_HPP__
#define __LI_OPTICAL_FLOW_HPP__

#include "LiMotion.hpp"
// type de noyau
#define FLAT 0
#define GAUSSIAN 1

// type pyramide
#define BURT 0
#define HAAR 1

// type derivation
#define SIMPLE 0
#define PREWITT 1
#define SOBEL 2

// type initialisation
#define SIMPLE 0
#define PROPA 1
#define HOMO 2


namespace LIVA 
{
class OpticalFlow
{
public:
	~OpticalFlow();

	/* INPUT */
	// initialisateurs des pyramides
	OpticalFlow();

	/* fonction pour regler les parametres d'algo qu'on change peut */
	void setTalon(float talon);
	void setPyrType(int pyr);

	int3 getImSize();
protected:
	/* parametres */
	unsigned int _nLevels;
	int3 *_levelSize;
	int _pyrType;

	void _fillPyramid(float **Pyr, float* buff1, float *buff2);
	/* fait remonter les info du niveau n au niveau n-1 */
	void _resample(float **Pyr, unsigned int n);

	void _copyDeviceToHost(float **d_ptr,float *h_ptr,unsigned int level, int3 h_imSize);
	void _copyHostToDevice(float **d_ptr,float *h_ptr,unsigned int level, int3 h_imSize);
	void _depPyram(float **Pyr,float *buff1, float *buff2,int lsrc, int ldest);
	void _updatenLevels(unsigned int nLevels, int3 imSize);

	/* les maniulation de memoires */
	void _pyrAlloc(float ***Pyr);
	void _pyrFree(float **Pyr);
	void _pyrReAlloc(float ***Pyr, unsigned int olbLevels);
	void _gradient(float *I,float *Ix,float *Iy,int3 imSize);
};
} // namespace LIVA
#endif
