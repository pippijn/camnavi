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

#ifndef __LI_CUDA_HPP
#define __LI_CUDA_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cuda_gl_interop.h>

enum memLoc  { Gpu, Cpu};
#undef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(X) (X)

#define MAX_KERNEL_RADIUS 32

// utilisation de boucle déroulées pour les convolutions (template)
//#define UNROLL_INNER
#define U_KERNEL_RADIUS 5 

// nombre de buffer utilisable sur la carte GPU (de la taille image au niv 0)
#define NB_GPU_BUFF 5

// parametre de la taille des blocks pour les opperations matriciele elmt a elmt
#define CB_TILE_W  16
#define CB_TILE_H  16


// valeurs choisis pour la convolution separable (derive)
#define ROW_W 256
#define MAXTH 512


#define COL_W 16
#define COL_H 16

// valeur d'allignement pour les objets 2D
#define ALLIGN_ROW 16

#include "../Core/LIVA.hpp"

#endif
