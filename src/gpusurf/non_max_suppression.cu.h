/*
Copyright (c) 2010, Paul Furgale and Chi Hay Tong
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are 
met:

* Redistributions of source code must retain the above copyright notice, 
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright 
  notice, this list of conditions and the following disclaimer in the 
  documentation and/or other materials provided with the distribution.
* The names of its contributors may not be used to endorse or promote 
  products derived from this software without specific prior written 
  permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "non_max_suppression.h"

namespace asrl {

  extern __shared__ float fh_vals[];
  __global__ void surf_nonmaxonly_kernel(float * d_hessian, int octave, int4 * d_maxmin, unsigned int * d_maxmin_counter, float threshold)
  {
    // The hidx variables are the indices to the hessian buffer.
    int hidx_x = threadIdx.x + __mul24(blockIdx.x, (blockDim.x-2));
    int hidx_y = threadIdx.y + __mul24(blockIdx.y, (blockDim.y-2));
    int hidx_z = threadIdx.z;
    int localLin = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // Is this thread within the hessian buffer?
    if(	 hidx_x < d_octave_params[octave].x_size && 
	 hidx_y < d_octave_params[octave].y_size && 
	 hidx_z < d_octave_params[octave].nIntervals ){
      // Get the linear index to the buffer.
      int hidx_lin =	hidx_x + 
	d_hessian_stride[0] * hidx_y + 
	d_hessian_stride[0] * d_octave_params[octave].y_size * hidx_z;
      fh_vals[localLin] = d_hessian[hidx_lin];
    }
    __syncthreads();
    
    // Is this location one of the ones being processed for nonmax suppression.
    // Blocks overlap by one so we don't process the border threads.
    bool inBounds2 = threadIdx.x > 0 && threadIdx.x < blockDim.x-1 && hidx_x < d_octave_params[octave].x_size - 1 
      &&             threadIdx.y > 0 && threadIdx.y < blockDim.y-1 && hidx_y < d_octave_params[octave].y_size - 1
      &&             threadIdx.z > 0 && threadIdx.z < blockDim.z-1;

  float val = fh_vals[localLin];

  if(inBounds2 && val >= threshold){
    // Check to see if we have a max (in its 26 neighbours)
    int zoff = __mul24(blockDim.x, blockDim.y);
    bool condmax  =    val > fh_vals[localLin                     + 1]
      &&               val > fh_vals[localLin                     - 1]
      &&               val > fh_vals[localLin        - blockDim.x + 1]
      &&               val > fh_vals[localLin        - blockDim.x    ]
      &&               val > fh_vals[localLin        - blockDim.x - 1]
      &&               val > fh_vals[localLin        + blockDim.x + 1]
      &&               val > fh_vals[localLin        + blockDim.x    ]
      &&               val > fh_vals[localLin        + blockDim.x - 1]
      
      &&               val > fh_vals[localLin - zoff              + 1]
      &&               val > fh_vals[localLin - zoff                 ]
      &&               val > fh_vals[localLin - zoff              - 1]
      &&               val > fh_vals[localLin - zoff - blockDim.x + 1]
      &&               val > fh_vals[localLin - zoff - blockDim.x    ]
      &&               val > fh_vals[localLin - zoff - blockDim.x - 1]
      &&               val > fh_vals[localLin - zoff + blockDim.x + 1]
      &&               val > fh_vals[localLin - zoff + blockDim.x    ]
      &&               val > fh_vals[localLin - zoff + blockDim.x - 1]
      
      &&               val > fh_vals[localLin + zoff              + 1]
      &&               val > fh_vals[localLin + zoff                 ]
      &&               val > fh_vals[localLin + zoff              - 1]
      &&               val > fh_vals[localLin + zoff - blockDim.x + 1]
      &&               val > fh_vals[localLin + zoff - blockDim.x    ]
      &&               val > fh_vals[localLin + zoff - blockDim.x - 1]
      &&               val > fh_vals[localLin + zoff + blockDim.x + 1]
      &&               val > fh_vals[localLin + zoff + blockDim.x    ]
      &&               val > fh_vals[localLin + zoff + blockDim.x - 1]
      ;

    if(condmax) {
      unsigned i = atomicInc(d_maxmin_counter,(unsigned int) -1);
      
      if(i < ASRL_SURF_MAX_CANDIDATES) {
	int4 f = {hidx_x, hidx_y, threadIdx.z, octave};
	d_maxmin[i] = f;
	
      } // end if the maximum number of maxima has been reached.
    } // end if is a maxima
  } // end in thread block bounds and threshold
  
  
}//end kernel

  void run_surf_nonmaxonly_kernel(dim3 grid, dim3 threads, size_t sharedBytes, float * d_hessian, int octave, int4 * d_maxmin, unsigned int * d_maxmin_counter, float threshold)
  {
    surf_nonmaxonly_kernel <<< grid, threads, sharedBytes >>>
          (d_hessian, octave, d_maxmin, d_maxmin_counter, threshold);
  }
  
} // namespace asrl
