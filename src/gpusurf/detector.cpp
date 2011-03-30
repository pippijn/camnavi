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

#include "detector.h"
#include "fasthessian.h"
#include "non_max_suppression.h"
#include "keypoint_interpolation.h"
#include "GpuSurfFeatures.hpp"
#include "GpuSurfOctave.hpp"

namespace asrl {
  void run_surf_detector(float * d_hessianBuffer, GpuSurfOctave & octave, int octaveIdx, GpuSurfFeatures & features, 
			 float threshold, int fh_x_threads, int fh_y_threads,
			 int nonmax_x_threads, int nonmax_y_threads)
  {
    /////////////////
    // FASTHESSIAN //
    /////////////////
    dim3 threads; 

    threads.x = fh_x_threads;
    threads.y = fh_y_threads;
    threads.z = octave.intervals();

    dim3 grid;
    grid.x = ( (octave.width()  + threads.x - 1) / threads.x);
    grid.y = ( (octave.height() + threads.y - 1) / threads.y);
    grid.z = 1;
	
    if(octave.valid()) {
      run_fasthessian_kernel(grid, threads, d_hessianBuffer, octaveIdx);
      ASRL_CHECK_CUDA_ERROR("Finding fasthessian");
    }

    // Reset the candidate count.
    features.featureCounterMem().pullFromDevice();
    features.featureCounterMem().h_get()[1] = 0;
    features.featureCounterMem().pushToDevice();
    

    ////////////
    // NONMAX //
    ////////////
    
    threads.x = nonmax_x_threads;
    threads.y = nonmax_y_threads;
    threads.z = octave.intervals();

    grid.x = ( (octave.width()  + (threads.x-2) - 1) / (threads.x-2));
    grid.y = ( (octave.height() + (threads.y-2) - 1) / (threads.y-2));
    grid.z = 1;

    size_t sharedBytes = threads.x*threads.y*threads.z*sizeof(float);
    run_surf_nonmaxonly_kernel(grid, threads, sharedBytes, d_hessianBuffer,
    					octaveIdx, features.rawFeatureMem().d_get(), features.featureCounterMem().d_get() + 1, 
    					threshold);
    ASRL_CHECK_CUDA_ERROR("Running Nonmax, octave " << octaveIdx);

    
    ///////////////////
    // INTERPOLATION //
    ///////////////////
    
    
    run_fh_interp_extremum(d_hessianBuffer,
					    features.deviceFeatures(), 
					    features.rawFeatureMem().d_get(), 
					    features.featureCounterMem().d_get(),
					    features.featureCounterMem().d_get() + 1);

    features.featureCounterMem().pullFromDevice();
    features.setDirty();
    

  } // run_surf_detector()

} // namespace asrl
