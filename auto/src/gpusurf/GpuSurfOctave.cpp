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

#include "GpuSurfOctave.hpp"
#include <cmath>

namespace asrl {
  GpuSurfOctave::GpuSurfOctave(){
    init(0,0,1.f,1.f,1.f,1.f,1.f,1.f,0,1,1);
  }

  GpuSurfOctave::GpuSurfOctave(int img_width, int img_height, float l1, float l2, float l3, float l4, float edge_scale, float base_scale, int octave, int baseStep, int nIntervals)
  {
    init(img_width, img_height, l1, l2, l3, l4, edge_scale, base_scale, octave, baseStep, nIntervals);
  }

  void GpuSurfOctave::init(int img_width, int img_height, float l1, float l2, float l3, float l4, float edge_scale, float base_scale, int octave, int baseStep, int nIntervals)
    {
      // Compute Dxx and Dyy filter half-widths
      m_mask_width = l2*0.5f;
      m_mask_height = 1.f + l1;
      
      // Compute step size
      m_step = baseStep * (1<<octave);
      
      // Compute scales

      float d = (base_scale * (1<<octave))/(nIntervals-2);
      for(int i = 0; i < nIntervals; i++)
	{
		m_scales[i] = base_scale * (1<<octave) + d * (i - 1.f) + 0.5f;	// scales with SURF-style overlap
	}
	
	  // Compute border required such that the filters don't overstep the image boundaries
      float smax = m_scales[nIntervals-1];
      m_border = (int) ceil(smax * std::max(std::max(m_mask_width, m_mask_height), l3+l4*0.5f));
	  
      // Hessian buffer size
      m_width = (img_width - 2*m_border)/m_step;
      m_height = (img_height - 2*m_border)/m_step;
      m_valid = m_width > 0 && m_height > 0;	// Ensure we have a valid Hessian before creating it
      //if(m_valid)
      //{
      //   m_hessian.init(m_width*m_height*nIntervals);
      //}
      m_intervals = nIntervals;
	
      // Store the filter parameters for weight computation
      m_l1 = l1;
      m_l2 = l2;
      m_l3 = l3;
      m_l4 = l4;
      m_edge_scale = edge_scale;
    }

  GpuSurfOctave::operator SurfOctaveParameters(){
      SurfOctaveParameters s;
      s.x_size = m_width;
      s.y_size = m_height;
      s.nIntervals = m_intervals;
      s.border = m_border;
      s.step = m_step;
      s.mask_width = m_mask_width;
      s.mask_height = m_mask_height;
      s.dxy_center_offset = 0.5f*(m_l4 + m_l3);       	// Dxy gap half-width
      s.dxy_half_width = 0.5f*m_l3;	       		// Dxy squares half-width
      s.dxy_scale = m_edge_scale * pow((2.f + 2.f*m_l1) * m_l2 / (4.f*m_l3*m_l3), 2.f);	// rescale edge_scale to fit with the filter dimensions
      
      return s;
    }
    
  // Lookup to m_hessian
//   float & GpuSurfOctave::operator()(int row, int col, int interval)
//     {
//       ASRL_ASSERT_GE_LT(row,0,height(),"row out of bounds");
//       ASRL_ASSERT_GE_LT(col,0,width(),"col out of bounds");
//       ASRL_ASSERT_GE_LT(interval,0,intervals(),"interval out of bounds");
      
//       return m_hessian[col + row * width() + interval * width() * height()];
//     }

} // namespace asrl
