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

#include "GpuSurfDetectorInternal.hpp"
#include "gpu_utils.h"
#include "assert_macros.hpp"
#include "fasthessian.h"
#include "gpu_area.h"
#include "keypoint_interpolation.h"
#include "non_max_suppression.h"
#include "orientation.h"
#include "descriptors.h"
#include <boost/lexical_cast.hpp>
#include <fstream>
#include "detector.h"

namespace asrl {

  GpuSurfDetectorInternal::GpuSurfDetectorInternal(GpuSurfConfiguration config) : 
    m_initialized(false),
    m_config(config)
  {
    int deviceCount;
    int device;
    cudaError_t err;
    cudaGetDeviceCount(&deviceCount);
    ASRL_ASSERT_GT(deviceCount,0,"There are no CUDA capable devices present");
    
	
    err = cudaGetDevice(&device);
    ASRL_ASSERT_EQ(err,cudaSuccess, "Unable to get the CUDA device: " << cudaGetErrorString(err));		
    //std::cout << "Found device " << device << std::endl;
    err = cudaGetDeviceProperties(&m_deviceProp,device);
    ASRL_ASSERT_EQ(err,cudaSuccess, "Unable to get the CUDA device properties: " << cudaGetErrorString(err));		

    // Some more checking...
    ASRL_ASSERT_GE(m_deviceProp.major,1,"Minimum compute capability 1.1 is necessary");
    ASRL_ASSERT_GE(m_deviceProp.minor,1,"Minimum compute capability 1.1 is necessary");

    m_maxmin.init(ASRL_SURF_MAX_CANDIDATES,false);
    m_maxmin.memset(0);

  }

  GpuSurfDetectorInternal::~GpuSurfDetectorInternal()
  {

  }

  void GpuSurfDetectorInternal::buildIntegralImage(cv::Mat const & image)
  {
    if(!m_initialized || m_intImg->width() != image.cols || m_intImg->height() != image.rows)
      {
	initDetector(image.cols,image.rows);
      }
    
    m_intProcessor->process(image, *m_intImg);
    texturize_integral_image(m_intImg->d_get());
    init_globals(m_intImg->width(), m_intImg->height(), m_octaves, m_config.nOctaves);
  }

  void GpuSurfDetectorInternal::saveIntegralImage(std::string const & basename)
  {
    float * iimg = m_intImg->h_get();
    std::stringstream sout;
    sout << basename << "-iimg.bin";
    std::ofstream fout(sout.str().c_str(),std::ios::binary);
    ASRL_ASSERT(fout.good(),"Unable to open file \"" << sout.str() << "\" for writing");
    int size[2];
    size[0] = m_intImg->width();
    size[1] = m_intImg->height();
    fout.write(reinterpret_cast<const char *>(&size[0]),2*sizeof(int));
    fout.write(reinterpret_cast<const char *>(iimg),m_intImg->width()*m_intImg->height() * sizeof(float));

  }


  void GpuSurfDetectorInternal::detectKeypoints()
  {
    detectKeypoints(m_config.threshold);
  }

  void GpuSurfDetectorInternal::detectKeypoints(float threshold)
  {
	m_features.featureCounterMem().memset(0);
	

	for(int o = 0; o < m_config.nOctaves; o++)
	  {
	    if(m_octaves[o].valid())
	      {
		run_surf_detector(m_interest.d_get(), m_octaves[o], o, m_features, 
				  threshold, m_config.detector_threads_x,
				  m_config.detector_threads_y, m_config.nonmax_threads_x,
				  m_config.nonmax_threads_y);
	      }
	  }
  }

  void GpuSurfDetectorInternal::findOrientation()
  {
    int nFeatures = m_features.ftCount();
    if(nFeatures > 0)
      {
	find_orientation(m_features.deviceFeatures(), m_features.ftCount());
	ASRL_CHECK_CUDA_ERROR_DBG("Find orientation");
      }
  }

  void GpuSurfDetectorInternal::findOrientationFast()
  {
    int nFeatures = m_features.ftCount();
    if(nFeatures > 0)
      {
	find_orientation_fast(m_features.deviceFeatures(), m_features.ftCount());
	ASRL_CHECK_CUDA_ERROR_DBG("Find orientation fast");
      }
  }

  void GpuSurfDetectorInternal::computeDescriptors()
  {
    int nFeatures = m_features.ftCount();
    
    if(nFeatures > 0)
      {
	compute_descriptors(m_features.deviceDescriptors(), m_features.deviceFeatures(), m_features.ftCount());
	ASRL_CHECK_CUDA_ERROR_DBG("compute descriptors");
      }
    
  }

  void GpuSurfDetectorInternal::computeUprightDescriptors()
  {
    ASRL_THROW("Not implemented");
  }

  void GpuSurfDetectorInternal::getKeypoints(std::vector<cv::KeyPoint> & outKeypoints)
  {
    int ftcount = m_features.ftCount();
    
    m_features.getKeypoints(outKeypoints);
    

  }


  void GpuSurfDetectorInternal::setKeypoints(std::vector<cv::KeyPoint> const & inKeypoints)
  {
    m_features.setKeypoints(inKeypoints);
  }

  void GpuSurfDetectorInternal::getDescriptors(std::vector<float> & outDescriptors)
  {
    int ftcount = m_features.ftCount();
    
    m_features.descriptorsMem().pullFromDevice();
    cudaThreadSynchronize();


    // Resize the destination buffer.
    outDescriptors.resize(descriptorSize() * ftcount);
    // Copy the descriptors into the buffer. AFAIK, all known std::vector implementations use
    // contiguous memory.
    memcpy(&outDescriptors[0],m_features.hostDescriptors(), descriptorSize() * ftcount * sizeof(float));
    
  }

  int GpuSurfDetectorInternal::descriptorSize()
  {
    return ASRL_SURF_DESCRIPTOR_DIM;
  }


  void GpuSurfDetectorInternal::initDetector(int width, int height) {
    
    m_intProcessor.reset(new GpuIntegralImageProcessor(width, height));
    
    m_intImg.reset(new GpuIntegralImage(width, height));
    
    // initialize the fast hessian parameters.
    m_dxx_width = 1 + m_config.l1;
    m_dxx_height = m_config.l2;
   
    for(int o = 0; o < m_config.nOctaves; o++)
      {
	m_octaves[o].init(width,height,m_config.l1,m_config.l2,m_config.l3, m_config.l4, m_config.edgeScale, m_config.initialScale, o, m_config.initialStep, m_config.nIntervals);
      }
    
    m_interest.init(m_octaves[0].stride() * m_octaves[0].height() * m_octaves[0].intervals());

    m_initialized = true;
  }

} // namespace asrl
