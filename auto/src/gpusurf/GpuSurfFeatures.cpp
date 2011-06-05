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

#include "GpuSurfFeatures.hpp"
#include "assert_macros.hpp"

#include <cuda.h>
#include <cudpp.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include "gpu_globals.h"

#include <opencv2/features2d/features2d.hpp>

namespace asrl {
  GpuSurfFeatures::GpuSurfFeatures()
  {
    m_descriptorsDirty = false;
    m_featuresDirty = false;
    m_countDirty = false;
    // Initialize a buffer for the feature points.
    m_features.init(ASRL_SURF_MAX_FEATURES);
    m_features.memset(0);
    m_descriptors.init(ASRL_SURF_MAX_FEATURES * ASRL_SURF_DESCRIPTOR_DIM);	
    m_feature_counter.init(2,false);
    m_feature_counter.memset(0);
    m_rawFeatures.init(ASRL_SURF_MAX_CANDIDATES);
  }

  GpuSurfFeatures::~GpuSurfFeatures()
  {

  }


  Keypoint * GpuSurfFeatures::downloadFeatures()
  {
    if(m_featuresDirty) {
	  
      m_features.pullFromDevice(ftCount());
      m_featuresDirty = false;
    }
    return m_features.h_get();

  }


  unsigned int GpuSurfFeatures::ftCount()
  {
    if(m_countDirty)
      {
	m_feature_counter.pullFromDevice();
	m_countDirty = false;
      }
	
    return m_feature_counter[0];
  }

  unsigned int GpuSurfFeatures::rawFeatureCount()
  {
    if(m_countDirty)
      {
	m_feature_counter.pullFromDevice();
	m_countDirty = false;
      }
	
    return m_feature_counter[1];
  }

  void GpuSurfFeatures::clearFeatureCounts()
  {
    m_feature_counter.memset(0);
  }

  float * GpuSurfFeatures::downloadDescriptors()
  {
    if(m_descriptorsDirty) {
      m_descriptors.pullFromDevice(ftCount() * ASRL_SURF_DESCRIPTOR_DIM);
      m_descriptorsDirty = false;
    } 

    return m_descriptors.h_get();
  }

  void GpuSurfFeatures::setDirty()
  {
    m_descriptorsDirty = true;
    m_featuresDirty = true;
    m_countDirty = true;
  }

  void GpuSurfFeatures::getKeypoints(std::vector<cv::KeyPoint> & outKeypoints)
  {
    // This has an implicit "Thread synchronize"
    int ftcount = ftCount();

    featuresMem().pullFromDevice();
    //m_features->descriptorsMem().pullFromDevice();

    outKeypoints.resize(ftcount);
    if(ftcount > 0)
      {

	for(int i = 0; i < ftcount; i++)
	  {
            cv::KeyPoint & k = outKeypoints[i];
	    Keypoint * ft = hostFeatures() + i;

	    k.pt.x = ft->x;
	    k.pt.y = ft->y;

	    k.size = ft->size;

	    k.octave= ft->octave;
	    k.angle = ft->angle;
	    k.response = ft->response;

	  }
      }

  }

  void GpuSurfFeatures::getRawKeypoints(std::vector<cv::KeyPoint> & outKeypoints)
  {
    // This has an implicit "Thread synchronize"
    m_feature_counter.pullFromDevice();
    m_rawFeatures.pullFromDevice();
    int ftcount = m_feature_counter[1];


    outKeypoints.resize(ftcount);


    for(int i = 0; i < ftcount; i++) {
      //  {hidx_x, hidx_y, threadIdx.z, octave};
      cv::KeyPoint & k = outKeypoints[i];

      k.pt.x = (float)m_rawFeatures[i].x;
      k.pt.y = (float)m_rawFeatures[i].y;

      k.response = (float)m_rawFeatures[i].z;

      k.octave= m_rawFeatures[i].w;
      k.angle = 0;

    }


  }

  void GpuSurfFeatures::setKeypoints(std::vector<cv::KeyPoint> const & inKeypoints){

    ASRL_ASSERT_LT(inKeypoints.size(), ASRL_SURF_MAX_FEATURES, "The device is unable to hold " << inKeypoints.size() << " features. Max: " << ASRL_SURF_MAX_FEATURES);
    
    m_feature_counter[0] = inKeypoints.size();
    m_feature_counter.pushToDevice();

    for(unsigned i = 0; i < inKeypoints.size(); i++)
      {
	cv::KeyPoint const & k = inKeypoints[i];
	Keypoint & ft = m_features[i];
	ft.x = k.pt.x;
	ft.y = k.pt.y;
	ft.size = k.size;
	ft.angle = k.angle;
	ft.response = k.response;
	ft.octave = k.octave;
      }

    m_features.pushToDevice(inKeypoints.size());
  }

} // namespace asrl
