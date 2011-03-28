#pragma once

#include <opencv2/core/core.hpp>

template<int N>
struct kernel
{
  kernel ()
    : K (new float[N * N])
    , P (K)
  {
  }

  operator cv::Mat ()
  {
    return cv::Mat (N, N, CV_32FC1, K).clone ();
  }

  cv::Mat operator / (int divisor)
  {
    cv::Mat mat = *this;
    mat /= divisor;
    return mat;
  }

  cv::Mat operator * (int divisor)
  {
    cv::Mat mat = *this;
    mat *= divisor;
    return mat;
  }

  template<typename... Args>
  kernel &operator () (Args... args)
  {
    static_assert (sizeof... (Args) == N, "invalid number of arguments");
    fill (args...);

    return *this;
  }

private:
  void fill (float v0)
  {
    printf ("%ld / %d\n", P - K, N * N);
    assert (P - K < N * N);
    *P++ = v0;
  }

  template<typename... Args>
  void fill (float v0, Args... args)
  {
    fill (v0);
    fill (args...);
  }

  float *K;
  float *P;
};

template<typename... Args>
kernel<sizeof... (Args)>
Kernel (Args... args)
{
  kernel<sizeof... (Args)> K;
  return K (args...);
}
