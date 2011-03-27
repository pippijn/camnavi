#pragma once

#include <boost/array.hpp>

#include <opencv2/core/core.hpp>

template<int Cols>
struct kernel_base
{
  kernel_base ()
  {
  }

  operator cv::Mat ()
  {
    return cv::Mat (K.size (), Cols, CV_32F, &K[0][0]).clone ();
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

protected:
  cv::vector<boost::array<float, Cols> > K;
};


template<int Cols>
struct kernel;


template<>
struct kernel<1>
  : kernel_base<1>
{;
  using kernel_base<1>::K;

  kernel &operator () (float v0)
  {
    boost::array<float, 1> R = { {
      v0,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<1>
Kernel (float v0)
{
  kernel<1> K;
  return K (v0);
}


template<>
struct kernel<2>
  : kernel_base<2>
{;
  using kernel_base<2>::K;

  kernel &operator () (float v0, float v1)
  {
    boost::array<float, 2> R = { {
      v0, v1,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<2>
Kernel (float v0, float v1)
{
  kernel<2> K;
  return K (v0, v1);
}


template<>
struct kernel<3>
  : kernel_base<3>
{;
  using kernel_base<3>::K;

  kernel &operator () (float v0, float v1, float v2)
  {
    boost::array<float, 3> R = { {
      v0, v1, v2,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<3>
Kernel (float v0, float v1, float v2)
{
  kernel<3> K;
  return K (v0, v1, v2);
}


template<>
struct kernel<4>
  : kernel_base<4>
{;
  using kernel_base<4>::K;

  kernel &operator () (float v0, float v1, float v2, float v3)
  {
    boost::array<float, 4> R = { {
      v0, v1, v2, v3,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<4>
Kernel (float v0, float v1, float v2, float v3)
{
  kernel<4> K;
  return K (v0, v1, v2, v3);
}


template<>
struct kernel<5>
  : kernel_base<5>
{;
  using kernel_base<5>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4)
  {
    boost::array<float, 5> R = { {
      v0, v1, v2, v3, v4,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<5>
Kernel (float v0, float v1, float v2, float v3, float v4)
{
  kernel<5> K;
  return K (v0, v1, v2, v3, v4);
}


template<>
struct kernel<6>
  : kernel_base<6>
{;
  using kernel_base<6>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5)
  {
    boost::array<float, 6> R = { {
      v0, v1, v2, v3, v4, v5,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<6>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5)
{
  kernel<6> K;
  return K (v0, v1, v2, v3, v4, v5);
}


template<>
struct kernel<7>
  : kernel_base<7>
{;
  using kernel_base<7>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6)
  {
    boost::array<float, 7> R = { {
      v0, v1, v2, v3, v4, v5, v6,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<7>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6)
{
  kernel<7> K;
  return K (v0, v1, v2, v3, v4, v5, v6);
}


template<>
struct kernel<8>
  : kernel_base<8>
{;
  using kernel_base<8>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7)
  {
    boost::array<float, 8> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<8>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7)
{
  kernel<8> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7);
}


template<>
struct kernel<9>
  : kernel_base<9>
{;
  using kernel_base<9>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8)
  {
    boost::array<float, 9> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<9>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8)
{
  kernel<9> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8);
}


template<>
struct kernel<10>
  : kernel_base<10>
{;
  using kernel_base<10>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9)
  {
    boost::array<float, 10> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<10>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9)
{
  kernel<10> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9);
}


template<>
struct kernel<11>
  : kernel_base<11>
{;
  using kernel_base<11>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10)
  {
    boost::array<float, 11> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<11>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10)
{
  kernel<11> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10);
}


template<>
struct kernel<12>
  : kernel_base<12>
{;
  using kernel_base<12>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11)
  {
    boost::array<float, 12> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<12>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11)
{
  kernel<12> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11);
}


template<>
struct kernel<13>
  : kernel_base<13>
{;
  using kernel_base<13>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12)
  {
    boost::array<float, 13> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<13>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12)
{
  kernel<13> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12);
}


template<>
struct kernel<14>
  : kernel_base<14>
{;
  using kernel_base<14>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13)
  {
    boost::array<float, 14> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<14>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13)
{
  kernel<14> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13);
}


template<>
struct kernel<15>
  : kernel_base<15>
{;
  using kernel_base<15>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14)
  {
    boost::array<float, 15> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<15>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14)
{
  kernel<15> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14);
}


template<>
struct kernel<16>
  : kernel_base<16>
{;
  using kernel_base<16>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15)
  {
    boost::array<float, 16> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<16>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15)
{
  kernel<16> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
}


template<>
struct kernel<17>
  : kernel_base<17>
{;
  using kernel_base<17>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16)
  {
    boost::array<float, 17> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<17>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16)
{
  kernel<17> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16);
}


template<>
struct kernel<18>
  : kernel_base<18>
{;
  using kernel_base<18>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17)
  {
    boost::array<float, 18> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<18>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17)
{
  kernel<18> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17);
}


template<>
struct kernel<19>
  : kernel_base<19>
{;
  using kernel_base<19>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18)
  {
    boost::array<float, 19> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<19>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18)
{
  kernel<19> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18);
}


template<>
struct kernel<20>
  : kernel_base<20>
{;
  using kernel_base<20>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19)
  {
    boost::array<float, 20> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<20>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19)
{
  kernel<20> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19);
}


template<>
struct kernel<21>
  : kernel_base<21>
{;
  using kernel_base<21>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20)
  {
    boost::array<float, 21> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<21>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20)
{
  kernel<21> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20);
}


template<>
struct kernel<22>
  : kernel_base<22>
{;
  using kernel_base<22>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21)
  {
    boost::array<float, 22> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<22>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21)
{
  kernel<22> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21);
}


template<>
struct kernel<23>
  : kernel_base<23>
{;
  using kernel_base<23>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22)
  {
    boost::array<float, 23> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<23>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22)
{
  kernel<23> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22);
}


template<>
struct kernel<24>
  : kernel_base<24>
{;
  using kernel_base<24>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23)
  {
    boost::array<float, 24> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<24>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23)
{
  kernel<24> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23);
}


template<>
struct kernel<25>
  : kernel_base<25>
{;
  using kernel_base<25>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23, float v24)
  {
    boost::array<float, 25> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<25>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23, float v24)
{
  kernel<25> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24);
}


template<>
struct kernel<26>
  : kernel_base<26>
{;
  using kernel_base<26>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23, float v24, float v25)
  {
    boost::array<float, 26> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<26>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23, float v24, float v25)
{
  kernel<26> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25);
}


template<>
struct kernel<27>
  : kernel_base<27>
{;
  using kernel_base<27>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23, float v24, float v25, float v26)
  {
    boost::array<float, 27> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<27>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23, float v24, float v25, float v26)
{
  kernel<27> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26);
}


template<>
struct kernel<28>
  : kernel_base<28>
{;
  using kernel_base<28>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23, float v24, float v25, float v26, float v27)
  {
    boost::array<float, 28> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<28>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23, float v24, float v25, float v26, float v27)
{
  kernel<28> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27);
}


template<>
struct kernel<29>
  : kernel_base<29>
{;
  using kernel_base<29>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23, float v24, float v25, float v26, float v27, float v28)
  {
    boost::array<float, 29> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<29>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23, float v24, float v25, float v26, float v27, float v28)
{
  kernel<29> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28);
}


template<>
struct kernel<30>
  : kernel_base<30>
{;
  using kernel_base<30>::K;

  kernel &operator () (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23, float v24, float v25, float v26, float v27, float v28, float v29)
  {
    boost::array<float, 30> R = { {
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
    } };
    K.push_back (R);
    return *this;
  }
};

kernel<30>
Kernel (float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23, float v24, float v25, float v26, float v27, float v28, float v29)
{
  kernel<30> K;
  return K (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29);
}
