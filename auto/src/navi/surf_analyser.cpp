#include "surf_analyser.h"

#include <numeric>

#include <boost/foreach.hpp>
#include <boost/thread.hpp>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "timer.h"

#define SIFT 0

using cv::Mat;

struct surf_analyser::pimpl
{
  struct distance_type
  {
    typedef float  ValueType;
    typedef double ResultType;

    ResultType operator () (ValueType const *a, ValueType const *b, int size) const
    {
      ResultType result = 0;
      for (int i = 0; i < size; i++)
        {
          ResultType diff = a[i] - b[i];
          result += fabs (diff);
        }
      return result;
    }
  };

#if SIFT
  typedef cv::SiftFeatureDetector               detector_type;
  typedef cv::SiftDescriptorExtractor           extractor_type;
#else
  typedef cv::SurfFeatureDetector               detector_type;
  typedef cv::SurfDescriptorExtractor           extractor_type;
#endif
  typedef cv::BruteForceMatcher<distance_type>  matcher_type;

  detector_type  detector;
  extractor_type extractor;
  matcher_type   matcher;

  struct history
  {
    Mat mat;
    cv::vector<cv::KeyPoint> keypoints;
    Mat descriptors;
  } prev;

  Mat const mask;

  pimpl ()
    : mask (cv::imread (SRCDIR"/masks/lower_mask.jpg", CV_LOAD_IMAGE_GRAYSCALE))
  {
  }

  void surf (Mat const &src, Mat &dst);
};

static double
add_distance (double lhs, cv::DMatch const &rhs)
{
  return lhs + rhs.distance;
}

/**
 * Removes matches that are likely to be incorrect.
 *
 * Currently, this function removes all matches whose distance is much larger
 * than the average distance. It does two cleaning cycles. The first time, all
 * matches that are more than 2 times the average distance are removed; the
 * second time, all that are more than 0.5 times the average distance larger
 * than the average distance are removed.
 *
 * If the matches were too diverse to make a decision, all matches are
 * discarded. If there was only one match, this single match is preserved.
 *
 * TODO: A better idea is to compare the actual difference vectors instead of
 * their magnitudes and weigh them using their y position on the image:
 * movement in the upper part of the image has a much larger effect on the
 * geological position of the feature being tracked.
 */
static void
clean_matches (Mat const &descriptors1, Mat const &descriptors2, cv::vector<cv::DMatch> &matches)
{
  std::sort (matches.begin (), matches.end ());
  for (int i = 0; i < 2; i++)
    {
      double avg_dist = std::accumulate (matches.begin (), matches.end (), 0.0, add_distance)
                      / matches.size ();
      while (!matches.empty () && matches.back ().distance > avg_dist * (2 - i / 2.))
        matches.pop_back ();
    }
}

void
surf_analyser::pimpl::surf (Mat const &src, Mat &dst)
{
  cv::vector<cv::KeyPoint> keypoints;
  {
    timer const T ("detecting keypoints");
    detector.detect (src, keypoints, mask);
  }

  Mat descriptors;
  {
    timer const T ("computing keypoints");
    extractor.compute (src, keypoints, descriptors);
  }

  if (prev.mat.size () != cv::Size ())
    {
      cv::vector<cv::DMatch> matches;
      {
        timer const T ("matching descriptors");
        matcher.match (prev.descriptors, descriptors, matches);
        clean_matches (prev.descriptors, descriptors, matches);
      }

      {
        timer const T ("drawing the results");
        drawMatches (prev.mat, prev.keypoints, src, keypoints, matches, dst);
      }

      if (matches.empty ())
        return;
    }

  prev.mat = src;
  prev.keypoints.swap (keypoints);
  swap (prev.descriptors, descriptors);
}


surf_analyser::surf_analyser ()
  : self (new pimpl)
{
}

surf_analyser::~surf_analyser ()
{
}

void
surf_analyser::operator () (Mat const &src, Mat &dst)
{
  self->surf (src, dst);
}
