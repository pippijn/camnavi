#include "gpusift.h"

#include <cstdio>
#include <map>

#include <boost/array.hpp>
#include <boost/circular_buffer.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <GL/gl.h>

#include "gpu/gpusift/SiftGPU.h"
#include "timer.h"
#include "foreach.h"

using cv::Mat;

static SiftGPU *
init_sift ()
{
  SiftGPU *sift = new SiftGPU;
  char const *argv[] = {
    "-fo", "-1",
    "-v", "1",
    "-d", "2",
    "-p", "640x480",
    "-di",
    "-lmp", "100",
    "-t", "0.01",
    //"-cuda",
  };
  size_t argc = sizeof argv / sizeof *argv;
  sift->ParseParam (argc, const_cast<char **> (argv));
  if (sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
    {
      delete sift;
      throw 0;
    }
  return sift;
}

static std::auto_ptr<SiftGPU> sift;
static std::auto_ptr<SiftMatchGPU> matcher;


struct gpusift::pimpl
{
  static size_t const history_size = 5;

  struct history_item
  {
    static size_t last_id;

    static size_t next_id ()
    {
      return ++last_id;
    }

    size_t id;

    Mat mat;
    std::vector<SiftGPU::SiftKeypoint> keys;
    std::vector<float> descriptors;
#if 0
    // matches with all other history_items
    boost::array<MatchBuffer, history_size - 1> match_vector;

    MatchBuffer &matches (history_item const &other)
    {
      int distance = other.id - id;
      distance -= distance > 0;
      distance += match_vector.size () - (last_id - id);
      return match_vector[distance];
    }
#endif

    void swap (history_item &other)
    {
      //printf ("%lu is now %lu\n", id, other.id);
      std::swap (id, other.id);
      cv::swap (mat, other.mat);
      keys.swap (other.keys);
      descriptors.swap (other.descriptors);
#if 0
      for (size_t i = 0; i < match_vector.size (); i++)
        match_vector[i].swap (other.match_vector[i]);
#endif
    }

    bool empty () const
    {
      return mat.empty ();
    }

    history_item (Mat const &mat = Mat (), int count = 0)
      : id (next_id ())
      , mat (mat)
      , keys (count)
      , descriptors (128 * count)
    {
    }
  };
  
  struct history
  {
    size_t size () const { return data.size (); }
    history_item       &operator [] (size_t i)       { return data[i]; }
    history_item const &operator [] (size_t i) const { return data[i]; }

    // returns true if the history is complete (all elements have been filled)
    bool push (history_item &newitem)
    {
      data.push_back (newitem);
      return data.size () == data.capacity ();
    }

    void print_ids () const
    {
      fputs ("history:", stdout);
      foreach (history_item const &item, data)
        printf (" %lu", item.id);
      fputc ('\n', stdout);
    }

    history ()
      : data (history_size)
    {
    }

    // TODO: replace with statically sized buffer
    boost::circular_buffer<history_item> data;
  } hist;

  void gpusift (Mat const &src, Mat &dst);
  static void match (history_item const &item0, history_item const &item1, MatchBuffer &match_buffer);
  void match (Mat &dst);
};

size_t gpusift::pimpl::history_item::last_id;

void
gpusift::pimpl::gpusift (Mat const &src, Mat &dst)
{
  if (!sift.get ())
    {
      sift.reset (init_sift ());
      matcher.reset (new SiftMatchGPU (4096));
      matcher->VerifyContextGL ();
    }

  imwrite ("tmp.jpg", src);

  timer const T (__func__);
  if (!sift->RunSIFT ("tmp.jpg"))
    throw 0;

  int count = sift->GetFeatureNum ();
  // TODO: don't create new ones all the time
  history_item cur (src, count);
  sift->GetFeatureVector (&cur.keys[0], &cur.descriptors[0]);

  if (hist.push (cur))
    match (dst);
}

void
gpusift::pimpl::match (history_item const &item0, history_item const &item1, MatchBuffer &match_buffer)
{
  printf ("matching %lu and %lu: ", item0.id, item1.id);
  matcher->SetDescriptors (0, item0.keys.size (), &item0.descriptors[0]);
  matcher->SetDescriptors (1, item1.keys.size (), &item1.descriptors[0]);

  matcher->GetSiftMatch (match_buffer);
  printf ("%lu matches\n", match_buffer.size ());
}

void
gpusift::pimpl::match (Mat &dst)
{
  //hist.print_ids ();
  //return;

  std::map<int, std::map<int, MatchBuffer> > matches;

  // match all items with all other items
  foreach (history_item &reference, hist.data)
    {
      foreach (history_item &other, hist.data)
        {
          if (reference.id < other.id)
            {
              MatchBuffer &match_buffer = matches[reference.id][other.id];
              match (reference, other, match_buffer);
            }
        }
    }

  // go through each item's match set and remove the ones that are not in all
  // match sets for the target
  // TODO
#if 0
  foreach (history_item &reference, hist.data)
    {
      foreach (history_item &other, hist.data)
        {
          if (reference.id < other.id)
            {
              MatchBuffer &match_buffer = matches[reference.id][other.id];
              for (size_t m = 0; m < match_buffer.size (); ++m)
                {
                  SiftGPU::SiftKeypoint const &key0 = reference.keys[match_buffer[m][0]];
                  SiftGPU::SiftKeypoint const &key1 = other.keys[match_buffer[m][1]];
                }
            }
        }
    }
#endif
}

#if 0
void
gpusift::pimpl::match_matches ()
{
  // enumerate all the feature matches
  for (size_t m = 0; m < match_buffer.size (); ++m)
    {
      // How to get the feature matches: 

      // key0 in the first image matches with key1 in the second image
      int const g = 50 * (i + 1);
      line (dst,
            cv::Point (key0.x, key0.y),
            cv::Point (key1.x, key1.y),
            cv::Scalar (g, g, g));
    }
}
#endif


gpusift::gpusift ()
  : self (new pimpl)
{
}

gpusift::~gpusift ()
{
}

void
gpusift::operator () (Mat const &src, Mat &dst)
{
  self->gpusift (src, dst);
}
