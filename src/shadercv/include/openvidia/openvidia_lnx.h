/**
 * This file is part of the OpenVIDIA project at http://openvidia.sf.net
 * Copyright (C) 2004, James Fung
 *
 * OpenVIDIA is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * OpenVIDIA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OpenVIDIA; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 **/
#ifndef _ONV_LBUFFER_H
#define _ONV_LBUFFER_H

#include <cc++/thread.h>
#include <cc++/config.h>
#include <cc++/exception.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <Imlib2.h>


using namespace ost;
/**\brief LBuffer is basically a malloc'd buffer with an associated mutex locking access
 *
 * LBuffer is basically a malloc'd buffer with an associated
 *   mutex lock that can be checked before/locked during 
 *   reads/writes.  It has an "width/height" 
 */
class LBuffer : public Mutex, public Semaphore {
  private:
    int Width, Height, ElementSz;
    void *Ptr;
  public:
   
    ///Initializes a lockable buffer, of size W*H*sz (just like a malloc) 
    LBuffer( int W,  ///< Width
             int H,  ///< Height
             int sz  )  ///< Size per element
    {
      Ptr = malloc(sz*W*H); 
      Width = (size_t)W;
      Height = (size_t)H;
      ElementSz = (size_t)sz ;
    };

    ///Deletes the buffer object, and frees its associated memory.
    ~LBuffer() { 
      free(Ptr); 
    };
    /// Lock the buffer.  Any subsequent attempts to lock it while it is 
    /// already locked will block.
    void lock() { ENTER_CRITICAL; }

    /// Unlock the buffer. 
    void unlock() { LEAVE_CRITICAL; }

    /// Return the "width" of the buffer 
    size_t width() { return (size_t)Width; }

    /// Return the "height" of the buffer 
    size_t height() { return (size_t)Height; }

    /// Return the number of bytes of each element
    size_t elmentsz() { return (size_t)ElementSz; }

    /// Return the size of the buffer (W*H*elementsz)
    size_t size() { return (size_t)(ElementSz*Width*Height); }

    /// Return a pointer to the associated memory
    virtual void *ptr() { return Ptr; }

    /// Resize the buffer.
    int resize( int W,  	///< Width
                int H,  	///< Height
                int sz  )  	///< Size per element
    {
      Ptr = realloc( Ptr, sz*W*H );
      Width = (size_t)W;
      Height = (size_t)H;
      ElementSz = (size_t)sz ;
      return (sz*W*H); 
    }

};
#endif

class MemoryImage : public LBuffer, public img_src {
  private :
    GLuint texObj;
    bool dirtyBit;
  public :
    MemoryImage(void *, int w, int h, int comp, GLenum fmt );
    void *ptr();
    ///\brief Get the pixel width of the loaded image.
    GLuint w();
    ///\brief Get the pixel height of the loaded image.
    GLuint h();
    ///\brief Get a texture handle of a texture that holds the image. 
    /// Mark as clean (used).
    GLuint tex();
    ///\brief Get a texture handle of a texture that holds the image. 
    GLuint tex_nodirty();
    ///\brief Get the dirty flag.  True means image is new and unprocesed by
    /// your program.  False (clean) means images has been read once by your
    /// system.
    bool dirty();
    ///\brief Reset this frame's dirty bit.
    void resetDirty(); //need this when multiple filters accessing the same source.
    
};

///\brief Load an image from disk and use it as an image source. (Linux)
class FileFrame : public LBuffer, public img_src {
  private : 
    Imlib_Image Im;
    GLuint texObj;
    bool dirtyBit;
  public :
    int load_image( char *fname );
    void load( char const *fname );

    ///\brief Constructor function.  Loads the named image, asccessible as an
    /// img_src object.  The image types compatible are those compatible with
    /// libImlib.
    FileFrame(char *fname);
    void *ptr();
    ///\brief Get the pixel width of the loaded image.
    GLuint w();
    ///\brief Get the pixel height of the loaded image.
    GLuint h();
    ///\brief Get a texture handle of a texture that holds the image. 
    /// Mark as clean (used).
    GLuint tex();
    ///\brief Get a texture handle of a texture that holds the image. 
    GLuint tex_nodirty();
    ///\brief Get the dirty flag.  True means image is new and unprocesed by
    /// your program.  False (clean) means images has been read once by your
    /// system.
    bool dirty();
    ///\brief Reset this frame's dirty bit.
    void resetDirty(); //need this when multiple filters accessing the same source.
};

class ListFile : public img_src {
  private :
    string listFileName;
    ifstream myfile;
    FileFrame *ff;
    bool dirtyBit;
    GLuint texObj;

  public :
    ListFile( char *fname );
    string getNextFilename() ;
    string advanceFile();
    int openListFile( char *fname ) ;
    void *ptr();
    ///\brief Get the pixel width of the loaded image.
    GLuint w();
    ///\brief Get the pixel height of the loaded image.
    GLuint h();
    ///\brief Get a texture handle of a texture that holds the image. 
    /// Mark as clean (used).
    GLuint tex();
    ///\brief Get a texture handle of a texture that holds the image. 
    GLuint tex_nodirty();
    ///\brief Get the dirty flag.  True means image is new and unprocesed by
    /// your program.  False (clean) means images has been read once by your
    /// system.
    bool dirty();
    ///\brief Reset this frame's dirty bit.
    void resetDirty(); 
    //need this when multiple filters accessing the same source.
};


/**
 * This file is part of the OpenVIDIA project at http://openvidia.sf.net
 * Copyright (C) 2004, James Fung
 *
 * OpenVIDIA is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * OpenVIDIA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OpenVIDIA; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 **/

#ifndef _DC1394_H
#define _DC1394_H


#include <cc++/thread.h>
//#include <thread.h>
#include <cc++/config.h>
#include <cc++/exception.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <libraw1394/raw1394.h>
#include <libdc1394/dc1394_control.h>

using namespace std;
using namespace ost;


/** \brief Dc1394 is a class which encapsulates the libdc1394 camera
    handling routines.  (Linux)

Dc1394 is a class which encapsulates the libdc1394 camera handling routines
    It is a "Thread" class (commonc++), which means that after it is 
    constructed, it's start() method must be called which starts 
    the thread running.  While dc1394 is running, it is continually
    grabbing images from a camera, and placing them in its lockable
    buffer.  When an image is recieved from the 
    camera, Dc1394 is locked, updated, and unlocked.  If another 
    thread wants to use this image, it should lock, use( ptr() ), then unlock.
   
    See the LBuffer class for more information.

  Typical usage:
 <pre>
<b>
  //construct a 1394 capture object, using 320x240 resolution.
  // options are 640x480, 320x240, 160x120
  Dc1394 CamSource(320,240);
</b>

  // make a texture
  glGenTextures(1, &tex);              // texture 
  glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,  GL_REPLACE );
  glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex);
  glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA, width,height, 0,
                GL_RGB, GL_UNSIGNED_BYTE,NULL );
<b>  
  //start the camera capture
  CamSource.start();
</b>

   // rendering loop
   while(1) {
<b>
        //wait for a new frame to be captured.
        // this can be removed if you dont mind re-using a previous frame.
        CamSource.wait();
        //lock the image data so it is not updated while we are capturing.
        CamSource.lock();
</b>
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex);
        glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0,0, width,height,
                         GL_RGB, GL_UNSIGNED_BYTE,<b>CamSource.ptr()</b>);
<b>
        //free the image data so it can be updated again.
        CamSource.unlock();
</b>
     
        //use the image...
   }
 </pre>
*/
//class Dc1394 : public Thread, public Mutex, public Semaphore
//class Dc1394 : public Thread, public LBuffer, public Semaphore
class Dc1394 : public LBuffer, public Thread, public img_src
{  
 private:
  
  void TellRWMHeCanUseImage(int numBufs, const char *dma_bufs_[] );
  const static int DefaultCaptureWidth=320;
  const static int DefaultCaptureHeight=240;
  int CaptureWidth,CaptureHeight;
  GLuint texture;
  
  dc1394_cameracapture cameras[8];
  raw1394handle_t handles[8]; // selected camera
  nodeid_t *camera_nodes; // all nodes on bus

  int bufferUsed;
  int numCamsToUse;
  int numCamsUsed;
  char *bufs[8] ; //array of adresses of the buffers where the images end up
  bool noDMA;
  bool dirtybit;
  bool doOHCI();
//  void lock() { ENTER_CRITICAL; }
//  void unlock() { LEAVE_CRITICAL; } 
  void tellThreadDoneWithBuffer();

  void releaseBarrier();
  
  /// Start Thread.  Camera will begin capturing, writing results to 
  /// memory at ptr().  To access image, lock() this object, use the ptr()
  /// data, then unlock() when finished with the buffer.
  void run(); 
  void createTexture();
 public:
  ///The lockable LBuffer which holds the camera image data.  When accessing it,
  /// remember to: lock(), use(), unlock()
  //LBuffer rgb_buffer;

  ///Initializes the camera capture.
  Dc1394( int W=320, ///< desired capture width, either 320 or 640
          int H=240  ///<desired capture height, either 240 or 480
        );
  ~Dc1394();
  GLuint w();
  GLuint h();
  GLuint tex();
  GLuint tex_nodirty();
  void resetDirty();
  bool dirty();
  
};

#endif

/**
 * This file is part of the OpenVIDIA project at http://openvidia.sf.net
 * Copyright (C) 2004, James Fung
 *
 * OpenVIDIA is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * OpenVIDIA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OpenVIDIA; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

 * based on Video4linux2 capture.c example code.
 **/

#ifndef _ONV_V4L1_H
#define _ONV_V4L1_H
#include <cc++/thread.h>
//#include <thread.h>
#include <cc++/config.h>
#include <cc++/exception.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <asm/types.h>          /* for videodev2.h */
#include <linux/videodev.h>
#include <libdc1394/dc1394_control.h>


typedef enum {
	IO_METHOD_READ,
	IO_METHOD_MMAP,
	IO_METHOD_USERPTR,
} io_method;

struct buffer {
        void *                  start;
        size_t                  length;
};

using namespace std;
using namespace ost;

class V4L2 : public LBuffer, public Thread, public img_src
{
    private:
        void TellRWMHeCanUseImage(int numBufs, const char *dma_bufs_[] );
        int bufferUsed;
        char *dev_name;
        io_method io;
        int fd;
        struct buffer *buffers;
        struct video_mbuf v4l1_mbuf;
        unsigned int n_buffers;
        unsigned int current_buffernum;
        unsigned	int format;
        char *newimgbuf_rgb;
        char *read_frame(void);
        void start_capturing(void);
        void stop_capturing(void);
        void uninit_device(void);
        void init_mmap(void);
        void init_read(unsigned int);
        void init_userp(unsigned int);
        void init_device(void);
        void close_device(void);
        void open_device(void);
        int  set_controls();

        const static int DefaultCaptureWidth=320;
        const static int DefaultCaptureHeight=240;
        int CaptureWidth,CaptureHeight;
        void run();
        void tellThreadDoneWithBuffer();

        GLuint texture;
        void createTexture();
        bool dirtybit;


    public:
        V4L2(int W, int H);
        ~V4L2();
        // Start Thread
        //void lock() { ENTER_CRITICAL; }
        //void unlock() { LEAVE_CRITICAL; }

        ///\brief Get the current pixel width of images captured.
        int getCaptureWidth();
        ///\brief Get the current pixel hieght of images captured.
        int getCaptureHeight();
        ///\brief Get the current bit depth of images captured.
        int getCaptureDepth();
        ///\brief Get pointers to each of the capture buffers.  Images are
        ///buffered in this array.
        char *bufs[8];

        ///\brief Get the pixel width of the loaded image.  
        GLuint w();
        ///\brief Get the pixel height of the loaded image. 
        GLuint h();
        ///\brief Get a texture handle of a texture that holds the most recent camera image. Mark as clean (used). 
        GLuint tex();
        ///\brief Get the dirty flag. True means image is new and unprocesed by your program. False (clean) means images has been read once by your system.
        bool dirty();
        ///\brief Get a texture handle of a texture that holds the image.  Do not
        /// modify the dirty bit.
        GLuint tex_nodirty();
        ///\brief Reset this frame's dirty bit. 
        void resetDirty();

        ///\brief Return a pointer to the associated image data memory.
        void *ptr();
};

///\brief Video4Linux camera input object that will initialize and 
///get frames of video from any compatible Video4Linux device. (This object is Linux only)
/**This object captures images from a Video4Linux device and 
 * places the images in a texture for processing.  It is a 
 * img_src class, and thus supports an interface which allows it to
 * act as a source of images.
 * It should be initialized after OpenGL is initialized.
 */
class V4L1 : public LBuffer, public Thread, public img_src
{
 private:
  void TellRWMHeCanUseImage(int numBufs, const char *dma_bufs_[] );
  int bufferUsed;
  char *dev_name;
  io_method io;
  int fd;
  struct buffer *buffers;
  struct video_mbuf v4l1_mbuf;
  unsigned int n_buffers;
  unsigned int current_buffernum;
  unsigned	int format;
  char *newimgbuf_rgb;
  char *read_frame(void);
  void start_capturing(void);
  void stop_capturing(void);
  void uninit_device(void);
  int  init_mmap(void);
  int init_read(unsigned int);
  void init_userp(unsigned int);
  int  init_device(void);
  void close_device(void);
  void open_device(void);
  int  set_controls();


 //protected:
  const static int DefaultCaptureWidth=320;
  const static int DefaultCaptureHeight=240;
  int CaptureWidth,CaptureHeight;
  void run();
  void tellThreadDoneWithBuffer();
  
  GLuint texture;
  void createTexture();
  bool dirtybit;

 public:
  V4L1(int W, int H);
  ~V4L1();
  // Start Thread
  //void lock() { ENTER_CRITICAL; }
  //void unlock() { LEAVE_CRITICAL; }

  ///\brief Get the current pixel width of images captured.
  int getCaptureWidth();
  ///\brief Get the current pixel hieght of images captured.
  int getCaptureHeight();
  ///\brief Get the current bit depth of images captured.
  int getCaptureDepth();
  ///\brief Get pointers to each of the capture buffers.  Images are
  ///buffered in this array.
  char *bufs[8];
 
  ///\brief Get the pixel width of the loaded image.  
  GLuint w();
  ///\brief Get the pixel height of the loaded image. 
  GLuint h();
  ///\brief Get a texture handle of a texture that holds the most recent camera image. Mark as clean (used). 
  GLuint tex();
  ///\brief Get the dirty flag. True means image is new and unprocesed by your program. False (clean) means images has been read once by your system.
  bool dirty();
  ///\brief Get a texture handle of a texture that holds the image.  Do not
  /// modify the dirty bit.
  GLuint tex_nodirty();
  ///\brief Reset this frame's dirty bit. 
  void resetDirty();

  ///\brief Return a pointer to the associated image data memory.
  void *ptr();

};


#include <cc++/thread.h>
#include <cc++/config.h>
#include <cc++/exception.h>
#include <semaphore.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>


using namespace std;
using namespace ost;

struct framestruct {
  ///width of the frame
  int width;
  ///height of the frame
  int height;
  ///ID number for misc. use.
  int IDnum;
  ///number of channels (greyscale,color,RGBA etc)
  int channels;
  ///Current transformation parameters.
  ///This is used to transfer initial guess parameters,
  ///and then hold the state of the current estimation afterwards,
  ///as better estimates are calculated.
  double params[8];
  ///used for parallel orbits.  Server sets requeested chirps to
  ///some number which it wants to client cards to run on this 
  ///image.  Then, the client sees this, and starts running the
  ///estimations, decrementing this each time - both can see
  ///how the work is progressing.  Results are in the params 
  ///member above
  int requested_chirps;


  ///flags if this frame is being placed into Shm, and should be 
  ///considered as a new request for work
  bool isNew;

  /*
   * pass skin tome information to other
   * people reading shmframe
   */
  //could use one int, TOP 16 bits are X, bottom 16 bits are Y
  int hand_x_320;
  int hand_x_640;
  int hand_y_240;
  int hand_y_480;

  // for rect server
  int rect_x_320[4];
  int rect_y_240[4];

  float thresh_x;//X: skin typically H value < 0.02
  float thresh_y;//Y: skin Sat.typically 0.3->0.6 (camera dependent)
  float thresh_z;//Z: variance allowed around Sat. value
  float thresh_w;//W: maximum allowable brightness (value)


  // Finger tracker sets these signals.  Signals are cleared by RWM once it consumes them.
  int MOUSE_SIGNAL;
  int MOUSE_SIGNAL_x_320;
  int MOUSE_SIGNAL_y_240;

  //information string. free form string about the image
  char infoString[1024];
};

/** \brief The ShmFrame class encapsulates the inter-GPU communication scheme by being essentially a shared object who's state is seen/updated by any interested GPU.  Mostly, it encapsulates the nitty gritty IPC shm instructions.  
*/

class ShmFrame : public LBuffer {
  private:
    ///Data pointer is stored here so that it is consistent 
    ///between different shared memory users (i.e. each will
    ///see their own internal address here, which points 
    ///always to the same shared memory in the end).
    ///This pointer is shm-alloc'd in the object initialization.
    unsigned char *data;
    bool attachShm();
    bool createShm(int w, int h, int nchans, int ID);
    void initFrameInfo(int w, int h, int nchans, int ID);
    int ipcid;
    int semid;
    bool has_lock;
  public:

    ///\brief This structure holds meta-information about the frame. 
    ///The framestruct structure can be defined as needed at add
    /// extra information associated with your frame.  Just recompile
    /// and delete existing memory segments, then run the programs to
    /// create new ones.
    struct framestruct *Frame;

    ///default constructor attempts to attach to an
    ///assumed exisiting shared memory segments
      //ShmFrame(int ipcidin, float b);
//      ShmFrame(int ipcidin);

    ///\brief Constructor function creates a shared memory area
    /// of a particular size and number of color channels.
    /// ipcidin is an integer uniquely identifying the shared memory area
    ///   ipcidin is usually 1-5 for instance, to have 5 different 
    ///   shared memory areas used by openvidia programs
    /// ID is a generic identification number stored inside the shared
    ///   memory area, which could be used as a unique framecounter. 
    ShmFrame( int ipcidin, int w, int h, int nchans, int ID ) ;

 
    ///\brief Construct a shared memory object, that is to be 
    /// attahed to some existing shared memory object, created by
    /// possibly another process.
    ShmFrame( int ipcidin ) ;

    ///\brief Set the data inside the shared memory area.  
    ///This copies the given frame data into shared memory
    int setFrame(int w, int h, int channels, int IDnum, unsigned char *);
  

    ///\brief Set a Projective Coordinate Transformation
    /// associated with this frame.  Useful for record keeping.
    void setParams( PCT pin );

    ///\brief Set the image data segment and an associated 
    /// Projective Coordinate Transformation
    /// associated with this frame. 
    int setFrame(int w, int h, int nchans, int ID, unsigned char *d,
                       int req_chirp, PCT Pin);

    ///\brief Get a pointer to the image data in the shared memory area.
    /// Changing this image data changes the image as seen by all
    /// other programs accessing it.
    unsigned char *getData() { return data; } 
    ///Get the ID number of the current ShmFrame content.
    int getID();
/*
            int theid;
            enterMutex();  //mm mutexes only matter to the process...
            theid = Frame->IDnum;
            leaveMutex(); //mutex probably not doign anything between process
            return theid;
*/
   
    ///Set the number of chirps the client is requested to do.
    void setRequestedChirps(int r){ Frame->requested_chirps = r; } 
    ///Get the number of chirps the client has left to do. 
    int  getRequestedChirps(){ return Frame->requested_chirps; } 
    /**\brief
     * Returns true if the shmframe is currently processing data
     * Returns false otherwise
     **/
    bool isProcessing();

    ///\brief  Wait to gain access to the shared memory exclusively (so
    /// only your calling process is changing its contents), and lock out
    /// other processes. 
    void lock();

    ///\brief Allow the shared memory to be accessible to other processes.
    void unlock();

    ///\brief Retrieve associated text information. Free form text as needed.
    char *getInfoString();
    ///\brief Set associated text information. Free form text as needed.
    void setInfoString(char *s);
    ///\brief Mark this frame as having been processed.
    void finished();

    ///\brief Determine if the current calling process has the lock
    /// for this particular shared memory frame.
    bool hasLock() { return has_lock; }

    ///\brief Wait to access the shared memory atomically
    void semWait();
    ///\brief Signal to other processes we are accessing the memory 
    void semPost();
    ///\brief Reset all locks to the shared memory
    void semClear();

    ///\brief Get a pointer to the image data in the shared memory area.
    /// Changing this image data changes the image as seen by all
    /// other programs accessing it.
    void *ptr();
};

class Timer {
        private :
                struct timeval tv_start, tv_end; 
        public :
                int elapsedTime();
                void start();
                void end();
};

class AutoTimer 
{
    private:
        Timer t;
        Recorder r;
        bool samplingState ;
        int  sleepTime;
        struct timeval tv; 
    public:
        AutoTimer();
        void start();
        void end();
};



#endif
