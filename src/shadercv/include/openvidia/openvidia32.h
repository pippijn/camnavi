#ifndef __OPENVIDIA32_H
#define __OPENVIDIA32_H

#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>

#ifdef WIN32
#include <GL/glew.h>
#endif

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>

#ifdef WIN32
#define GLEW_STATIC 1
#endif

#include <GL/glut.h>
#include <Cg/cgGL.h>

using namespace std;

#ifdef WIN32
# ifdef DLL_EXPORT
#   define DECLSPEC __declspec(dllexport)
#   define EXPIMP_TEMPLATE
# else
#  define DECLSPEC __declspec(dllimport)
#  define EXPIMP_TEMPLATE extern
#endif

  EXPIMP_TEMPLATE template class DECLSPEC std::vector<float>;
  EXPIMP_TEMPLATE template class DECLSPEC std::pair<float,float>;

#else
# define DECLSPEC
# define EXPIMP_TEMPLATE
#endif

///\brief Class to hold coordinates (ordered STL pairs of floats)
class DECLSPEC Coords : public pair<float,float>{
  public : 
  	///Constructor.  Constructs coordinates (x,y) = (0,0) by default.  Optionally 
  	///given the coordinates to create.  This class provides get/set functions 
  	///for coordinates.
    Coords( 
    		float a=0,  ///< First Coordinate
    		float b=0 ) ///< Second Coordinate
    { 
    	first = a; 
    	second = b; 
    }
    //look like the vgl libs version for future association
    ///Returns the first, "x", coordinate.
    float x() { return first; }
    ///Returns the second, "y", coordinate.
    float y() { return second; }
    //Set the coordinates
    void set(
    	float in1,  ///< x coordinate value
    	float in2 ) {  ///< y coordinate value
      first = in1; 
      second = in2;
    }
    ///Print out the coordinates to stderr.
    void print() {  
      cerr<<" "<<x()<<" "<<y()<<" ";
    }
};

///\brief The feature class holds the information about a feature found in an image.
/// The feature class holds the information about a feature found in an image.
/// It inherits from the Coords class, so has an x and y denoting its position
/// in the image.    See the fvutil.h file for some utility functions that operate
/// on features.  Currently these are created by the featureTrack object.
class DECLSPEC Feature : public Coords {
  public:
///\brief win32 and linux use descArray.
/// element feature descriptor (SIFT "key")
///  It can be considered to "match" another
///  another feature by calculating typical euclidean distance.
///linux g++4.0 needs the descriptor though
///because if no push back call is made , -o3 optimizes the 
///assignement out and results in 0 (see featureTracker.getScene() )
	float descArray[128];
#ifndef WIN32
///\brief 128 element feature descriptor (SIFT "key")
///  It can be considered to "match" another
///  another feature by calculating typical euclidean distance.
   vector<float> descriptor;        
#endif
    float orientation;               ///< \brief Orientation of the feature = atan2f( dy, dx )
    float dx;                        ///< \brief X derivative in a 16x16 region around the feature centre.
    float dy;                        ///< \brief Y derivative in a 16x16 region around the feature centre. 
    float magnitude;                 ///< \brief Magnitude of the gradient = sqrt( dx^2 + dy^2 ) 

    ///\brief Construct a feature object.
    Feature() {
      //descriptor.reserve(128*sizeof(float));
    }
    ///\brief returns the norm of the feature descriptor.
    double norm() {                  
      double n=0.0;
      for( int i=0 ;i<128; i++ ) { 
        //n += descriptor[i]*descriptor[i];
		n+=descArray[i]*descArray[i];
      } 
      return sqrt(n);
    }
    void print() {
      cerr<<"[";
      for( int i = 0 ; i < 4 ; i ++ ) {
        cerr<<descArray[i]<<" ";
      } 
      cerr<<"]"<<endl;
    }

};

//typedef pair<Feature *, Feature *> Match ;
#ifdef WIN32
EXPIMP_TEMPLATE template class DECLSPEC std::pair<Feature *, Feature *>;
#endif
/// a Match is an STL pair of pointers to two features.
class DECLSPEC Match : public pair<Feature *, Feature *> {
  public:
	  Match() {};
    Match( Feature *a, Feature *b ) : pair<Feature *,Feature*>( a, b ) {} 

    /// Returns true if the given match, op,  is the same set of pointers.
    //bool Match::operator==(Match op) {    
    bool operator==(Match op) {    
      return ( ( first==op.first )&&( second==op.second ) );
    }
    //void Match::print() {
    void print() {
      cerr<<first->x()<<","<<first->y()<<" -> "<<second->x()<<","<<second->y()<<endl;
    }
};

/// Matches is a STL vector of Match classes.
typedef vector<Match> Matches ;

#ifdef WIN32
EXPIMP_TEMPLATE template class DECLSPEC std::vector<Match>;
#endif


#define METHOD_EXACT 0
#define METHOD_SVD   1

//class DECLSPEC PCT : public vector<float>{
class DECLSPEC PCT : public vector<float>{
  private :
    float _FirstSingVal;
    void set( double a11, double a12, double b1, 
         double a21, double a22, double b2, 
         double c1,  double c2 ) ;
  public :
    PCT( ); 
    PCT( Matches &, int, int, int );
    PCT( float a11in, float a12in, float b1in, float a21in, 
         float a22in, float b2in,  float c1in, float c2in );
    
    float a11() { return (*this)[0]; } 
    float a12() { return (*this)[1]; } 
    float b1()  { return (*this)[2]; } 

    float a21() { return (*this)[3]; } 
    float a22() { return (*this)[4]; } 
    float b2()  { return (*this)[5]; } 

    float c1() { return (*this)[6]; } 
    float c2() { return (*this)[7]; } 

    void print() {
      PCT::iterator it = begin();
      while( it != end() ) {
        cerr<<*it++<<" ";
      }
      cerr<<endl;
    }

    //void PCT::operator!();
    void operator!();
    Coords project( Coords &p, int W, int H ) ;

    float norm() {
     return (a11()-1.0)*(a11()-1.0) + a12()*a12() + a21()*a21() +
            (a22()-1.0)*(a22()-1.0) + b1()*b1() + b2()*b2() +
            c1()*c1() + c2()*c2() ;
    }

    PCT operator-(PCT pct2);
    double FirstSingVal();
    
};




EXPIMP_TEMPLATE template class DECLSPEC std::vector<Feature>;

///\brief A "Scene" stores a set of features from a given picture/scene
class DECLSPEC Scene {
  public:
  
  
    vector<Feature> features;	///< features is an STL vector of features for the scene

    /// \brief Saves a scene to a disk.  
    /// The format is ASCII text integer of the number of 
    /// features in the scene, followed by
    /// the feature vectors, 1 per line as:
    /// X Y Orientation dx dy magnitude [ 128 element descriptor ]
    /// (elements separated by spaces)
    void saveToDisk(const char *filename) ;

    ///\brief Default constructor creates empty scene 
    /// (empty vector of features )
    Scene() ;
    
    /// \brief Constructor with an integer argument creates a scene with a preallocated 
    /// memory area for the features (for faster performance).  This is  
    /// the preferrred method for initializing a scene structure.
	Scene(int x);

    ///\brief Constructor is given a filename, which is loaded.
    Scene(const char *filename) ;
  
    ///\brief Matches this scene to another scene.  Must be
    /// given the width/height the images the scenes are from.  Fills
    /// in a best PCT that describes the motion of features between scenes. 
    /// Returns a list of matching features.  
    Matches MatchTo( Scene *s , int w, int h, PCT &bestP) ;

    ///\brief Determines how many matched features
    /// moved according to  the motion of a particular PCT.  A 
    /// radius around the predicted and matched locations can be specified
    /// to deal with some noise in the estimations (so exact locations do 
    /// not have to match).
    Matches getSupport( PCT &P, Matches &m, float radius, int w, int h );

    ///\brief This function trims out the given matches OUT of the scene.
    /// ** not implemented **
    int  TrimOutMatches( Matches *m);

    ///\brief Adds a new set of features to a scene.  New features 
    /// may occur because of camera motion panning in new areas while
    /// it pans around a particular scene.
    void AddFeaturesToScene( Scene *toAdd, Matches *m, PCT P, int w, int h );

  private :
    PCT RANSAC( Matches &m, int w, int h );
 

};


///\brief Virtual interface class, functions that all image sources support.
/** This class encapsulates functions that allow an object to behave as a 
 * source of images.  Files, cameras, and filtered images can be image 
 * sources.  In particular it is written to build a "filter graph"
 * type pipeline of sets of image sources and filters.   Typically,
 * this graph would be constructed once, with each filter being assigned
 * a previous filter as its input source, and some image souce (like a camera
 * or file input) as the original input.  Then, the tex() function of the 
 * last filter object can be called to cause execution of the entire graph to
 * produce the lastest image result.
 */
class DECLSPEC img_src {
public :
	///\brief Return the pixel width of the image source.
	///@return The pixel width of the image source.
    virtual GLuint w()=0;
    
    ///\brief Return the height of the image source.
    ///@return The height of the image source.
    virtual GLuint h()=0;
    
    ///\brief Return the texture handle containing the image from the source.
    ///If the class returning the tex() is a filter or does some processing,
    ///it returns the processed version of the texture, possibly performing
    ///its filtering operations before returning from this call.
    ///@return The GL texture handle (integer) of a texture with the image. 
    virtual GLuint tex()=0;
    
    ///\brief Asks if the current image source is "dirty."  A Dirty
    ///value of true indicates that the source has updated the image
    ///and this new image has not yet been processed.
    ///
    ///@return The true/false "Dirty" flag for the source image.
    virtual bool dirty()=0;
    
    ///\brief Return the texture handle containing the image from this source,
    ///without doing any "dirty" test.  This function just returns the
    ///texture handle, without doing any processing.  Thus, it just
    ///retrieves whatever is currently in the source without processing.
    ///
    ///@return The GL texture handle (int) of a texture with this image.
    virtual GLuint tex_nodirty()=0;
    
    
    ///\brief Resets the source's dirty bit.  For filters, it should just ask its source to
    ///reset.  For real sources, they should actually havea dirty bit that is reset.
    virtual void resetDirty()=0;

    virtual ~img_src(){};

};

///\brief Performs image filtering.  This object applies a Cg program (a
/// filter) to an image.  The Cg program is run on the image at each pixel.  The
/// result is rendered to a texture.
class DECLSPEC FBO_Filter : public img_src {
private :
    CGprogram cgProgram;
    CGprogram cgDisplayProgram;
    CGprofile cgProfile;
    CGcontext cgContext;

    GLuint fb, oTex;
    GLuint pboBuf;
    int tWidth; int tHeight;
    int previousViewportDims[4] ;

    CGprogram load_cgprogram(CGprofile prof, char *name, char **args) ;

    void renderBegin() ;
    //return the rendering state to what it was before.
    void renderEnd() ;

    void drawQuadTex(int,int);

   void onvTapCoords(float baseX, float baseY ) ;

    void emitOffsets(float,float,vector<Coords>);

    vector<Coords> taps;
    vector<img_src *> srcs;
    char *fpname;

    GLint outputPrecision ;
    
protected :
    void createTexture(int W, int H, GLint fmt) ;
    img_src *source_ptr;
public :

    ///\brief Set the program this filter uses.  It will reload this program
    /// at this point and images will be filtered with it from here in.
    void setProgram(char *fname);

    ///\brief Constructor.  This object filters images according to a
    /// user written Cg program.  It can act upon any valid img_src, and
    /// by default, the action will only occur if the dirty bit is set 
    /// (i.e. new data appears at the source).
    FBO_Filter(
                char *name,         ///< filename of the Cg program.  Note that
                                    /// the entry point function should be named
                                    /// "FragmentProgram" in the Cg 
                                    /*
                 GLuint outputTex,  ///< the open GL texture object 
                                    /// to which the results will go 
                 int W,             ///< width of the output texture, 
                 int H,             ///< height of the output texture
                 */
                 img_src *src,
                 
                 /// Output format.  If using 32-bit floats, remember that these do not display
                 /// directly on the screen - they require a Fragment Program to convert the RGBA32
                 /// to RGBA8 for display.  If you just need to see the pixels, use a 8 bit, displayable
                 /// format like GL_RGBA
                 GLint outputFmt = GL_RGBA,
                 vector<Coords> *taps = NULL,

                 /// (Optional) arguments to the cgc compiler. Useful for
                 /// example to specify compile time #defines for example
                 char **args  = NULL );

    ///\brief Set a 4-element floating point input variable to the Cg program.
    void setCGParameter(char *name, float *value);

    ///\brief Set a 4-element double input variable to the Cg program. Note:
    /// the hardware does not support double precision, this is mainly for
    /// avoiding casting requirements in the calling program.
    void setCGParameter(char *name, double *value);

    ///\brief Flargle.
    GLuint apply( GLuint iTex, GLuint W, GLuint H, GLvoid *rb_buf, GLenum fmt );
    GLuint apply( img_src *, GLvoid *rb_buf = NULL, GLenum fmt =GL_RGBA  );
      GLuint w();
    GLuint h();
    GLuint tex();
    bool dirty();
    //void addSource( img_src *);
    void resetDirty();
    ~FBO_Filter();
    GLuint tex_nodirty();
    GLuint tex_apply();

    ///\brief Sets/changes the image source this filter operates on.
    void setSource(img_src *s);

    ///\brief Sets/changes the numerical precision used in this filter.
    void setOutputPrecision(GLint fmt ) ;

    ///\brief Resize the size of the texture this filter produces.
    void resize(GLint w, GLint h);

    ///\brief Activates Cg program to allow viewing of the resulting texture data.
    /// This is necessary when 32-bit precision floating point data is used ,since 
    /// it cannot be displayed without a Cg program.
    /// If no viewer is needed this returns false.
    /// If a viewer is activated this returns true.
    bool activateViewerProgram();
    ///\brief DeActivates the Cg program that allowed viewing of the resulting texture data.
    void deactivateViewerProgram();

};


class DECLSPEC DownFilter : public FBO_Filter {
public :
    DownFilter( char *name, img_src *src, GLint fmt, vector<Coords> *taps = NULL,
                char **args = NULL );
};

///\brief Finds the numerical sum of a texture's data.
class DECLSPEC Sum {

    private :

        vector<Coords> sizes;
        int width, height;
        GLuint  *fb;
        GLuint  *tex;
        float *rbbuf;
        int N_REDUCE; 

        /* vars for Cg */
        CGcontext cgContext;
        CGprogram fragmentProgram4tap, fragmentProgram8tap, fragmentProgram8tapY;
        CGprofile cgProfile;
        static GLenum errCode;
        const GLubyte *errString;

        void reduce(GLuint texture, int w, int h, int, int);
        void reduce4w(GLuint texture, int w, int h );

        void errcheck();
        void reshape(int , int);
        void init_cg();
        
        double sumBuf( float *b, int w, int h, int c) ;
        void sumBuf4( float *b, int w, int h, int c, double *r1, double *r2, double *r3, double *r4) ;
        bool isValidDimensions(GLuint w , GLuint h );
        int numLevels(int , int );
        bool makeStrategy(GLuint, GLuint, int );
        int getBestFold( int, int );

    public :
        ///\brief Constructs a summation object whose purpose is to sum
        /// up texture data, of a set size.  Optionally you can
        /// specify a desired number of reduction, otherwise it will
        /// utilize an efficient number of reductions.
        Sum(GLuint W, GLuint H, int numReduce=-1 ); 
   
        ///\brief Get the numerical sum of a given texture's data.  Optionally,
        /// specify the perChannel flag to return a set of numbers which are
        /// only the sums of each channel.
        void getSum(GLuint texInput,double *result,bool perChannel=false) ;
        ~Sum();
        ///\brief Determine the number of reductions this summation object is
        /// currently performing.
        int nReductions();
        //int setReductions(int);
};


class DECLSPEC CamParams {
private:
  float paramf1, paramf2, paramox, paramoy, paramk1, paramk2, paramk3, paramk4;
public:
  CamParams(float f1, float f2, float ox, float oy, 
            float k1, float k2, float k3, float k4) {
    paramf1 = f1;
    paramf2 = f2;
    paramox = ox;
    paramoy = oy;
    paramk1 = k1;
    paramk2 = k2;
    paramk3 = k3;
    paramk4 = k4;
  }
  float f1() { return paramf1; }
  float f2() { return paramf2; }
  float ox() { return paramox; }
  float oy() { return paramoy; }
  float k1() { return paramk1; }
  float k2() { return paramk2; } 
  float k3() { return paramk3; } 
  float k4() { return paramk4; }
};
///\brief Find natural image features (SIFT keypoints) and descriptors.
class DECLSPEC featureTrack {
  private :
    GLuint fb[10],depth_rb;
    GLuint inputfb, testTex; 

    //orientation/feature vector resources
    GLuint occlusionQueries;
    GLuint pixelCount, gaussian16x16tex;
    GLuint orientationfb, orientationQuadTex[5], orientationSumTex; 
    GLuint orientationTex, orientationfb2;
    GLuint featureCoordsLUTTex;

    GLuint featurefb, featureTex[16];
    float *featureCoordsBuf, *featureCoordsTex;
    unsigned char *stencilbuf;
    float *orientationsBuf;
    GLuint featureWorkBufs[4], featureWorkTex[16];
	
	//to take the place of pBuffer
	GLuint descTex;

    /* vars for Cg */
    CGcontext cgContext;
    CGprogram undistortProgram, gaussianderivXProgram, magdirProgram,
              gaussianderivYProgram,
              cannysearchProgram, dxdyProgram, gaussianXProgram, gaussianYProgram,
              derandomProgram, decideProgram, basicProgram;

    /* Cg for feature vector computations */
    CGprogram passredProgram, orientationProgram, featureProgram, sum4texProgram;

    CGprofile cgProfile;
    
 
    int numCorners; // = 0;
    float featureBuf0[16*640*4];

//    float cannythresh[4] = {2.0, 4.0, 4.0, 4.0};
    float cannythresh[4] ;
    float derandomthresh[4];// = {0.5, 0.25, 0.25, 0.25};
    float cornerthresh[4];// = {11.0, 0.25, 0.25, 0.25};


    unsigned char rbbuf[640*480*4];

    int Width;
    int Height;
    float depth;

    static GLenum errCode;
    const GLubyte *errString;

    void reshape(int,int);
    void reshapeDescFBO(int,int);
    void drawQuadTex();
    void drawQuadFBT();
    void makeLUT(unsigned char *f);
    CGprogram load_cgprogram(CGprofile prof, char *name);
    void init_cg();
    void locateFeatures( GLuint texIn );
    void makeLookupTexture();
	void makeLookupTextureNoStencil();
    void calcOrientations();
    void calcFeatures();
    void createScene( int num, float *coords, float *orients, float *buf, Scene &s );
    void render_redirect(GLuint texIn);

  public : 
    ///\brief OpenGL textures which hold intermediate results.  Can be 
    /// used to view intermediate processing.
    GLuint tex[10];   

	
    ///\brief Construct a feature tracking object that will operate on
    /// input images of a particular width and height 
	featureTrack(int width,   
                 int height); 

    ///\brief Find all the features and their SIFT descriptors, from a 
    /// image in a texture.  Store them in a Scene structure, which contains a 
    /// vector of Features.  Maximum number of features recordable is
    /// equal to the height of the frame, to be extended in future releases.
    void getScene( GLuint texIn, Scene &s );
    ///Get the width of images upon which the feature track object is set to operate
    ///@return The pixel width of the current images.
    ///\brief get the pixel width of the images.
    int width() { return Width; } 
    ///Get the height of images upon which the feature track object is set to operate
    ///@return The pixel height of the current images.
    ///\brief Get the pixel height of the images.
    int height() { return Height; }

    ///\brief Set paramters for radial distortion correction.
    void setCamParams(CamParams cp);
};


Matches DECLSPEC findMatches( Scene &s0,
                      Scene &s1,
                      float magRadius ,  
                      float improvement  ) ;
double DECLSPEC distSq( Coords &c0, Coords &c1 ) ;
Matches DECLSPEC &findMatchesByPosition( Scene &s0,
                                Scene &s1,
                                float radius,
                                PCT P, 
                                int width, 
                                int height );

class DECLSPEC Recorder {
    private :
        int num;
        double sum;
    public :    
        double addSample(double n ) { 
            num++;
            sum+= n; 
            return sum;
        }
        Recorder() {
            num = 0;
            sum = 0;
        }
        double mean() {
            return sum/(double)num;
        }
        void reset() {
            num = 0;
            sum = 0;
        }
        int numSamples() {
            return num;
        }
        double getSum() {
            return sum;
        }
};


#ifndef WIN32
#include <openvidia/openvidia_lnx.h>
#endif

#endif
