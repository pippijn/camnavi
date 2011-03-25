%typemap(javacode) SWIGTYPE
%{
  static
  {
    RobotinoLibraryLoader.loadLibraries();
  }
%}

%typemap(jni) (const unsigned char* data, unsigned int dataSize) "jbyteArray"
%typemap(jtype) (const unsigned char* data, unsigned int dataSize) "byte[]"
%typemap(jstype) (const unsigned char* data, unsigned int dataSize) "java.awt.Image"

%typemap(directorin,descriptor="Ljava/awt/Image;") (const unsigned char* data, unsigned int dataSize)
%{
    jbyteArray jbuffer = jenv->NewByteArray($2);
    jenv->SetByteArrayRegion(jbuffer, 0, $2, (jbyte*) $1);
    $input = jbuffer;
%}

%typemap(javadirectorin) (const unsigned char* data, unsigned int dataSize)
%{
  ImageLoader.loadImage( $jniinput, width, height, numChannels, bitsPerChannel, step)
%}

%typemap(in) (const unsigned char* data, unsigned int dataSize) 
%{
  $1 = 0;
%}

%typemap(javain) (const unsigned char* data, unsigned int dataSize)
%{
  null
%}
