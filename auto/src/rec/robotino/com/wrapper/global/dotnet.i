%typemap(csbase) rec::robotino::com::Exception "System.Exception";

%typemap(ctype) (const unsigned char* data) "const unsigned char*"
%typemap(imtype) (const unsigned char* data) "System.IntPtr"
%typemap(cstype) (const unsigned char* data) "System.Drawing.Image"

%typemap(directorin,descriptor="LSystem/Drawing/Image;") (const unsigned char* data)
%{
    $input = $1;
%}

%typemap(csdirectorin) (const unsigned char* data)
%{
  ImageLoader.loadImage($iminput, dataSize, width, height, numChannels, bitsPerChannel, step)
%}

%typemap(in) (const unsigned char* data) 
%{
  $1 = 0;
%}

%typemap(csin) (const unsigned char* data)
%{
  new System.IntPtr()
%}