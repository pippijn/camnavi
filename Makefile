PKGS = QtCore QtNetwork QtXml ImageMagick++ fftw3 opencv x11 gl glew IL

CXXFLAGS	:= -MD -fPIC -O0 -mfpmath=sse -march=native -fno-inline -Wall -ggdb3
CFLAGS		:= -MD -fPIC -O3 -mfpmath=sse -march=native -fno-inline -ggdb3
CPPFLAGS	:= -Isrc -Isrc/gpu/cudpp -Isrc/gpu/common
CPPFLAGS	+= -DTIXML_USE_STL -DTIXML_USE_TICPP
CPPFLAGS	+= -DCL_SIFTGPU_ENABLED
CPPFLAGS	+= -DCUDA_SIFTGPU_ENABLED
CPPFLAGS	+= -DSRCDIR='"$(PWD)"'
PKGFLAGS	+= $(shell pkg-config $(PKGS) --cflags)

LDFLAGS		:= -lpthread -lboost_system-mt -lboost_thread-mt -lboost_date_time-mt -lopencv_gpu -lcudart -lOpenCL
LDFLAGS		+= $(shell pkg-config $(PKGS) --libs)

LIBS		:=

MOC_SOURCES :=						\
	src/rec/robotino/com/ComImpl.moc.cpp		\
	src/rec/robotino/com/v1/Com.moc.cpp		\
	src/rec/robotino/com/v1/ImageServer.moc.cpp	\
	src/rec/robotino/imagesender/Manager.moc.cpp	\
	src/rec/robotino/imagesender/Sender.moc.cpp	\

SOURCES := $(shell find src -name "*.cpp" -or -name "*.cxx" -or -name "*.c" -or -name "*.cu") $(MOC_SOURCES)
OBJECTS := $(addsuffix .o,$(basename $(SOURCES)))

V_MOC	= $(V_MOC_$(V))
V_CC	= $(V_CC_$(V))
V_CPP	= $(V_CPP_$(V))
V_CXX	= $(V_CXX_$(V))
V_CU	= $(V_CU_$(V))
V_LD	= $(V_LD_$(V))

V_MOC_	= @echo "   MOC   " $@;
V_CC_	= @echo "   CC    " $@;
V_CPP_	= @echo "   CXX   " $@;
V_CXX_	= @echo "   CXX0X " $@;
V_CU_	= @echo "   CUDA  " $@;
V_LD_	= @echo "   LD    " $@;

PROGRAMS = $(patsubst programs/%.cpp,bin/%,$(wildcard programs/*.cpp))

CUSTOMLIBS = $(shell head -n1 ${<:.o=.cpp} | grep '^//>' | cut -b1,2,3 --complement)

programs: $(PROGRAMS)

run: programs
	bin/client #|| $(MAKE)

bin/%: programs/%.o bin/libnavi.so
	$(V_LD)$(LINK.cpp) $< -o $@ -lnavi -Lbin -Wl,-rpath,$(PWD)/bin $(LIBS) $(CUSTOMLIBS)

bin/libnavi.so: $(OBJECTS)
	$(V_LD)$(LINK.cpp) -shared -o $@ $^

clean:
	$(RM) $(wildcard $(OBJECTS) $(OBJECTS:.o=.d) wildcard bin/*)

%.moc.cpp: %.h
	$(V_MOC)moc $< -o $@

%.moc.cpp: %.hh
	$(V_MOC)moc $< -o $@

%.o: %.c
	$(V_CC)$(COMPILE.c) $(PKGFLAGS) $< -o $@

%.o: %.cpp
	$(V_CPP)$(COMPILE.cpp) $(PKGFLAGS) $< -o $@ -std=c++98

%.o: %.cxx
	$(V_CXX)$(COMPILE.cpp) $(PKGFLAGS) $< -o $@ -std=c++0x

%.o: %.cu
	$(V_CU)nvcc $(CPPFLAGS) -c $< -o $@ -O3 -Xcompiler ',"-g","-fno-strict-aliasing","-fPIC"' -arch=sm_11

-include $(shell find src programs -name "*.d")
