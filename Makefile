PKGS = QtCore QtNetwork QtXml ImageMagick++ fftw3 gtk+-2.0 opencv

CXXFLAGS	:= -Wall -ggdb3
CFLAGS		:= -O3 -ggdb3 -fno-inline
INCLUDES	:= -Isrc/gpu/cudpp -Isrc/gpu/common
CPPFLAGS	:= -MD $(shell pkg-config $(PKGS) --cflags) $(INCLUDES)
CPPFLAGS	+= -DTIXML_USE_STL -DTIXML_USE_TICPP

LDFLAGS		:= -lpthread -lboost_system-mt -lboost_thread-mt -lboost_date_time-mt -lopencv_gpu -lcudart
LDFLAGS		+= $(shell pkg-config $(PKGS) --libs)

MOC_SOURCES :=						\
	src/rec/robotino/com/ComImpl.moc.cpp		\
	src/rec/robotino/com/v1/Com.moc.cpp		\
	src/rec/robotino/com/v1/ImageServer.moc.cpp	\
	src/rec/robotino/imagesender/Manager.moc.cpp	\
	src/rec/robotino/imagesender/Sender.moc.cpp	\

SOURCES := $(shell find src -name "*.cpp" -or -name "*.cxx" -or -name "*.c" -or -name "*.cu") $(MOC_SOURCES)
OBJECTS := $(addsuffix .o,$(basename $(SOURCES)))

all: client
	./$< #|| $(MAKE)

client: $(OBJECTS)
	$(LINK.cpp) $^ -o $@ $(LIBS)

clean:
	$(RM) client $(OBJECTS) $(OBJECTS:.o=.d)

%.moc.cpp: %.h
	moc $< -o $@

%.moc.cpp: %.hh
	moc $< -o $@

%.o: %.cpp
	$(COMPILE.cpp) $< -o $@ -std=c++98

%.o: %.cxx
	$(COMPILE.cpp) $< -o $@ -std=c++0x

%.o: %.cu
	nvcc $(INCLUDES) -c $< -o $@ -O3 -Xcompiler ',"-g","-fno-strict-aliasing"' -arch=sm_11

-include $(shell find src -name "*.d")
