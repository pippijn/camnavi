PKGS = QtCore QtNetwork QtXml ImageMagick++ fftw3 gtk+-2.0 opencv

CXXFLAGS	:= -Wall -ggdb3
CFLAGL		:= -O3
CPPFLAGS	:= -MD -Isrc $(shell pkg-config $(PKGS) --cflags)
CPPFLAGS	+= -DTIXML_USE_STL -DTIXML_USE_TICPP

LDFLAGS		:= -lpthread -lboost_system-mt -lboost_thread-mt -lboost_date_time-mt -lopencv_gpu
LDFLAGS		+= $(shell pkg-config $(PKGS) --libs)

MOC_SOURCES :=						\
	src/rec/robotino/com/ComImpl.moc.cpp		\
	src/rec/robotino/com/v1/Com.moc.cpp		\
	src/rec/robotino/com/v1/ImageServer.moc.cpp	\
	src/rec/robotino/imagesender/Manager.moc.cpp	\
	src/rec/robotino/imagesender/Sender.moc.cpp	\

SOURCES := $(shell find src -name "*.cpp" -or -name "*.c") $(MOC_SOURCES)
OBJECTS := $(addsuffix .o,$(basename $(SOURCES)))

all: client
	./$< #|| $(MAKE)

client: $(OBJECTS)
	$(LINK.cpp) $^ -o $@

clean:
	$(RM) client $(OBJECTS) $(OBJECTS:.o=.d)

%.moc.cpp: %.h
	moc $< -o $@

%.moc.cpp: %.hh
	moc $< -o $@

-include $(shell find src -name "*.d")
