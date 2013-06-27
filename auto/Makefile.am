ACLOCAL_AMFLAGS		= -I m4/libtool -I m4
AUTOMAKE_OPTIONS	= -Wall -Wno-portability

lib_LTLIBRARIES		= # empty
bin_PROGRAMS		= # empty
noinst_PROGRAMS		= # empty
pkgdata_DATA		= # empty
include_HEADERS		= # empty
noinst_HEADERS		= # empty

BUILT_SOURCES		= # empty

EXTRA_DIST		= COPYING.AGPL

CLEANFILES		= # empty
DISTCLEANFILES		= # empty
MAINTAINERCLEANFILES	= # empty

AM_CPPFLAGS		= # empty
AM_CXXFLAGS		= # empty
AM_LDFLAGS		= -Wl,-z,defs

AM_CPPFLAGS		+= -I$(srcdir)/src
AM_CPPFLAGS		+= -I$(srcdir)/src/cudpp
AM_CPPFLAGS		+= -I$(srcdir)/src/cutil
AM_CPPFLAGS		+= -I$(srcdir)/src/ticpp
AM_CPPFLAGS		+= -DTIXML_USE_TICPP
AM_CPPFLAGS		+= -DSRCDIR='"$(abs_srcdir)"'

OPTFLAGS		= -pipe -O0 -ggdb3 -fno-inline
WARNINGS		= # empty
WARNINGS		+= -isystem /usr/include/qt4/QtCore
WARNINGS		+= -isystem /usr/include/qt4/QtGui
WARNINGS		+= -isystem /usr/include/qt4
WARNINGS		+= -isystem /opt/qtsdk-2010.05/qt/include/QtCore
WARNINGS		+= -isystem /opt/qtsdk-2010.05/qt/include/QtGui
WARNINGS		+= -isystem /opt/qtsdk-2010.05/qt/include
WARNINGS		+= -isystem $(builddir)
WARNINGS		+= -Wall -Wextra -pedantic
WARNINGS		+= -Wno-unused-parameter
#WARNINGS		+= -Wpadded -Wno-error=padded
#WARNINGS		+= -Wpacked
WARNINGS		+= -Wconversion
WARNINGS		+= -Weffc++
WARNINGS		+= -Wstrict-overflow=4
WARNINGS		+= -Wstrict-aliasing=3
#WARNINGS		+= -Werror
WARNINGS		+= -w

AM_CXXFLAGS		+= $(OPTFLAGS) $(WARNINGS) #-std=c++0x

# QtCore
AM_CXXFLAGS		+= $(QTCORE_CXXFLAGS)
AM_LDFLAGS		+= $(QTCORE_LIBS)

include res/Rules.am
include src/Rules.am
include rules.am
