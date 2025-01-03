# makefile for SerializationTest
# testing boost library

# need to use llvm clang compiler instead of apple clang or g++ in order to get both openmp and mpi to work
# so brew install llvm, then brew info llvm to set the path to the compiler
# then install sundials and boost using brew
# that should also compile open-mpi and libomp
CPP = mpicxx
#CPP = g++

# compiler flags:
# -Wall turns on most compiler warning
# WARNING = -w
WARNING = -Wall
CFLAGS = $(WARNING) -std=c++17 -fopenmp -MMD -MP -O3 -DNDEBUG

# linker flags
LFLAGS = -std=c++17

projectDir :=$(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../../)


boostIncludeDir = /opt/homebrew/Cellar/boost/1.87.0/include
cvodeIncludeDir = /opt/homebrew/Cellar/sundials/7.2.1/include

boostLibDir = /opt/homebrew/Cellar/boost/1.87.0/lib
cvodeLibDir = /opt/homebrew/Cellar/sundials/7.2.1/lib

libompDir = /opt/homebrew/opt/libomp/lib

INCLUDES = -I$(boostIncludeDir) -I$(cvodeIncludeDir) -I"$(projectDir)"
LIBDIR = -L$(libompDir) -L$(boostLibDir) -L$(cvodeLibDir)

LIBS = -lboost_serialization -lboost_system -lboost_filesystem\
	 -lboost_date_time -lboost_program_options\
	 -lsundials_core -lsundials_cvode -lsundials_nvecserial
# dynamicLibs = 
# LIBS = $(staticLibs)  #-Wl, $(dynamicLibs)

# execuatble file
MAIN = nsclc_sim_single 

# list of source files for dependency generation
CPPFILES = $(shell find ../ -name "*.cpp")
CPPFILES += $(shell find ../../SP_QSP_NSCLC/ -name "*.cpp")
CPPFILES += $(shell find ../../../BioFVM/ -name "*.cpp")
CPPFILES += $(shell find ../../../SP_QSP_shared/ -name "*.cpp")
CPPFILES:=$(filter-out ../GenericSolidTumor/*, $(CPPFILES))

SRCS = $(CPPFILES)

OBJS = $(SRCS:.cpp=.o)
DEPS = $(SRCS:.cpp=.d)

#
# The following part of the makefile is generic; it can be used to 
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#

$(MAIN): $(OBJS) 
	$(CPP) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBDIR) $(LIBS)

%.o: %.cpp
	$(CPP) $(CFLAGS) $(INCLUDES) -c $<  -o $@

profile: CFLAGS += -pg 
profile: clean
profile: $(MAIN)

.PHONY: clean

clean:
	$(RM) $(OBJS) $(DEPS) $(MAIN)

-include $(DEPS)
# DO NOT DELETE THIS LINE -- make depend needs it
