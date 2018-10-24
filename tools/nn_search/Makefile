PROJECT := topk

# BUILD_DIR = ./build

INCLUDE_DIRS :=
LIBRARY_DIRS :=

LIBRARIES += boost_system boost_thread stdc++ pthread
CXX = g++

# COMMON_FLAGS += -g -O0
COMMON_FLAGS += -O3
CXXFLAGS += -MMD -MP
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS)
LINKFLAGS += -pthread -fPIC $(COMMON_FLAGS)
LDFLAGS += $(foreach library,$(LIBRARIES),-l$(library))

DYNAMIC_NAME := libtopk.so

SRCS := $(shell find -maxdepth 1 -name "*.cpp")
OBJS := $(SRCS:.cpp=.o)

# DEPS := ${OBJS:.o=.d}
all: $(DYNAMIC_NAME)

$(DYNAMIC_NAME): $(OBJS)
	$(CXX) -shared -o $@ $(OBJS) $(COMMON_FLAGS) $(LINKFLAGS) $(LDFLAGS)
	rm *.o
	rm *.d

-include $(DEPS)

clean:
	rm -rf *.o
	rm -rf *.d
	rm -rf *.so
