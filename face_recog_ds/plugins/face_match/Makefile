# Build libface_match.so — custom DeepStream classifier parser
# qui matche le tensor 512-D ArcFace contre face-index.npz via cuBLAS.

CUDA_VER       ?= 12.6
CUDA_PATH      ?= /usr/local/cuda-$(CUDA_VER)
DEEPSTREAM     ?= /opt/nvidia/deepstream/deepstream

CXX            := g++
CXXFLAGS       := -std=c++17 -fPIC -O2 -Wall -Wextra
INCLUDES       := -I$(DEEPSTREAM)/sources/includes \
                  -I$(CUDA_PATH)/include
LDFLAGS        := -shared
LIBS           := -L$(CUDA_PATH)/lib64 -lcudart -lcublas

TARGET         := libface_match.so
SRC            := face_match.cpp
OBJ            := $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(TARGET) $(OBJ)

.PHONY: all clean
