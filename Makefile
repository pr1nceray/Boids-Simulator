TARGET  :=  -o Boids-Simulator
CC      := nvcc

SOURCE_DIR := Simulator
CUH_SOURCES := $(wildcard $(addsuffix /*.cuh, $(SOURCE_DIR)))
CU_SOURCES := $(wildcard $(addsuffix /*.cu, $(SOURCE_DIR)))
FILES := $(CUH_SOURCES) $(CU_SOURCES)

#CFLAGS  := -std=c++20 -pedantic -Wall -Wextra -Wshadow -Wwrite-strings -O3
LDFLAGS := -lsfml-graphics -lsfml-window -lsfml-system -L./lib -lsfml-widgets -lGL

STYLE_CHECKER := cpplint
STYLE_HEADERS := --headers=cuh,h,hpp
STYLE_EXTENSIONS := --extensions=c,c++,cc,cpp,cu,cuh

all: compile
# Compile
compile:
	@echo "Compiling"
	$(CC) $(TARGET) $(CU_SOURCES) $(LDFLAGS) 

#Style Check
style:
	@echo "Style Checking"
	$(STYLE_CHECKER) $(STYLE_HEADERS) $(STYLE_EXTENSIONS) $(FILES)

clean:
	rm -fr *.o