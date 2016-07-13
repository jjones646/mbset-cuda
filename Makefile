CC=/usr/local/cuda/bin/nvcc
LIBS= -lglut -lGL -lGLU
INCLUDES=-I/usr/include -I/usr/local/cuda/include
OBJECTS= mbset.o
CCFLAGS=

#######################################

all: mbset

mbset: $(OBJECTS)
	$(CC) -o mbset $(CCFLAGS) $(INCLUDES) $(OBJECTS) $(LIBS) 

mbset.o: mbset.cu
	$(CC) $(CCFLAGS) $(INCLUDES) -c mbset.cu

clean:
	rm -f mbset $(OBJECTS)
