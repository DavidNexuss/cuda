GCC 	     = gcc
NVCC         = $(CUDA_HOME)/bin/nvcc
INCLUDE_DIRS = -I$(CUDA_HOME)/include -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc -I include -I src -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib -I/opt/cuda/include

NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -gencode arch=compute_86,code=sm_86 --ptxas-options=-v $(INCLUDE_DIRS)
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 $(INCLUDE_DIRS)

GCC_FLAGS   = -O3 $(INCLUDE_DIRS)

BIN = bin
ODIR = obj
IDIR = src
SDIR = as

OUT = tracer

C_SOURCES = $(shell find $(IDIR) -type f -name *.c -printf "%f\n")
CUDA_SOURCES = $(shell find $(IDIR) -type f -name *.cu -printf "%f\n")

OBJECTS = $(patsubst %.c, $(ODIR)/%.o,$(C_SOURCES))
CUDA_OBJECTS = $(patsubst %.cu, $(ODIR)/%.o,$(CUDA_SOURCES))

all : $(OUT) $(ODIR) $(BIN)

$(ODIR)/%.o : $(IDIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@

$(ODIR)/%.o : $(IDIR)/**/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@

$(ODIR)/%.o : $(IDIR)/%.c
	$(GCC) $(GCC_FLAGS) -c $^ -o $@

$(ODIR)/%.o : $(IDIR)/**/%.c
	$(GCC) $(GCC_FLAGS) -c $^ -o $@

$(OUT): $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $^ -o $@ 


clean: $(ODIR) $(BIN)
	rm -rf $(ODIR)
	rm -rf $(BIN)
	rm -rf $(SDIR)

$(ODIR):
	mkdir -p $(ODIR)
$(BIN):
	mkdir -p $(BIN)
$(IDIR):
	mkdir -p $(IDIR)
$(SDIR):
	mkdir -p $(SDIR)
